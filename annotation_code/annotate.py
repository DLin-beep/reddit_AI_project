#!/usr/bin/env python3
"""Score blinded annotation posts with a chat model behind the UNC AI gateway.

Reads the blinded post file produced by prepare_annotation.py, sends one post
per request, validates each response against the frozen ACSI schema, and
appends results to a JSONL file. The output is append-only so an interrupted
run resumes without rescoring anything.

The API key is read from the environment variable named in the config
(default UNC_AI_API_KEY). It is never written to any output file.

Usage:
    export UNC_AI_API_KEY='...'
    python annotate.py --config sampling_config_creation_relative.json
    python annotate.py --config ... --run 2 --only-scored-in 1   # self-consistency
    python annotate.py --config ... --limit 20 --dry-run         # inspect prompts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# This module lives in annotation_code/ but pipeline_utils sits at the repo
# root, so make the parent importable whether the script is run from here or
# from the root via the .sh wrapper.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline_utils import load_config, resolve_path, sha256_file, sha256_text, write_json

SCORE_FIELDS = (
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
)
FLAG_FIELDS = ("ai_related_flag",)
JSON_FIELDS = ("post_id", *SCORE_FIELDS, *FLAG_FIELDS, "rationale")

RESPONSE_INSTRUCTION = (
    "You are coding a single Reddit post with the rubric above.\n"
    "Return exactly one JSON object and nothing else. No prose, no code fence.\n"
    'Use the post_id given in the input as the "post_id" value.\n'
    "Required keys, exactly these and no others: "
    + ", ".join(JSON_FIELDS)
    + ".\n"
    + "The five score fields are integers 0-3, ai_related_flag is integer 0 or 1, "
    "and rationale is one specific evidence-based sentence."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sampling_config_creation_relative.json")
    parser.add_argument("--run", type=int, default=1, help="Scoring pass number; separate output per run")
    parser.add_argument("--limit", type=int, default=None, help="Score at most this many posts")
    parser.add_argument("--only-scored-in", type=int, default=None,
                        help="Restrict to posts already scored in this run (for self-consistency subsets)")
    parser.add_argument("--only-ids", default=None,
                        help="CSV with an annotation_id column; score only those posts "
                             "(use the subsets from select_subsets.py)")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and exit without calling the API")
    return parser.parse_args()


def build_system_prompt(rubric_text: str) -> str:
    return f"{rubric_text.strip()}\n\n---\n\n{RESPONSE_INSTRUCTION}"


def build_user_prompt(row: dict[str, str]) -> str:
    return f"post_id: {row['annotation_id']}\nyear_month: {row['year_month']}\ntext: {row['text']}"


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("No JSON object in model response")
        return json.loads(text[start : end + 1])


def validate_record(record: Any, expected_id: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Model response is not a JSON object")
    keys = set(record)
    expected = set(JSON_FIELDS)
    if keys != expected:
        raise ValueError(f"Wrong keys: missing={sorted(expected - keys)} extra={sorted(keys - expected)}")
    if str(record["post_id"]) != expected_id:
        raise ValueError(f"post_id mismatch: expected {expected_id}, got {record['post_id']}")
    for field in SCORE_FIELDS:
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1, 2, 3}:
            raise ValueError(f"{field} must be integer 0-3, got {value!r}")
    for field in FLAG_FIELDS:
        value = record[field]
        if not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1}:
            raise ValueError(f"{field} must be integer 0 or 1, got {value!r}")
    rationale = str(record["rationale"]).strip()
    if not rationale:
        raise ValueError("rationale cannot be blank")
    record["post_id"] = str(record["post_id"])
    record["rationale"] = rationale
    return record


class GatewayClient:
    """Chat-completions client with bounded retries and 429 handling."""

    def __init__(self, api: dict[str, Any], api_key: str) -> None:
        self.url = api["endpoint"]
        self.model = api["model"]
        # GPT-5.5 rejects temperature and top_p outright, so both are optional:
        # a null in the config means the parameter is omitted from the request.
        self.temperature = api.get("temperature")
        self.top_p = api.get("top_p")
        self.max_completion_tokens = api["max_completion_tokens"]
        self.timeout = api["request_timeout_seconds"]
        self.max_attempts = api["max_attempts"]
        self.backoff_base = api["retry_backoff_seconds"]
        self.key_header = api.get("api_key_header", "api-key")
        self.api_key = api_key
        self.use_json_mode = bool(api.get("use_json_mode", True))
        self.observed_model = ""
        self._json_mode_lock = threading.Lock()

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_completion_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _disable_json_mode(self) -> bool:
        with self._json_mode_lock:
            if not self.use_json_mode:
                return False
            self.use_json_mode = False
            print("Gateway rejected response_format; continuing without JSON mode", file=sys.stderr)
            return True

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            body = json.dumps(self._payload(system_prompt, user_prompt)).encode("utf-8")
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache",
                    self.key_header: self.api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                self.observed_model = parsed.get("model", "")
                choice = parsed["choices"][0]
                finish_reason = choice.get("finish_reason")
                content = choice["message"]["content"]
                if finish_reason == "content_filter":
                    raise ValueError("Content filter blocked the response")
                if finish_reason == "length":
                    raise ValueError(
                        "Response hit max_completion_tokens before finishing; "
                        "raise api.max_completion_tokens"
                    )
                if not content:
                    raise ValueError(f"Empty content (finish_reason={finish_reason})")
                return content, parsed.get("usage", {})
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")[:400]
                last_error = f"HTTP {error.code}: {detail}"
                if error.code == 400 and "response_format" in detail and self._disable_json_mode():
                    continue
                if error.code in {408, 409, 429} or error.code >= 500:
                    retry_after = error.headers.get("Retry-After") if error.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else None
                else:
                    raise RuntimeError(last_error) from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, ValueError) as error:
                last_error = f"{type(error).__name__}: {error}"
                delay = None
            if attempt == self.max_attempts:
                break
            wait = delay if delay is not None else self.backoff_base * (2 ** (attempt - 1))
            time.sleep(wait + random.uniform(0, 0.5))
        raise RuntimeError(f"Failed after {self.max_attempts} attempts. Last error: {last_error}")


def read_blinded_posts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    salvaged: list[str] = []
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["post_id"])
        except (json.JSONDecodeError, KeyError):
            # A run killed mid-write can leave one truncated trailing line.
            # Drop it and rewrite the file so the post is simply rescored;
            # anything malformed elsewhere is a real corruption and must stop.
            if number != len(lines):
                raise ValueError(f"Malformed JSON at line {number} of {path}")
            print(f"Discarding truncated final line of {path.name}", file=sys.stderr)
            salvaged = lines[: number - 1]
    if salvaged:
        path.write_text("".join(salvaged), encoding="utf-8")
    return done


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    api = config.get("api")
    if not api:
        raise ValueError(f"Add an 'api' block to {args.config} before scoring")

    run_dir = resolve_path(base_dir, config["paths"]["run_dir"])
    prep_dir = run_dir / "05_annotation_prep"
    input_path = prep_dir / "annotation_posts_blinded.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Run prepare_annotation.py first.")
    output_dir = run_dir / "06_annotation_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / f"scores_run{args.run}.jsonl"
    failures_path = output_dir / f"failures_run{args.run}.csv"

    rubric_path = resolve_path(base_dir, api["rubric_file"])
    system_prompt = build_system_prompt(rubric_path.read_text(encoding="utf-8"))

    posts = read_blinded_posts(input_path)
    if args.only_ids:
        with Path(args.only_ids).open(newline="", encoding="utf-8") as handle:
            wanted = {row["annotation_id"] for row in csv.DictReader(handle)}
        posts = [row for row in posts if row["annotation_id"] in wanted]
        if len(posts) != len(wanted):
            raise ValueError(f"{len(wanted) - len(posts)} ids in {args.only_ids} are not in the post file")
    if args.only_scored_in is not None:
        subset = completed_ids(output_dir / f"scores_run{args.only_scored_in}.jsonl")
        posts = [row for row in posts if row["annotation_id"] in subset]
    already = completed_ids(scores_path)
    pending = [row for row in posts if row["annotation_id"] not in already]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"run={args.run} input={len(posts)} already_scored={len(already)} pending={len(pending)}")
    if args.dry_run:
        if pending:
            print("\n--- SYSTEM PROMPT ---\n" + system_prompt)
            print("\n--- USER PROMPT (first pending post) ---\n" + build_user_prompt(pending[0]))
        print(f"\nprompt_sha256={sha256_text(system_prompt)}")
        return
    if not pending:
        print("Nothing to score.")
        return

    api_key = os.environ.get(api["api_key_env"], "").strip()
    if not api_key:
        raise ValueError(f"Set the {api['api_key_env']} environment variable")

    client = GatewayClient(api, api_key)
    work: queue.Queue[dict[str, str] | None] = queue.Queue()
    for row in pending:
        work.put(row)
    write_lock = threading.Lock()
    counters = {
        "ok": 0,
        "failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "excluded": 0,
        "low_confidence": 0,
    }
    failures: list[dict[str, str]] = []
    started = time.time()

    def worker() -> None:
        while True:
            try:
                row = work.get_nowait()
            except queue.Empty:
                return
            annotation_id = row["annotation_id"]
            try:
                content, usage = client.complete(system_prompt, build_user_prompt(row))
                record = validate_record(extract_json_object(content), annotation_id)
            except Exception as error:  # noqa: BLE001 - every failure is logged, never fatal
                with write_lock:
                    counters["failed"] += 1
                    failures.append(
                        {
                            "annotation_id": annotation_id,
                            "error_type": type(error).__name__,
                            "error": str(error)[:500],
                        }
                    )
                continue
            record["year_month"] = row["year_month"]
            rationale = record["rationale"].strip().upper()
            with write_lock:
                if rationale.startswith("EXCLUDE"):
                    counters["excluded"] += 1
                elif "CONFIDENCE IS LOW" in rationale or "LOW CONFIDENCE" in rationale:
                    counters["low_confidence"] += 1
                with scores_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                counters["ok"] += 1
                counters["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                counters["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                counters["reasoning_tokens"] += int(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                )
                counters["cached_tokens"] += int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
                )
                done = counters["ok"] + counters["failed"]
                if done % 100 == 0 or done == len(pending):
                    rate = done / max(time.time() - started, 1e-9)
                    remaining = (len(pending) - done) / rate if rate else 0
                    print(
                        f"{done}/{len(pending)} ok={counters['ok']} failed={counters['failed']} "
                        f"{rate:.1f}/s eta={remaining / 60:.0f}m",
                        flush=True,
                    )

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(int(api["concurrency"]))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with failures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["annotation_id", "error_type", "error"])
        writer.writeheader()
        writer.writerows(failures)

    write_json(
        output_dir / f"scoring_summary_run{args.run}.json",
        {
            "status": "complete" if not failures else "complete_with_failures",
            "run": args.run,
            "model_requested": api["model"],
            "model_snapshot_returned": client.observed_model,
            "endpoint": api["endpoint"],
            "temperature": api.get("temperature"),
            "top_p": api.get("top_p"),
            "max_completion_tokens": api["max_completion_tokens"],
            "json_mode_used": client.use_json_mode,
            "concurrency": api["concurrency"],
            "max_attempts": api["max_attempts"],
            "posts_attempted": len(pending),
            "posts_scored": counters["ok"],
            "posts_failed": counters["failed"],
            "posts_marked_exclude": counters["excluded"],
            "posts_low_confidence": counters["low_confidence"],
            "total_scored_in_run": len(completed_ids(scores_path)),
            "prompt_tokens": counters["prompt_tokens"],
            "completion_tokens": counters["completion_tokens"],
            "reasoning_tokens": counters["reasoning_tokens"],
            "cached_prompt_tokens": counters["cached_tokens"],
            "elapsed_seconds": round(time.time() - started, 1),
            "rubric_file": str(rubric_path.name),
            "rubric_sha256": sha256_file(rubric_path),
            "system_prompt_sha256": sha256_text(system_prompt),
            "input_file_sha256": sha256_file(input_path),
        },
    )
    print(f"Done: scored={counters['ok']} failed={counters['failed']} -> {scores_path}")
    if failures:
        print(f"Failures logged to {failures_path}; rerun the same command to retry them.")


if __name__ == "__main__":
    main()
