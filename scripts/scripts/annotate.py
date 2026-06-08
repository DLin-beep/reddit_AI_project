#!/usr/bin/env python3
"""Single command for building and scoring ACSI annotation samples.

This file owns the annotation workflow end to end:
- `build` creates or tops up `data/acsi_data.csv`
- `next`, `append`, `summary`, `combine`, and `aggregate` manage scoring
"""


# ---------------------------------------------------------------------------
# Annotation sample construction.
# ---------------------------------------------------------------------------

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path(__file__).parent.parent.resolve()
BUILD_DEFAULT_INPUT = ROOT / "output" / "latest" / "posts_clean_all.parquet"
BUILD_DEFAULT_OUTPUT = ROOT / "data" / "acsi_data.csv"
BUILD_DEFAULT_RAW_DATA_DIR = ROOT / "data" / "raw_files"
BUILD_START_DATE = datetime(2020, 1, 1)
BUILD_END_DATE_EXCLUSIVE = datetime(2025, 1, 1)
BUILD_SHOCK_MONTH = pd.Timestamp("2022-12-01")
BUILD_PRE_SHOCK_ONLY_START = pd.Timestamp("2022-01-01")
BUILD_PRE_SHOCK_ONLY_END_EXCLUSIVE = pd.Timestamp("2022-11-01")
BUILD_PRE_SHOCK_ONLY_TARGET_ROWS = 100

BUILD_SCORE_COLUMNS = [
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
    "ai_related_flag",
]


def clean_text(value, max_chars):
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    if text in {"[removed]", "[deleted]"}:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def acsi_output_columns():
    return [
        "subreddit",
        "post_id",
        "period",
        "created_date",
        "year_month",
        "title_excerpt",
        "selftext_excerpt",
        *BUILD_SCORE_COLUMNS,
        "rationale",
        "version",
    ]


def allocate_counts(counts, target):
    """Allocate target rows as evenly as possible without exceeding counts."""
    counts = counts.sort_index().astype(int)
    if int(counts.sum()) <= target:
        return counts

    base = target // len(counts)
    alloc = counts.clip(upper=base)
    remaining = int(target - alloc.sum())

    while remaining > 0:
        capacity = counts - alloc
        eligible = capacity[capacity > 0].sort_values(ascending=False)
        if eligible.empty:
            break
        for key in eligible.index:
            if remaining <= 0:
                break
            alloc.loc[key] += 1
            remaining -= 1

    return alloc.astype(int)


def sample_period(df, period, target, seed):
    period_df = df[df["period"].eq(period)].copy()
    counts = period_df.groupby("subreddit").size()
    alloc = allocate_counts(counts, target)

    pieces = []
    for i, (subreddit, n) in enumerate(alloc.items()):
        if n <= 0:
            continue
        group = period_df[period_df["subreddit"].eq(subreddit)]
        pieces.append(group.sample(n=int(n), random_state=seed + i))

    if not pieces:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(pieces, ignore_index=True)


def raw_subreddit_from_path(path):
    name = path.name
    return name[len("r_") : -len("_posts.jsonl")]


def parse_raw_post_line(line, subreddit, existing_post_ids, candidate_post_ids):
    try:
        post = json.loads(line)
    except json.JSONDecodeError:
        return None

    post_id = str(post.get("id") or "")
    if not post_id or post_id in existing_post_ids or post_id in candidate_post_ids:
        return None

    created_utc = post.get("created_utc")
    if created_utc is None:
        return None
    try:
        created_at = datetime.utcfromtimestamp(int(created_utc))
    except (TypeError, ValueError, OSError):
        return None
    if created_at < BUILD_START_DATE or created_at >= BUILD_END_DATE_EXCLUSIVE:
        return None

    title = clean_text(post.get("title"), 280)
    if not title:
        return None

    month = pd.Timestamp(created_at.strftime("%Y-%m-01"))
    period = "post_gpt" if month >= BUILD_SHOCK_MONTH else "pre_gpt"
    return {
        "subreddit": subreddit,
        "post_id": post_id,
        "period": period,
        "created_date": created_at.strftime("%Y-%m-%d"),
        "year_month": created_at.strftime("%Y-%m"),
        "title_excerpt": title,
        "selftext_excerpt": clean_text(post.get("selftext"), 2000),
    }


def add_candidate(row, rows_by_period, max_candidates):
    if row is None:
        return False
    bucket = rows_by_period[row["period"]]
    if len(bucket) >= max_candidates:
        return False
    bucket.append(row)
    return True


def candidate_buckets_full(rows_by_period, max_candidates):
    return all(len(bucket) >= max_candidates for bucket in rows_by_period.values())


def tail_lines(path, max_lines, max_bytes=20_000_000):
    file_size = path.stat().st_size
    if file_size <= 0:
        return []

    with path.open("rb") as handle:
        start = max(0, file_size - max_bytes)
        handle.seek(start)
        if start:
            handle.readline()
        lines = handle.readlines()
    return [line.decode("utf-8", errors="ignore").strip() for line in lines[-max_lines:]]


def raw_post_rows(subreddit, path, existing_post_ids, max_candidates, seed):
    rows_by_period = {"pre_gpt": [], "post_gpt": []}
    candidate_post_ids = set()
    line_limit = max_candidates * 4

    def consume(lines):
        for line in lines:
            if candidate_buckets_full(rows_by_period, max_candidates):
                break
            line = str(line).strip()
            if not line:
                continue
            row = parse_raw_post_line(line, subreddit, existing_post_ids, candidate_post_ids)
            if add_candidate(row, rows_by_period, max_candidates):
                candidate_post_ids.add(row["post_id"])

    with path.open("r", encoding="utf-8") as handle:
        head = []
        for _, line in zip(range(line_limit), handle):
            head.append(line)
    consume(head)
    consume(tail_lines(path, line_limit))

    rows = [*rows_by_period["pre_gpt"], *rows_by_period["post_gpt"]]
    return pd.DataFrame(rows), len(candidate_post_ids)


def created_utc_from_raw_line(line):
    try:
        post = json.loads(line)
        created_utc = post.get("created_utc")
        if created_utc is None:
            return None
        return int(created_utc)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def seek_position_for_created_utc(path, target_created_utc):
    file_size = path.stat().st_size
    if file_size <= 0:
        return 0

    low = 0
    high = file_size
    candidate = 0
    with path.open("rb") as handle:
        for _ in range(40):
            if low >= high:
                break
            mid = (low + high) // 2
            handle.seek(mid)
            if mid:
                handle.readline()
            pos = handle.tell()
            line = handle.readline()
            if not line:
                high = mid
                continue
            created_utc = created_utc_from_raw_line(line)
            if created_utc is None:
                low = handle.tell()
                continue
            if created_utc < target_created_utc:
                low = handle.tell()
            else:
                candidate = pos
                high = mid

    # Step back a little so small local ordering irregularities do not skip
    # eligible rows immediately before the binary-search boundary.
    return max(0, candidate - 1_000_000)


def pre_shock_only_raw_post_rows(subreddit, path, existing_post_ids, max_candidates=None):
    rows = []
    candidate_post_ids = set()
    if not path.exists():
        return pd.DataFrame(columns=acsi_output_columns()), 0

    start_utc = int(BUILD_PRE_SHOCK_ONLY_START.tz_localize("UTC").timestamp())
    start_pos = seek_position_for_created_utc(path, start_utc)
    with path.open("rb") as handle:
        handle.seek(start_pos)
        if start_pos:
            handle.readline()
        for raw_line in handle:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            row = parse_raw_post_line(line, subreddit, existing_post_ids, candidate_post_ids)
            if row is None:
                continue
            created_date = pd.to_datetime(row["created_date"], errors="coerce")
            if pd.isna(created_date):
                continue
            if not (BUILD_PRE_SHOCK_ONLY_START <= created_date < BUILD_PRE_SHOCK_ONLY_END_EXCLUSIVE):
                if created_date >= BUILD_PRE_SHOCK_ONLY_END_EXCLUSIVE:
                    break
                continue
            rows.append(row)
            candidate_post_ids.add(row["post_id"])
            if max_candidates is not None and len(rows) >= max_candidates:
                break

    return pd.DataFrame(rows), len(candidate_post_ids)


def sample_one_subreddit(df, target, seed):
    if len(df) <= target:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pre = df[df["period"].eq("pre_gpt")]
    post = df[df["period"].eq("post_gpt")]
    target_pre = target // 2
    target_post = target - target_pre

    take_pre = min(len(pre), target_pre)
    take_post = min(len(post), target_post)
    spare = target - take_pre - take_post
    if spare > 0:
        pre_capacity = len(pre) - take_pre
        add_pre = min(pre_capacity, spare)
        take_pre += add_pre
        spare -= add_pre
    if spare > 0:
        post_capacity = len(post) - take_post
        take_post += min(post_capacity, spare)

    pieces = []
    if take_pre:
        pieces.append(pre.sample(n=take_pre, random_state=seed))
    if take_post:
        pieces.append(post.sample(n=take_post, random_state=seed + 100_000))
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed + 200_000)


def load_existing_acsi_data(output_csv):
    out_cols = acsi_output_columns()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv, dtype=str).fillna("")
        for col in out_cols:
            if col not in existing.columns:
                existing[col] = ""
        return existing[out_cols].copy()
    return pd.DataFrame(columns=out_cols)


BUILD_SUMMARY_COLUMNS = [
    "subreddit",
    "existing_rows",
    "added_rows",
    "final_rows",
    "target",
    "available_candidates",
]


def count_pre_shock_rows(existing):
    columns = [
        "subreddit",
        "existing_pre_shock_rows",
        "existing_coded_pre_shock_rows",
        "existing_pending_pre_shock_rows",
    ]
    if existing.empty:
        return pd.DataFrame(columns=columns)
    frame = existing.copy()
    frame["created_date_dt"] = pd.to_datetime(frame["created_date"], errors="coerce")
    personal_req = frame["personal_req_score"].astype(str).str.strip()
    frame["is_pre_shock"] = (
        (frame["created_date_dt"] >= BUILD_PRE_SHOCK_ONLY_START)
        & (frame["created_date_dt"] < BUILD_PRE_SHOCK_ONLY_END_EXCLUSIVE)
    )
    frame["is_coded_pre_shock"] = frame["is_pre_shock"] & personal_req.ne("")
    frame["is_pending_pre_shock"] = frame["is_pre_shock"] & personal_req.eq("")
    counts = (
        frame[frame["is_pre_shock"]]
        .groupby("subreddit", as_index=False)
        .agg(
            existing_pre_shock_rows=("post_id", "size"),
            existing_coded_pre_shock_rows=("is_coded_pre_shock", "sum"),
            existing_pending_pre_shock_rows=("is_pending_pre_shock", "sum"),
        )
    )
    for column in columns[1:]:
        counts[column] = counts[column].astype(int)
    return counts[columns]


def top_up_pre_shock_only_acsi_data(output_csv, raw_data_dir, target_rows_per_subreddit, seed):
    out_cols = acsi_output_columns()
    existing = load_existing_acsi_data(output_csv)
    existing_post_ids = set(existing["post_id"].astype(str))
    existing_counts = count_pre_shock_rows(existing).set_index("subreddit")
    additions = []
    summary_rows = []

    target_subreddits = set(existing["subreddit"].astype(str).str.strip())
    raw_paths = [
        path
        for path in sorted(
            raw_data_dir.glob("r_*_posts.jsonl"),
            key=lambda path: raw_subreddit_from_path(path).lower(),
        )
        if raw_subreddit_from_path(path) in target_subreddits
    ]
    for index, path in enumerate(raw_paths):
        subreddit = raw_subreddit_from_path(path)
        if subreddit in existing_counts.index:
            existing_count = int(existing_counts.loc[subreddit, "existing_pre_shock_rows"])
        else:
            existing_count = 0
        needed = max(0, target_rows_per_subreddit - existing_count)
        added = 0
        available_count = 0
        if needed > 0:
            candidates, available_count = pre_shock_only_raw_post_rows(
                subreddit=subreddit,
                path=path,
                existing_post_ids=existing_post_ids,
                max_candidates=needed,
            )
            if not candidates.empty:
                sampled = candidates.sample(
                    n=min(needed, len(candidates)),
                    random_state=seed + index,
                ).reset_index(drop=True)
                for col in BUILD_SCORE_COLUMNS:
                    sampled[col] = ""
                sampled["rationale"] = ""
                sampled["version"] = ""
                additions.append(sampled[out_cols])
                existing_post_ids.update(sampled["post_id"].astype(str))
                added = int(len(sampled))
        summary_rows.append({
            "subreddit": subreddit,
            "existing_rows": existing_count,
            "added_rows": added,
            "final_rows": existing_count + added,
            "target": target_rows_per_subreddit,
            "available_candidates": available_count,
        })

    if additions:
        output = pd.concat([existing, *additions], ignore_index=True)
    else:
        output = existing

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_name(output_csv.name + ".tmp")
    output.to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)

    summary = pd.DataFrame(summary_rows, columns=BUILD_SUMMARY_COLUMNS)
    return {
        "csv": output_csv,
        "raw_data_dir": raw_data_dir,
        "target_pre_shock_rows_per_subreddit": target_rows_per_subreddit,
        "starting_rows": len(existing),
        "added_rows": int(0 if summary.empty else summary["added_rows"].sum()),
        "ending_rows": len(output),
        "subreddits_topped_up": int(0 if summary.empty else summary[summary["added_rows"].gt(0)]["subreddit"].nunique()),
        "summary": summary,
    }


def augment_acsi_data_from_raw(output_csv, raw_data_dir, target_rows_per_subreddit, seed):
    out_cols = acsi_output_columns()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv, dtype=str).fillna("")
        for col in out_cols:
            if col not in existing.columns:
                existing[col] = ""
        existing = existing[out_cols].copy()
    else:
        existing = pd.DataFrame(columns=out_cols)

    existing_post_ids = set(existing["post_id"].astype(str))
    existing_counts = existing.groupby("subreddit").size()
    additions = []
    summary_rows = []

    raw_paths = sorted(raw_data_dir.glob("r_*_posts.jsonl"), key=lambda path: raw_subreddit_from_path(path).lower())
    for index, path in enumerate(raw_paths):
        subreddit = raw_subreddit_from_path(path)
        existing_count = int(existing_counts.get(subreddit, 0))
        needed = max(0, target_rows_per_subreddit - existing_count)
        if needed <= 0:
            continue

        candidates, available_count = raw_post_rows(
            subreddit=subreddit,
            path=path,
            existing_post_ids=existing_post_ids,
            max_candidates=target_rows_per_subreddit,
            seed=seed + index,
        )
        if candidates.empty:
            summary_rows.append(
                {
                    "subreddit": subreddit,
                    "existing_rows": existing_count,
                    "added_rows": 0,
                    "final_rows": existing_count,
                    "target": target_rows_per_subreddit,
                    "available_candidates": available_count,
                }
            )
            continue

        sampled = sample_one_subreddit(candidates, needed, seed + index)
        for col in BUILD_SCORE_COLUMNS:
            sampled[col] = ""
        sampled["rationale"] = ""
        sampled["version"] = ""
        additions.append(sampled[out_cols])
        existing_post_ids.update(sampled["post_id"].astype(str))
        summary_rows.append(
            {
                "subreddit": subreddit,
                "existing_rows": existing_count,
                "added_rows": len(sampled),
                "final_rows": existing_count + len(sampled),
                "target": target_rows_per_subreddit,
                "available_candidates": available_count,
            }
        )

    if additions:
        output = pd.concat([existing, *additions], ignore_index=True)
    else:
        output = existing

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_name(output_csv.name + ".tmp")
    output.to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)

    summary = pd.DataFrame(summary_rows, columns=BUILD_SUMMARY_COLUMNS)
    return {
        "csv": output_csv,
        "raw_data_dir": raw_data_dir,
        "target_rows_per_subreddit": target_rows_per_subreddit,
        "starting_rows": len(existing),
        "added_rows": int(0 if summary.empty else summary["added_rows"].sum()),
        "ending_rows": len(output),
        "subreddits_added_or_topped_up": int(0 if summary.empty else summary[summary["added_rows"].gt(0)]["subreddit"].nunique()),
        "summary": summary,
    }


def build_acsi_data(input_path, output_csv, total_rows, seed):
    if total_rows % 2:
        raise ValueError("--total-rows must be even so pre/post periods are balanced.")
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned post output not found: {input_path}")

    columns = [
        "subreddit",
        "post_id",
        "date",
        "year_month",
        "post_shock_exact",
        "title",
        "selftext",
    ]
    df = pd.read_parquet(input_path, columns=columns)
    df["period"] = df["post_shock_exact"].map({0: "pre_gpt", 1: "post_gpt"})
    df = df[df["period"].isin(["pre_gpt", "post_gpt"])].copy()

    df["created_date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["title_excerpt"] = df["title"].map(lambda x: clean_text(x, 280))
    df["selftext_excerpt"] = df["selftext"].map(lambda x: clean_text(x, 2000))
    df = df[df["title_excerpt"].ne("")].copy()

    per_period = total_rows // 2
    pre = sample_period(df, "pre_gpt", per_period, seed)
    post = sample_period(df, "post_gpt", per_period, seed + 100_000)
    sampled = pd.concat([pre, post], ignore_index=True)
    if len(sampled) != total_rows:
        raise RuntimeError(
            f"Expected {total_rows:,} sampled rows but got {len(sampled):,}. "
            "The cleaned post output does not have enough usable rows."
        )

    sampled = sampled.sample(frac=1.0, random_state=seed + 200_000).reset_index(drop=True)
    sampled["rationale"] = ""
    sampled["version"] = ""

    for col in BUILD_SCORE_COLUMNS:
        sampled[col] = ""

    out_cols = acsi_output_columns()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_name(output_csv.name + ".tmp")
    sampled[out_cols].to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)

    summary = sampled.groupby(["period", "subreddit"]).size().rename("n").reset_index()
    return {
        "csv": output_csv,
        "input": input_path,
        "rows": len(sampled),
        "pre_rows": int(sampled["period"].eq("pre_gpt").sum()),
        "post_rows": int(sampled["period"].eq("post_gpt").sum()),
        "subreddits": int(sampled["subreddit"].nunique()),
        "min_per_subreddit_period": int(summary["n"].min()),
        "max_per_subreddit_period": int(summary["n"].max()),
    }


def build_main(argv=None):
    parser = argparse.ArgumentParser(description="Build run-once ACSI measurement input.")
    parser.add_argument("--input", type=Path, default=BUILD_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=BUILD_DEFAULT_OUTPUT)
    parser.add_argument("--total-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment-raw-subreddits",
        action="store_true",
        help="Append/top up rows from local data/raw_files/r_*_posts.jsonl files instead of rebuilding from parquet.",
    )
    parser.add_argument(
        "--pre-shock-only",
        action="store_true",
        help="Append Jan-Oct 2022 raw post rows so each subreddit has enough pre-shock ACSI coding rows.",
    )
    parser.add_argument("--raw-data-dir", type=Path, default=BUILD_DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--target-rows-per-subreddit", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.pre_shock_only:
        summary = top_up_pre_shock_only_acsi_data(
            output_csv=args.output,
            raw_data_dir=args.raw_data_dir,
            target_rows_per_subreddit=BUILD_PRE_SHOCK_ONLY_TARGET_ROWS,
            seed=args.seed,
        )
    elif args.augment_raw_subreddits:
        summary = augment_acsi_data_from_raw(
            output_csv=args.output,
            raw_data_dir=args.raw_data_dir,
            target_rows_per_subreddit=args.target_rows_per_subreddit,
            seed=args.seed,
        )
    else:
        summary = build_acsi_data(
            input_path=args.input,
            output_csv=args.output,
            total_rows=args.total_rows,
            seed=args.seed,
        )
    for key, value in summary.items():
        if key == "summary":
            if not value.empty:
                print("\nsubreddit_additions")
                print(value.sort_values("subreddit").to_string(index=False))
            continue
        print(f"{key}: {value}")

# ---------------------------------------------------------------------------
# Manual dual-pass coding workflow.
# ---------------------------------------------------------------------------

DEFAULT_INPUT = ROOT / "data" / "acsi_data.csv"
DEFAULT_RUN1_OUTPUT = ROOT / "data" / "acsi_annotated.csv"
DEFAULT_RUN2_OUTPUT = ROOT / "data" / "acsi_measurement_sample_run2.csv"
DEFAULT_FINAL_OUTPUT = ROOT / "data" / "acsi_measurement_sample.csv"
DEFAULT_ACSI_SCORE_OUTPUT = ROOT / "data" / "acsi_scores.csv"
DEFAULT_FLOOR_TARGET = 300
PRE_SHOCK_ONLY_START = pd.Timestamp("2022-01-01")
PRE_SHOCK_ONLY_END_EXCLUSIVE = pd.Timestamp("2022-11-01")

SCORE_COLUMNS = [
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
    "ai_related_flag",
]

SCORE_VALUE_COLUMNS = [
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
]

FLAG_COLUMNS = ["ai_related_flag"]

DUAL_RUN_DIAGNOSTIC_COLUMNS = [
    "max_score_disagreement",
    "mean_score_disagreement",
    "hard_case_flag",
]

OUTPUT_COLUMNS = [
    "subreddit",
    "post_id",
    "period",
    "created_date",
    "year_month",
    "title_excerpt",
    "selftext_excerpt",
    *SCORE_COLUMNS,
    *DUAL_RUN_DIAGNOSTIC_COLUMNS,
    "rationale",
    "version",
]

JSON_KEYS = [
    "post_id",
    *SCORE_COLUMNS,
    "rationale",
]


def output_for_run(args):
    if args.output is not None:
        return args.output
    return DEFAULT_RUN1_OUTPUT if args.run == 1 else DEFAULT_RUN2_OUTPUT


def ensure_output_file(path):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def empty_output_frame():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def read_output(path, create=True):
    if create:
        ensure_output_file(path)
    elif not path.exists() or path.stat().st_size == 0:
        return empty_output_frame()
    df = pd.read_csv(path, dtype=str).fillna("")
    if "version" not in df.columns and "coding_instruction_version" in df.columns:
        df["version"] = (
            df["coding_instruction_version"]
            .astype(str)
            .str.replace("acsi_data_v2_", "", regex=False)
            .str.replace("dual_avg", "avg", regex=False)
        )
    if "version" not in df.columns:
        df["version"] = ""
    for col in DUAL_RUN_DIAGNOSTIC_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df[OUTPUT_COLUMNS].copy()


def coded_ids(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path, usecols=["post_id"], dtype=str)["post_id"].astype(str))


def load_remaining(input_path, output_path):
    df = pd.read_csv(input_path, dtype=str).fillna("")
    done = coded_ids(output_path)
    return df[~df["post_id"].astype(str).isin(done)].copy(), len(done), len(df)


def load_acsi_target_subreddits(path):
    if not path.exists():
        raise FileNotFoundError(f"ACSI target subreddit file not found: {path}")
    targets = pd.read_csv(path, dtype=str, usecols=["subreddit"]).fillna("")
    return set(targets["subreddit"].astype(str).str.strip())


def apply_scope_filters(df, args):
    scoped = df.copy()
    if getattr(args, "acsi_targets_only", False):
        targets = load_acsi_target_subreddits(args.acsi_target_input)
        scoped = scoped[scoped["subreddit"].astype(str).str.strip().isin(targets)].copy()
    if getattr(args, "pre_shock_only", False):
        created_date = pd.to_datetime(scoped["created_date"], errors="coerce")
        scoped = scoped[
            (created_date >= PRE_SHOCK_ONLY_START)
            & (created_date < PRE_SHOCK_ONLY_END_EXCLUSIVE)
        ].copy()
    return scoped


def load_remaining_scoped(args, output_path):
    df = pd.read_csv(args.input, dtype=str).fillna("")
    scoped = apply_scope_filters(df, args)
    scoped_ids = set(scoped["post_id"].astype(str))
    done_ids = coded_ids(output_path)
    done_scoped = len(scoped_ids & done_ids)
    remaining = scoped[~scoped["post_id"].astype(str).isin(done_ids)].copy()
    return remaining, done_scoped, len(scoped)


def measured_progress(output_path, args=None):
    measured = read_output(output_path, create=False)
    if measured.empty:
        return pd.DataFrame(columns=["subreddit", "coded_count", "usable_count"])

    measured = measured.copy()
    if args is not None:
        measured = apply_scope_filters(measured, args)
    measured["ai_related_flag"] = pd.to_numeric(measured["ai_related_flag"], errors="coerce").fillna(0)
    coded = measured.groupby("subreddit").size().rename("coded_count")
    usable = measured[measured["ai_related_flag"].ne(1)].groupby("subreddit").size().rename("usable_count")
    return (
        pd.concat([coded, usable], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )


def choose_balanced_batch(remaining, output_path, batch_size, floor_target, target_count, args=None):
    if remaining.empty:
        return remaining, None

    progress = measured_progress(output_path, args=args).set_index("subreddit")
    remaining_counts = remaining.groupby("subreddit").size().rename("remaining_count")
    selector = pd.DataFrame(index=remaining_counts.index)
    selector["remaining_count"] = remaining_counts
    selector["coded_count"] = progress["coded_count"].reindex(selector.index).fillna(0).astype(int)
    selector["usable_count"] = progress["usable_count"].reindex(selector.index).fillna(0).astype(int)
    selector["progress_count"] = selector[target_count + "_count"]
    selector["below_floor"] = selector["progress_count"].lt(floor_target)
    selector["floor_priority"] = selector["below_floor"].map({True: 0, False: 1})
    selector["subreddit_sort"] = selector.index.str.lower()

    selector = selector.sort_values(
        ["floor_priority", "progress_count", "coded_count", "subreddit_sort"],
        kind="mergesort",
    )
    target_subreddit = str(selector.index[0])
    batch = remaining[remaining["subreddit"].eq(target_subreddit)].head(batch_size)
    target_status = selector.loc[target_subreddit].to_dict()
    target_status["subreddit"] = target_subreddit
    target_status["under_floor_subreddits"] = int(selector["below_floor"].sum())
    target_status["eligible_subreddits"] = int(len(selector))
    return batch, target_status


def validate_json_record(record):
    keys = set(record)
    expected = set(JSON_KEYS)
    if keys != expected:
        raise ValueError(
            f"post_id={record.get('post_id')} has wrong keys. "
            f"missing={sorted(expected - keys)} extra={sorted(keys - expected)}"
        )

    for col in SCORE_VALUE_COLUMNS:
        value = record[col]
        if not isinstance(value, int) or value not in {0, 1, 2, 3}:
            raise ValueError(f"post_id={record['post_id']} {col} must be integer 0, 1, 2, or 3")

    for col in FLAG_COLUMNS:
        value = record[col]
        if not isinstance(value, int) or value not in {0, 1}:
            raise ValueError(f"post_id={record['post_id']} {col} must be integer 0 or 1")

    rationale = str(record["rationale"]).strip()
    if not rationale:
        raise ValueError(f"post_id={record['post_id']} rationale cannot be blank")
    record["rationale"] = rationale


def load_json_records(path):
    text = sys.stdin.read() if str(path) == "-" else Path(path).read_text()
    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("JSON input must be an array of post objects.")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every JSON array item must be an object.")
        validate_json_record(record)
        post_id = str(record["post_id"])
        if post_id in seen:
            raise ValueError(f"Duplicate post_id in JSON batch: {post_id}")
        record["post_id"] = post_id
        seen.add(post_id)
    return records


def print_next(args):
    output_path = output_for_run(args)
    remaining, done, total = load_remaining_scoped(args, output_path)
    if args.subreddit:
        requested_subreddits = {subreddit.strip().lower() for subreddit in args.subreddit if subreddit.strip()}
        remaining = remaining[remaining["subreddit"].str.lower().isin(requested_subreddits)].copy()
    if args.selection_strategy == "input-order":
        batch = remaining.head(args.n)
        target_status = None
    else:
        batch, target_status = choose_balanced_batch(
            remaining=remaining,
            output_path=output_path,
            batch_size=args.n,
            floor_target=args.floor_target,
            target_count=args.target_count,
            args=args,
        )
    print(f"run={args.run} output={output_path}")
    print(f"done={done} total={total} remaining={len(remaining)} batch={len(batch)}")
    scope = []
    if getattr(args, "pre_shock_only", False):
        scope.append("pre_shock_only=2022-01_to_2022-10")
    if getattr(args, "acsi_targets_only", False):
        scope.append(f"acsi_targets_only={args.acsi_target_input}")
    if scope:
        print("scope=" + " ".join(scope))
    print(f"selection_strategy={args.selection_strategy}")
    if target_status is not None:
        print(
            "target="
            f"{target_status['subreddit']} "
            f"{args.target_count}_count={int(target_status['progress_count'])} "
            f"coded_count={int(target_status['coded_count'])} "
            f"usable_count={int(target_status['usable_count'])} "
            f"remaining_for_target={int(target_status['remaining_count'])} "
            f"floor_target={args.floor_target} "
            f"under_floor_subreddits={target_status['under_floor_subreddits']} "
            f"eligible_subreddits={target_status['eligible_subreddits']}"
        )
    print("\nUse data/acsi_data_rubric.md. Return a JSON array only.")
    print("Code this as an independent pass; do not look at the other run's scores.")
    print("Each object must have exactly:")
    print(
        '{"post_id":"...","direct_gen_score":0,"usefulness_score":0,'
        '"quality_comp_score":0,"physical_req_score":0,"personal_req_score":0,'
        '"ai_related_flag":0,"rationale":"specific evidence-based sentence"}'
    )
    for i, row in enumerate(batch.to_dict(orient="records"), start=1):
        print("\n---")
        print(f"batch_index: {i}")
        print(f"post_id: {row['post_id']}")
        print(f"subreddit: {row['subreddit']}")
        print(f"period: {row['period']}")
        print(f"date: {row['created_date']}")
        print(f"title: {row['title_excerpt']}")
        body = row["selftext_excerpt"]
        if body:
            print(f"selftext: {body}")


def append_scores(args):
    output_path = output_for_run(args)
    records = load_json_records(args.json_file)
    input_df = pd.read_csv(args.input, dtype=str).fillna("")
    input_by_id = input_df.set_index("post_id", drop=False)
    existing = read_output(output_path)
    existing_ids = set(existing["post_id"].astype(str))

    rows = []
    for record in records:
        post_id = record["post_id"]
        if post_id not in input_by_id.index:
            raise ValueError(f"post_id={post_id} is not in {args.input}")
        if post_id in existing_ids:
            raise ValueError(f"post_id={post_id} is already present in {output_path}")

        source = input_by_id.loc[post_id]
        row = {col: source.get(col, "") for col in OUTPUT_COLUMNS}
        for col in SCORE_COLUMNS:
            row[col] = record[col]
        row["rationale"] = record["rationale"]
        row["version"] = f"run{args.run}"
        rows.append(row)

    combined = pd.concat([existing, pd.DataFrame(rows, columns=OUTPUT_COLUMNS)], ignore_index=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    combined.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"appended={len(rows)} run={args.run} output={output_path}")


def invalid_score_values(coded, averaged=False):
    bad_scores = {}
    allowed_scores = {0.0, 1.0, 2.0, 3.0}
    if averaged:
        allowed_scores |= {0.5, 1.5, 2.5}
    for col in SCORE_VALUE_COLUMNS:
        vals = set(pd.to_numeric(coded[col], errors="coerce").dropna().unique())
        bad_scores[col] = sorted(vals - allowed_scores)
    for col in FLAG_COLUMNS:
        vals = set(pd.to_numeric(coded[col], errors="coerce").dropna().unique())
        bad_scores[col] = sorted(vals - {0.0, 1.0})
    return bad_scores


def validate_coded_dataframe(label, df, input_ids=None, averaged=False):
    duplicate_count = int(df["post_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{label} has {duplicate_count} duplicate post_id values.")

    if not df.empty:
        required_value_cols = [*SCORE_VALUE_COLUMNS, *FLAG_COLUMNS, "rationale"]
        missing_values = {
            col: int(df[col].astype(str).str.strip().eq("").sum())
            for col in required_value_cols
        }
        missing_values = {col: count for col, count in missing_values.items() if count}
        if missing_values:
            raise ValueError(f"{label} has missing required values: {missing_values}")

        for col in [*SCORE_VALUE_COLUMNS, *FLAG_COLUMNS]:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.isna().any():
                bad_ids = df.loc[numeric.isna(), "post_id"].head(10).tolist()
                raise ValueError(f"{label} has nonnumeric values in {col}: {bad_ids}")

    if input_ids is not None:
        extra_ids = sorted(set(df["post_id"].astype(str)) - input_ids)
        if extra_ids:
            preview = ", ".join(extra_ids[:10])
            raise ValueError(f"{label} has post_id values not in input data: {preview}")

    bad_scores = invalid_score_values(df, averaged=averaged)
    if any(bad_scores.values()):
        raise ValueError(f"{label} has invalid score values: {bad_scores}")


def print_one_summary(label, output_path, input_path, averaged=False):
    remaining, done, total = load_remaining(input_path, output_path)
    print(f"\n{label}")
    print(f"output={output_path}")
    print(f"done={done}")
    print(f"remaining={len(remaining)}")
    print(f"total={total}")

    measured = read_output(output_path, create=False)
    if measured.empty:
        if output_path.exists():
            print("measurement file has header only")
        else:
            print("measurement file not found")
        return

    duplicates = int(measured["post_id"].duplicated().sum())
    print(f"duplicate_post_ids={duplicates}")
    print("\nby_period")
    print(measured.groupby("period").size().to_string())
    print("\nby_subreddit_top")
    print(measured.groupby("subreddit").size().sort_values(ascending=False).head(20).to_string())

    bad_scores = invalid_score_values(measured, averaged=averaged)
    if any(bad_scores.values()):
        print("\nbad_scores")
        print(bad_scores)


def print_summary(args):
    if args.all_runs:
        print_one_summary("run_1", DEFAULT_RUN1_OUTPUT, args.input)
        print_one_summary("run_2", DEFAULT_RUN2_OUTPUT, args.input)
        print_one_summary("final_averaged", DEFAULT_FINAL_OUTPUT, args.input, averaged=True)
        return

    output_path = output_for_run(args)
    print_one_summary(f"run_{args.run}", output_path, args.input)


def combine_runs(args):
    input_df = pd.read_csv(args.input, dtype=str).fillna("")
    run1 = read_output(args.run1_output)
    run2 = read_output(args.run2_output)
    input_ids = set(input_df["post_id"].astype(str))

    for label, df in [("run1", run1), ("run2", run2)]:
        validate_coded_dataframe(label, df, input_ids=input_ids)

    if not args.allow_partial:
        run1_ids = set(run1["post_id"].astype(str))
        run2_ids = set(run2["post_id"].astype(str))
        missing_run1 = len(input_ids - run1_ids)
        missing_run2 = len(input_ids - run2_ids)
        mismatched = len(run1_ids.symmetric_difference(run2_ids))
        if missing_run1 or missing_run2 or mismatched:
            raise ValueError(
                "Both full coding runs must be complete before combine. "
                f"missing_run1={missing_run1} missing_run2={missing_run2} "
                f"run_id_mismatches={mismatched}. Pass --allow-partial only "
                "for calibration output."
            )

    common_ids = set(run1["post_id"]) & set(run2["post_id"])
    ordered_ids = [post_id for post_id in input_df["post_id"] if post_id in common_ids]
    run1 = run1.set_index("post_id")
    run2 = run2.set_index("post_id")
    input_by_id = input_df.set_index("post_id", drop=False)

    rows = []
    for post_id in ordered_ids:
        source = input_by_id.loc[post_id]
        r1 = run1.loc[post_id]
        r2 = run2.loc[post_id]
        row = {col: source.get(col, "") for col in OUTPUT_COLUMNS}
        score_differences = []
        for col in SCORE_VALUE_COLUMNS:
            run1_score = float(pd.to_numeric(r1[col], errors="raise"))
            run2_score = float(pd.to_numeric(r2[col], errors="raise"))
            row[col] = (run1_score + run2_score) / 2
            score_differences.append(abs(run1_score - run2_score))
        row["ai_related_flag"] = int(max(int(r1["ai_related_flag"]), int(r2["ai_related_flag"])))
        row["max_score_disagreement"] = max(score_differences)
        row["mean_score_disagreement"] = sum(score_differences) / len(score_differences)
        row["hard_case_flag"] = int(row["max_score_disagreement"] > 1)
        row["rationale"] = "Averaged from two independent coding passes."
        row["version"] = "avg"
        rows.append(row)

    final = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    args.final_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.final_output.with_name(args.final_output.name + ".tmp")
    final.to_csv(tmp_path, index=False)
    tmp_path.replace(args.final_output)

    print(f"combined_rows={len(final)}")
    print(f"run1_rows={len(run1)}")
    print(f"run2_rows={len(run2)}")
    print(f"missing_run1={len(input_df) - len(run1)}")
    print(f"missing_run2={len(input_df) - len(run2)}")
    print(f"final_output={args.final_output}")

    if not final.empty:
        disagreement_rows = []
        for col in SCORE_VALUE_COLUMNS:
            diff = (
                pd.to_numeric(run1.loc[ordered_ids, col], errors="coerce")
                - pd.to_numeric(run2.loc[ordered_ids, col], errors="coerce")
            ).abs()
            disagreement_rows.append(
                {
                    "score": col,
                    "mean_abs_diff": round(float(diff.mean()), 4),
                    "pct_diff_2_or_more": round(float((diff >= 2).mean() * 100), 2),
                }
            )
        print("\ndisagreement_diagnostics")
        print(pd.DataFrame(disagreement_rows).to_string(index=False))
        hard_cases = int(pd.to_numeric(final["hard_case_flag"], errors="coerce").fillna(0).sum())
        print(f"\nhard_case_rows={hard_cases}")
        if hard_cases:
            hard_case_summary = (
                final[pd.to_numeric(final["hard_case_flag"], errors="coerce").fillna(0).eq(1)]
                .groupby("subreddit")
                .size()
                .sort_values(ascending=False)
                .head(20)
                .rename("hard_cases")
                .reset_index()
            )
            print("\nhard_cases_by_subreddit_top")
            print(hard_case_summary.to_string(index=False))


def weighted_mean(group, col):
    weights = group["_weight"].astype(float)
    total_weight = weights.sum()
    if total_weight <= 0:
        return float("nan")
    return float((group[col].astype(float) * weights).sum() / total_weight)


def to_run_py_component(avg_0_to_3):
    return 1 + (avg_0_to_3 * 4 / 3)


def aggregate_scores(args):
    input_df = pd.read_csv(args.input, dtype=str).fillna("")
    coded = read_output(args.final_output)
    if coded.empty:
        raise ValueError(f"{args.final_output} has no measurement rows to aggregate.")
    versions = set(coded["version"].astype(str).str.strip())
    allow_averaged_scores = bool(versions) and versions.issubset({"avg"})
    validate_coded_dataframe(
        "measurement score file",
        coded,
        input_ids=set(input_df["post_id"]),
        averaged=allow_averaged_scores,
    )
    if len(coded) != len(input_df) and not args.allow_partial:
        raise ValueError(
            f"{args.final_output} has {len(coded)} measurement rows but {args.input} has "
            f"{len(input_df)} rows. Finish both full coding runs first, or pass "
            "--allow-partial for calibration output."
        )

    for col in SCORE_VALUE_COLUMNS + FLAG_COLUMNS:
        coded[col] = pd.to_numeric(coded[col], errors="raise")

    coded["_weight"] = 1.0
    if "hard_case_flag" in coded.columns:
        coded["hard_case_flag"] = pd.to_numeric(coded["hard_case_flag"], errors="coerce").fillna(0).astype(int)
    else:
        coded["hard_case_flag"] = 0

    if args.hard_case_policy == "exclude":
        coded["_excluded_hard_case"] = coded["hard_case_flag"].eq(1)
    else:
        coded["_excluded_hard_case"] = False
        if args.hard_case_policy == "downweight":
            coded.loc[coded["hard_case_flag"].eq(1), "_weight"] = args.hard_case_weight

    n_coded = coded.groupby("subreddit").size().rename("n_coded")
    n_ai = coded[coded["ai_related_flag"].eq(1)].groupby("subreddit").size().rename("n_ai_related_excluded")
    n_hard = coded[coded["hard_case_flag"].eq(1)].groupby("subreddit").size().rename("n_hard_cases")
    used = coded.copy()
    if not args.include_ai_related:
        used = used[used["ai_related_flag"].ne(1)].copy()
    if args.hard_case_policy == "exclude":
        used = used[used["hard_case_flag"].ne(1)].copy()

    rows = []
    missing = []
    for subreddit in sorted(input_df["subreddit"].unique()):
        group = used[used["subreddit"].eq(subreddit)]
        if group.empty:
            missing.append(subreddit)
            continue

        avg = {col: weighted_mean(group, col) for col in SCORE_VALUE_COLUMNS}
        direct_gen = to_run_py_component(avg["direct_gen_score"])
        usefulness = to_run_py_component(avg["usefulness_score"])
        quality_comp = to_run_py_component(avg["quality_comp_score"])
        physical_req = to_run_py_component(avg["physical_req_score"])
        personal_req = to_run_py_component(avg["personal_req_score"])
        raw_gse = direct_gen + usefulness + quality_comp + (6 - physical_req) + (6 - personal_req)
        gse = max(0.0, min(1.0, (raw_gse - 5) / 20))

        rows.append(
            {
                "subreddit": subreddit,
                "direct_gen": direct_gen,
                "usefulness": usefulness,
                "quality_comp": quality_comp,
                "physical_req": physical_req,
                "personal_req": personal_req,
                "raw_gse": raw_gse,
                "gse": gse,
                "n_coded": int(n_coded.get(subreddit, 0)),
                "n_used": int(len(group)),
                "n_ai_related_excluded": int(0 if args.include_ai_related else n_ai.get(subreddit, 0)),
                "n_hard_cases": int(n_hard.get(subreddit, 0)),
                "hard_case_policy": args.hard_case_policy,
                "score_reliability": "low" if len(group) < 50 else ("medium" if len(group) < 100 else "high"),
                "low_n_flag": int(len(group) < 50),
                "avg_direct_gen_0_to_3": avg["direct_gen_score"],
                "avg_usefulness_0_to_3": avg["usefulness_score"],
                "avg_quality_comp_0_to_3": avg["quality_comp_score"],
                "avg_physical_req_0_to_3": avg["physical_req_score"],
                "avg_personal_req_0_to_3": avg["personal_req_score"],
            }
        )

    if missing and not args.allow_partial:
        raise ValueError(
            "No usable measurement rows for these subreddits after exclusions: "
            + ", ".join(missing)
        )
    if missing:
        print(
            "skipped_uncoded_or_unusable_subreddits="
            + ",".join(missing)
        )

    out = pd.DataFrame(rows).sort_values("subreddit")
    args.acsi_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.acsi_output.with_name(args.acsi_output.name + ".tmp")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(args.acsi_output)
    print(f"wrote={args.acsi_output}")
    print(f"subreddits={len(out)}")
    print(f"include_ai_related={args.include_ai_related}")
    print(f"hard_case_policy={args.hard_case_policy}")
    if args.hard_case_policy == "downweight":
        print(f"hard_case_weight={args.hard_case_weight}")
    print(f"low_reliability_subreddits={int(out['low_n_flag'].sum())}")


def coding_main(argv=None):
    parser = argparse.ArgumentParser(description="Manual dual-pass ACSI coding helper.")
    parser.add_argument("command", choices=["next", "append", "summary", "combine", "aggregate"])
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--run", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-file", type=Path, default=Path("-"))
    parser.add_argument("--subreddit", action="append", default=[])
    parser.add_argument(
        "--pre-shock-only",
        action="store_true",
        help="For next, restrict selection and progress counts to Jan-Oct 2022 rows.",
    )
    parser.add_argument(
        "--acsi-targets-only",
        action="store_true",
        help="For next, restrict selection and progress counts to subreddits in the ACSI score table.",
    )
    parser.add_argument("--acsi-target-input", type=Path, default=DEFAULT_ACSI_SCORE_OUTPUT)
    parser.add_argument(
        "--selection-strategy",
        choices=["balanced", "input-order"],
        default="balanced",
        help="How next chooses rows. balanced targets the lowest-count subreddit first.",
    )
    parser.add_argument(
        "--floor-target",
        type=int,
        default=DEFAULT_FLOOR_TARGET,
        help="Minimum per-subreddit count to prioritize before even rotation.",
    )
    parser.add_argument(
        "--target-count",
        choices=["usable", "coded"],
        default="usable",
        help="Progress count used by balanced selection. usable excludes ai_related_flag=1 rows.",
    )
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--run1-output", type=Path, default=DEFAULT_RUN1_OUTPUT)
    parser.add_argument("--run2-output", type=Path, default=DEFAULT_RUN2_OUTPUT)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument("--acsi-output", type=Path, default=DEFAULT_ACSI_SCORE_OUTPUT)
    parser.add_argument("--gse-output", dest="acsi_output", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--include-ai-related", action="store_true")
    parser.add_argument("--hard-case-policy", choices=["downweight", "exclude", "keep"], default="downweight")
    parser.add_argument("--hard-case-weight", type=float, default=0.5)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "next":
        print_next(args)
    elif args.command == "append":
        append_scores(args)
    elif args.command == "summary":
        print_summary(args)
    elif args.command == "aggregate":
        aggregate_scores(args)
    else:
        combine_runs(args)

ANNOTATION_COMMANDS = ("next", "append", "summary", "combine", "aggregate")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build annotation samples and manage dual-pass ACSI coding. "
            "Use `build` to create/top up data/acsi_data.csv; use next, append, "
            "summary, combine, and aggregate for scoring workflow steps."
        )
    )
    parser.add_argument("command", choices=("build", *ANNOTATION_COMMANDS))
    parser.add_argument("command_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command == "build":
        return build_main(args.command_args)
    return coding_main([args.command, *args.command_args])


if __name__ == "__main__":
    main()
