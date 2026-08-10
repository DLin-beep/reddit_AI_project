#!/usr/bin/env python3
"""Compute agreement between two sets of ACSI scores.

Handles all three comparisons the protocol requires:

  * self-consistency   two passes of the primary model over the same posts
  * cross-model        primary vs secondary model over the same posts
  * human validation   human raters vs each other and vs the model, evaluated
                       against the acceptance thresholds

Agreement is quadratic-weighted Cohen's kappa on the 0-3 ordinal scales, plus
exact and within-one agreement for interpretability, and community-level
composite correlations where both sides cover enough communities.

Sources may be scored-run JSONL files (scores_runN.jsonl) or human rating CSVs
with columns annotation_id and the five dimension names.

Examples:
    python agreement.py --a 1 --b 2                       # self-consistency
    python agreement.py --a 1 --b 3 --label cross_model   # cross-model
    python agreement.py --a 1 --b-file human_rater1.csv --label human_vs_model
    python agreement.py --a-file human_rater1.csv --b-file human_rater2.csv \
        --label human_vs_human --acceptance human
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# This module lives in annotation_code/ but pipeline_utils sits at the repo
# root, so make the parent importable whether the script is run from here or
# from the root via the .sh wrapper.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline_utils import load_config, resolve_path, sha256_file, write_json

DIMS = (
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
)
GEN_DIMS = DIMS[:3]

# Acceptance thresholds fixed in the protocol.
HUMAN_HUMAN_FLOOR = 0.50
RELATIVE_TOLERANCE = 0.05
MODEL_FLOOR = {d: 0.50 for d in DIMS} | {"personal_req_score": 0.60}

csv.field_size_limit(10_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sampling_config_creation_relative.json")
    parser.add_argument("--a", type=int, help="Run number for the first score set")
    parser.add_argument("--b", type=int, help="Run number for the second score set")
    parser.add_argument("--a-file", help="CSV of ratings for the first set")
    parser.add_argument("--b-file", help="CSV of ratings for the second set")
    parser.add_argument("--label", default=None, help="Name for this comparison")
    parser.add_argument("--acceptance", choices=["none", "human", "model"], default="none",
                        help="'human' applies the rubric-validity floor; 'model' applies the "
                             "absolute floors and, with --benchmark, the relative criterion")
    parser.add_argument("--benchmark", help="JSON from a human-vs-human run, for the relative criterion")
    return parser.parse_args()


def read_jsonl(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if str(record.get("rationale", "")).strip().upper().startswith("EXCLUDE"):
                continue
            out[record["post_id"]] = {d: int(record[d]) for d in DIMS}
    return out


def read_ratings_csv(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("annotation_id") or row.get("post_id") or "").strip()
            if not key:
                raise ValueError(f"{path} needs an annotation_id (or post_id) column")
            missing = [d for d in DIMS if not (row.get(d) or "").strip()]
            if missing:
                continue
            out[key] = {d: int(float(row[d])) for d in DIMS}
    return out


def quadratic_weighted_kappa(a: list[int], b: list[int], categories: int = 4) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = [[0] * categories for _ in range(categories)]
    for x, y in zip(a, b):
        observed[x][y] += 1
    hist_a = [a.count(i) for i in range(categories)]
    hist_b = [b.count(i) for i in range(categories)]
    denominator = (categories - 1) ** 2
    num = den = 0.0
    for i in range(categories):
        for j in range(categories):
            w = (i - j) ** 2 / denominator
            num += w * observed[i][j]
            den += w * hist_a[i] * hist_b[j] / n
    if den == 0:
        # No expected disagreement: perfect agreement on a constant vector.
        return 1.0 if num == 0 else 0.0
    return 1 - num / den


def composites(scores: dict[str, dict[str, int]], community_of: dict[str, str]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for post_id, record in scores.items():
        community = community_of.get(post_id)
        if community is None:
            continue
        for d in DIMS:
            buckets[community][d].append(record[d])
    out: dict[str, dict[str, float]] = {}
    for community, dims in buckets.items():
        gen = sum(sum(dims[d]) / len(dims[d]) / 3 for d in GEN_DIMS) / len(GEN_DIMS)
        out[community] = {
            "gen_cap": gen,
            "phys_free": 1 - sum(dims["physical_req_score"]) / len(dims["physical_req_score"]) / 3,
            "pers_free": 1 - sum(dims["personal_req_score"]) / len(dims["personal_req_score"]) / 3,
        }
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_dir = resolve_path(base_dir, config["paths"]["run_dir"])
    score_dir = run_dir / "06_annotation_scores"
    out_dir = run_dir / "09_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(run: int | None, file: str | None, side: str) -> tuple[dict[str, dict[str, int]], str, Path]:
        if run is not None:
            path = score_dir / f"scores_run{run}.jsonl"
            return read_jsonl(path), f"run{run}", path
        if file:
            path = Path(file) if Path(file).is_absolute() else base_dir / file
            return read_ratings_csv(path), path.stem, path
        raise ValueError(f"Provide --{side} or --{side}-file")

    a_scores, a_name, a_path = load(args.a, args.a_file, "a")
    b_scores, b_name, b_path = load(args.b, args.b_file, "b")
    label = args.label or f"{a_name}_vs_{b_name}"

    shared = sorted(set(a_scores) & set(b_scores))
    if not shared:
        raise RuntimeError("The two score sets share no posts")

    community_of: dict[str, str] = {}
    key_path = run_dir / "05_annotation_prep" / "annotation_posts_key.csv"
    if key_path.exists():
        with key_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                community_of[row["annotation_id"]] = row["subreddit"]

    per_dim: dict[str, dict[str, float]] = {}
    for d in DIMS:
        xs = [a_scores[p][d] for p in shared]
        ys = [b_scores[p][d] for p in shared]
        exact = sum(x == y for x, y in zip(xs, ys)) / len(shared)
        within1 = sum(abs(x - y) <= 1 for x, y in zip(xs, ys)) / len(shared)
        per_dim[d] = {
            "qwk": round(quadratic_weighted_kappa(xs, ys), 4),
            "exact_agreement": round(exact, 4),
            "within_one": round(within1, 4),
            "mean_a": round(sum(xs) / len(xs), 3),
            "mean_b": round(sum(ys) / len(ys), 3),
        }

    comp_a, comp_b = composites(a_scores, community_of), composites(b_scores, community_of)
    shared_communities = sorted(set(comp_a) & set(comp_b))
    comp_stats: dict[str, Any] = {}
    for name in ("gen_cap", "phys_free", "pers_free"):
        if len(shared_communities) >= 3:
            xs = [comp_a[c][name] for c in shared_communities]
            ys = [comp_b[c][name] for c in shared_communities]
            comp_stats[name] = {
                "pearson_r": round(pearson(xs, ys), 4),
                "mean_absolute_difference": round(sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs), 4),
            }

    acceptance: dict[str, Any] = {}
    if args.acceptance == "human":
        acceptance = {
            "criterion": "rubric validity gate: human-to-human QWK >= 0.50 on every dimension",
            "results": {d: {"qwk": per_dim[d]["qwk"], "passes": per_dim[d]["qwk"] >= HUMAN_HUMAN_FLOOR}
                        for d in DIMS},
        }
        acceptance["all_pass"] = all(v["passes"] for v in acceptance["results"].values())
    elif args.acceptance == "model":
        benchmark = {}
        if args.benchmark:
            path = Path(args.benchmark) if Path(args.benchmark).is_absolute() else base_dir / args.benchmark
            benchmark = {d: v["qwk"] for d, v in json.loads(path.read_text())["per_dimension"].items()}
        results = {}
        for d in DIMS:
            k = per_dim[d]["qwk"]
            floor_ok = k >= MODEL_FLOOR[d]
            entry = {"qwk": k, "absolute_floor": MODEL_FLOOR[d], "passes_floor": floor_ok}
            if benchmark:
                needed = benchmark[d] - RELATIVE_TOLERANCE
                entry |= {
                    "human_human_qwk": benchmark[d],
                    "relative_requirement": round(needed, 4),
                    "passes_relative": k >= needed,
                    "passes": floor_ok and k >= needed,
                }
            else:
                entry["passes"] = floor_ok
            results[d] = entry
        acceptance = {
            "criterion": "absolute floors (0.60 personal-context, 0.50 others)"
                         + ("; relative criterion vs human-human benchmark" if benchmark else
                            "; relative criterion NOT evaluated (no --benchmark given)"),
            "results": results,
            "all_pass": all(v["passes"] for v in results.values()),
            "failing_dimensions": [d for d, v in results.items() if not v["passes"]],
        }

    summary = {
        "label": label,
        "a": {"name": a_name, "posts": len(a_scores), "sha256": sha256_file(a_path)},
        "b": {"name": b_name, "posts": len(b_scores), "sha256": sha256_file(b_path)},
        "shared_posts": len(shared),
        "shared_communities": len(shared_communities),
        "per_dimension": per_dim,
        "community_composites": comp_stats,
        "acceptance": acceptance,
    }
    out_path = out_dir / f"agreement_{label}.json"
    write_json(out_path, summary)

    print(f"{label}: {len(shared):,} shared posts, {len(shared_communities)} communities")
    print(f"  {'dimension':22} {'QWK':>7} {'exact':>7} {'within1':>8}")
    for d in DIMS:
        v = per_dim[d]
        print(f"  {d:22} {v['qwk']:7.3f} {v['exact_agreement']:7.1%} {v['within_one']:8.1%}")
    for name, v in comp_stats.items():
        print(f"  community {name:12} r={v['pearson_r']:+.3f}  mean|diff|={v['mean_absolute_difference']:.4f}")
    if acceptance:
        print(f"  acceptance: all_pass={acceptance['all_pass']}")
        if acceptance.get("failing_dimensions"):
            print(f"  FAILING: {', '.join(acceptance['failing_dimensions'])}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
