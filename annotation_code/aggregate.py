#!/usr/bin/env python3
"""Aggregate post-level ACSI scores into community-level exposure measures.

Joins the blinded scores back to the unblinding key, computes the three
pre-specified composites per community, and runs the measurement-review
checks fixed in the protocol:

  * per-community unscoreable and low-confidence rates against their triggers
  * the correlation between a community's unscoreable rate and its PersFree
    score, which tests whether content filtering falls disproportionately on
    the communities carrying the primary result

Outputs to <run_dir>/07_community_scores/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

GEN_DIMS = ("direct_gen_score", "usefulness_score", "quality_comp_score")
PHYS_DIM = "physical_req_score"
PERS_DIM = "personal_req_score"

UNSCOREABLE_TRIGGER = 0.05
LOW_CONFIDENCE_TRIGGER = 0.15
STUDY_UNSCOREABLE_TRIGGER = 0.03

csv.field_size_limit(10_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sampling_config_creation_relative.json")
    parser.add_argument("--run", type=int, default=1)
    return parser.parse_args()


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def two_sided_p(r: float, n: int) -> float:
    """Normal approximation to the t test on a correlation coefficient."""
    if not math.isfinite(r) or n < 4 or abs(r) >= 1:
        return float("nan")
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    return math.erfc(t / math.sqrt(2))


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_dir = resolve_path(base_dir, config["paths"]["run_dir"])
    prep_dir = run_dir / "05_annotation_prep"
    score_dir = run_dir / "06_annotation_scores"
    out_dir = run_dir / "07_community_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_path = prep_dir / "annotation_posts_key.csv"
    scores_path = score_dir / f"scores_run{args.run}.jsonl"
    failures_path = score_dir / f"failures_run{args.run}.csv"
    for path in (key_path, scores_path):
        if not path.exists():
            raise FileNotFoundError(path)

    community: dict[str, dict[str, str]] = {}
    with key_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            community[row["annotation_id"]] = {
                "subreddit": row["subreddit"],
                "category": row["category"],
            }
    assigned = len(community)

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"scores": defaultdict(list), "ai_flagged": 0, "excluded": 0, "low_conf": 0, "failed": 0}
    )
    for meta in community.values():
        _ = stats[meta["subreddit"]]

    scored = 0
    with scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            meta = community.get(record["post_id"])
            if meta is None:
                raise ValueError(f"Scored post {record['post_id']} is not in the key file")
            entry = stats[meta["subreddit"]]
            scored += 1
            rationale = str(record.get("rationale", "")).strip().upper()
            if rationale.startswith("EXCLUDE"):
                entry["excluded"] += 1
                continue
            if "CONFIDENCE IS LOW" in rationale or "LOW CONFIDENCE" in rationale:
                entry["low_conf"] += 1
            if int(record.get("ai_related_flag", 0)) == 1:
                entry["ai_flagged"] += 1
                continue
            for dim in (*GEN_DIMS, PHYS_DIM, PERS_DIM):
                entry["scores"][dim].append(float(record[dim]))

    if failures_path.exists():
        with failures_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                meta = community.get((row.get("annotation_id") or "").strip())
                if meta:
                    stats[meta["subreddit"]]["failed"] += 1

    rows: list[dict[str, Any]] = []
    for subreddit in sorted(stats, key=str.casefold):
        entry = stats[subreddit]
        used = len(entry["scores"][PERS_DIM])
        assigned_here = sum(1 for m in community.values() if m["subreddit"] == subreddit)
        unscoreable = entry["failed"] + entry["excluded"]
        if used == 0:
            raise RuntimeError(f"No usable scored posts for {subreddit}")
        gen = sum(sum(entry["scores"][d]) / len(entry["scores"][d]) / 3 for d in GEN_DIMS) / len(GEN_DIMS)
        phys = sum(entry["scores"][PHYS_DIM]) / used / 3
        pers = sum(entry["scores"][PERS_DIM]) / used / 3
        rows.append(
            {
                "subreddit": subreddit,
                "category": next(m["category"] for m in community.values() if m["subreddit"] == subreddit),
                "posts_assigned": assigned_here,
                "posts_used": used,
                "posts_failed": entry["failed"],
                "posts_excluded": entry["excluded"],
                "posts_ai_flagged": entry["ai_flagged"],
                "posts_low_confidence": entry["low_conf"],
                "unscoreable_rate": round(unscoreable / assigned_here, 6),
                "low_confidence_rate": round(entry["low_conf"] / assigned_here, 6),
                "gen_cap": round(gen, 6),
                "phys_free": round(1 - phys, 6),
                "pers_free": round(1 - pers, 6),
            }
        )

    scores_csv = out_dir / f"community_scores_run{args.run}.csv"
    with scores_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flagged = [
        {
            "subreddit": r["subreddit"],
            "reason": "unscoreable" if r["unscoreable_rate"] > UNSCOREABLE_TRIGGER else "low_confidence",
            "unscoreable_rate": r["unscoreable_rate"],
            "low_confidence_rate": r["low_confidence_rate"],
            "posts_used": r["posts_used"],
        }
        for r in rows
        if r["unscoreable_rate"] > UNSCOREABLE_TRIGGER or r["low_confidence_rate"] > LOW_CONFIDENCE_TRIGGER
    ]
    flagged_csv = out_dir / f"flagged_communities_run{args.run}.csv"
    with flagged_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["subreddit", "reason", "unscoreable_rate", "low_confidence_rate", "posts_used"]
        )
        writer.writeheader()
        writer.writerows(flagged)

    # Pre-committed check: does filtering fall on the communities that carry the result?
    unscore = [r["unscoreable_rate"] for r in rows]
    lowconf = [r["low_confidence_rate"] for r in rows]
    checks: dict[str, Any] = {}
    for label, xs in (("unscoreable_rate", unscore), ("low_confidence_rate", lowconf)):
        for composite in ("pers_free", "gen_cap", "phys_free"):
            ys = [r[composite] for r in rows]
            r_p = pearson(xs, ys)
            r_s = pearson(ranks(xs), ranks(ys))
            checks[f"{label}_vs_{composite}"] = {
                "pearson_r": round(r_p, 4),
                "pearson_p": round(two_sided_p(r_p, len(rows)), 5),
                "spearman_rho": round(r_s, 4),
                "spearman_p": round(two_sided_p(r_s, len(rows)), 5),
            }

    total_assigned = assigned
    total_unscoreable = sum(r["posts_failed"] + r["posts_excluded"] for r in rows)
    summary = {
        "run": args.run,
        "communities": len(rows),
        "posts_assigned": total_assigned,
        "posts_scored": scored,
        "posts_used_in_aggregation": sum(r["posts_used"] for r in rows),
        "posts_failed": sum(r["posts_failed"] for r in rows),
        "posts_excluded": sum(r["posts_excluded"] for r in rows),
        "posts_ai_flagged": sum(r["posts_ai_flagged"] for r in rows),
        "posts_low_confidence": sum(r["posts_low_confidence"] for r in rows),
        "study_unscoreable_rate": round(total_unscoreable / total_assigned, 6),
        "study_trigger_exceeded": total_unscoreable / total_assigned > STUDY_UNSCOREABLE_TRIGGER,
        "communities_flagged": len(flagged),
        "min_posts_used": min(r["posts_used"] for r in rows),
        "median_posts_used": sorted(r["posts_used"] for r in rows)[len(rows) // 2],
        "filtering_bias_checks": checks,
        "input_scores_sha256": sha256_file(scores_path),
        "outputs": {scores_csv.name: sha256_file(scores_csv), flagged_csv.name: sha256_file(flagged_csv)},
    }
    write_json(out_dir / f"aggregation_summary_run{args.run}.json", summary)

    print(f"communities={len(rows)} posts_used={summary['posts_used_in_aggregation']:,}")
    print(f"study unscoreable rate={summary['study_unscoreable_rate']:.4f} "
          f"(trigger {STUDY_UNSCOREABLE_TRIGGER}) exceeded={summary['study_trigger_exceeded']}")
    print(f"communities flagged={len(flagged)}  min posts used={summary['min_posts_used']}")
    for name, c in checks.items():
        if name.startswith("unscoreable"):
            print(f"  {name}: r={c['pearson_r']:+.3f} (p={c['pearson_p']}) rho={c['spearman_rho']:+.3f}")


if __name__ == "__main__":
    main()
