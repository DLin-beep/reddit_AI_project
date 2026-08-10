#!/usr/bin/env python3
"""Draw the reliability and human-validation subsets specified in the protocol.

Both draws are deterministic and depend only on the frozen annotation set, so
they can be reproduced from the archived inputs:

  * reliability subset: exactly 30 posts from each of the 200 communities
    (6,000 posts), NumPy default_rng(seed=42), used for BOTH the
    self-consistency and cross-model checks.
  * human-validation subset: 300 posts allocated across categories by the
    largest-remainder proportional rule with a floor of one per category,
    NumPy default_rng(seed=44).

Candidates are sorted by annotation identifier before every draw so the result
does not depend on file order. Writes id lists that annotate.py consumes with
--only-ids, keeping the scoring script blind to community identity.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# This module lives in annotation_code/ but pipeline_utils sits at the repo
# root, so make the parent importable whether the script is run from here or
# from the root via the .sh wrapper.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline_utils import load_config, resolve_path, sha256_file, write_json

RELIABILITY_PER_COMMUNITY = 30
RELIABILITY_SEED = 42
VALIDATION_TOTAL = 300
VALIDATION_SEED = 44
VALIDATION_FLOOR = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sampling_config_creation_relative.json")
    return parser.parse_args()


def largest_remainder(sizes: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """Allocate `total` seats proportional to `sizes`, floor per stratum, ties alphabetical."""
    population = sum(sizes.values())
    if total > population:
        raise ValueError(f"Cannot draw {total} posts from a population of {population}")
    quota = {k: total * v / population for k, v in sizes.items()}
    alloc = {k: min(sizes[k], max(floor, int(q))) for k, q in quota.items()}
    while sum(alloc.values()) != total:
        short = total - sum(alloc.values())
        if short > 0:
            pool = [k for k in sizes if alloc[k] < sizes[k]]
            pool.sort(key=lambda k: (-(quota[k] - int(quota[k])), k))
            for k in pool[:short]:
                alloc[k] += 1
        else:
            pool = [k for k in sizes if alloc[k] > floor]
            pool.sort(key=lambda k: (quota[k] - int(quota[k]), k))
            for k in pool[: -short]:
                alloc[k] -= 1
        if not pool:
            raise RuntimeError("Cannot reach the target allocation under the floor")
    return alloc


def write_ids(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_dir = resolve_path(base_dir, config["paths"]["run_dir"])
    key_path = run_dir / "05_annotation_prep" / "annotation_posts_key.csv"
    out_dir = run_dir / "08_subsets"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_community: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    category_of: dict[str, str] = {}
    with key_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            aid = row["annotation_id"]
            by_community[row["subreddit"]].append(aid)
            by_category[row["category"]].append(aid)
            category_of[aid] = row["category"]

    # --- reliability subset: fixed count per community
    rng = np.random.default_rng(RELIABILITY_SEED)
    reliability: list[dict[str, str]] = []
    for subreddit in sorted(by_community, key=str.casefold):
        candidates = sorted(by_community[subreddit])
        if len(candidates) < RELIABILITY_PER_COMMUNITY:
            raise RuntimeError(f"{subreddit} has only {len(candidates)} posts")
        picked = rng.choice(len(candidates), size=RELIABILITY_PER_COMMUNITY, replace=False)
        for index in sorted(picked):
            reliability.append({"annotation_id": candidates[index], "subreddit": subreddit})

    # --- human-validation subset: proportional across categories
    sizes = {c: len(v) for c, v in by_category.items()}
    alloc = largest_remainder(sizes, VALIDATION_TOTAL, VALIDATION_FLOOR)
    rng = np.random.default_rng(VALIDATION_SEED)
    validation: list[dict[str, str]] = []
    for category in sorted(by_category):
        candidates = sorted(by_category[category])
        picked = rng.choice(len(candidates), size=alloc[category], replace=False)
        for index in sorted(picked):
            validation.append({"annotation_id": candidates[index], "category": category})

    reliability_path = out_dir / "reliability_subset.csv"
    validation_path = out_dir / "human_validation_subset.csv"
    write_ids(reliability_path, reliability, ["annotation_id", "subreddit"])
    write_ids(validation_path, validation, ["annotation_id", "category"])

    overlap = {r["annotation_id"] for r in reliability} & {v["annotation_id"] for v in validation}
    write_json(
        out_dir / "subset_summary.json",
        {
            "reliability": {
                "posts": len(reliability),
                "per_community": RELIABILITY_PER_COMMUNITY,
                "communities": len(by_community),
                "seed": RELIABILITY_SEED,
                "sha256": sha256_file(reliability_path),
            },
            "human_validation": {
                "posts": len(validation),
                "categories": len(alloc),
                "seed": VALIDATION_SEED,
                "floor": VALIDATION_FLOOR,
                "allocation": dict(sorted(alloc.items())),
                "sha256": sha256_file(validation_path),
            },
            "overlap_posts": len(overlap),
            "input_key_sha256": sha256_file(key_path),
        },
    )
    print(f"reliability={len(reliability)} posts over {len(by_community)} communities")
    print(f"human_validation={len(validation)} posts over {len(alloc)} categories")
    print(f"overlap between the two subsets: {len(overlap)} posts")


if __name__ == "__main__":
    main()
