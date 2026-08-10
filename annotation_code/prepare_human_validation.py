#!/usr/bin/env python3
"""Produce blinded scoring sheets for the human validation raters.

Takes the 300-post validation subset drawn by select_subsets.py and writes one
sheet per rater containing the post text and empty score columns. The sheets
carry no community name, no category, and no model score, and each rater's
sheet is shuffled with its own seed so the two raters see the posts in
different orders. Both properties are required by the protocol: a rater who
can see the community can score its reputation rather than the post, and a
shared order lets fatigue or drift correlate across raters.

The unblinding key is written separately and should not be opened until both
sheets are complete.

Usage:
    python annotation_code/prepare_human_validation.py --config sampling_config_creation_relative.json
    # each rater fills the five score columns, saves as CSV, then:
    python agreement.py --a-file rater_derek.csv --b-file rater_arya.csv \
        --label human_vs_human --acceptance human
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# This module lives in annotation_code/ but pipeline_utils sits at the repo
# root, so make the parent importable whether the script is run from here or
# from the root via the .sh wrapper.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline_utils import load_config, resolve_path, sha256_file, write_json

SCORE_FIELDS = (
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
)
RATERS = {"derek": 101, "arya": 202}

csv.field_size_limit(10_000_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sampling_config_creation_relative.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    run_dir = resolve_path(base_dir, config["paths"]["run_dir"])
    subset_path = run_dir / "08_subsets" / "human_validation_subset.csv"
    posts_path = run_dir / "05_annotation_prep" / "annotation_posts_blinded.csv"
    out_dir = run_dir / "17_human_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (subset_path, posts_path):
        if not path.exists():
            raise FileNotFoundError(path)

    wanted: dict[str, str] = {}
    with subset_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            wanted[row["annotation_id"]] = row.get("category", "")

    posts: dict[str, dict[str, str]] = {}
    with posts_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["annotation_id"] in wanted:
                posts[row["annotation_id"]] = row
    missing = set(wanted) - set(posts)
    if missing:
        raise ValueError(f"{len(missing)} validation posts are not in the blinded post file")

    for rater, seed in RATERS.items():
        ids = sorted(posts)
        random.Random(seed).shuffle(ids)
        path = out_dir / f"rater_{rater}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "annotation_id", "year_month", "text", *SCORE_FIELDS])
            for i, aid in enumerate(ids, start=1):
                writer.writerow([i, aid, posts[aid]["year_month"], posts[aid]["text"],
                                 *[""] * len(SCORE_FIELDS)])
        print(f"wrote {path.name}: {len(ids)} posts, shuffle seed {seed}")

    key_path = out_dir / "validation_key.csv"
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["annotation_id", "category"])
        for aid in sorted(wanted):
            writer.writerow([aid, wanted[aid]])

    write_json(out_dir / "packet_summary.json", {
        "posts": len(posts),
        "raters": list(RATERS),
        "shuffle_seeds": RATERS,
        "blinded_fields": ["subreddit", "category", "author", "model scores"],
        "outputs": {f"rater_{r}.csv": sha256_file(out_dir / f"rater_{r}.csv") for r in RATERS},
        "key_sha256": sha256_file(key_path),
    })

    print(f"\nkey written separately to {key_path.name}; do not open it until scoring is done")
    print("\nEach rater scores five dimensions 0-3 per the frozen rubric, working from the")
    print("text alone. Do not confer, and do not look at the other sheet.")


if __name__ == "__main__":
    main()
