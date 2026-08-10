#!/usr/bin/env python3
"""Prepare a blinded, reproducible post sample for the annotation stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# This module lives in annotation_code/ but pipeline_utils sits at the repo
# root, so make the parent importable whether the script is run from here or
# from the root via the .sh wrapper.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from pipeline_utils import (
    iter_zstd_json,
    load_config,
    reddit_post_id,
    resolve_path,
    sha256_file,
    sha256_text,
    submission_files,
    subreddit_key,
    usable_text,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sampling_config_creation_relative.json")
    return parser.parse_args()


def read_final_sample(path: Path) -> dict[str, dict[str, str]]:
    sample: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = subreddit_key(row.get("subreddit"))
            if not key or key in sample:
                raise ValueError(f"Blank or duplicate final-sample row: {row}")
            sample[key] = {"subreddit": row["subreddit"].strip(), "category": row["category"].strip()}
    return sample


def update_reservoir(
    reservoir: list[dict[str, str]],
    item: dict[str, str],
    seen_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    index = rng.randrange(seen_count)
    if index < capacity:
        reservoir[index] = item


def priority(seed: int, subreddit: str, post_id: str) -> str:
    value = f"{seed}|{subreddit}|{post_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    paths = config["paths"]
    rule = config["annotation"]
    if rule.get("maximum_posts_per_author") is None:
        raise ValueError(
            "Set annotation.maximum_posts_per_author in sampling_config.json "
            "before preparing annotation posts"
        )
    author_cap = int(rule["maximum_posts_per_author"])
    if author_cap <= 0:
        raise ValueError("annotation.maximum_posts_per_author must be positive")
    if rule.get("sampling_mode") != "uniform_over_eligible_posts":
        raise ValueError(f"Unsupported annotation sampling mode: {rule.get('sampling_mode')!r}")

    run_dir = resolve_path(base_dir, paths["run_dir"])
    sample_path = run_dir / "04_sample" / "final_sample.csv"
    output_dir = run_dir / "05_annotation_prep"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = read_final_sample(sample_path)
    posts_per_subreddit = int(rule["posts_per_subreddit"])
    oversample_factor = int(rule["reservoir_oversample_factor"])
    capacity = posts_per_subreddit * oversample_factor
    rng = random.Random(int(rule["reservoir_seed"]))
    files = submission_files(
        resolve_path(base_dir, paths["submissions_dir"]),
        rule["start_month"],
        rule["end_month"],
    )

    reservoirs: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_usable: dict[str, int] = defaultdict(int)
    for file_number, path in enumerate(files, start=1):
        year_month = path.stem.removeprefix("RS_")
        print(f"Annotation reservoir: {year_month} ({file_number}/{len(files)})", flush=True)
        for post in iter_zstd_json(path):
            key = subreddit_key(post.get("subreddit"))
            if key not in sample:
                continue
            author = post.get("author")
            if not isinstance(author, str) or not author.strip() or author.casefold() == "[deleted]":
                continue
            post_id = reddit_post_id(post)
            text = usable_text(
                post,
                min_chars=int(rule["minimum_text_characters"]),
                max_chars=int(rule["maximum_text_characters"]),
            )
            if not post_id or text is None:
                continue
            seen_usable[key] += 1
            update_reservoir(
                reservoirs[key],
                {
                    "post_id": post_id,
                    "year_month": year_month,
                    "author": author.strip(),
                    "text": text,
                    "text_sha256": sha256_text(text),
                },
                seen_count=seen_usable[key],
                capacity=capacity,
                rng=rng,
            )

    selected: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    seed = int(rule["reservoir_seed"])
    for key, metadata in sorted(sample.items(), key=lambda item: item[0]):
        candidates = sorted(
            reservoirs.get(key, []),
            key=lambda row: priority(seed, key, row["post_id"]),
        )
        author_counts: dict[str, int] = defaultdict(int)
        chosen: list[dict[str, str]] = []
        for row in candidates:
            author_key = row["author"].casefold()
            if author_counts[author_key] >= author_cap:
                continue
            author_counts[author_key] += 1
            chosen.append(row)
            if len(chosen) == posts_per_subreddit:
                break
        if len(chosen) < posts_per_subreddit:
            shortfalls.append(
                {
                    "subreddit": metadata["subreddit"],
                    "category": metadata["category"],
                    "usable_posts_seen": seen_usable.get(key, 0),
                    "reservoir_posts": len(candidates),
                    "selected_after_author_cap": len(chosen),
                    "required": posts_per_subreddit,
                }
            )
        for row in chosen:
            annotation_id = priority(seed, key, row["post_id"])[:20]
            selected.append(
                {
                    "annotation_id": annotation_id,
                    "subreddit": metadata["subreddit"],
                    "category": metadata["category"],
                    **row,
                }
            )

    shortfall_path = output_dir / "annotation_shortfalls.csv"
    with shortfall_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "subreddit",
            "category",
            "usable_posts_seen",
            "reservoir_posts",
            "selected_after_author_cap",
            "required",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shortfalls)
    if shortfalls:
        write_json(
            output_dir / "annotation_prep_summary.json",
            {
                "status": "blocked_by_shortfalls",
                "shortfall_communities": len(shortfalls),
                "shortfall_file_sha256": sha256_file(shortfall_path),
            },
        )
        raise RuntimeError(
            f"{len(shortfalls)} communities have fewer than {posts_per_subreddit} posts "
            "after the author cap; see annotation_shortfalls.csv"
        )

    random.Random(seed + 1).shuffle(selected)
    blinded_path = output_dir / "annotation_posts_blinded.csv"
    key_path = output_dir / "annotation_posts_key.csv"
    with blinded_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["annotation_id", "post_id", "year_month", "text"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    with key_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "annotation_id",
            "subreddit",
            "category",
            "post_id",
            "year_month",
            "author",
            "text_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "status": "complete",
        "communities": len(sample),
        "posts_per_subreddit": posts_per_subreddit,
        "total_posts": len(selected),
        "window": {"start": rule["start_month"], "end": rule["end_month"]},
        "sampling_mode": rule["sampling_mode"],
        "maximum_posts_per_author": author_cap,
        "reservoir_seed": seed,
        "reservoir_oversample_factor": oversample_factor,
        "input_sample_sha256": sha256_file(sample_path),
        "outputs": {
            blinded_path.name: sha256_file(blinded_path),
            key_path.name: sha256_file(key_path),
            shortfall_path.name: sha256_file(shortfall_path),
        },
    }
    write_json(output_dir / "annotation_prep_summary.json", summary)
    print(f"Annotation preparation complete: {len(selected):,} blinded posts")


if __name__ == "__main__":
    main()
