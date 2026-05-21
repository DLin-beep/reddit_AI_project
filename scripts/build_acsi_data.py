#!/usr/bin/env python3
"""Build run-once ACSI measurement input from cleaned post outputs.

The output is for mechanism scoring, not regression estimation. It is balanced
across pre/post ChatGPT periods and spread as evenly as the available cached
data allow across active subreddits.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_INPUT = ROOT / "output" / "latest" / "posts_clean_all.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "acsi_data.csv"

SCORE_COLUMNS = [
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

    for col in SCORE_COLUMNS:
        sampled[col] = ""

    out_cols = [
        "subreddit",
        "post_id",
        "period",
        "created_date",
        "year_month",
        "title_excerpt",
        "selftext_excerpt",
        *SCORE_COLUMNS,
        "rationale",
        "version",
    ]

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


def main():
    parser = argparse.ArgumentParser(description="Build run-once ACSI measurement input.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = build_acsi_data(
        input_path=args.input,
        output_csv=args.output,
        total_rows=args.total_rows,
        seed=args.seed,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
