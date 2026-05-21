#!/usr/bin/env python3
"""Manual-batch helper for dual-pass ACSI measurement inside Codex/ChatGPT.

Use `next` to print unmeasured rows for one scoring run, `append` to add a JSON
batch to that run's measurement CSV, `summary` to inspect progress, `combine` to
create the final averaged file after both runs are complete, and `aggregate` to
write subreddit-level ACSI component scores.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_INPUT = ROOT / "data" / "acsi_data.csv"
DEFAULT_RUN1_OUTPUT = ROOT / "data" / "acsi_measurement_sample_run1.csv"
DEFAULT_RUN2_OUTPUT = ROOT / "data" / "acsi_measurement_sample_run2.csv"
DEFAULT_FINAL_OUTPUT = ROOT / "data" / "acsi_measurement_sample.csv"
DEFAULT_ACSI_SCORE_OUTPUT = ROOT / "data" / "acsi_scores.csv"

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
    remaining, done, total = load_remaining(args.input, output_path)
    if args.subreddit:
        requested_subreddits = {subreddit.strip().lower() for subreddit in args.subreddit if subreddit.strip()}
        remaining = remaining[remaining["subreddit"].str.lower().isin(requested_subreddits)].copy()
    batch = remaining.head(args.n)
    print(f"run={args.run} output={output_path}")
    print(f"done={done} total={total} remaining={len(remaining)} batch={len(batch)}")
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

    if missing:
        raise ValueError(
            "No usable measurement rows for these subreddits after exclusions: "
            + ", ".join(missing)
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


def main():
    parser = argparse.ArgumentParser(description="Manual dual-pass ACSI coding helper.")
    parser.add_argument("command", choices=["next", "append", "summary", "combine", "aggregate"])
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--run", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-file", type=Path, default=Path("-"))
    parser.add_argument("--subreddit", action="append", default=[])
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
