import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "latest"
TABLES_DIR = OUTPUT_DIR / "tables"

MEASUREMENT_PATH = DATA_DIR / "acsi_measurement_sample_run1.csv"
RAW_METADATA_PATH = TABLES_DIR / "post_level_raw_metadata.parquet"
MODEL_TABLE_PATH = TABLES_DIR / "post_level_acsi_models.csv"
SAMPLE_TABLE_PATH = TABLES_DIR / "post_level_acsi_sample_summary.csv"

SCORE_COLUMNS = [
    "direct_gen_score",
    "usefulness_score",
    "quality_comp_score",
    "physical_req_score",
    "personal_req_score",
]


def load_measurement_posts(measurement_path):
    measurement = pd.read_csv(measurement_path)
    required = {
        "subreddit",
        "post_id",
        "year_month",
        "created_date",
        "ai_related_flag",
        *SCORE_COLUMNS,
    }
    missing = sorted(required - set(measurement.columns))
    if missing:
        raise ValueError(f"{measurement_path} missing required columns: {missing}")

    measurement["subreddit"] = measurement["subreddit"].astype(str).str.strip()
    measurement["post_id"] = measurement["post_id"].astype(str).str.strip()
    measurement = measurement[measurement["ai_related_flag"].fillna(0).astype(int).ne(1)].copy()
    measurement = measurement.drop_duplicates(["subreddit", "post_id"], keep="last")

    for column in SCORE_COLUMNS:
        measurement[column] = pd.to_numeric(measurement[column], errors="coerce")
        bad = measurement[~measurement[column].between(0, 3)]
        if not bad.empty:
            raise ValueError(f"{column} has values outside 0-3")

    measurement["created_date"] = pd.to_datetime(measurement["created_date"], errors="coerce")
    measurement["year_month"] = measurement["year_month"].astype(str)
    measurement["post_shock"] = (
        pd.to_datetime(measurement["year_month"] + "-01") >= pd.Timestamp("2022-11-01")
    ).astype(int)

    generation_sum = (
        measurement["direct_gen_score"]
        + measurement["usefulness_score"]
        + measurement["quality_comp_score"]
    )
    measurement["post_acsi"] = (
        generation_sum
        + (3 - measurement["physical_req_score"])
        + (3 - measurement["personal_req_score"])
    ) / 15.0
    measurement["generation_capability"] = generation_sum / 9.0
    measurement["low_physical_constraint"] = (3 - measurement["physical_req_score"]) / 3.0
    measurement["low_personal_context_need"] = (3 - measurement["personal_req_score"]) / 3.0

    for column in [
        "direct_gen_score",
        "usefulness_score",
        "quality_comp_score",
        "physical_req_score",
        "personal_req_score",
    ]:
        measurement[column.replace("_score", "_norm")] = measurement[column] / 3.0

    return measurement


def metadata_has_all_posts(metadata, measurement):
    if metadata is None or metadata.empty:
        return False
    metadata_keys = set(zip(metadata["subreddit"].astype(str), metadata["post_id"].astype(str)))
    measurement_keys = set(zip(measurement["subreddit"].astype(str), measurement["post_id"].astype(str)))
    return measurement_keys.issubset(metadata_keys)


def load_raw_metadata(measurement, metadata_path=RAW_METADATA_PATH):
    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        if metadata_has_all_posts(metadata, measurement):
            return metadata

    targets_by_subreddit = {
        subreddit: set(group["post_id"].astype(str))
        for subreddit, group in measurement.groupby("subreddit", sort=True)
    }
    rows = []
    missing_files = []
    missing_ids = {}

    for subreddit, wanted_ids in targets_by_subreddit.items():
        path = DATA_DIR / f"r_{subreddit}_posts.jsonl"
        if not path.exists():
            missing_files.append(subreddit)
            continue

        found_ids = set()
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(found_ids) == len(wanted_ids):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    post = json.loads(line)
                except Exception:
                    continue
                post_id = str(post.get("id") or "")
                if post_id not in wanted_ids:
                    continue
                rows.append(
                    {
                        "subreddit": subreddit,
                        "post_id": post_id,
                        "author": str(post.get("author") or ""),
                        "score": safe_numeric(post.get("score")),
                        "num_comments": safe_numeric(post.get("num_comments")),
                        "created_utc": safe_numeric(post.get("created_utc")),
                        "subreddit_subscribers": safe_numeric(post.get("subreddit_subscribers")),
                    }
                )
                found_ids.add(post_id)

        missing = sorted(wanted_ids - found_ids)
        if missing:
            missing_ids[subreddit] = len(missing)

    if missing_files:
        raise FileNotFoundError(f"Missing raw post files for subreddits: {missing_files}")
    if not rows:
        raise ValueError("No measurement posts were found in raw JSONL files.")

    metadata = pd.DataFrame(rows)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(metadata_path, index=False)

    if missing_ids:
        print(f"WARNING: missing raw metadata for {sum(missing_ids.values())} measurement posts: {missing_ids}")
    return metadata


def safe_numeric(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def prepare_model_data(measurement, raw_metadata):
    raw_cols = [
        "subreddit",
        "post_id",
        "author",
        "score",
        "num_comments",
        "subreddit_subscribers",
    ]
    data = measurement.merge(raw_metadata[raw_cols], on=["subreddit", "post_id"], how="inner")
    data = data[~data["author"].isin(["", "[deleted]", "AutoModerator"])].copy()
    data = data[~data["author"].str.lower().str.endswith("bot", na=False)].copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["num_comments"] = pd.to_numeric(data["num_comments"], errors="coerce")
    data["log_score"] = np.log1p(data["score"].fillna(0).clip(lower=0))
    data["log_comments"] = np.log1p(data["num_comments"].fillna(0).clip(lower=0))
    data["post_acsi_x_post"] = data["post_acsi"] * data["post_shock"]
    data["generation_capability_x_post"] = data["generation_capability"] * data["post_shock"]
    data["low_physical_constraint_x_post"] = data["low_physical_constraint"] * data["post_shock"]
    data["low_personal_context_need_x_post"] = data["low_personal_context_need"] * data["post_shock"]
    for column in [
        "direct_gen_norm",
        "usefulness_norm",
        "quality_comp_norm",
        "physical_req_norm",
        "personal_req_norm",
    ]:
        data[f"{column}_x_post"] = data[column] * data["post_shock"]
    return data


def residualize_against_fixed_effects(frame, columns, fixed_effects, max_iter=200, tol=1e-10):
    residuals = frame[columns].astype(float).to_numpy(copy=True)
    residuals -= np.nanmean(residuals, axis=0)

    fe_codes = [pd.Categorical(frame[fe]).codes for fe in fixed_effects]
    for _ in range(max_iter):
        before = residuals.copy()
        for codes in fe_codes:
            tmp = pd.DataFrame(residuals)
            means = tmp.groupby(codes, sort=False).transform("mean").to_numpy()
            residuals -= means
        max_change = np.max(np.abs(residuals - before))
        if max_change < tol:
            break

    return pd.DataFrame(residuals, columns=columns, index=frame.index)


def fit_absorbed_ols(data, outcome, regressors, fixed_effects, cluster_col):
    required = [outcome, cluster_col, *regressors, *fixed_effects]
    model_data = data.dropna(subset=required).copy()
    if model_data.empty:
        return None

    residualized = residualize_against_fixed_effects(model_data, [outcome, *regressors], fixed_effects)
    y = residualized[outcome]
    x = residualized[regressors]
    nonzero = x.abs().sum(axis=1).gt(1e-12) | y.abs().gt(1e-12)
    model_data = model_data.loc[nonzero].copy()
    y = y.loc[nonzero]
    x = x.loc[nonzero]
    if len(model_data) <= len(regressors):
        return None

    fit = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_data[cluster_col].astype(str)},
    )
    return fit, model_data


def result_rows(data, outcome, model_name, regressors, terms_of_interest, fixed_effects, cluster_col):
    fit_result = fit_absorbed_ols(data, outcome, regressors, fixed_effects, cluster_col)
    if fit_result is None:
        return []
    fit, model_data = fit_result
    rows = []
    for term in terms_of_interest:
        coef = safe_float_from_series(fit.params, term)
        rows.append(
            {
                "outcome": outcome,
                "model": model_name,
                "term": term,
                "coef": coef,
                "se": safe_float_from_series(fit.bse, term),
                "p_value": safe_float_from_series(fit.pvalues, term),
                "full_exposure_effect_pct": None if coef is None else 100 * (np.exp(coef) - 1),
                "n_obs": int(fit.nobs),
                "n_authors": int(model_data["author"].nunique()),
                "n_subreddits": int(model_data["subreddit"].nunique()),
                "n_months": int(model_data["year_month"].nunique()),
                "fixed_effects": "+".join(fixed_effects),
                "cluster": cluster_col,
            }
        )
    return rows


def safe_float_from_series(series, key):
    try:
        value = float(series.get(key, np.nan))
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def write_sample_summary(data):
    summary = pd.DataFrame(
        [
            {"metric": "measurement_non_ai_posts_with_raw_metadata", "value": len(data)},
            {"metric": "authors", "value": data["author"].nunique()},
            {"metric": "authors_with_2plus_measurement_posts", "value": int(data.groupby("author").size().ge(2).sum())},
            {"metric": "subreddits", "value": data["subreddit"].nunique()},
            {"metric": "months", "value": data["year_month"].nunique()},
            {"metric": "pre_posts", "value": int((data["post_shock"] == 0).sum())},
            {"metric": "post_posts", "value": int((data["post_shock"] == 1).sum())},
            {"metric": "post_acsi_mean", "value": float(data["post_acsi"].mean())},
            {"metric": "post_acsi_sd", "value": float(data["post_acsi"].std())},
            {"metric": "score_mean", "value": float(data["score"].mean())},
            {"metric": "num_comments_mean", "value": float(data["num_comments"].mean())},
        ]
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SAMPLE_TABLE_PATH, index=False)


def run_models(data):
    rows = []
    outcomes = ["log_score", "log_comments"]
    fixed_effect_sets = [
        ("subreddit_month_fe", ["subreddit", "year_month"], "subreddit"),
        ("author_month_fe", ["author", "year_month"], "subreddit"),
        ("author_subreddit_month_fe", ["author", "subreddit", "year_month"], "subreddit"),
    ]

    for outcome in outcomes:
        for fe_name, fixed_effects, cluster_col in fixed_effect_sets:
            rows.extend(
                result_rows(
                    data=data,
                    outcome=outcome,
                    model_name=f"{fe_name}: post-level composite",
                    regressors=["post_acsi", "post_acsi_x_post"],
                    terms_of_interest=["post_acsi_x_post"],
                    fixed_effects=fixed_effects,
                    cluster_col=cluster_col,
                )
            )
            rows.extend(
                result_rows(
                    data=data,
                    outcome=outcome,
                    model_name=f"{fe_name}: joint mechanisms",
                    regressors=[
                        "generation_capability",
                        "low_physical_constraint",
                        "low_personal_context_need",
                        "generation_capability_x_post",
                        "low_physical_constraint_x_post",
                        "low_personal_context_need_x_post",
                    ],
                    terms_of_interest=[
                        "generation_capability_x_post",
                        "low_physical_constraint_x_post",
                        "low_personal_context_need_x_post",
                    ],
                    fixed_effects=fixed_effects,
                    cluster_col=cluster_col,
                )
            )
            rows.extend(
                result_rows(
                    data=data,
                    outcome=outcome,
                    model_name=f"{fe_name}: five dimensions",
                    regressors=[
                        "direct_gen_norm",
                        "usefulness_norm",
                        "quality_comp_norm",
                        "physical_req_norm",
                        "personal_req_norm",
                        "direct_gen_norm_x_post",
                        "usefulness_norm_x_post",
                        "quality_comp_norm_x_post",
                        "physical_req_norm_x_post",
                        "personal_req_norm_x_post",
                    ],
                    terms_of_interest=[
                        "direct_gen_norm_x_post",
                        "usefulness_norm_x_post",
                        "quality_comp_norm_x_post",
                        "physical_req_norm_x_post",
                        "personal_req_norm_x_post",
                    ],
                    fixed_effects=fixed_effects,
                    cluster_col=cluster_col,
                )
            )

    output = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(MODEL_TABLE_PATH, index=False)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run post-level ACSI regressions without subreddit-level aggregation.")
    parser.add_argument("--measurement-path", dest="measurement_path", type=Path, default=MEASUREMENT_PATH)
    parser.add_argument("--coded-path", dest="measurement_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--metadata-path", type=Path, default=RAW_METADATA_PATH)
    parser.add_argument("--rebuild-metadata", dest="rebuild_metadata", action="store_true")
    parser.add_argument("--rebuild-cache", dest="rebuild_metadata", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/pycache")
    args = parse_args()
    if args.rebuild_metadata and args.metadata_path.exists():
        args.metadata_path.unlink()

    measurement = load_measurement_posts(args.measurement_path)
    raw_metadata = load_raw_metadata(measurement, args.metadata_path)
    data = prepare_model_data(measurement, raw_metadata)
    write_sample_summary(data)
    results = run_models(data)

    print(f"measurement_non_ai_posts={len(measurement):,}")
    print(f"post_level_model_posts={len(data):,}")
    print(f"authors={data['author'].nunique():,}")
    print(f"authors_with_2plus_measurement_posts={int(data.groupby('author').size().ge(2).sum()):,}")
    print(f"wrote={MODEL_TABLE_PATH}")
    print(f"wrote={SAMPLE_TABLE_PATH}")
    focus = results[
        (results["outcome"].eq("log_score"))
        & (results["model"].str.contains("author_subreddit_month_fe"))
    ]
    if not focus.empty:
        print("\nlog_score, strict author+subreddit+month FE:")
        print(
            focus[
                ["model", "term", "coef", "se", "p_value", "full_exposure_effect_pct", "n_obs"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
