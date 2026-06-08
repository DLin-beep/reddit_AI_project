"""Runtime validation and test gates for run.py.

This module owns the pipeline's validation/test checks; run.py binds its live
context here before calling these functions.
"""


def bind_context(ctx):
    for name in dir(ctx):
        if name.startswith("__"):
            continue
        existing = globals().get(name)
        if getattr(existing, "__module__", None) == __name__:
            continue
        globals()[name] = getattr(ctx, name)


def compute_content_validation_sample(top_subs=None):
    print("\n=== content validation sample ===")
    if top_subs is None:
        top_subs = ["art", "writing", "applyingtocollege", "poetry", "fanfiction"]

    rows = []
    rng_cv = random.Random(RANDOM_SEED)
    for sub in top_subs:
        path = raw_post_path(sub)
        if not path.exists(): continue
        pre_pool, post_pool = [], []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if MAX_LINES_PER_FILE is not None and i >= MAX_LINES_PER_FILE:
                    break
                line = line.strip()
                if not line: continue
                try: p = json.loads(line)
                except Exception: continue
                
                ts = p.get("created_utc")
                if ts is None: continue
                try: dt = datetime.utcfromtimestamp(int(ts))
                except Exception: continue
                if dt < START_DATE or dt >= END_DATE_EXCLUSIVE: continue
                
                author = str(p.get("author") or "")
                if author in EXCLUDED_AUTHORS or author.lower().endswith("bot"): continue
                
                title    = str(p.get("title") or "")
                selftext = str(p.get("selftext") or "")
                if selftext in {"[removed]", "[deleted]"}: selftext = ""
                full_text = (title + " " + selftext).strip()
                if len(full_text) < 10: continue
                
                words = full_text.split()
                ttr   = len(set(w.lower() for w in words)) / max(len(words), 1)
                month = month_start_for_datetime(dt)
                rec   = {
                    "subreddit":     sub,
                    "author":        author,
                    "date":          dt.strftime("%Y-%m-%d"),
                    "post_shock":    int(month >= SHOCK_MONTH),
                    "post_shock_exact": int(dt >= EXACT_SHOCK_DATE),
                    "word_count":    len(words),
                    "ttr":           round(ttr, 4),
                    "title_preview": title[:120],
                }
                if month < SHOCK_MONTH: pre_pool.append(rec)
                else: post_pool.append(rec)

        sampled = rng_cv.sample(pre_pool,  min(60, len(pre_pool))) + \
                  rng_cv.sample(post_pool, min(60, len(post_pool)))
        rows.extend(sampled)

    if not rows: return {"n_sampled": 0, "caution": "No text posts found."}

    df_cv = pd.DataFrame(rows)
    out_path = TABLES_DIR / "content_validation_sample.csv"
    emit_output_table(df_cv, out_path, index=False)

    result = {"n_sampled": len(df_cv), "caution": "Heuristic only — not diagnostic of AI use."}
    for label, col in [("word_count", "word_count"), ("ttr", "ttr")]:
        pre_v  = df_cv[df_cv["post_shock"] == 0][col].values
        post_v = df_cv[df_cv["post_shock"] == 1][col].values
        if len(pre_v) > 1 and len(post_v) > 1:
            t, p = stats.ttest_ind(pre_v, post_v, equal_var=False)
            result[f"pre_mean_{label}"]  = float(np.mean(pre_v))
            result[f"post_mean_{label}"] = float(np.mean(post_v))
            result[f"pvalue_{label}"]    = float(p)
    print(f"  prepared {len(df_cv)} sampled posts")
    return result

def subreddit_set_from_frame(frame):
    if frame is None or frame.empty or "subreddit" not in frame.columns:
        return set()
    return set(frame["subreddit"].dropna().astype(str).unique())

def format_subreddit_diagnostic_list(subreddits):
    if not subreddits:
        return "none"
    return ", ".join(sorted(subreddits))

def print_subreddit_coverage_diagnostics(
    ecosystem_posts,
    acsi_scores,
    creator_posts=None,
    ecosystem_label="ecosystem-clean post sample",
):
    ecosystem_subreddits = subreddit_set_from_frame(ecosystem_posts)
    acsi_subreddits = subreddit_set_from_frame(acsi_scores)
    creator_subreddits = subreddit_set_from_frame(creator_posts)
    scored_not_ecosystem = acsi_subreddits - ecosystem_subreddits
    ecosystem_not_scored = ecosystem_subreddits - acsi_subreddits

    print("\n=== Subreddit coverage diagnostics ===")
    print(f"  Unique subreddits in {ecosystem_label}: {len(ecosystem_subreddits):,}")
    print(f"  Unique subreddits in ACSI scores dataframe: {len(acsi_subreddits):,}")
    print(
        f"  Subreddits in ACSI scores but not in {ecosystem_label} "
        f"({len(scored_not_ecosystem):,}): {format_subreddit_diagnostic_list(scored_not_ecosystem)}"
    )
    print(
        f"  Subreddits in {ecosystem_label} but not in ACSI scores "
        f"({len(ecosystem_not_scored):,}): {format_subreddit_diagnostic_list(ecosystem_not_scored)}"
    )
    if creator_posts is None:
        print("  Unique subreddits in creator-clean sample: unavailable in this run mode")
    else:
        print(f"  Unique subreddits in creator-clean sample: {len(creator_subreddits):,}")

def require_file(path, validation_errors, label=None):
    required_path = Path(path)
    display_label = label or str(required_path)
    if not required_path.exists():
        validation_errors.append(f"Missing required output: {display_label}")
        return
    if required_path.is_file() and required_path.stat().st_size == 0:
        validation_errors.append(f"Required output is empty: {display_label}")

def validate_numeric_range(table, column_name, minimum_value, maximum_value, validation_errors, table_label):
    if column_name not in table.columns:
        validation_errors.append(f"{table_label} missing required column: {column_name}")
        return
    numeric_values = pd.to_numeric(table[column_name], errors="coerce")
    invalid_value_mask = (
        numeric_values.isna()
        | ~np.isfinite(numeric_values)
        | (numeric_values < minimum_value)
        | (numeric_values > maximum_value)
    )
    if bool(invalid_value_mask.any()):
        example_columns = ["subreddit", column_name] if "subreddit" in table.columns else [column_name]
        examples = table.loc[invalid_value_mask, example_columns].head(10).to_dict("records")
        validation_errors.append(
            f"{table_label} column {column_name} has values outside "
            f"[{minimum_value}, {maximum_value}]; examples={examples}"
        )

def validate_nonnegative_numeric(table, column_name, validation_errors, table_label):
    if column_name not in table.columns:
        validation_errors.append(f"{table_label} missing required column: {column_name}")
        return
    numeric_values = pd.to_numeric(table[column_name], errors="coerce")
    invalid_value_mask = numeric_values.isna() | ~np.isfinite(numeric_values) | (numeric_values < 0)
    if bool(invalid_value_mask.any()):
        validation_errors.append(f"{table_label} column {column_name} has missing, infinite, or negative values")

def validate_finite_numeric(table, column_name, validation_errors, table_label):
    if column_name not in table.columns:
        validation_errors.append(f"{table_label} missing required column: {column_name}")
        return
    numeric_values = pd.to_numeric(table[column_name], errors="coerce")
    invalid_value_mask = numeric_values.isna() | ~np.isfinite(numeric_values)
    if bool(invalid_value_mask.any()):
        validation_errors.append(f"{table_label} column {column_name} has missing, infinite, or nonnumeric values")

def expected_panel_subreddits(score_table=None):
    subreddits = set(available_post_subreddits())
    if score_table is not None and "subreddit" in score_table.columns:
        scored_subreddits = set(score_table["subreddit"].astype(str))
        subreddits &= scored_subreddits
    return sorted(subreddits)

def validate_subreddit_month_panel(
    panel,
    score_table,
    validation_errors,
    panel_label="subreddit-month panel",
    expected_subreddits=None,
):
    if panel is None or panel.empty:
        validation_errors.append(f"{panel_label} is empty")
        return

    required_columns = [
        "subreddit", "year_month", "year_month_dt", "post_shock",
        "posts", "active_creators", "calibrated_output", "avg_score",
        "log_posts", "log_active_creators", "log_calibrated_output", "log_avg_score",
        "gse", "raw_gse", "gse_post",
        *ACSI_COMPONENT_COLUMNS,
        "generation_capability", "generation_capability_norm", "generation_capability_post",
        "physical_free", "non_personal",
        *[spec["norm"] for spec in ACSI_DIMENSION_SPECS],
        *[spec["post"] for spec in ACSI_DIMENSION_SPECS],
    ]
    missing_columns = [column_name for column_name in required_columns if column_name not in panel.columns]
    if missing_columns:
        validation_errors.append(f"{panel_label} missing required columns: {missing_columns}")
        return

    duplicate_row_count = int(panel.duplicated(["subreddit", "year_month"]).sum())
    if duplicate_row_count:
        validation_errors.append(f"{panel_label} has {duplicate_row_count:,} duplicate subreddit-month rows")

    zero_post_subreddits = []
    if "posts" in panel.columns:
        post_totals = panel.groupby("subreddit")["posts"].sum()
        zero_post_subreddits = sorted(post_totals[post_totals <= 0].index.astype(str).tolist())
        if zero_post_subreddits:
            validation_errors.append(
                f"{panel_label} includes zero-post subreddits that contribute no in-window outcome rows: "
                f"{zero_post_subreddits}"
            )

    if expected_subreddits is None:
        expected_subreddits = expected_panel_subreddits(score_table)
    else:
        expected_subreddits = sorted(str(subreddit) for subreddit in expected_subreddits)
    actual_subreddits = sorted(panel["subreddit"].astype(str).unique())
    if actual_subreddits != expected_subreddits:
        missing_subreddits = sorted(set(expected_subreddits) - set(actual_subreddits))
        extra_subreddits = sorted(set(actual_subreddits) - set(expected_subreddits))
        validation_errors.append(
            f"{panel_label} subreddit coverage mismatch; "
            f"missing={missing_subreddits}, extra={extra_subreddits}"
        )

    expected_row_count = len(expected_subreddits) * len(ALL_MONTHS)
    if len(panel) != expected_row_count:
        validation_errors.append(
            f"{panel_label} row count is {len(panel):,}; expected {expected_row_count:,} "
            f"({len(expected_subreddits)} subreddits x {len(ALL_MONTHS)} months)"
        )

    month_counts_by_subreddit = panel.groupby("subreddit")["year_month"].nunique()
    incomplete_subreddit_month_counts = month_counts_by_subreddit[month_counts_by_subreddit != len(ALL_MONTHS)]
    if not incomplete_subreddit_month_counts.empty:
        validation_errors.append(
            f"{panel_label} has subreddits without all {len(ALL_MONTHS)} months: "
            f"{incomplete_subreddit_month_counts.head(10).to_dict()}"
        )

    parsed_month_dates = pd.to_datetime(panel["year_month_dt"], errors="coerce")
    if parsed_month_dates.isna().any():
        validation_errors.append(f"{panel_label} has invalid year_month_dt values")
    else:
        expected_month_labels = set(pd.Series(ALL_MONTHS).dt.strftime("%Y-%m"))
        actual_month_labels = set(parsed_month_dates.dt.strftime("%Y-%m"))
        if actual_month_labels != expected_month_labels:
            validation_errors.append(
                f"{panel_label} month coverage mismatch; "
                f"missing={sorted(expected_month_labels - actual_month_labels)}, "
                f"extra={sorted(actual_month_labels - expected_month_labels)}"
            )
        month_label_mismatch = panel["year_month"].astype(str) != parsed_month_dates.dt.strftime("%Y-%m")
        if bool(month_label_mismatch.any()):
            validation_errors.append(f"{panel_label} has year_month values that do not match year_month_dt")

    observed_shock_values = set(pd.to_numeric(panel["post_shock"], errors="coerce").dropna().astype(int).unique())
    if not observed_shock_values.issubset({0, 1}):
        validation_errors.append(f"{panel_label} post_shock has values outside {{0, 1}}: {sorted(observed_shock_values)}")

    nonnegative_columns = [
        "posts", "active_creators", "calibrated_output",
        "log_posts", "log_active_creators", "log_calibrated_output", "log_avg_score",
    ]
    for column_name in nonnegative_columns:
        validate_nonnegative_numeric(panel, column_name, validation_errors, panel_label)
    validate_finite_numeric(panel, "avg_score", validation_errors, panel_label)

    component_score_columns = ACSI_COMPONENT_COLUMNS + ["generation_capability", "physical_free", "non_personal"]
    normalized_score_columns = (
        ["gse", "gse_post", "generation_capability_norm", "generation_capability_post"]
        + [spec["norm"] for spec in ACSI_DIMENSION_SPECS]
        + [spec["post"] for spec in ACSI_DIMENSION_SPECS]
    )
    for column_name in component_score_columns:
        validate_numeric_range(panel, column_name, 1, 5, validation_errors, panel_label)
    for column_name in normalized_score_columns:
        validate_numeric_range(panel, column_name, 0, 1, validation_errors, panel_label)
    validate_numeric_range(panel, "raw_gse", 5, 25, validation_errors, panel_label)

    if score_table is not None and not score_table.empty:
        duplicate_score_subreddits = sorted(score_table.loc[score_table["subreddit"].duplicated(), "subreddit"].unique())
        if duplicate_score_subreddits:
            validation_errors.append(f"{INDEX_SHORT} score data has duplicate subreddit rows: {duplicate_score_subreddits}")
        missing_scores = sorted(set(actual_subreddits) - set(score_table["subreddit"].astype(str)))
        if missing_scores:
            validation_errors.append(f"{INDEX_SHORT} score data missing panel subreddits: {missing_scores}")
        if "n_used" in score_table.columns:
            n_used = pd.to_numeric(score_table["n_used"], errors="coerce")
            if n_used.isna().any():
                bad_subreddits = score_table.loc[n_used.isna(), "subreddit"].astype(str).tolist()
                validation_errors.append(f"{INDEX_SHORT} score data has invalid n_used for: {bad_subreddits}")

def validate_required_artifacts(analysis_results, include_content_validation, validation_errors):
    small_multiples_result = (analysis_results or {}).get("subreddit_small_multiples")
    small_multiples_skipped = (
        isinstance(small_multiples_result, dict)
        and small_multiples_result.get("status") == "skipped"
    )
    core_tables = [
        "acsi_three_dimensional_main_model.tex",
        "acsi_three_dimensional_covariate_adjusted.tex",
        "gse_main_dose_response.tex",
        "gse_covariate_adj.tex",
    ]
    if WRITE_OUTPUT_CSVS:
        core_tables.extend(
            [
                "acsi_scores_computed.csv",
                "acsi_three_dimensional_main_model.csv",
                "acsi_three_dimensional_covariate_adjusted.csv",
                "acsi_three_dimensional_leave_one_out.csv",
                "acsi_component_correlations.csv",
                "gse_permutation_coefs.csv",
                "gse_secondary_outcomes.csv",
                "gse_quartile_check.csv",
                "gse_event_study.csv",
            ]
        )
    core_figures = [
        "gse_quartile_check.png",
        "robust_time_varying_personal_context.png",
        "robust_event_study_full.png",
        "robust_matched_strict_event_study.png",
        "robust_placebo_permutation.png",
        "robust_placebo_nov2023.png",
    ]
    if not small_multiples_skipped:
        core_figures.append("subreddit_small_multiples.png")
    for name in core_tables:
        require_file(TABLES_DIR / name, validation_errors)
    for name in core_figures:
        require_file(FIGURES_DIR / name, validation_errors)
    for required_path in [SUBMONTH_PANEL_PATH, SUBMONTH_PANEL_META_PATH]:
        require_file(required_path, validation_errors)

    required_results = [
        "acsi_three_dimensional",
        "acsi_three_dimensional_covariate_adj",
        "acsi_three_dimensional_influence",
        "gse_main",
        "gse_covariate_adj",
        "acsi_component_correlations",
        "gse_permutation",
        "gse_secondary",
        "gse_quartiles",
        "gse_event_study",
        "robust_time_varying_personal_context",
        "robust_event_study_full",
        "robust_matched_strict_event_study",
        "robust_placebo_permutation",
        "robust_placebo_nov2023",
        "subreddit_small_multiples",
    ]
    missing_results = [result_key for result_key in required_results if not analysis_results.get(result_key)]
    if missing_results:
        validation_errors.append(f"Analysis results missing required entries: {missing_results}")

    expected_three_dimensional_results = len(ACSI_MECHANISM_SPECS)
    if (
        analysis_results.get("acsi_three_dimensional")
        and len(analysis_results["acsi_three_dimensional"]) != expected_three_dimensional_results
    ):
        validation_errors.append(
            f"Expected {expected_three_dimensional_results} three-dimensional model results; "
            f"got {len(analysis_results['acsi_three_dimensional'])}"
        )
    if analysis_results.get("gse_secondary") and len(analysis_results["gse_secondary"]) != 3:
        validation_errors.append(f"Expected 3 secondary outcome results; got {len(analysis_results['gse_secondary'])}")
    if analysis_results.get("gse_quartiles") and len(analysis_results["gse_quartiles"]) != 3:
        validation_errors.append(f"Expected 3 quartile results; got {len(analysis_results['gse_quartiles'])}")

    if WRITE_OUTPUT_CSVS and analysis_results.get("post_ai_adoption"):
        require_file(TABLES_DIR / "post_ai_adoption_check.csv", validation_errors)
        require_file(TABLES_DIR / "post_ai_adoption_by_subreddit.csv", validation_errors)

    if WRITE_OUTPUT_CSVS and analysis_results.get("q2_survival"):
        require_file(TABLES_DIR / "q2_survival.csv", validation_errors)
    if WRITE_OUTPUT_CSVS and analysis_results.get("q2_survival_moderation"):
        require_file(TABLES_DIR / "q2_survival_moderation.csv", validation_errors)
    if WRITE_OUTPUT_CSVS and analysis_results.get("q3_engagement"):
        require_file(TABLES_DIR / "q3_per_creator_engagement_did.csv", validation_errors)

    if (
        WRITE_OUTPUT_CSVS
        and include_content_validation
        and analysis_results.get("content_validation", {}).get("n_sampled", 0) > 0
    ):
        require_file(TABLES_DIR / "content_validation_sample.csv", validation_errors)

def validate_cache_metadata(author_cap_enabled, validation_errors):
    try:
        if not submonth_panel_cache_is_current(author_cap_enabled):
            validation_errors.append("Subreddit-month panel metadata is missing or does not match current raw files/settings")
    except Exception as exc:
        validation_errors.append(f"Could not validate subreddit-month panel metadata: {exc}")

    if use_persistent_raw_cache() and POST_MONTHLY_AGG_PATH.exists():
        try:
            if not post_monthly_agg_cache_is_current(author_cap_enabled):
                validation_errors.append("Post monthly aggregate cache metadata does not match current raw files/settings")
        except Exception as exc:
            validation_errors.append(f"Could not validate post monthly aggregate cache metadata: {exc}")

def validate_main_regression_sample(submonth_panel, analysis_results, validation_errors):
    rows = (analysis_results or {}).get("acsi_three_dimensional") or []
    if submonth_panel is None or not rows:
        return
    n_obs_values = {
        safe_int(row.get("n_obs"))
        for row in rows
        if row.get("n_obs") is not None
    }
    n_obs_values.discard(None)
    if len(n_obs_values) != 1:
        return
    n_obs = next(iter(n_obs_values))
    expected_n = len(submonth_panel)
    if n_obs != expected_n:
        excluded_months = rows[0].get("excluded_months", "unknown")
        validation_errors.append(
            f"Main three-dimensional model N is {n_obs:,}, but the panel has "
            f"{expected_n:,} rows; excluded months: {excluded_months}"
        )

def validate_run_outputs(
    submonth_panel,
    score_table,
    analysis_results,
    author_cap_enabled,
    require_cache_metadata=True,
    include_content_validation=False,
    expected_panel_subreddits_override=None,
):
    print("\n=== Final validation gate ===")
    validation_errors = []

    validate_subreddit_month_panel(
        submonth_panel,
        score_table,
        validation_errors,
        expected_subreddits=expected_panel_subreddits_override,
    )
    if SUBMONTH_PANEL_PATH.exists():
        try:
            saved_panel = pd.read_parquet(SUBMONTH_PANEL_PATH)
            validate_subreddit_month_panel(
                saved_panel,
                score_table,
                validation_errors,
                panel_label="saved subreddit-month panel",
                expected_subreddits=expected_panel_subreddits_override,
            )
            if submonth_panel is not None and len(saved_panel) != len(submonth_panel):
                validation_errors.append(
                    f"Saved panel row count {len(saved_panel):,} does not match "
                    f"in-memory panel row count {len(submonth_panel):,}"
                )
        except Exception as exc:
            validation_errors.append(f"Could not read saved subreddit-month panel: {exc}")

    validate_main_regression_sample(submonth_panel, analysis_results, validation_errors)
    validate_required_artifacts(analysis_results, include_content_validation, validation_errors)
    if require_cache_metadata:
        validate_cache_metadata(author_cap_enabled, validation_errors)

    if validation_errors:
        print("  VALIDATION FAILED:")
        for validation_error in validation_errors:
            print(f"  - {validation_error}")
        raise RuntimeError(f"Final validation failed with {len(validation_errors)} issue(s).")

    print(
        f"  validation passed: {len(submonth_panel):,} panel rows, "
        f"{submonth_panel['subreddit'].nunique():,} subreddits, "
        f"{submonth_panel['year_month'].nunique():,} months"
    )

def run_pytest_suite():
    print("\n=== Final test gate: pytest ===")
    result = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT)
    if result.returncode != 0:
        print(f"  pytest failed with exit code {result.returncode}")
        raise RuntimeError("pytest failed")
    print("  pytest passed")


if "pytest" in __import__("sys").modules:
    import argparse
    import contextlib
    import io
    import shutil
    import tempfile
    import unittest
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    import scripts.annotate as annotate
    TEST_SUBREDDITS = ["alpha", "beta"]
    TEST_MONTHS = pd.date_range("2022-11-01", periods=2, freq="MS")
    run = None
    robustness_checks = None
    run_test = None

    def ensure_runtime_test_context():
        global run, robustness_checks, run_test
        if run is None:
            import scripts.robustness_checks as _robustness_checks
            import scripts.run as _run

            run = _run
            robustness_checks = _robustness_checks
            run_test = run.pipeline_tests
        return run, robustness_checks, run_test

    def make_source_row(subreddit, post_id):
        return {
            "subreddit": subreddit,
            "post_id": post_id,
            "period": "pre_gpt",
            "created_date": "2022-01-01",
            "year_month": "2022-01",
            "title_excerpt": f"Title {post_id}",
            "selftext_excerpt": "",
        }

    def make_coded_row(subreddit, post_id, ai_related_flag=0):
        return {
            **make_source_row(subreddit, post_id),
            "direct_gen_score": 1,
            "usefulness_score": 1,
            "quality_comp_score": 1,
            "physical_req_score": 1,
            "personal_req_score": 1,
            "ai_related_flag": ai_related_flag,
            "max_score_disagreement": "",
            "mean_score_disagreement": "",
            "hard_case_flag": "",
            "rationale": "Nonblank rationale.",
            "version": "run1",
        }

    def make_score_table():
        ensure_runtime_test_context()
        rows = []
        for subreddit in TEST_SUBREDDITS:
            row = {
                "subreddit": subreddit,
                "direct_gen": 3,
                "usefulness": 3,
                "quality_comp": 3,
                "physical_req": 2,
                "personal_req": 2,
                "n_used": 50,
                "n_coded": 50,
                "n_ai_related_excluded": 0,
                "n_hard_cases": 0,
                "score_reliability": "medium",
                "low_n_flag": 0,
            }
            row["physical_free"] = 6 - row["physical_req"]
            row["non_personal"] = 6 - row["personal_req"]
            row["raw_gse"] = (
                row["direct_gen"]
                + row["usefulness"]
                + row["quality_comp"]
                + row["physical_free"]
                + row["non_personal"]
            )
            row["gse"] = (row["raw_gse"] - 5) / 20
            for dimension_spec in run.ACSI_DIMENSION_SPECS:
                row[dimension_spec["norm"]] = (row[dimension_spec["source"]] - 1) / 4
            row["generation_capability_norm"] = (
                row["direct_gen_norm"] + row["usefulness_norm"] + row["quality_comp_norm"]
            ) / 3
            row["generation_capability"] = 1 + 4 * row["generation_capability_norm"]
            rows.append(row)
        return pd.DataFrame(rows)

    def make_valid_panel():
        ensure_runtime_test_context()
        score_table = make_score_table()
        score_by_subreddit = score_table.set_index("subreddit").to_dict("index")
        rows = []
        for subreddit in TEST_SUBREDDITS:
            score_row = score_by_subreddit[subreddit]
            for month in TEST_MONTHS:
                post_shock = int(month >= run.SHOCK_MONTH)
                row = {
                    "subreddit": subreddit,
                    "year_month_dt": month,
                    "year_month": month.strftime("%Y-%m"),
                    "post_shock": post_shock,
                    "posts": 1,
                    "active_creators": 1,
                    "calibrated_output": 1.0,
                    "avg_score": 0.0,
                    "log_posts": np.log1p(1),
                    "log_active_creators": np.log1p(1),
                    "log_calibrated_output": np.log1p(1.0),
                    "log_avg_score": 0.0,
                    "gse": score_row["gse"],
                    "raw_gse": score_row["raw_gse"],
                    "direct_gen": score_row["direct_gen"],
                    "usefulness": score_row["usefulness"],
                    "quality_comp": score_row["quality_comp"],
                    "generation_capability": score_row["generation_capability"],
                    "generation_capability_norm": score_row["generation_capability_norm"],
                    "generation_capability_post": score_row["generation_capability_norm"] * post_shock,
                    "physical_req": score_row["physical_req"],
                    "personal_req": score_row["personal_req"],
                    "physical_free": score_row["physical_free"],
                    "non_personal": score_row["non_personal"],
                    "gse_post": score_row["gse"] * post_shock,
                }
                for dimension_spec in run.ACSI_DIMENSION_SPECS:
                    normalized_score = score_row[dimension_spec["norm"]]
                    row[dimension_spec["norm"]] = normalized_score
                    row[dimension_spec["post"]] = normalized_score * post_shock
                rows.append(row)
        return pd.DataFrame(rows)

    def make_required_results(n_obs=None):
        ensure_runtime_test_context()
        acsi_row = {"coef": 0.0}
        if n_obs is not None:
            acsi_row["n_obs"] = n_obs
        return {
            "gse_main": {"coef": 0.0},
            "gse_covariate_adj": {"coef": 0.0},
            "acsi_component_correlations": [{"component": "test"}],
            "acsi_three_dimensional": [dict(acsi_row) for _ in range(len(run.ACSI_MECHANISM_SPECS))],
            "gse_permutation": {"observed_coef": 0.0},
            "gse_secondary": [{"coef": 0.0} for _ in range(3)],
            "gse_quartiles": [{"coef": 0.0} for _ in range(3)],
            "gse_event_study": {"rows": []},
        }

    class AcsiManualBatchTests(unittest.TestCase):
        def test_next_prioritizes_subreddit_below_floor_target(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                input_path = tmpdir / "acsi_data.csv"
                run1_path = tmpdir / "measurement_sample_run1.csv"

                source_rows = [
                    *(make_source_row("oldsub", f"old{i}") for i in range(5)),
                    *(make_source_row("newsub", f"new{i}") for i in range(5)),
                ]
                pd.DataFrame(source_rows).to_csv(input_path, index=False)
                pd.DataFrame(
                    [make_coded_row("oldsub", f"old{i}") for i in range(3)],
                    columns=annotate.OUTPUT_COLUMNS,
                ).to_csv(run1_path, index=False)

                args = argparse.Namespace(
                    input=input_path,
                    run=1,
                    output=run1_path,
                    subreddit=[],
                    n=2,
                    selection_strategy="balanced",
                    floor_target=3,
                    target_count="usable",
                )

                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    annotate.print_next(args)

                text = out.getvalue()
                self.assertIn("target=newsub", text)
                self.assertIn("post_id: new0", text)
                self.assertIn("post_id: new1", text)

        def test_next_uses_lowest_count_after_floor_target_is_met(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                input_path = tmpdir / "acsi_data.csv"
                run1_path = tmpdir / "measurement_sample_run1.csv"

                source_rows = [
                    *(make_source_row("alpha", f"alpha{i}") for i in range(6)),
                    *(make_source_row("beta", f"beta{i}") for i in range(6)),
                ]
                pd.DataFrame(source_rows).to_csv(input_path, index=False)
                coded_rows = [
                    *(make_coded_row("alpha", f"alpha{i}") for i in range(4)),
                    *(make_coded_row("beta", f"beta{i}") for i in range(3)),
                ]
                pd.DataFrame(coded_rows, columns=annotate.OUTPUT_COLUMNS).to_csv(run1_path, index=False)

                args = argparse.Namespace(
                    input=input_path,
                    run=1,
                    output=run1_path,
                    subreddit=[],
                    n=2,
                    selection_strategy="balanced",
                    floor_target=3,
                    target_count="usable",
                )

                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    annotate.print_next(args)

                text = out.getvalue()
                self.assertIn("target=beta", text)
                self.assertIn("post_id: beta3", text)

        def test_aggregate_rejects_half_point_scores_in_single_run_file(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                input_path = tmpdir / "acsi_data.csv"
                coded_path = tmpdir / "measurement_sample_run1.csv"
                acsi_output_path = tmpdir / "acsi_scores.csv"

                source_row = {
                    "subreddit": "example",
                    "post_id": "p1",
                    "period": "pre_gpt",
                    "created_date": "2022-01-01",
                    "year_month": "2022-01",
                    "title_excerpt": "Example",
                    "selftext_excerpt": "",
                }
                pd.DataFrame([source_row]).to_csv(input_path, index=False)

                coded_row = {
                    **source_row,
                    "direct_gen_score": 1.5,
                    "usefulness_score": 1,
                    "quality_comp_score": 1,
                    "physical_req_score": 1,
                    "personal_req_score": 1,
                    "ai_related_flag": 0,
                    "max_score_disagreement": "",
                    "mean_score_disagreement": "",
                    "hard_case_flag": "",
                    "rationale": "Nonblank rationale.",
                    "version": "run1",
                }
                pd.DataFrame([coded_row], columns=annotate.OUTPUT_COLUMNS).to_csv(coded_path, index=False)

                args = argparse.Namespace(
                    input=input_path,
                    final_output=coded_path,
                    allow_partial=True,
                    include_ai_related=False,
                    hard_case_policy="downweight",
                    hard_case_weight=0.5,
                    acsi_output=acsi_output_path,
                )

                with self.assertRaisesRegex(ValueError, "invalid score values"):
                    annotate.aggregate_scores(args)

        def test_summary_all_runs_does_not_create_missing_measurement_files(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                input_path = tmpdir / "acsi_data.csv"
                run1_path = tmpdir / "measurement_sample_run1.csv"
                run2_path = tmpdir / "measurement_sample_run2.csv"
                final_path = tmpdir / "measurement_sample.csv"

                pd.DataFrame(
                    [
                        {
                            "subreddit": "example",
                            "post_id": "p1",
                            "period": "pre_gpt",
                            "created_date": "2022-01-01",
                            "year_month": "2022-01",
                            "title_excerpt": "Example",
                            "selftext_excerpt": "",
                        }
                    ]
                ).to_csv(input_path, index=False)

                args = argparse.Namespace(
                    input=input_path,
                    run=1,
                    output=None,
                    all_runs=True,
                )

                with patch.object(annotate, "DEFAULT_RUN1_OUTPUT", run1_path), patch.object(
                    annotate, "DEFAULT_RUN2_OUTPUT", run2_path
                ), patch.object(annotate, "DEFAULT_FINAL_OUTPUT", final_path), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    annotate.print_summary(args)

                self.assertFalse(run1_path.exists())
                self.assertFalse(run2_path.exists())
                self.assertFalse(final_path.exists())

    class RunValidationTests(unittest.TestCase):
        def setUp(self):
            ensure_runtime_test_context()
            self.previous_output_dir = run.OUTPUT_DIR
            self.test_output_dir = run.OUTPUT_ROOT / "test_validation"
            shutil.rmtree(self.test_output_dir, ignore_errors=True)
            run.configure_output_dir(self.test_output_dir)
            run.bind_check_modules()

        def tearDown(self):
            run.configure_output_dir(self.previous_output_dir)
            run.bind_check_modules()
            shutil.rmtree(self.test_output_dir, ignore_errors=True)

        def panel_context(self):
            return patch.multiple(
                run_test,
                ALL_MONTHS=TEST_MONTHS,
                available_post_subreddits=lambda: list(TEST_SUBREDDITS),
            )

        def test_output_directory_guard_rejects_paths_outside_project_output(self):
            with self.assertRaises(ValueError):
                run.validate_generated_output_dir(Path("/private/tmp/acsi-outside-output"))

            accepted_path = run.validate_generated_output_dir(run.OUTPUT_ROOT / "unit-test-output")
            self.assertEqual(accepted_path, run.OUTPUT_ROOT / "unit-test-output")

        def test_subreddit_month_panel_validation_accepts_clean_panel(self):
            with self.panel_context():
                validation_errors = []
                run_test.validate_subreddit_month_panel(make_valid_panel(), make_score_table(), validation_errors)

            self.assertEqual(validation_errors, [])

        def test_subreddit_month_panel_validation_detects_duplicate_rows(self):
            panel = make_valid_panel()
            panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

            with self.panel_context():
                validation_errors = []
                run_test.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

            self.assertTrue(any("duplicate subreddit-month rows" in error for error in validation_errors))

        def test_subreddit_month_panel_validation_detects_out_of_range_scores(self):
            panel = make_valid_panel()
            panel.loc[0, "gse"] = 1.5

            with self.panel_context():
                validation_errors = []
                run_test.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

            self.assertTrue(any("column gse has values outside" in error for error in validation_errors))

        def test_subreddit_month_panel_validation_rejects_zero_post_subreddits(self):
            panel = make_valid_panel()
            beta_mask = panel["subreddit"] == "beta"
            panel.loc[beta_mask, ["posts", "active_creators", "calibrated_output"]] = 0
            panel.loc[beta_mask, ["avg_score", "log_posts", "log_active_creators"]] = 0.0
            panel.loc[beta_mask, ["log_calibrated_output", "log_avg_score"]] = 0.0

            with self.panel_context():
                validation_errors = []
                run_test.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

            self.assertTrue(any("zero-post subreddits" in error for error in validation_errors))

        def test_subreddit_month_panel_validation_accepts_negative_raw_average_score(self):
            panel = make_valid_panel()
            panel.loc[0, "avg_score"] = -0.25
            panel.loc[0, "log_avg_score"] = 0.0

            with self.panel_context():
                validation_errors = []
                run_test.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

            self.assertEqual(validation_errors, [])

        def test_load_acsi_scores_rejects_duplicate_subreddit_rows(self):
            duplicate_scores_path = self.test_output_dir / "duplicate_scores.csv"
            pd.DataFrame(
                [
                    {
                        "subreddit": "alpha",
                        "direct_gen": 3,
                        "usefulness": 3,
                        "quality_comp": 3,
                        "physical_req": 2,
                        "personal_req": 2,
                        "n_used": 50,
                    },
                    {
                        "subreddit": "alpha",
                        "direct_gen": 4,
                        "usefulness": 3,
                        "quality_comp": 3,
                        "physical_req": 2,
                        "personal_req": 2,
                        "n_used": 50,
                    },
                    {
                        "subreddit": "beta",
                        "direct_gen": 3,
                        "usefulness": 3,
                        "quality_comp": 3,
                        "physical_req": 2,
                        "personal_req": 2,
                        "n_used": 50,
                    },
                ]
            ).to_csv(duplicate_scores_path, index=False)

            with patch.object(run, "ACSI_SCORE_PATH", duplicate_scores_path), patch.object(
                run, "SUBREDDITS", {"alpha": "treatment", "beta": "control"}
            ):
                with self.assertRaisesRegex(ValueError, "duplicate subreddit rows"):
                    run.load_acsi_scores()

        def test_regression_sample_summary_reports_excluded_months(self):
            panel = make_valid_panel()
            model_data = panel[panel["year_month"] != "2022-11"].copy()

            summary = run.regression_sample_summary(panel, model_data)

            self.assertEqual(summary["n_panel_rows"], 4)
            self.assertEqual(summary["n_model_rows"], 2)
            self.assertEqual(summary["excluded_months"], "2022-11")
            self.assertFalse(summary["transition_month_included"])

        def test_run_analysis_uses_provided_acsi_scores(self):
            with patch.object(
                run,
                "recompute_acsi_scores_from_run1",
                side_effect=AssertionError("run_analysis should use provided scores"),
            ):
                results = run.run_analysis(
                    df_all=pd.DataFrame(),
                    creators=pd.DataFrame(),
                    acsi_scores=make_score_table(),
                    submonth_panel=pd.DataFrame(),
                )

            self.assertIn("acsi_component_correlations", results)

        def test_final_validation_rejects_main_model_n_mismatch(self):
            panel = make_valid_panel()
            score_table = make_score_table()
            panel.to_parquet(run.SUBMONTH_PANEL_PATH, index=False)
            analysis_results = make_required_results(n_obs=len(panel) - len(TEST_SUBREDDITS))
            for row in analysis_results["acsi_three_dimensional"]:
                row["excluded_months"] = "2022-11"

            with self.panel_context(), patch.object(run_test, "validate_required_artifacts"), patch.object(
                run_test, "validate_cache_metadata"
            ):
                with self.assertRaisesRegex(RuntimeError, "Final validation failed"):
                    run_test.validate_run_outputs(
                        submonth_panel=panel,
                        score_table=score_table,
                        analysis_results=analysis_results,
                        author_cap_enabled=True,
                    )

        def test_creator_exit_moderation_uses_primary_subreddit_persfree(self):
            class FakeLogit:
                def __init__(self, formula, data):
                    self.formula = formula
                    self.data = data

                def fit(self, disp=0):
                    class Fit:
                        params = {
                            "log_pre_shock_posting_rate": -0.1,
                            "non_personal_norm": 0.2,
                            "log_pre_shock_posting_rate:non_personal_norm": 0.3,
                        }
                        bse = {
                            "log_pre_shock_posting_rate": 0.01,
                            "non_personal_norm": 0.02,
                            "log_pre_shock_posting_rate:non_personal_norm": 0.03,
                        }
                        pvalues = {
                            "log_pre_shock_posting_rate": 0.10,
                            "non_personal_norm": 0.20,
                            "log_pre_shock_posting_rate:non_personal_norm": 0.04,
                        }
                        nobs = 3

                    return Fit()

            posts = pd.DataFrame(
                [
                    {"author": "a1", "subreddit": "writing", "post_id": "p1", "post_shock": 0},
                    {"author": "a1", "subreddit": "writing", "post_id": "p2", "post_shock": 1},
                    {"author": "a2", "subreddit": "art", "post_id": "p3", "post_shock": 0},
                    {"author": "a3", "subreddit": "art", "post_id": "p4", "post_shock": 0},
                    {"author": "a3", "subreddit": "art", "post_id": "p5", "post_shock": 1},
                    {"author": "a4", "subreddit": "woodworking", "post_id": "p6", "post_shock": 0},
                ]
            )
            creators = pd.DataFrame([{"author": "a1", "is_stable": 0}])
            scores = pd.DataFrame(
                [
                    {"subreddit": "writing", "non_personal_norm": 0.25, "generation_capability_norm": 0.90},
                    {"subreddit": "art", "non_personal_norm": 0.75, "generation_capability_norm": 0.80},
                    {"subreddit": "woodworking", "non_personal_norm": 0.10, "generation_capability_norm": 0.10},
                ]
            )

            with patch.object(run.smf, "logit", side_effect=lambda formula, data: FakeLogit(formula, data)), patch.object(
                robustness_checks, "emit_output_table"
            ):
                results = robustness_checks.compute_creator_level_checks(posts, creators, acsi_scores=scores)

            moderation = results["q2_survival_moderation"]
            self.assertIsNotNone(moderation)
            self.assertEqual(moderation["term"], "log_pre_shock_posting_rate:non_personal_norm")
            self.assertEqual(moderation["n_authors"], 3)
            self.assertEqual(moderation["n_primary_subreddits"], 2)
            self.assertEqual(moderation["coef"], 0.3)

        def test_subreddit_role_map_rejects_duplicate_assignments(self):
            role_groups = {
                "treatment": ("alpha", "shared"),
                "control": ("beta", "shared"),
            }

            with self.assertRaisesRegex(ValueError, "assigned to multiple role groups: shared"):
                run.build_subreddit_role_map(role_groups)

        def test_final_validation_gate_reads_saved_panel_and_passes_clean_outputs(self):
            panel = make_valid_panel()
            score_table = make_score_table()
            panel.to_parquet(run.SUBMONTH_PANEL_PATH, index=False)

            with self.panel_context(), patch.object(run_test, "validate_required_artifacts"), patch.object(
                run_test, "validate_cache_metadata"
            ):
                run_test.validate_run_outputs(
                    submonth_panel=panel,
                    score_table=score_table,
                    analysis_results=make_required_results(),
                    author_cap_enabled=True,
                )

        def test_final_validation_accepts_explicit_active_subreddit_target(self):
            panel = make_valid_panel()
            panel = panel[panel["subreddit"] == "alpha"].copy()
            score_table = make_score_table()
            panel.to_parquet(run.SUBMONTH_PANEL_PATH, index=False)

            with self.panel_context(), patch.object(run_test, "validate_required_artifacts"), patch.object(
                run_test, "validate_cache_metadata"
            ):
                run_test.validate_run_outputs(
                    submonth_panel=panel,
                    score_table=score_table,
                    analysis_results=make_required_results(),
                    author_cap_enabled=True,
                    expected_panel_subreddits_override=["alpha"],
                )

        def test_gen_cap_simex_helper_returns_corrected_estimate_and_plot(self):
            subreddits = ["alpha", "beta", "gamma", "delta", "epsilon"]
            months = pd.to_datetime(["2022-09-01", "2022-10-01", "2022-12-01", "2023-01-01"])
            score_rows = []
            score_lookup = {}
            for i, subreddit in enumerate(subreddits):
                run1_gen = i % 4
                run2_gen = min(3, run1_gen + (i % 2))
                run1_use = (i + 1) % 4
                run2_use = run1_use
                run1_qual = (i + 2) % 4
                run2_qual = max(0, run1_qual - (i % 2))
                run1_phys = i % 3
                run2_phys = min(3, run1_phys + 1)
                run1_pers = (i + 1) % 3
                run2_pers = run1_pers
                for post_index in range(2):
                    score_rows.append({
                        "post_id": f"{subreddit}_{post_index}",
                        "subreddit": subreddit,
                        "run1_gen": run1_gen,
                        "run1_use": run1_use,
                        "run1_qual": run1_qual,
                        "run2_gen": run2_gen,
                        "run2_use": run2_use,
                        "run2_qual": run2_qual,
                        "run1_pers": run1_pers,
                        "run2_pers": run2_pers,
                        "run1_phys": run1_phys,
                        "run2_phys": run2_phys,
                    })
                gen_cap = np.mean([
                    np.mean([run1_gen, run1_use, run1_qual]) / 3,
                    np.mean([run2_gen, run2_use, run2_qual]) / 3,
                ])
                phys_free = 1 - np.mean([run1_phys, run2_phys]) / 3
                pers_free = 1 - np.mean([run1_pers, run2_pers]) / 3
                score_lookup[subreddit] = (gen_cap, phys_free, pers_free)

            panel_rows = []
            for i, subreddit in enumerate(subreddits):
                gen_cap, phys_free, pers_free = score_lookup[subreddit]
                for j, month in enumerate(months):
                    post_shock = int(month >= run.SHOCK_MONTH)
                    panel_rows.append({
                        "subreddit": subreddit,
                        "year_month": month.strftime("%Y-%m"),
                        "post_shock": post_shock,
                        "log_posts": (
                            0.4 * i
                            + 0.1 * j
                            + post_shock * (0.8 * gen_cap - 0.2 * phys_free + 0.3 * pers_free)
                        ),
                    })

            output_path = run.FIGURES_DIR / "unit_gen_cap_simex.png"
            result = robustness_checks.compute_gen_cap_simex_correction(
                pd.DataFrame(score_rows),
                pd.DataFrame(panel_rows),
                lambdas=(0.5, 1.0),
                n_simulations=3,
                n_bootstrap=5,
                random_seed=123,
                output_path=output_path,
            )

            self.assertIn("corrected_b1", result)
            self.assertTrue(np.isfinite(result["corrected_b1"]))
            self.assertGreater(result["sigma2_u"], 0)
            self.assertEqual(result["bootstrap_n"], 5)
            self.assertTrue(output_path.exists())
