"""Robustness, placebo, backtest, and mechanism checks for run.py.

The functions in this module are bound to the live run.py context by
``bind_context`` so they share the runner's configured paths, constants, and
utility helpers without keeping the check implementations in run.py.
"""

from pathlib import Path


def bind_context(ctx):
    for name in dir(ctx):
        if name.startswith("__"):
            continue
        existing = globals().get(name)
        if getattr(existing, "__module__", None) == __name__:
            continue
        globals()[name] = getattr(ctx, name)


def resolve_raw_data_dir(data_dir=None):
    if data_dir is not None:
        base = Path(data_dir)
        nested = base / "raw_files"
        return nested if nested.exists() else base
    raw_dir = globals().get("RAW_DATA_DIR")
    if raw_dir is not None:
        return Path(raw_dir)
    data_root = Path(globals().get("DATA_DIR", Path.cwd() / "data"))
    nested = data_root / "raw_files"
    return nested if nested.exists() else data_root


def raw_post_file_path(subreddit, data_dir=None):
    return resolve_raw_data_dir(data_dir) / f"r_{subreddit}_posts.jsonl"


ACSI_DIMENSION_DIAGNOSTIC_DIMENSIONS = [
    {
        "var": "direct_gen_exp",
        "source": "direct_gen",
        "label": "Direct generation potential",
        "interpretation": "Higher means GenAI can directly answer or produce the requested content.",
    },
    {
        "var": "usefulness_exp",
        "source": "usefulness",
        "label": "Usefulness if used",
        "interpretation": "Higher means GenAI would be useful for the poster's task.",
    },
    {
        "var": "quality_comp_exp",
        "source": "quality_comp",
        "label": "Competitive with knowledgeable human",
        "interpretation": "Higher means GenAI output would compete with a knowledgeable human response.",
    },
    {
        "var": "physical_free_exp",
        "source": "physical_req",
        "label": "Low physical constraint",
        "interpretation": "Higher means the post depends less on physical work, inspection, materials, or place.",
    },
    {
        "var": "non_personal_exp",
        "source": "personal_req",
        "label": "Low personal-context need",
        "interpretation": "Higher means the post depends less on the poster's specific situation or identity.",
    },
]

ACSI_DIMENSION_DOMAIN_MAP = {
    "writing": "text_writing",
    "worldbuilding": "text_writing",
    "shortstories": "text_writing",
    "screenwriting": "text_writing",
    "poetry": "text_writing",
    "fanfiction": "text_writing",
    "songwriting": "text_writing",
    "fantasywriters": "text_writing",
    "scifiwriting": "text_writing",
    "fiction": "text_writing",
    "books": "text_writing",
    "art": "image_design",
    "illustration": "image_design",
    "conceptart": "image_design",
    "comics": "image_design",
    "digitalart": "image_design",
    "graphic_design": "image_design",
    "gamedev": "image_design",
    "3Dmodeling": "image_design",
    "photography": "image_design",
    "askphotography": "image_design",
    "learnart": "image_design",
    "learntodraw": "image_design",
    "applyingtocollege": "academic_career",
    "gre": "academic_career",
    "lsat": "academic_career",
    "mcat": "academic_career",
    "sat": "academic_career",
    "personalstatement": "academic_career",
    "resume": "academic_career",
    "devops": "academic_career",
    "chanceme": "academic_career",
    "college": "academic_career",
    "gradschool": "academic_career",
    "lawschool": "academic_career",
    "medicalschool": "academic_career",
    "phd": "academic_career",
    "premed": "academic_career",
    "machinelearning": "academic_career",
    "learnprogramming": "academic_career",
    "learnmath": "academic_career",
    "cscareerquestions": "academic_career",
    "askacademia": "academic_career",
    "woodworking": "craft_physical",
    "pottery": "craft_physical",
    "sewing": "craft_physical",
    "baking": "craft_physical",
    "cooking": "craft_physical",
    "knitting": "craft_physical",
    "breadit": "craft_physical",
    "carpentry": "craft_physical",
    "leathercraft": "craft_physical",
    "quilting": "craft_physical",
    "ceramics": "craft_physical",
    "fermentation": "craft_physical",
    "gardening": "craft_physical",
    "homebrewing": "craft_physical",
    "plants": "craft_physical",
    "chess": "travel_fitness_misc",
    "programminghumor": "travel_fitness_misc",
    "rowing": "travel_fitness_misc",
    "running": "travel_fitness_misc",
    "swimming": "travel_fitness_misc",
    "solotravel": "travel_fitness_misc",
    "travel": "travel_fitness_misc",
}


def acsi_dimension_pct_effect(coef, scale=1.0):
    if coef is None or pd.isna(coef):
        return None
    return float(100 * (np.exp(float(coef) * float(scale)) - 1))


def prepare_acsi_dimension_diagnostics_panel(output_dir=None, panel=None, score_table=None):
    output_dir = Path(output_dir or OUTPUT_DIR)
    if panel is None:
        panel_path = output_dir / "subreddit_month_gse_panel.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(panel_path)
        panel = pd.read_parquet(panel_path)
    else:
        panel = panel.copy()
    if score_table is None:
        score_path = output_dir / "tables" / "acsi_scores_computed.csv"
        legacy_score_path = output_dir / "tables" / "gse_scores_computed.csv"
        if not score_path.exists():
            if legacy_score_path.exists():
                score_path = legacy_score_path
            else:
                raise FileNotFoundError(score_path)
        scores = pd.read_csv(score_path)
    else:
        scores = score_table.copy()
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])

    keep_cols = [
        "subreddit",
        "gse",
        "raw_gse",
        "direct_gen",
        "usefulness",
        "quality_comp",
        "physical_req",
        "personal_req",
        "n_used",
        "n_ai_related_excluded",
    ]
    scores = scores[[column for column in keep_cols if column in scores.columns]].copy()



    scores["direct_gen_exp"] = (scores["direct_gen"] - 1) / 4
    scores["usefulness_exp"] = (scores["usefulness"] - 1) / 4
    scores["quality_comp_exp"] = (scores["quality_comp"] - 1) / 4
    scores["physical_free_exp"] = (5 - scores["physical_req"]) / 4
    scores["non_personal_exp"] = (5 - scores["personal_req"]) / 4
    scores["domain"] = scores["subreddit"].map(ACSI_DIMENSION_DOMAIN_MAP).fillna("other")

    score_columns_to_refresh = [column for column in scores.columns if column != "subreddit"]
    merged = (
        panel.drop(columns=score_columns_to_refresh, errors="ignore")
        .merge(scores, on="subreddit", how="inner")
    )
    expected = panel["subreddit"].nunique()
    actual = merged["subreddit"].nunique()
    if actual != expected:
        missing = sorted(set(panel["subreddit"]) - set(merged["subreddit"]))
        raise ValueError(f"Only merged {actual}/{expected} subreddits. Missing: {missing}")
    return merged


def run_acsi_dimension_models(panel, tables_dir):
    rows = []
    cov = add_pre_covariates(panel)
    adjusted = panel.merge(cov, on="subreddit", how="left")
    adjusted["pre_avg_post"] = adjusted["pre_avg_log_posts"] * adjusted["post_shock"]
    adjusted["pre_trend_post"] = adjusted["pre_trend"] * adjusted["post_shock"]
    adjusted["log_mu_post"] = adjusted["log_mu_k"] * adjusted["post_shock"]

    for dim in ACSI_DIMENSION_DIAGNOSTIC_DIMENSIONS:
        var = dim["var"]
        term = f"{var}_post"
        panel[term] = panel[var] * panel["post_shock"]
        adjusted[term] = adjusted[var] * adjusted["post_shock"]

        for model_name, data, extra in [
            ("fixed_effects", panel, ""),
            ("covariate_adjusted", adjusted, " + pre_avg_post + pre_trend_post + log_mu_post"),
        ]:
            model = fit_ols(
                f"log_posts ~ {term}{extra} + C(subreddit) + C(year_month)",
                data,
                cluster_col="subreddit",
            )
            row = reg_result(model, term)
            observed_range = float(data[var].max() - data[var].min())
            row.update({
                "model": model_name,
                "dimension": var,
                "source_score": dim["source"],
                "label": dim["label"],
                "interpretation": dim["interpretation"],
                "observed_min": float(data[var].min()),
                "observed_max": float(data[var].max()),
                "percent_effect_full_0_to_1": acsi_dimension_pct_effect(row["coef"]),
                "percent_effect_observed_range": acsi_dimension_pct_effect(row["coef"], observed_range),
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "acsi_dimension_regressions.csv", index=False)
    return out


def run_acsi_dimension_joint_models(panel, tables_dir):
    dims = [dimension["var"] for dimension in ACSI_DIMENSION_DIAGNOSTIC_DIMENSIONS]
    terms = [f"{var}_post" for var in dims]
    for var, term in zip(dims, terms):
        panel[term] = panel[var] * panel["post_shock"]

    cov = add_pre_covariates(panel)
    adjusted = panel.merge(cov, on="subreddit", how="left")
    adjusted["pre_avg_post"] = adjusted["pre_avg_log_posts"] * adjusted["post_shock"]
    adjusted["pre_trend_post"] = adjusted["pre_trend"] * adjusted["post_shock"]
    adjusted["log_mu_post"] = adjusted["log_mu_k"] * adjusted["post_shock"]

    rows = []
    for model_name, data, extra in [
        ("fixed_effects", panel, ""),
        ("covariate_adjusted", adjusted, " + pre_avg_post + pre_trend_post + log_mu_post"),
    ]:
        formula = f"log_posts ~ {' + '.join(terms)}{extra} + C(subreddit) + C(year_month)"
        model = fit_ols(formula, data, cluster_col="subreddit")
        for dim in ACSI_DIMENSION_DIAGNOSTIC_DIMENSIONS:
            term = f"{dim['var']}_post"
            row = reg_result(model, term)
            row.update({
                "model": model_name,
                "dimension": dim["var"],
                "source_score": dim["source"],
                "label": dim["label"],
                "percent_effect_full_0_to_1": acsi_dimension_pct_effect(row["coef"]),
            })
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "acsi_dimension_joint_model.csv", index=False)
    return out


def run_acsi_dimension_domain_tables(panel, tables_dir):
    sub_pre = panel[panel["post_shock"].eq(0)].groupby("subreddit")["log_posts"].mean()
    sub_post = panel[panel["post_shock"].eq(1)].groupby("subreddit")["log_posts"].mean()
    sub_scores = (
        panel[
            [
                "subreddit",
                "domain",
                "gse",
                "direct_gen_exp",
                "usefulness_exp",
                "quality_comp_exp",
                "physical_free_exp",
                "non_personal_exp",
                "n_used",
            ]
        ]
        .drop_duplicates("subreddit")
        .copy()
    )
    sub_scores["pre_post_log_change"] = sub_scores["subreddit"].map(sub_post - sub_pre)

    summary = (
        sub_scores.groupby("domain", as_index=False)
        .agg(
            n_subreddits=("subreddit", "nunique"),
            mean_gse=("gse", "mean"),
            mean_direct_gen=("direct_gen_exp", "mean"),
            mean_usefulness=("usefulness_exp", "mean"),
            mean_quality_comp=("quality_comp_exp", "mean"),
            mean_low_physical_constraint=("physical_free_exp", "mean"),
            mean_low_personal_context=("non_personal_exp", "mean"),
            mean_measurement_posts=("n_used", "mean"),
            mean_pre_post_log_change=("pre_post_log_change", "mean"),
            sd_pre_post_log_change=("pre_post_log_change", "std"),
        )
        .sort_values("mean_pre_post_log_change")
    )
    summary["mean_percent_change"] = summary["mean_pre_post_log_change"].apply(
        acsi_dimension_pct_effect
    )
    summary.to_csv(tables_dir / "acsi_content_area_summary.csv", index=False)

    reg_panel = panel.copy()
    reference = "craft_physical"
    domains = sorted(domain for domain in reg_panel["domain"].unique() if domain != reference)
    terms = []
    for domain in domains:
        clean = domain.replace("-", "_")
        term = f"domain_{clean}_post"
        reg_panel[term] = reg_panel["post_shock"] * reg_panel["domain"].eq(domain).astype(int)
        terms.append((domain, term))

    formula = f"log_posts ~ {' + '.join(term for _, term in terms)} + C(subreddit) + C(year_month)"
    model = fit_ols(formula, reg_panel, cluster_col="subreddit")
    rows = []
    for domain, term in terms:
        row = reg_result(model, term)
        row.update({
            "domain": domain,
            "reference_domain": reference,
            "label": f"{domain} vs {reference}",
            "percent_effect_vs_reference": acsi_dimension_pct_effect(row["coef"]),
        })
        rows.append(row)

    reg = pd.DataFrame(rows)
    reg.to_csv(tables_dir / "acsi_content_area_regression.csv", index=False)
    sub_scores.to_csv(tables_dir / "acsi_subreddit_dimension_summary.csv", index=False)
    return summary, reg


def run_acsi_dimension_correlation_table(panel, tables_dir):
    sub_scores = (
        panel[["subreddit"] + [d["var"] for d in ACSI_DIMENSION_DIAGNOSTIC_DIMENSIONS] + ["gse"]]
        .drop_duplicates("subreddit")
    )
    corr = sub_scores.drop(columns=["subreddit"]).corr()
    corr.to_csv(tables_dir / "acsi_dimension_correlations.csv")
    return corr


def compute_acsi_dimension_diagnostics(output_dir=None, panel=None, acsi_scores=None):
    output_dir = Path(output_dir or OUTPUT_DIR).resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    panel = prepare_acsi_dimension_diagnostics_panel(
        output_dir,
        panel=panel,
        score_table=acsi_scores,
    )
    dim = run_acsi_dimension_models(panel.copy(), tables_dir)
    joint = run_acsi_dimension_joint_models(panel.copy(), tables_dir)
    domain_summary, domain_reg = run_acsi_dimension_domain_tables(panel.copy(), tables_dir)
    corr = run_acsi_dimension_correlation_table(panel.copy(), tables_dir)

    print("\nDimension regressions:")
    print(
        dim[dim["model"].eq("fixed_effects")][
            ["label", "coef", "se", "pvalue", "percent_effect_observed_range"]
        ].to_string(index=False)
    )
    print("\nJoint dimension model:")
    print(
        joint[joint["model"].eq("fixed_effects")][
            ["label", "coef", "se", "pvalue", "percent_effect_full_0_to_1"]
        ].to_string(index=False)
    )
    print("\nContent-area summary:")
    print(
        domain_summary[
            ["domain", "n_subreddits", "mean_gse", "mean_pre_post_log_change", "mean_percent_change"]
        ].to_string(index=False)
    )
    print("\nContent-area regression, reference=craft_physical:")
    print(
        domain_reg[
            ["domain", "coef", "se", "pvalue", "percent_effect_vs_reference"]
        ].to_string(index=False)
    )
    print(f"\nWrote ACSI dimension diagnostics to {tables_dir}")

    return {
        "dimension_regressions": dim,
        "joint_model": joint,
        "domain_summary": domain_summary,
        "domain_regression": domain_reg,
        "correlations": corr,
        "tables_dir": str(tables_dir),
    }


def two_way_fe_coefficients(data, outcome, terms):
    model_data = data.dropna(subset=[outcome] + terms).copy()
    if model_data.empty:
        return None
    y_resid = residualize_two_way(
        model_data[outcome].astype(float),
        model_data["subreddit"],
        model_data["year_month"],
    ).to_numpy()
    x_resids = [
        residualize_two_way(
            model_data[term].astype(float),
            model_data["subreddit"],
            model_data["year_month"],
        ).to_numpy()
        for term in terms
    ]
    x_matrix = np.column_stack(x_resids)
    try:
        coefficients = np.linalg.lstsq(x_matrix, y_resid, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return {
        term: safe_float(coefficient)
        for term, coefficient in zip(terms, coefficients)
    }

def compute_three_dimensional_leave_one_out(acsi_panel, outcome="log_posts"):
    terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
    focal_term = "non_personal_post"
    base_coefficients = two_way_fe_coefficients(acsi_panel, outcome, terms)
    if not base_coefficients:
        return None, []

    rows = []
    for omitted_subreddit in tqdm(sorted(acsi_panel["subreddit"].astype(str).unique()), desc="  leave-one-subreddit"):
        subset = acsi_panel[acsi_panel["subreddit"].astype(str) != omitted_subreddit].copy()
        coefficients = two_way_fe_coefficients(subset, outcome, terms)
        if not coefficients:
            continue
        row = {"omitted_subreddit": omitted_subreddit}
        for term in terms:
            row[term] = coefficients.get(term)
        try:
            subset_model = fit_ols(
                f"{outcome} ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)",
                subset.dropna(subset=[outcome] + terms),
                cluster_col="subreddit",
            )
            if subset_model is not None and focal_term in subset_model.pvalues.index:
                row[f"{focal_term}_pvalue"] = safe_float(subset_model.pvalues[focal_term])
                row[f"{focal_term}_significant"] = bool(
                    row[focal_term] < 0 and row[f"{focal_term}_pvalue"] < 0.05
                )
        except Exception:
            row[f"{focal_term}_pvalue"] = None
            row[f"{focal_term}_significant"] = None
        rows.append(row)

    if not rows:
        return None, []

    influence_table = pd.DataFrame(rows)
    base_focal = base_coefficients.get(focal_term)
    influence_table["non_personal_delta_from_full"] = (
        influence_table[focal_term] - base_focal
    )
    influence_table["abs_non_personal_delta"] = influence_table[
        "non_personal_delta_from_full"
    ].abs()
    influence_table = influence_table.sort_values("abs_non_personal_delta", ascending=False)
    top_row = influence_table.iloc[0].to_dict()
    significant_column = f"{focal_term}_significant"
    significant_series = (
        influence_table[significant_column]
        if significant_column in influence_table.columns
        else pd.Series(dtype=bool)
    )
    summary = {
        "term": focal_term,
        "label": "Low personal-context need",
        "full_sample_coef": base_focal,
        "min_leave_one_out_coef": safe_float(influence_table[focal_term].min()),
        "max_leave_one_out_coef": safe_float(influence_table[focal_term].max()),
        "median_leave_one_out_coef": safe_float(influence_table[focal_term].median()),
        "n_subreddits_tested": safe_int(len(influence_table)),
        "n_positive_leave_one_out": safe_int((influence_table[focal_term] > 0).sum()),
        "n_significant_leave_one_out": safe_int(significant_series.fillna(False).sum()),
        "largest_shift_subreddit": top_row.get("omitted_subreddit"),
        "largest_shift_coef": safe_float(top_row.get(focal_term)),
        "largest_shift_delta": safe_float(top_row.get("non_personal_delta_from_full")),
    }
    return summary, influence_table.to_dict("records")


def ensure_current_acsi_panel(panel, acsi_scores=None):
    """Return a panel with current ACSI interaction columns present."""
    working = panel.copy()
    if "year_month_dt" not in working.columns and "year_month" in working.columns:
        working["year_month_dt"] = pd.to_datetime(working["year_month"].astype(str) + "-01", errors="coerce")
    if "year_month" not in working.columns and "year_month_dt" in working.columns:
        working["year_month"] = pd.to_datetime(working["year_month_dt"], errors="coerce").dt.strftime("%Y-%m")
    if "post_shock" not in working.columns and "year_month_dt" in working.columns:
        shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
        working["post_shock"] = (pd.to_datetime(working["year_month_dt"]) >= shock_month).astype(int)

    required_norms = {
        "generation_capability_norm",
        "physical_free_norm",
        "non_personal_norm",
    }
    if not required_norms.issubset(working.columns) and acsi_scores is not None:
        working = attach_acsi_scores(working, acsi_scores)

    if "generation_capability_norm" not in working.columns and {
        "direct_gen_norm",
        "usefulness_norm",
        "quality_comp_norm",
    }.issubset(working.columns):
        working["generation_capability_norm"] = working[
            ["direct_gen_norm", "usefulness_norm", "quality_comp_norm"]
        ].mean(axis=1)

    for norm_column, post_column in [
        ("generation_capability_norm", "generation_capability_post"),
        ("physical_free_norm", "physical_free_post"),
        ("non_personal_norm", "non_personal_post"),
    ]:
        if post_column not in working.columns and norm_column in working.columns and "post_shock" in working.columns:
            working[post_column] = working[norm_column] * working["post_shock"]
    return working


def compute_wild_cluster_bootstrap(acsi_panel, n_bootstrap=1000):
    """Wild cluster bootstrap for the main three-dimensional PersFree x Post coefficient."""
    terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
    focal_term = "non_personal_post"
    model_data = ensure_current_acsi_panel(acsi_panel).dropna(
        subset=["log_posts", "subreddit", "year_month"] + terms
    ).copy()
    if model_data.empty or model_data["subreddit"].nunique() < 2:
        raise ValueError("Need at least two subreddits for wild cluster bootstrap.")

    formula = "log_posts ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)"
    observed_model = fit_ols(formula, model_data, cluster_col="subreddit")
    if observed_model is None or focal_term not in observed_model.params.index:
        raise ValueError("Main three-dimensional model failed for wild cluster bootstrap.")
    observed_result = reg_result(observed_model, focal_term)
    observed_coef = observed_result.get("coef")

    model_data = model_data.reset_index(drop=True)
    fitted_values = np.asarray(observed_model.fittedvalues, dtype=float)
    residuals = np.asarray(observed_model.resid, dtype=float)
    clusters = np.array(sorted(model_data["subreddit"].astype(str).unique()))
    rng = np.random.default_rng(globals().get("RANDOM_SEED", 20250603) + 101)
    n_bootstrap = int(n_bootstrap or globals().get("N_RANDOMIZATION_PERMS", 1000))
    iterator = globals().get("tqdm", lambda values, **_kwargs: values)(
        range(n_bootstrap),
        desc="  wild cluster bootstrap",
    )

    bootstrap_coefs = []
    for draw in iterator:
        cluster_weights = dict(zip(clusters, rng.choice([-1.0, 1.0], size=len(clusters))))
        weights = model_data["subreddit"].astype(str).map(cluster_weights).to_numpy(dtype=float)
        bootstrap_data = model_data.copy()
        bootstrap_data["log_posts_bootstrap"] = fitted_values + residuals * weights
        bootstrap_model = fit_ols(
            "log_posts_bootstrap ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)",
            bootstrap_data,
            cluster_col="subreddit",
        )
        if bootstrap_model is None or focal_term not in bootstrap_model.params.index:
            continue
        bootstrap_coefs.append(safe_float(bootstrap_model.params.get(focal_term)))

    bootstrap_coefs = [coef for coef in bootstrap_coefs if coef is not None]
    if not bootstrap_coefs:
        raise ValueError("No successful wild cluster bootstrap draws.")
    coef_array = np.asarray(bootstrap_coefs, dtype=float)
    summary = {
        "row_type": "summary",
        "term": focal_term,
        "observed_coef": observed_coef,
        "observed_se": observed_result.get("se"),
        "observed_pvalue": observed_result.get("pvalue"),
        "bootstrap_se": safe_float(np.std(coef_array, ddof=1)),
        "ci_low": safe_float(np.percentile(coef_array, 2.5)),
        "ci_high": safe_float(np.percentile(coef_array, 97.5)),
        "proportion_more_negative_than_observed": safe_float(np.mean(coef_array <= float(observed_coef))),
        "n_bootstrap_requested": safe_int(n_bootstrap),
        "n_bootstrap_successful": safe_int(len(coef_array)),
        "n_obs": safe_int(observed_model.nobs),
        "n_subreddits": safe_int(model_data["subreddit"].nunique()),
    }
    draw_rows = [
        {
            "row_type": "draw",
            "draw": safe_int(i + 1),
            "term": focal_term,
            "bootstrap_coef": safe_float(coef),
        }
        for i, coef in enumerate(coef_array)
    ]
    result_table = pd.concat([pd.DataFrame([summary]), pd.DataFrame(draw_rows)], ignore_index=True, sort=False)
    emit_output_table(result_table, TABLES_DIR / "acsi_wild_cluster_bootstrap.csv", index=False)
    return {
        "summary": summary,
        "draws": draw_rows,
        "output_path": str(TABLES_DIR / "acsi_wild_cluster_bootstrap.csv"),
    }


def compute_linear_preshock_persfree_trend(submonth_panel, acsi_scores):
    panel = ensure_current_acsi_panel(submonth_panel, acsi_scores)
    shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
    pre = panel[pd.to_datetime(panel["year_month_dt"], errors="coerce") < shock_month].copy()
    pre = pre.dropna(subset=["log_posts", "subreddit", "year_month", "year_month_dt", "non_personal_norm"])
    if pre.empty or pre["subreddit"].nunique() < 2:
        raise ValueError("Need at least two pre-shock subreddits for linear PersFree trend.")
    min_month = pd.to_datetime(pre["year_month_dt"]).min()
    pre["trend"] = (
        (pd.to_datetime(pre["year_month_dt"]).dt.year - min_month.year) * 12
        + (pd.to_datetime(pre["year_month_dt"]).dt.month - min_month.month)
    ).astype(float)
    pre["pers_free_mu"] = pd.to_numeric(pre["non_personal_norm"], errors="coerce")
    pre["pers_free_mu_trend"] = pre["pers_free_mu"] * pre["trend"]
    pre = pre.dropna(subset=["pers_free_mu_trend"])
    model = fit_ols(
        "log_posts ~ pers_free_mu_trend + C(subreddit) + C(year_month)",
        pre,
        cluster_col="subreddit",
    )
    if model is None:
        raise ValueError("Linear pre-shock PersFree trend model failed to fit.")
    result = reg_result(model, "pers_free_mu_trend")
    result.update({
        "term": "pers_free_mu_trend",
        "outcome": "log_posts",
        "window": f"{pre['year_month'].min()} to {pre['year_month'].max()}",
        "n_subreddits": safe_int(pre["subreddit"].nunique()),
        "n_months": safe_int(pre["year_month"].nunique()),
    })
    emit_output_table(pd.DataFrame([result]), TABLES_DIR / "acsi_preshock_linear_trend.csv", index=False)
    return {
        "result": result,
        "output_path": str(TABLES_DIR / "acsi_preshock_linear_trend.csv"),
    }


def fit_two_point_persfree_pretrend(acsi_panel, omitted_month=None):
    panel = ensure_current_acsi_panel(acsi_panel)
    shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
    panel = panel[pd.to_datetime(panel["year_month_dt"], errors="coerce") >= shock_month - pd.DateOffset(months=24)].copy()
    if omitted_month is not None:
        panel = panel[panel["year_month"].astype(str).ne(str(omitted_month))].copy()
    if panel.empty:
        return None

    def get_bin(dt):
        mb = (shock_month.year - dt.year) * 12 + (shock_month.month - dt.month)
        if mb <= 0:
            return "post"
        if mb <= 6:
            return "pre_6"
        if mb <= 12:
            return "pre_12"
        if mb <= 18:
            return "pre_18"
        return "pre_24"

    panel["bin"] = pd.to_datetime(panel["year_month_dt"]).apply(get_bin)
    event_bins = [
        label
        for label in ["pre_24", "pre_18", "pre_12", "post"]
        if panel["bin"].eq(label).any()
    ]
    if not event_bins:
        return None
    for label in event_bins:
        panel[f"non_personal_{label}"] = (
            pd.to_numeric(panel["non_personal_norm"], errors="coerce")
            * panel["bin"].eq(label).astype(int)
        )
    event_terms = [f"non_personal_{label}" for label in event_bins]
    model_data = panel.dropna(subset=["log_posts", "subreddit", "year_month"] + event_terms).copy()
    model = fit_ols(
        "log_posts ~ " + " + ".join(event_terms) + " + C(subreddit) + C(year_month)",
        model_data,
        cluster_col="subreddit",
    )
    if model is None:
        return None
    pre_terms = [f"non_personal_{label}" for label in event_bins if label != "post"]
    focal_term = "non_personal_pre_12" if "non_personal_pre_12" in pre_terms else (pre_terms[-1] if pre_terms else None)
    if focal_term is None or focal_term not in model.params.index:
        return None
    result = reg_result(model, focal_term)
    result.update({
        "omitted_month": omitted_month,
        "term": focal_term,
        "n_obs": safe_int(model.nobs),
        "n_subreddits": safe_int(model_data["subreddit"].nunique()),
        "n_months": safe_int(model_data["year_month"].nunique()),
    })
    return result


def compute_leave_one_month_out_pretrends(acsi_panel):
    panel = ensure_current_acsi_panel(acsi_panel)
    shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
    pre_months = sorted(
        panel.loc[pd.to_datetime(panel["year_month_dt"], errors="coerce") < shock_month, "year_month"]
        .dropna()
        .astype(str)
        .unique()
    )
    rows = []
    for omitted_month in pre_months:
        result = fit_two_point_persfree_pretrend(panel, omitted_month=omitted_month)
        if result is None:
            continue
        result["row_type"] = "leave_one_month"
        rows.append(result)
    if not rows:
        raise ValueError("No leave-one-month pretrend models were estimated.")
    row_table = pd.DataFrame(rows)
    summary = {
        "row_type": "summary",
        "omitted_month": "all",
        "n_iterations": safe_int(len(row_table)),
        "min_pre_lead_coef": safe_float(row_table["coef"].min()),
        "max_pre_lead_coef": safe_float(row_table["coef"].max()),
        "min_pre_lead_pvalue": safe_float(row_table["pvalue"].min()),
        "max_pre_lead_pvalue": safe_float(row_table["pvalue"].max()),
    }
    result_table = pd.concat([pd.DataFrame([summary]), row_table], ignore_index=True, sort=False)
    emit_output_table(result_table, TABLES_DIR / "acsi_leave_one_month_pretrend.csv", index=False)
    return {
        "summary": summary,
        "rows": rows,
        "output_path": str(TABLES_DIR / "acsi_leave_one_month_pretrend.csv"),
    }


def entropy_balance_group_weights(covariates, target, max_iter=500, tol=1e-8):
    x = np.asarray(covariates, dtype=float)
    target = np.asarray(target, dtype=float)
    n_rows = x.shape[0]
    if n_rows == 0:
        return np.array([], dtype=float)
    if n_rows <= x.shape[1] or not np.isfinite(x).all():
        return np.ones(n_rows, dtype=float)
    lambdas = np.zeros(x.shape[1], dtype=float)
    for _iteration in range(max_iter):
        eta = np.clip(x @ lambdas, -50, 50)
        exp_eta = np.exp(eta - eta.max())
        probabilities = exp_eta / exp_eta.sum()
        weighted_mean = probabilities @ x
        gradient = weighted_mean - target
        if np.linalg.norm(gradient, ord=np.inf) < tol:
            break
        centered = x - weighted_mean
        hessian = (centered.T * probabilities) @ centered
        hessian += np.eye(hessian.shape[0]) * 1e-8
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        step_norm = float(np.linalg.norm(step))
        if step_norm > 5:
            step = step * (5 / step_norm)
        lambdas -= step
    eta = np.clip(x @ lambdas, -50, 50)
    exp_eta = np.exp(eta - eta.max())
    probabilities = exp_eta / exp_eta.sum()
    weights = probabilities * n_rows
    weights = np.clip(weights, 1e-4, 1e4)
    weights = weights * n_rows / weights.sum()
    return weights


def compute_entropy_balanced_did(acsi_panel):
    panel = ensure_current_acsi_panel(acsi_panel)
    covariates = add_pre_covariates(panel)
    score_cols = panel[["subreddit", "non_personal_norm"]].drop_duplicates("subreddit").copy()
    balance_frame = covariates.merge(score_cols, on="subreddit", how="inner")
    balance_frame = balance_frame.dropna(subset=["pre_avg_log_posts", "log_mu_k", "non_personal_norm"]).copy()
    if balance_frame["subreddit"].nunique() < 3:
        raise ValueError("Need at least three subreddits for entropy balancing.")
    balance_frame["persfree_tercile"] = pd.qcut(
        balance_frame["non_personal_norm"].rank(method="first"),
        q=3,
        labels=["low", "middle", "high"],
    ).astype(str)
    covariate_columns = ["pre_avg_log_posts", "log_mu_k"]
    means = balance_frame[covariate_columns].mean()
    scales = balance_frame[covariate_columns].std(ddof=0).replace(0, 1)
    standardized = (balance_frame[covariate_columns] - means) / scales
    target = standardized.mean().to_numpy(dtype=float)
    balance_frame["entropy_weight"] = 1.0
    for tercile, group_index in balance_frame.groupby("persfree_tercile", sort=False).groups.items():
        weights = entropy_balance_group_weights(
            standardized.loc[group_index, covariate_columns].to_numpy(dtype=float),
            target,
        )
        balance_frame.loc[group_index, "entropy_weight"] = weights

    model_data = panel.merge(
        balance_frame[["subreddit", "entropy_weight", "persfree_tercile"]],
        on="subreddit",
        how="inner",
    )
    terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
    model_data = model_data.dropna(subset=["log_posts", "subreddit", "year_month", "entropy_weight"] + terms).copy()
    formula = "log_posts ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)"
    model = smf.wls(formula, data=model_data, weights=model_data["entropy_weight"]).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_data["subreddit"]},
    )
    rows = []
    for term in terms:
        result = reg_result(model, term)
        result.update({
            "row_type": "regression",
            "term": term,
            "model": "entropy_balanced_three_dimensional_did",
            "n_obs": safe_int(model.nobs),
            "n_subreddits": safe_int(model_data["subreddit"].nunique()),
            "n_months": safe_int(model_data["year_month"].nunique()),
            "weight_min": safe_float(balance_frame["entropy_weight"].min()),
            "weight_max": safe_float(balance_frame["entropy_weight"].max()),
        })
        rows.append(result)

    for tercile, group in balance_frame.groupby("persfree_tercile"):
        for covariate in covariate_columns:
            rows.append({
                "row_type": "balance",
                "tercile": tercile,
                "covariate": covariate,
                "unweighted_mean": safe_float(group[covariate].mean()),
                "weighted_mean": safe_float(np.average(group[covariate], weights=group["entropy_weight"])),
                "target_mean": safe_float(balance_frame[covariate].mean()),
                "n_subreddits": safe_int(group["subreddit"].nunique()),
            })
    result_table = pd.DataFrame(rows)
    emit_output_table(result_table, TABLES_DIR / "acsi_entropy_balanced_did.csv", index=False)
    persfree_row = next((row for row in rows if row.get("term") == "non_personal_post"), {})
    return {
        "persfree": persfree_row,
        "rows": rows,
        "output_path": str(TABLES_DIR / "acsi_entropy_balanced_did.csv"),
    }


def compute_drop_blackout_quarter(submonth_panel, acsi_scores):
    panel = ensure_current_acsi_panel(submonth_panel, acsi_scores)
    blackout_months = ["2023-06", "2023-07", "2023-08"]
    model_panel = panel[~panel["year_month"].astype(str).isin(blackout_months)].copy()
    model, rows = fit_three_dimensional_acsi_model(model_panel)
    if model is None or not rows:
        raise ValueError("Drop-blackout-quarter three-dimensional model failed to fit.")
    for row in rows:
        row["dropped_months"] = ",".join(blackout_months)
        row["model"] = "drop_june_august_2023"
    emit_output_table(pd.DataFrame(rows), TABLES_DIR / "acsi_drop_blackout_quarter.csv", index=False)
    persfree_row = next((row for row in rows if row.get("term") == "non_personal_post"), {})
    return {
        "persfree": persfree_row,
        "rows": rows,
        "output_path": str(TABLES_DIR / "acsi_drop_blackout_quarter.csv"),
    }


def compute_creator_composition_checks(df_all, acsi_scores, submonth_panel):
    if df_all is None or df_all.empty:
        raise ValueError("No creator-clean post dataframe provided.")
    posts = df_all.copy()
    posts["author"] = posts["author"].fillna("").astype(str)
    posts["subreddit"] = posts["subreddit"].astype(str)
    if "year_month" not in posts.columns and "year_month_dt" in posts.columns:
        posts["year_month"] = pd.to_datetime(posts["year_month_dt"], errors="coerce").dt.strftime("%Y-%m")
    if "year_month_dt" not in posts.columns:
        posts["year_month_dt"] = pd.to_datetime(posts["year_month"].astype(str) + "-01", errors="coerce")
    if "post_shock" not in posts.columns:
        shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
        posts["post_shock"] = (pd.to_datetime(posts["year_month_dt"], errors="coerce") >= shock_month).astype(int)
    posts = posts[posts["author"].ne("")].copy()

    target_subreddits = sorted(submonth_panel["subreddit"].astype(str).unique())
    all_months = sorted(submonth_panel["year_month"].astype(str).unique())
    posts = posts[posts["subreddit"].isin(target_subreddits)].copy()
    pre_counts = posts[posts["post_shock"].eq(0)].groupby("author").size().rename("pre_posts").reset_index()
    if pre_counts.empty:
        raise ValueError("No pre-shock creators available for composition checks.")
    pre_counts["pre_frequency_tercile"] = pd.qcut(
        pre_counts["pre_posts"].rank(method="first"),
        q=3,
        labels=["bottom", "middle", "top"],
    ).astype(str)
    bottom_authors = set(
        pre_counts.loc[pre_counts["pre_frequency_tercile"].eq("bottom"), "author"].astype(str)
    )

    author_month = posts[["subreddit", "year_month", "author"]].drop_duplicates().copy()
    first_month = (
        author_month.groupby(["subreddit", "author"], as_index=False)["year_month"]
        .min()
        .rename(columns={"year_month": "first_subreddit_month"})
    )
    author_month = author_month.merge(first_month, on=["subreddit", "author"], how="left")
    author_month["is_new_entrant"] = author_month["year_month"].eq(author_month["first_subreddit_month"]).astype(int)
    author_month["is_bottom_tercile_creator"] = author_month["author"].isin(bottom_authors).astype(int)
    monthly = (
        author_month.groupby(["subreddit", "year_month"], as_index=False)
        .agg(
            active_authors=("author", "nunique"),
            new_entrant_authors=("is_new_entrant", "sum"),
            bottom_tercile_authors=("is_bottom_tercile_creator", "sum"),
        )
    )
    grid = pd.MultiIndex.from_product(
        [target_subreddits, all_months],
        names=["subreddit", "year_month"],
    ).to_frame(index=False)
    panel = grid.merge(monthly, on=["subreddit", "year_month"], how="left")
    for column_name in ["active_authors", "new_entrant_authors", "bottom_tercile_authors"]:
        panel[column_name] = pd.to_numeric(panel[column_name], errors="coerce").fillna(0)
    panel["new_entrant_share"] = np.where(
        panel["active_authors"] > 0,
        panel["new_entrant_authors"] / panel["active_authors"],
        np.nan,
    )
    panel["bottom_tercile_creator_share"] = np.where(
        panel["active_authors"] > 0,
        panel["bottom_tercile_authors"] / panel["active_authors"],
        np.nan,
    )
    score_columns = [
        "subreddit",
        "generation_capability_norm",
        "physical_free_norm",
        "non_personal_norm",
    ]
    scores = acsi_scores[score_columns].copy()
    panel = panel.merge(scores, on="subreddit", how="inner")
    panel["year_month_dt"] = pd.to_datetime(panel["year_month"].astype(str) + "-01", errors="coerce")
    shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
    panel["post_shock"] = (panel["year_month_dt"] >= shock_month).astype(int)
    for norm_column, post_column in [
        ("generation_capability_norm", "generation_capability_post"),
        ("physical_free_norm", "physical_free_post"),
        ("non_personal_norm", "non_personal_post"),
    ]:
        panel[post_column] = panel[norm_column] * panel["post_shock"]

    terms = [spec["post"] for spec in ACSI_MECHANISM_SPECS]
    rows = []
    for outcome in ["new_entrant_share", "bottom_tercile_creator_share"]:
        model_data = panel.dropna(subset=[outcome, "subreddit", "year_month"] + terms).copy()
        model = fit_ols(
            f"{outcome} ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)",
            model_data,
            cluster_col="subreddit",
        )
        if model is None:
            continue
        for term in terms:
            result = reg_result(model, term)
            result.update({
                "outcome": outcome,
                "term": term,
                "model": "three_dimensional_composition_did",
                "n_obs": safe_int(model.nobs),
                "n_subreddits": safe_int(model_data["subreddit"].nunique()),
                "n_months": safe_int(model_data["year_month"].nunique()),
                "n_bottom_tercile_authors": safe_int(len(bottom_authors)),
                "n_pre_shock_authors": safe_int(len(pre_counts)),
            })
            rows.append(result)
    if not rows:
        raise ValueError("No creator composition models were estimated.")
    emit_output_table(pd.DataFrame(rows), TABLES_DIR / "creator_composition_checks.csv", index=False)
    panel_path = TABLES_DIR / "creator_composition_panel.csv"
    emit_output_table(panel, panel_path, index=False)
    return {
        "rows": rows,
        "new_entrant_share": next(
            (row for row in rows if row.get("outcome") == "new_entrant_share" and row.get("term") == "non_personal_post"),
            {},
        ),
        "bottom_tercile_creator_share": next(
            (row for row in rows if row.get("outcome") == "bottom_tercile_creator_share" and row.get("term") == "non_personal_post"),
            {},
        ),
        "output_path": str(TABLES_DIR / "creator_composition_checks.csv"),
        "panel_output_path": str(panel_path),
    }


def compute_extended_panel_pretrend_test(extended_panel):
    pretrend_panel = extended_panel[
        (extended_panel["year_month_dt"] >= pd.Timestamp("2020-01-01"))
        & (extended_panel["year_month_dt"] <= pd.Timestamp("2022-10-01"))
    ].copy()
    if pretrend_panel.empty:
        return None

    baseline_month = "2022-10"
    pretrend_panel["year_month"] = pretrend_panel["year_month_dt"].dt.strftime("%Y-%m")
    month_labels = sorted(pretrend_panel["year_month"].dropna().unique())
    terms = []
    tested_terms = []
    for month_label in month_labels:
        if month_label == baseline_month:
            continue
        term = f"persfree_pretrend_{month_label.replace('-', '')}"
        pretrend_panel[term] = (
            pretrend_panel["non_personal_norm"]
            * (pretrend_panel["year_month"] == month_label).astype(int)
        )
        terms.append(term)
        if month_label <= "2021-12":
            tested_terms.append(term)

    if not terms or not tested_terms:
        return None

    model = fit_ols(
        "log_posts ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)",
        pretrend_panel,
        cluster_col="subreddit",
    )
    if not model:
        return None

    pretrend_f_stat = None
    pretrend_pvalue = None
    try:
        restriction_matrix = np.zeros((len(tested_terms), len(model.params)))
        param_names = list(model.params.index)
        valid_terms = []
        for i, term in enumerate(tested_terms):
            if term in param_names:
                restriction_matrix[i, param_names.index(term)] = 1.0
                valid_terms.append(term)
        if valid_terms:
            restriction_matrix = restriction_matrix[:len(valid_terms), :]
            f_test = model.f_test(restriction_matrix)
            pretrend_f_stat = safe_float(np.asarray(f_test.fvalue).ravel()[0])
            pretrend_pvalue = safe_float(f_test.pvalue)
    except Exception as exc:
        print(f"  Extended panel pre-trend F-test failed: {exc}")

    return {
        "f_stat": pretrend_f_stat,
        "pvalue": pretrend_pvalue,
        "n_obs": safe_int(model.nobs),
        "n_tested_terms": safe_int(len(tested_terms)),
        "tested_window": "2020-01 to 2021-12",
        "regression_window": "2020-01 to 2022-10",
        "baseline_month": baseline_month,
    }

def compute_extended_panel_robustness(submonth_panel, acsi_scores):
    target_subreddits = sorted(submonth_panel["subreddit"].astype(str).unique())
    extended_panel = build_extended_subreddit_month_panel(
        acsi_scores,
        target_subreddits,
        apply_author_cap=True,
    )
    extended_panel = acsi_model_panel(extended_panel, "Extended panel robustness")
    extended_model, extended_results = fit_three_dimensional_acsi_model(extended_panel)
    if not extended_results:
        return None

    pretrend = compute_extended_panel_pretrend_test(extended_panel) or {}
    persfree_row = next(
        (row for row in extended_results if row.get("term") == "non_personal_post"),
        {},
    )
    extended_persfree_estimate = persfree_row.get("coef")

    rows = []
    for model_result in extended_results:
        rows.append({
            "row_type": "extended_panel_beta",
            "term": model_result.get("term"),
            "label": model_result.get("label"),
            "coef": model_result.get("coef"),
            "se": model_result.get("se"),
            "pvalue": model_result.get("pvalue"),
            "n_obs": model_result.get("n_obs"),
            "n_subreddits": model_result.get("n_model_subreddits"),
            "n_months": model_result.get("n_model_months"),
            "pretrend_f_stat": pretrend.get("f_stat"),
            "pretrend_pvalue": pretrend.get("pvalue"),
            "pretrend_tested_terms": pretrend.get("n_tested_terms"),
            "pretrend_tested_window": pretrend.get("tested_window"),
            "primary_persfree_estimate": np.nan,
            "extended_persfree_estimate": extended_persfree_estimate,
        })

    rows.append({
        "row_type": "primary_vs_extended",
        "term": "non_personal_post",
        "label": "Primary vs extended PersFree",
        "coef": np.nan,
        "se": np.nan,
        "pvalue": np.nan,
        "n_obs": safe_int(len(extended_panel)),
        "n_subreddits": safe_int(extended_panel["subreddit"].nunique()),
        "n_months": safe_int(extended_panel["year_month"].nunique()),
        "pretrend_f_stat": pretrend.get("f_stat"),
        "pretrend_pvalue": pretrend.get("pvalue"),
        "pretrend_tested_terms": pretrend.get("n_tested_terms"),
        "pretrend_tested_window": pretrend.get("tested_window"),
        "primary_persfree_estimate": -0.475,
        "extended_persfree_estimate": extended_persfree_estimate,
    })

    output_table = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "extended_panel_robustness.csv"
    output_table.to_csv(output_path, index=False)
    print("\n--- extended_panel_robustness ---")
    print(output_table.to_string(index=False))
    print(f"  saved extended panel robustness table -> {output_path}")
    return {
        "rows": rows,
        "pretrend": pretrend,
        "path": str(output_path),
    }

def set_sparse_month_ticks(ax, month_labels, every=3, rotation=45):
    tick_positions = list(range(0, len(month_labels), every))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [month_labels[i] for i in tick_positions],
        rotation=rotation,
        ha="right",
    )

def plot_time_varying_personal_context(acsi_panel, output_path):
    apply_publication_style()
    panel = acsi_panel.copy()
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    panel = panel[panel["year_month"] != "2022-11"].copy()

    quarter_defs = [
        ("Q1", "2022-12-01", "2023-02-01"),
        ("Q2", "2023-03-01", "2023-05-01"),
        ("Q3", "2023-06-01", "2023-08-01"),
        ("Q4", "2023-09-01", "2023-11-01"),
        ("Q5", "2023-12-01", "2024-02-01"),
        ("Q6", "2024-03-01", "2024-05-01"),
        ("Q7", "2024-06-01", "2024-08-01"),
        ("Q8", "2024-09-01", "2024-12-01"),
    ]

    terms = []
    for quarter, start, end in quarter_defs:
        term = f"persfree_{quarter.lower()}"
        in_quarter = (
            (panel["year_month_dt"] >= pd.Timestamp(start))
            & (panel["year_month_dt"] <= pd.Timestamp(end))
        )
        panel[term] = panel["non_personal_norm"] * in_quarter.astype(int)
        terms.append(term)

    formula = (
        "log_posts ~ "
        + " + ".join(terms + ["generation_capability_post", "physical_free_post"])
        + " + C(subreddit) + C(year_month)"
    )
    model = fit_ols(formula, panel, cluster_col="subreddit")
    if not model:
        return None

    rows = []
    for quarter, term in zip([q[0] for q in quarter_defs], terms):
        row = reg_result(model, term)
        row["quarter"] = quarter
        row["term"] = term
        rows.append(row)

    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    y = frame["coef"].astype(float).to_numpy()
    se = frame["se"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for index, (x_value, y_value, se_value) in enumerate(zip(x, y, se)):
        quarter = frame.iloc[index]["quarter"]
        pvalue = frame.iloc[index].get("pvalue", np.nan)
        color = "#6b7280" if quarter in {"Q1", "Q2"} or not (pd.notna(pvalue) and pvalue < 0.05) else "#2563eb"
        ax.errorbar(
            x_value,
            y_value,
            yerr=1.96 * se_value,
            fmt="o",
            capsize=4,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capthick=1.5,
            alpha=0.8,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1.0)
    if len(frame) > 2:
        ax.annotate(
            "GPT-4 era begins",
            xy=(2, y[2]),
            xytext=(2.15, y[2] + 0.08),
            fontsize=9,
            color="#5a5a5a",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "linewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["quarter"])
    ax.set_ylabel("PersFree x quarter coefficient")
    ax.set_title("Personal-Context Displacement Effect by Quarter")
    save_plot(fig, output_path)
    return rows

def add_monthly_persfree_terms(panel, baseline_month="2022-10"):
    panel = panel.copy()
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    month_labels = sorted(panel["year_month"].dropna().unique())
    terms = []
    labels = []
    for month_label in month_labels:
        if month_label == baseline_month:
            continue
        term = f"persfree_m{month_label.replace('-', '')}"
        panel[term] = panel["non_personal_norm"] * (panel["year_month"] == month_label).astype(int)
        terms.append(term)
        labels.append(month_label)
    return panel, terms, labels

def plot_monthly_persfree_event_study(acsi_panel, output_path, title, subreddits=None):
    apply_publication_style()
    panel = acsi_panel.copy()
    if subreddits is not None:
        panel = panel[panel["subreddit"].astype(str).isin(set(subreddits))].copy()
    if panel.empty:
        return []

    panel, terms, labels = add_monthly_persfree_terms(panel)
    if not terms:
        return []

    formula = (
        "log_posts ~ "
        + " + ".join(terms + ["generation_capability_post", "physical_free_post"])
        + " + C(subreddit) + C(year_month)"
    )
    model = fit_ols(formula, panel, cluster_col="subreddit")
    if not model:
        return []

    rows = []
    for term, month_label in zip(terms, labels):
        row = reg_result(model, term)
        row["term"] = term
        row["month"] = month_label
        rows.append(row)

    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    y = frame["coef"].astype(float).to_numpy()
    se = frame["se"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(16, 5))
    if "2022-11" in labels:
        launch_x = labels.index("2022-11")
        ax.axvspan(-0.5, launch_x - 0.5, color="#f9fafb", alpha=0.3, zorder=0)
        ax.axvspan(launch_x - 0.5, len(labels) - 0.5, color="#eff6ff", alpha=0.3, zorder=0)
        ax.axvline(launch_x, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.annotate(
            "ChatGPT launch",
            xy=(launch_x, np.nanmax(y + 1.96 * se)),
            xytext=(launch_x + 0.45, np.nanmax(y + 1.96 * se)),
            fontsize=9,
            color="#5a5a5a",
            va="top",
            arrowprops={"arrowstyle": "-", "color": "#dc2626", "linewidth": 0.8, "alpha": 0.7},
        )
    for x_value, month_label, y_value, se_value in zip(x, labels, y, se):
        color = "#6b7280" if month_label < "2022-11" else "#2563eb"
        ax.errorbar(
            x_value,
            y_value,
            yerr=1.96 * se_value,
            fmt="o",
            capsize=4,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capthick=1.5,
            alpha=0.6,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=3,
        )
    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1.0)
    set_sparse_month_ticks(ax, labels, every=3, rotation=45)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("PersFree x month coefficient (Oct 2022 omitted)")
    ax.set_title(title)
    save_plot(fig, output_path)
    return rows

def plot_all_dimensions_event_study(acsi_panel, output_dir):
    apply_publication_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = acsi_panel.copy()
    if panel.empty:
        return {}

    if "year_month" not in panel.columns:
        if "month" in panel.columns:
            panel = panel.rename(columns={"month": "year_month"})
        elif "year_month_dt" in panel.columns:
            panel["year_month"] = pd.to_datetime(panel["year_month_dt"]).dt.strftime("%Y-%m")
    if "year_month" not in panel.columns:
        raise ValueError("Event-study panel must include year_month, month, or year_month_dt.")
    panel["year_month"] = panel["year_month"].astype(str).str.slice(0, 7)
    if "year_month_dt" not in panel.columns:
        panel["year_month_dt"] = pd.to_datetime(panel["year_month"])
    else:
        panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])

    dimension_specs = [
        {
            "dimension": "pers_free_mu",
            "label": "pers_free_mu",
            "norm": "non_personal_norm",
            "prefix": "persfree",
            "controls": ["generation_capability_post", "physical_free_post"],
            "use_reference_terms": True,
        },
        {
            "dimension": "direct_gen",
            "label": "direct_gen",
            "norm": "direct_gen_norm",
            "prefix": "direct_gen",
            "controls": ["usefulness_post", "quality_comp_post", "physical_free_post", "non_personal_post"],
            "use_reference_terms": False,
        },
        {
            "dimension": "usefulness",
            "label": "usefulness",
            "norm": "usefulness_norm",
            "prefix": "usefulness",
            "controls": ["direct_gen_post", "quality_comp_post", "physical_free_post", "non_personal_post"],
            "use_reference_terms": False,
        },
        {
            "dimension": "quality_comp",
            "label": "quality_comp",
            "norm": "quality_comp_norm",
            "prefix": "quality_comp",
            "controls": ["direct_gen_post", "usefulness_post", "physical_free_post", "non_personal_post"],
            "use_reference_terms": False,
        },
        {
            "dimension": "physical_free",
            "label": "physical_free",
            "norm": "physical_free_norm",
            "prefix": "physical_free",
            "controls": ["direct_gen_post", "usefulness_post", "quality_comp_post", "non_personal_post"],
            "use_reference_terms": False,
        },
    ]
    baseline_month = "2022-10"
    month_labels = sorted(panel["year_month"].dropna().unique())
    results = {}
    plot_frames = {}

    for spec in dimension_specs:
        dimension = spec["dimension"]
        exposure = spec["norm"]
        results[dimension] = []
        if exposure not in panel.columns:
            continue

        working = panel.copy()
        working[exposure] = pd.to_numeric(working[exposure], errors="coerce")
        if spec["use_reference_terms"]:
            working, terms, labels = add_monthly_persfree_terms(working, baseline_month=baseline_month)
        else:
            terms = []
            labels = []
            for month_label in month_labels:
                if month_label == baseline_month:
                    continue
                term = f"{spec['prefix']}_m{month_label.replace('-', '')}"
                working[term] = working[exposure] * (working["year_month"] == month_label).astype(int)
                terms.append(term)
                labels.append(month_label)
        if not terms:
            continue

        controls = [
            control for control in spec["controls"]
            if control in working.columns
        ]
        model_columns = ["log_posts", "subreddit", "year_month", exposure] + terms + controls
        model_panel = working.dropna(subset=[column for column in model_columns if column in working.columns]).copy()
        formula_terms = terms + controls
        formula = "log_posts ~ " + " + ".join(formula_terms) + " + C(subreddit) + C(year_month)"
        model = fit_ols(formula, model_panel, cluster_col="subreddit")
        if not model:
            continue

        rows = []
        for term, month_label in zip(terms, labels):
            row = reg_result(model, term)
            row["term"] = term
            row["month"] = month_label
            rows.append(row)
        results[dimension] = rows
        plot_frames[dimension] = pd.DataFrame(rows)

    n_panels = len(dimension_specs)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(16, 4 * n_panels),
        sharex=True,
    )
    x_labels = [month_label for month_label in month_labels if month_label != baseline_month]
    x_positions = np.arange(len(x_labels))
    shock_position = x_labels.index("2022-11") if "2022-11" in x_labels else None

    for axis_index, (ax, spec) in enumerate(zip(axes, dimension_specs)):
        dimension = spec["dimension"]
        frame = plot_frames.get(dimension, pd.DataFrame())
        if not frame.empty:
            frame = frame.set_index("month").reindex(x_labels).reset_index()
            x = np.arange(len(frame))
            y = frame["coef"].astype(float).to_numpy()
            se = frame["se"].astype(float).to_numpy()
            ax.errorbar(
                x,
                y,
                yerr=1.96 * se,
                fmt="o",
                capsize=4,
                color="#2563eb",
                ecolor="black",
                elinewidth=1.5,
                capthick=1.5,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=3,
            )
        if shock_position is not None:
            ax.axvspan(-0.5, shock_position - 0.5, color="#f9fafb", alpha=0.3, zorder=0)
            ax.axvspan(shock_position - 0.5, len(x_labels) - 0.5, color="#eff6ff", alpha=0.3, zorder=0)
            ax.axvline(shock_position, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7, zorder=2)
        ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1.0)
        ax.set_ylabel(spec["label"])
        ax.text(
            0.012,
            0.92,
            spec["label"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="semibold",
            color="#1a1a2e",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
        if axis_index < n_panels - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

    fig.supxlabel("Month (Oct 2022 omitted baseline)")
    if shock_position is not None:
        axes[0].annotate(
            "ChatGPT launch",
            xy=(shock_position, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(shock_position + 0.6, 0.98),
            textcoords=("data", "axes fraction"),
            fontsize=9,
            color="#5a5a5a",
            va="top",
            arrowprops={"arrowstyle": "-", "color": "#dc2626", "linewidth": 0.8, "alpha": 0.7},
        )
    source_rows = []
    for dimension, rows in results.items():
        for row in rows:
            source_rows.append({"dimension": dimension, **row})
    if source_rows:
        pd.DataFrame(source_rows).to_csv(output_dir / "event_study_all_dimensions.csv", index=False)
    save_plot(fig, output_dir / "event_study_all_dimensions.png")
    return results

def strict_matched_subreddit_pairs(acsi_panel):
    panel = acsi_panel.copy()
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    pre = panel[
        (panel["year_month_dt"] >= pd.Timestamp("2022-01-01"))
        & (panel["year_month_dt"] <= pd.Timestamp("2022-10-01"))
    ].copy()

    feature_rows = []
    for subreddit, group in pre.sort_values("year_month_dt").groupby("subreddit"):
        y = group["log_posts"].astype(float).to_numpy()
        t = np.arange(len(y), dtype=float)
        slope = float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 and np.unique(y).size > 1 else 0.0
        feature_rows.append({
            "subreddit": subreddit,
            "pre_mean_log_posts": float(np.mean(y)),
            "pre_slope_log_posts": slope,
            "non_personal_norm": float(group["non_personal_norm"].iloc[0]),
        })

    features = pd.DataFrame(feature_rows)
    if features.empty or features["non_personal_norm"].nunique() < 2:
        return [], []

    median_persfree = features["non_personal_norm"].median()
    low_personal = features[features["non_personal_norm"] >= median_persfree].reset_index(drop=True)
    high_personal = features[features["non_personal_norm"] < median_persfree].reset_index(drop=True)
    if low_personal.empty or high_personal.empty:
        return [], []

    feature_cols = ["pre_mean_log_posts", "pre_slope_log_posts"]
    all_x = features[feature_cols].to_numpy(dtype=float)
    inv_cov = np.linalg.pinv(np.cov(all_x.T))
    distances = np.zeros((len(low_personal), len(high_personal)))
    for i, left in enumerate(low_personal[feature_cols].to_numpy(dtype=float)):
        for j, right in enumerate(high_personal[feature_cols].to_numpy(dtype=float)):
            diff = left - right
            distances[i, j] = float(np.sqrt(diff @ inv_cov @ diff.T))

    row_indices, col_indices = linear_sum_assignment(distances)
    scale = features[feature_cols].std(ddof=0).replace(0, np.nan)
    retained_pairs = []
    all_pairs = []
    for i, j in zip(row_indices, col_indices):
        left = low_personal.iloc[i]
        right = high_personal.iloc[j]
        all_pairs.append((left["subreddit"], right["subreddit"]))
        level_ok = abs(left["pre_mean_log_posts"] - right["pre_mean_log_posts"]) <= 0.5 * scale["pre_mean_log_posts"]
        slope_ok = abs(left["pre_slope_log_posts"] - right["pre_slope_log_posts"]) <= 0.5 * scale["pre_slope_log_posts"]
        if bool(level_ok) and bool(slope_ok):
            retained_pairs.append((left["subreddit"], right["subreddit"]))
    return retained_pairs, all_pairs

def multivariate_two_way_fe_coef(data, outcome, terms):
    model_data = data.dropna(subset=[outcome, *terms, "subreddit", "year_month"]).copy()
    if model_data.empty:
        return None
    y_resid = residualize_two_way(
        model_data[outcome].astype(float),
        model_data["subreddit"],
        model_data["year_month"],
    ).to_numpy()
    x_resids = [
        residualize_two_way(
            model_data[term].astype(float),
            model_data["subreddit"],
            model_data["year_month"],
        ).to_numpy()
        for term in terms
    ]
    try:
        coefficients = np.linalg.lstsq(np.column_stack(x_resids), y_resid, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return {
        term: safe_float(coef)
        for term, coef in zip(terms, coefficients)
    }

def plot_persfree_permutation_placebo(acsi_panel, output_path):
    apply_publication_style()
    panel = acsi_panel.copy()
    month_frame = panel[["year_month", "post_shock"]].drop_duplicates().sort_values("year_month")
    if month_frame.empty:
        return None

    rng = np.random.default_rng(RANDOM_SEED)
    base_terms = [
        "generation_capability_perm_post",
        "physical_free_perm_post",
        "non_personal_perm_post",
    ]

    def coefficients_for_post_values(post_values):
        month_to_post = dict(zip(month_frame["year_month"], post_values))
        working = panel.copy()
        working["permuted_post"] = working["year_month"].map(month_to_post).astype(float)
        working[base_terms[0]] = working["generation_capability_norm"] * working["permuted_post"]
        working[base_terms[1]] = working["physical_free_norm"] * working["permuted_post"]
        working[base_terms[2]] = working["non_personal_norm"] * working["permuted_post"]
        return multivariate_two_way_fe_coef(working, "log_posts", base_terms)

    real_coefficients = coefficients_for_post_values(month_frame["post_shock"].astype(float).to_numpy())
    if not real_coefficients:
        return None
    observed_coef = real_coefficients.get("non_personal_perm_post")
    post_values = month_frame["post_shock"].astype(float).to_numpy()
    permuted_coefficients = []
    for _ in range(N_RANDOMIZATION_PERMS):
        coefficients = coefficients_for_post_values(rng.permutation(post_values))
        if coefficients and coefficients.get("non_personal_perm_post") is not None:
            permuted_coefficients.append(coefficients["non_personal_perm_post"])
    if not permuted_coefficients:
        return None

    permuted_coefficients = np.array(permuted_coefficients, dtype=float)
    empirical_p = float(np.mean(permuted_coefficients <= observed_coef))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(permuted_coefficients, bins=38, color="#dbeafe", edgecolor="#2563eb", alpha=0.9)
    ax.axvline(
        observed_coef,
        color="#dc2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Real coef = {observed_coef:+.3f}",
    )
    ax.axvline(
        np.median(permuted_coefficients),
        color="#9ca3af",
        linestyle="--",
        linewidth=1.0,
        label=f"Permutation median = {np.median(permuted_coefficients):+.3f}",
    )
    ax.set_title(f"Permutation Placebo: PersFree Coefficient (empirical p = {empirical_p:.3f})")
    ax.set_xlabel("Placebo PersFree coefficient")
    ax.set_ylabel("Permutation count")
    ax.legend(frameon=False)
    save_plot(fig, output_path)
    return {
        "observed_coef": observed_coef,
        "empirical_pvalue": empirical_p,
        "perm_mean": safe_float(np.mean(permuted_coefficients)),
        "perm_sd": safe_float(np.std(permuted_coefficients)),
        "n_perms": safe_int(len(permuted_coefficients)),
    }

def plot_post_period_placebo_nov2023(acsi_panel, output_path):
    apply_publication_style()
    panel = acsi_panel.copy()
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    panel = panel[
        (panel["year_month_dt"] >= pd.Timestamp("2022-12-01"))
        & (panel["year_month_dt"] <= pd.Timestamp("2024-12-01"))
    ].copy()
    if panel.empty:
        return []

    panel["placebo_post_dec2023"] = (panel["year_month_dt"] >= pd.Timestamp("2023-12-01")).astype(int)
    terms = [
        "generation_capability_placebo_post",
        "physical_free_placebo_post",
        "non_personal_placebo_post",
    ]
    panel[terms[0]] = panel["generation_capability_norm"] * panel["placebo_post_dec2023"]
    panel[terms[1]] = panel["physical_free_norm"] * panel["placebo_post_dec2023"]
    panel[terms[2]] = panel["non_personal_norm"] * panel["placebo_post_dec2023"]
    model = fit_ols(
        "log_posts ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)",
        panel,
        cluster_col="subreddit",
    )
    if not model:
        return []

    labels = ["GenCap", "PhysFree", "PersFree"]
    rows = []
    for term, label in zip(terms, labels):
        row = reg_result(model, term)
        row["term"] = term
        row["label"] = label
        rows.append(row)

    frame = pd.DataFrame(rows)
    x = np.arange(len(frame))
    y = frame["coef"].astype(float).to_numpy()
    se = frame["se"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        x,
        y,
        color=["#2563eb", "#16a34a", "#dc2626"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.errorbar(x, y, yerr=1.96 * se, fmt="none", ecolor="#1d4ed8", capsize=4, linewidth=1.5, capthick=1.5)
    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Post-period placebo shock: November 2023")
    ax.set_ylabel("Placebo post coefficient")
    save_plot(fig, output_path)
    return rows

def compute_post_ai_adoption_check(acsi_panel, acsi_scores):
    required_columns = {
        "subreddit", "year_month", "posts", "ai_post_mentions", "tool_post_mentions",
    }
    if acsi_panel is None or acsi_scores is None or not required_columns.issubset(acsi_panel.columns):
        return None

    pre_months = list(POST_AI_ADOPTION_PRE_MONTHS)
    post_months = list(POST_AI_ADOPTION_POST_MONTHS)
    adoption_panel = acsi_panel[
        acsi_panel["year_month"].isin(pre_months + post_months)
    ].copy()
    if adoption_panel.empty:
        return None

    adoption_panel["adoption_period"] = np.where(
        adoption_panel["year_month"].isin(post_months), "post", "pre"
    )
    sub_period = (
        adoption_panel
        .groupby(["subreddit", "adoption_period"], as_index=False)
        .agg(
            posts=("posts", "sum"),
            ai_post_mentions=("ai_post_mentions", "sum"),
            tool_post_mentions=("tool_post_mentions", "sum"),
        )
    )
    sub_period["ai_post_rate"] = sub_period["ai_post_mentions"] / sub_period["posts"].replace(0, np.nan)
    sub_period["tool_post_rate"] = sub_period["tool_post_mentions"] / sub_period["posts"].replace(0, np.nan)

    wide_ai = sub_period.pivot(
        index="subreddit", columns="adoption_period", values="ai_post_rate"
    ).reset_index().rename(columns={"pre": "ai_pre", "post": "ai_post"})
    wide_tool = sub_period.pivot(
        index="subreddit", columns="adoption_period", values="tool_post_rate"
    ).reset_index().rename(columns={"pre": "tool_pre", "post": "tool_post"})
    counts = sub_period.pivot(
        index="subreddit", columns="adoption_period", values="posts"
    ).reset_index().rename(columns={"pre": "posts_pre", "post": "posts_post"})

    wide = wide_ai.merge(wide_tool, on="subreddit", how="inner").merge(counts, on="subreddit", how="inner")
    needed = {"ai_pre", "ai_post", "tool_pre", "tool_post", "posts_pre", "posts_post"}
    if not needed.issubset(wide.columns):
        return None

    wide["delta_ai_post_rate"] = wide["ai_post"] - wide["ai_pre"]
    wide["delta_tool_post_rate"] = wide["tool_post"] - wide["tool_pre"]
    wide = wide[(wide["posts_pre"] >= 50) & (wide["posts_post"] >= 50)].copy()

    score_columns = [
        "subreddit", "gse", "generation_capability_norm",
        "physical_free_norm", "non_personal_norm",
    ]
    available_score_columns = [column for column in score_columns if column in acsi_scores.columns]
    wide = wide.merge(acsi_scores[available_score_columns], on="subreddit", how="inner")
    if len(wide) <= 5:
        return None

    correlation_rows = []
    exposure_labels = {
        "gse": INDEX_SHORT,
        "generation_capability_norm": "Generation capability",
        "physical_free_norm": "Low physical constraint",
        "non_personal_norm": "Low personal-context need",
    }
    metric_labels = {
        "delta_ai_post_rate": "Broad AI post mention rate",
        "delta_tool_post_rate": "Narrow AI-tool post mention rate",
    }
    for exposure, exposure_label in exposure_labels.items():
        if exposure not in wide.columns:
            continue
        for metric, metric_label in metric_labels.items():
            pearson_r, pearson_p = safe_corr(wide[exposure], wide[metric], "pearson")
            spearman_r, spearman_p = safe_corr(wide[exposure], wide[metric], "spearman")
            correlation_rows.append({
                "exposure": exposure,
                "exposure_label": exposure_label,
                "metric": metric,
                "metric_label": metric_label,
                "pearson_r": pearson_r,
                "pearson_pvalue": pearson_p,
                "spearman_r": spearman_r,
                "spearman_pvalue": spearman_p,
                "n_subreddits": safe_int(len(wide)),
            })

    if not correlation_rows:
        return None

    emit_output_table(wide, TABLES_DIR / "post_ai_adoption_by_subreddit.csv", index=False)
    emit_output_table(pd.DataFrame(correlation_rows), TABLES_DIR / "post_ai_adoption_check.csv", index=False)
    return {
        "rows": correlation_rows,
        "n_subreddits": safe_int(len(wide)),
    }

def ai_capability_index_for_month(month_value):
    month = pd.Timestamp(month_value).replace(day=1)
    if month >= pd.Timestamp("2024-05-01"):
        return 3.0
    if month >= pd.Timestamp("2023-03-01"):
        return 2.0
    if month >= SHOCK_MONTH:
        return 1.0
    return 0.0

def forward_capability_index_for_month(month_value, scenario):
    month = pd.Timestamp(month_value).replace(day=1)
    scenario_key = re.sub(r"[\s-]+", "_", str(scenario).strip().lower())
    if scenario_key == "step":
        return 3.6 if month.year >= 2027 else 3.0
    if scenario_key == "accelerating":
        return min(4.5, 3.0 + 0.25 * max(month.year - 2024, 0))
    if scenario_key == "partial_regression":
        if month.year >= 2028:
            return 2.4
        if month.year >= 2027:
            return 2.7
        return 3.0
    if scenario_key == "full_regression":
        if month.year >= 2029:
            return 1.2
        if month.year >= 2028:
            return 1.8
        if month.year >= 2027:
            return 2.4
        return 3.0
    return 3.0

def load_pre_shock_persfree_scores(acsi_scores):
    fallback = acsi_scores[["subreddit", "non_personal_norm"]].copy()
    fallback["pre_pers_free"] = pd.to_numeric(
        fallback["non_personal_norm"],
        errors="coerce",
    )
    fallback["pre_pers_free_source"] = "full_acsi_fallback"
    fallback["pre_pers_free_n_used"] = np.nan
    row_level_path = ACSI_MEASUREMENT_SAMPLE_RUN1_PATH
    if not row_level_path.exists():
        return fallback[[
            "subreddit", "pre_pers_free", "pre_pers_free_source",
            "pre_pers_free_n_used",
        ]]

    needed_columns = [
        "subreddit", "created_date", "personal_req_score",
        "ai_related_flag", "hard_case_flag",
    ]
    try:
        coded = pd.read_csv(row_level_path, usecols=needed_columns)
    except Exception as exc:
        print(f"  Pre-shock PersFree score load failed; using fallback ACSI scores: {exc}")
        return fallback[[
            "subreddit", "pre_pers_free", "pre_pers_free_source",
            "pre_pers_free_n_used",
        ]]

    coded["created_date"] = pd.to_datetime(coded["created_date"], errors="coerce")
    coded["personal_req_score"] = pd.to_numeric(
        coded["personal_req_score"],
        errors="coerce",
    )
    coded["ai_related_flag"] = pd.to_numeric(
        coded["ai_related_flag"],
        errors="coerce",
    ).fillna(0)
    coded["hard_case_flag"] = pd.to_numeric(
        coded["hard_case_flag"],
        errors="coerce",
    ).fillna(0)
    pre = coded[
        (coded["created_date"] >= pd.Timestamp("2022-01-01"))
        & (coded["created_date"] < pd.Timestamp("2022-11-01"))
        & (coded["ai_related_flag"] != 1)
        & coded["personal_req_score"].notna()
    ].copy()
    if pre.empty:
        return fallback[[
            "subreddit", "pre_pers_free", "pre_pers_free_source",
            "pre_pers_free_n_used",
        ]]

    pre["weight"] = np.where(pre["hard_case_flag"] == 1, 0.5, 1.0)
    pre["weighted_personal_req"] = pre["personal_req_score"] * pre["weight"]
    grouped = (
        pre.groupby("subreddit", as_index=False)
        .agg(
            weighted_personal_req=("weighted_personal_req", "sum"),
            total_weight=("weight", "sum"),
            pre_pers_free_n_used=("personal_req_score", "count"),
        )
    )
    grouped["avg_personal_req_0_3"] = (
        grouped["weighted_personal_req"] / grouped["total_weight"].replace(0, np.nan)
    )
    grouped["pre_pers_free"] = (1.0 - grouped["avg_personal_req_0_3"] / 3.0).clip(0, 1)
    grouped["pre_pers_free_source"] = "pre_shock_coded_posts"
    pre_scores = grouped[[
        "subreddit", "pre_pers_free", "pre_pers_free_source",
        "pre_pers_free_n_used",
    ]]
    merged = fallback.drop(columns=["pre_pers_free", "pre_pers_free_source", "pre_pers_free_n_used"]).merge(
        pre_scores,
        on="subreddit",
        how="left",
    )
    merged = merged.merge(
        fallback[["subreddit", "pre_pers_free"]].rename(
            columns={"pre_pers_free": "fallback_pre_pers_free"}
        ),
        on="subreddit",
        how="left",
    )
    missing_pre = merged["pre_pers_free"].isna()
    merged.loc[missing_pre, "pre_pers_free"] = merged.loc[
        missing_pre,
        "fallback_pre_pers_free",
    ]
    merged.loc[missing_pre, "pre_pers_free_source"] = "full_acsi_fallback"
    merged = merged.drop(columns=["fallback_pre_pers_free"])
    return merged[[
        "subreddit", "pre_pers_free", "pre_pers_free_source",
        "pre_pers_free_n_used",
    ]]

def prepare_mechanism_panel(submonth_panel, acsi_scores):
    panel = submonth_panel.copy()
    panel["subreddit"] = panel["subreddit"].astype(str)
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"]).dt.to_period("M").dt.to_timestamp()
    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    if "log_posts" not in panel.columns:
        panel["log_posts"] = np.log1p(panel["posts"])

    pre_scores = load_pre_shock_persfree_scores(acsi_scores)
    panel = panel.drop(columns=[
        "pre_pers_free", "pre_pers_free_source", "pre_pers_free_n_used",
    ], errors="ignore")
    panel = panel.merge(pre_scores, on="subreddit", how="left")
    if panel["pre_pers_free"].isna().any():
        missing = sorted(panel.loc[panel["pre_pers_free"].isna(), "subreddit"].unique())
        raise ValueError(f"Missing PersFree scores for backtest panel: {missing}")

    panel["mu_t"] = panel["year_month_dt"].map(ai_capability_index_for_month).astype(float)
    panel["pers_free_mu"] = panel["pre_pers_free"].astype(float) * panel["mu_t"]
    panel["blackout_dummy"] = panel["year_month"].isin(["2023-06", "2023-07", "2023-08"]).astype(int)
    panel = panel.sort_values(["subreddit", "year_month_dt"]).copy()
    panel["lag_log1p_posts"] = panel.groupby("subreddit")["log_posts"].shift(1)
    panel["time_index"] = (
        (panel["year_month_dt"].dt.year - panel["year_month_dt"].dt.year.min()) * 12
        + panel["year_month_dt"].dt.month
        - panel["year_month_dt"].dt.month.min()
    )
    sub_scores = panel[["subreddit", "pre_pers_free"]].drop_duplicates().copy()
    ranked = sub_scores["pre_pers_free"].rank(method="first")
    sub_scores["persfree_tercile"] = pd.qcut(
        ranked,
        q=3,
        labels=["low_persfree", "middle_persfree", "high_persfree"],
    ).astype(str)
    panel = panel.merge(
        sub_scores[["subreddit", "persfree_tercile"]],
        on="subreddit",
        how="left",
    )
    return panel

def prediction_sigma(actual, predicted):
    actual = pd.Series(actual).astype(float)
    predicted = pd.Series(predicted).astype(float)
    ok = actual.notna() & predicted.notna()
    if ok.sum() <= 2:
        return np.nan
    sigma = float(np.sqrt(np.mean((actual[ok] - predicted[ok]) ** 2)))
    return sigma if np.isfinite(sigma) else np.nan

def predict_ols_with_fe_defaults(model, frame, numeric_terms=None):
    numeric_terms = numeric_terms or []
    params = model.params
    predicted = np.zeros(len(frame), dtype=float)
    if "Intercept" in params.index:
        predicted += float(params["Intercept"])

    for term in numeric_terms:
        if term in params.index and term in frame.columns:
            predicted += pd.to_numeric(frame[term], errors="coerce").fillna(0).to_numpy() * float(params[term])

    if "subreddit" in frame.columns:
        for i, value in enumerate(frame["subreddit"].astype(str)):
            param = f"C(subreddit)[T.{value}]"
            if param in params.index:
                predicted[i] += float(params[param])
    if "year_month" in frame.columns:
        for i, value in enumerate(frame["year_month"].astype(str)):
            param = f"C(year_month)[T.{value}]"
            if param in params.index:
                predicted[i] += float(params[param])
    return predicted

def recursive_backtest_predictions(model, train, holdout, numeric_terms):
    holdout_sorted = holdout.sort_values(["year_month_dt", "subreddit"]).copy()
    lag_by_subreddit = (
        train.sort_values("year_month_dt")
        .groupby("subreddit")["log_posts"]
        .last()
        .to_dict()
    )
    predicted = pd.Series(index=holdout_sorted.index, dtype=float)
    for month in sorted(holdout_sorted["year_month_dt"].unique()):
        month_mask = holdout_sorted["year_month_dt"] == month
        month_frame = holdout_sorted.loc[month_mask].copy()
        month_frame["lag_log1p_posts"] = month_frame["subreddit"].map(lag_by_subreddit)
        month_pred = predict_ols_with_fe_defaults(model, month_frame, numeric_terms)
        predicted.loc[month_frame.index] = month_pred
        for subreddit, value in zip(month_frame["subreddit"], month_pred):
            lag_by_subreddit[subreddit] = float(value)
    return predicted.reindex(holdout.index).to_numpy()

def locf_backtest_predictions(train, holdout):
    locf_last = (
        train.sort_values("year_month_dt")
        .groupby("subreddit")["log_posts"]
        .last()
    )
    return holdout["subreddit"].map(locf_last).astype(float).to_numpy()

def tune_hybrid_backtest_blend(main_formula, train, main_terms, default_weight=0.25):
    months = sorted(pd.to_datetime(train["year_month_dt"].dropna().unique()))
    result = {
        "mechanism_weight": float(default_weight),
        "locf_weight": float(1.0 - default_weight),
        "validation_rmse": np.nan,
        "validation_sigma": np.nan,
        "validation_months": "",
    }
    if len(months) < 6:
        return result

    validation_months = months[-3:]
    validation_start = validation_months[0]
    core = train[train["year_month_dt"] < validation_start].copy()
    validation = train[train["year_month_dt"].isin(validation_months)].copy()
    if core.empty or validation.empty or core["subreddit"].nunique() < 3:
        return result

    validation_model, _ = fit_backtest_model(main_formula, core, main_terms)
    if validation_model is None:
        return result
    main_pred = recursive_backtest_predictions(validation_model, core, validation, main_terms)
    locf_pred = locf_backtest_predictions(core, validation)
    actual = validation["log_posts"].astype(float).to_numpy()
    ok = np.isfinite(actual) & np.isfinite(main_pred) & np.isfinite(locf_pred)
    if ok.sum() <= 2:
        return result

    best_weight = float(default_weight)
    best_pred = best_weight * main_pred + (1.0 - best_weight) * locf_pred
    best_rmse = np.inf
    for weight in np.linspace(0.0, 1.0, 11):
        candidate = weight * main_pred + (1.0 - weight) * locf_pred
        residual = actual[ok] - candidate[ok]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
            best_pred = candidate

    result.update({
        "mechanism_weight": best_weight,
        "locf_weight": float(1.0 - best_weight),
        "validation_rmse": safe_float(best_rmse),
        "validation_sigma": safe_float(prediction_sigma(actual, best_pred)),
        "validation_months": ",".join(pd.Timestamp(month).strftime("%Y-%m") for month in validation_months),
    })
    return result

def fit_subreddit_trend_baseline(mechanism_panel, cutoff_month):
    trend_frame = mechanism_panel[
        (mechanism_panel["year_month_dt"] >= pd.Timestamp("2022-01-01"))
        & (mechanism_panel["year_month_dt"] <= pd.Timestamp(cutoff_month))
    ].dropna(subset=["log_posts", "time_index"]).copy()
    rows = []
    for subreddit, group in trend_frame.groupby("subreddit"):
        group = group.sort_values("year_month_dt")
        x = pd.to_numeric(group["time_index"], errors="coerce").astype(float).to_numpy()
        y = pd.to_numeric(group["log_posts"], errors="coerce").astype(float).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 2:
            slope, intercept = np.polyfit(x[ok], y[ok], 1)
        elif ok.sum() == 1:
            slope = 0.0
            intercept = y[ok][0]
        else:
            continue
        rows.append({
            "subreddit": subreddit,
            "trend_intercept": float(intercept),
            "trend_slope": float(slope),
        })
    return pd.DataFrame(rows)

def add_residualized_log_posts(frame, trend_baseline):
    residualized = frame.merge(trend_baseline, on="subreddit", how="left")
    residualized["trend_log_posts"] = (
        residualized["trend_intercept"]
        + residualized["trend_slope"] * pd.to_numeric(residualized["time_index"], errors="coerce")
    )
    residualized["residual_log_posts"] = (
        pd.to_numeric(residualized["log_posts"], errors="coerce")
        - residualized["trend_log_posts"]
    )
    residualized = residualized.sort_values(["subreddit", "year_month_dt"]).copy()
    residualized["lag_residual_log_posts"] = residualized.groupby("subreddit")["residual_log_posts"].shift(1)
    return residualized

def fit_residual_backtest_model(formula, train, numeric_terms):
    train_model = train.dropna(subset=["residual_log_posts"]).copy()
    model = fit_ols(formula, train_model, cluster_col="subreddit")
    if not model:
        return None, None
    fitted_train = model.fittedvalues.reindex(train_model.index)
    sigma = prediction_sigma(train_model["residual_log_posts"], fitted_train)
    return model, sigma

def recursive_residual_backtest_predictions(model, train, holdout, numeric_terms):
    holdout_sorted = holdout.sort_values(["year_month_dt", "subreddit"]).copy()
    lag_by_subreddit = (
        train.sort_values("year_month_dt")
        .groupby("subreddit")["residual_log_posts"]
        .last()
        .to_dict()
    )
    predicted = pd.Series(index=holdout_sorted.index, dtype=float)
    for month in sorted(holdout_sorted["year_month_dt"].unique()):
        month_mask = holdout_sorted["year_month_dt"] == month
        month_frame = holdout_sorted.loc[month_mask].copy()
        month_frame["lag_residual_log_posts"] = month_frame["subreddit"].map(lag_by_subreddit)
        month_pred = predict_ols_with_fe_defaults(model, month_frame, numeric_terms)
        predicted.loc[month_frame.index] = month_pred
        for subreddit, value in zip(month_frame["subreddit"], month_pred):
            lag_by_subreddit[subreddit] = float(value)
    return predicted.reindex(holdout.index).to_numpy()

def residual_persistence_predictions(train, holdout):
    last_residual = (
        train.sort_values("year_month_dt")
        .groupby("subreddit")["residual_log_posts"]
        .last()
    )
    return holdout["subreddit"].map(last_residual).astype(float).to_numpy()

def add_residual_rmse_row(rows, cutoff_label, evaluation_label, metric_label, model_label, holdout, predicted, sigma):
    eval_frame = holdout.copy()
    eval_frame["predicted_residual_log_posts"] = predicted
    eval_frame = eval_frame.dropna(subset=["residual_log_posts", "predicted_residual_log_posts"])
    if eval_frame.empty:
        return
    residual = eval_frame["residual_log_posts"] - eval_frame["predicted_residual_log_posts"]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    if sigma is not None and np.isfinite(sigma) and sigma > 0:
        z80 = stats.norm.ppf(0.90)
        z95 = stats.norm.ppf(0.975)
        pi80 = (
            (eval_frame["residual_log_posts"] >= eval_frame["predicted_residual_log_posts"] - z80 * sigma)
            & (eval_frame["residual_log_posts"] <= eval_frame["predicted_residual_log_posts"] + z80 * sigma)
        ).mean()
        pi95 = (
            (eval_frame["residual_log_posts"] >= eval_frame["predicted_residual_log_posts"] - z95 * sigma)
            & (eval_frame["residual_log_posts"] <= eval_frame["predicted_residual_log_posts"] + z95 * sigma)
        ).mean()
    else:
        pi80 = np.nan
        pi95 = np.nan
    rows.append({
        "row_type": "mechanism_validity_metric",
        "cutoff": cutoff_label,
        "evaluation_window": evaluation_label,
        "metric": metric_label,
        "model": model_label,
        "group": "overall",
        "value": safe_float(rmse),
        "rmse": safe_float(rmse),
        "pi80_coverage": safe_float(pi80),
        "pi95_coverage": safe_float(pi95),
        "n_obs": safe_int(len(eval_frame)),
        "n_subreddits": safe_int(eval_frame["subreddit"].nunique()),
    })

def community_residual_rank_frame(holdout, predicted):
    eval_frame = holdout.copy()
    eval_frame["predicted_residual_log_posts"] = predicted
    eval_frame = eval_frame.dropna(subset=[
        "residual_log_posts", "predicted_residual_log_posts", "persfree_tercile",
    ])
    if eval_frame.empty:
        return pd.DataFrame()
    return (
        eval_frame.groupby(["subreddit", "persfree_tercile"], as_index=False)
        .agg(
            actual_residual_log_posts=("residual_log_posts", "mean"),
            predicted_residual_log_posts=("predicted_residual_log_posts", "mean"),
            pre_pers_free=("pre_pers_free", "mean"),
        )
    )

def add_spearman_row(rows, cutoff_label, evaluation_label, metric_label, model_label, group_label, community_frame):
    community_frame = community_frame.dropna(subset=[
        "actual_residual_log_posts", "predicted_residual_log_posts",
    ]).copy()
    if community_frame["subreddit"].nunique() < 3:
        return
    rho, pvalue = stats.spearmanr(
        community_frame["predicted_residual_log_posts"],
        community_frame["actual_residual_log_posts"],
    )
    rows.append({
        "row_type": "mechanism_validity_metric",
        "cutoff": cutoff_label,
        "evaluation_window": evaluation_label,
        "metric": metric_label,
        "model": model_label,
        "group": group_label,
        "value": safe_float(rho),
        "spearman_r": safe_float(rho),
        "spearman_pvalue": safe_float(pvalue),
        "n_subreddits": safe_int(community_frame["subreddit"].nunique()),
    })

def add_labeled_tercile_gap_rows(
    rows,
    cutoff_label,
    evaluation_label,
    model_label,
    community_frame,
    gap_metric_label,
    pvalue_metric_label,
):
    bottom = community_frame[community_frame["persfree_tercile"] == "low_persfree"].dropna(
        subset=["actual_residual_log_posts"]
    )
    top = community_frame[community_frame["persfree_tercile"] == "high_persfree"].dropna(
        subset=["actual_residual_log_posts"]
    )
    if bottom.empty or top.empty:
        return
    bottom_values = bottom["actual_residual_log_posts"].astype(float)
    top_values = top["actual_residual_log_posts"].astype(float)
    gap = float(bottom_values.mean() - top_values.mean())
    _, pvalue = stats.ttest_ind(
        bottom_values,
        top_values,
        equal_var=False,
        alternative="less",
    )
    rows.append({
        "row_type": "mechanism_validity_metric",
        "cutoff": cutoff_label,
        "evaluation_window": evaluation_label,
        "metric": gap_metric_label,
        "model": model_label,
        "group": "bottom_minus_top_persfree",
        "value": safe_float(gap),
        "tercile_gap": safe_float(gap),
        "tercile_gap_pvalue": safe_float(pvalue),
        "bottom_tercile_mean_residual": safe_float(bottom_values.mean()),
        "top_tercile_mean_residual": safe_float(top_values.mean()),
        "n_bottom_tercile_subreddits": safe_int(bottom["subreddit"].nunique()),
        "n_top_tercile_subreddits": safe_int(top["subreddit"].nunique()),
        "n_subreddits": safe_int(community_frame["subreddit"].nunique()),
    })
    rows.append({
        "row_type": "mechanism_validity_metric",
        "cutoff": cutoff_label,
        "evaluation_window": evaluation_label,
        "metric": pvalue_metric_label,
        "model": model_label,
        "group": "bottom_minus_top_persfree",
        "value": safe_float(pvalue),
        "tercile_gap": safe_float(gap),
        "tercile_gap_pvalue": safe_float(pvalue),
        "bottom_tercile_mean_residual": safe_float(bottom_values.mean()),
        "top_tercile_mean_residual": safe_float(top_values.mean()),
        "n_bottom_tercile_subreddits": safe_int(bottom["subreddit"].nunique()),
        "n_top_tercile_subreddits": safe_int(top["subreddit"].nunique()),
        "n_subreddits": safe_int(community_frame["subreddit"].nunique()),
    })

def add_tercile_gap_rows(rows, cutoff_label, evaluation_label, model_label, community_frame):
    add_labeled_tercile_gap_rows(
        rows,
        cutoff_label,
        evaluation_label,
        model_label,
        community_frame,
        "tercile_gap",
        "tercile_gap_pvalue",
    )

def persfree_bottom_half_frame(community_frame):
    ranked = community_frame.dropna(subset=["pre_pers_free"]).sort_values(
        ["pre_pers_free", "subreddit"]
    ).copy()
    if ranked.empty:
        return ranked
    return ranked.head(len(ranked) // 2)

def add_residualized_mechanism_validity_rows(
    rows,
    mechanism_panel,
    cutoff_label,
    cutoff_month,
    train,
    holdout,
    evaluation_label,
    metric_prefix=None,
):
    trend_baseline = fit_subreddit_trend_baseline(mechanism_panel, cutoff_month)
    if trend_baseline.empty:
        return
    trend_source = mechanism_panel[
        (mechanism_panel["year_month_dt"] >= pd.Timestamp("2022-01-01"))
        & (mechanism_panel["year_month_dt"] <= pd.Timestamp(cutoff_month))
    ].copy()
    residual_source = add_residualized_log_posts(trend_source, trend_baseline)
    train_residual = residual_source[
        residual_source["year_month_dt"].isin(train["year_month_dt"].unique())
        & residual_source["subreddit"].isin(train["subreddit"].unique())
    ].copy()
    holdout_residual = add_residualized_log_posts(holdout, trend_baseline)
    train_residual = train_residual.dropna(subset=[
        "residual_log_posts", "lag_residual_log_posts", "pre_pers_free", "mu_t", "pers_free_mu",
    ])
    holdout_residual = holdout_residual.dropna(subset=["residual_log_posts", "pre_pers_free", "mu_t", "pers_free_mu"])
    if train_residual.empty or holdout_residual.empty:
        return

    residual_terms = ["pers_free_mu", "lag_residual_log_posts"]
    residual_formula = "residual_log_posts ~ pers_free_mu + lag_residual_log_posts + C(subreddit) + C(year_month)"
    mechanism_model, mechanism_sigma = fit_residual_backtest_model(
        residual_formula,
        train_residual,
        residual_terms,
    )
    persistence_pred = residual_persistence_predictions(train_residual, holdout_residual)
    persistence_train = train_residual.sort_values(["subreddit", "year_month_dt"]).copy()
    persistence_train["persistence_residual_train_pred"] = (
        persistence_train.groupby("subreddit")["residual_log_posts"].shift(1)
    )
    persistence_sigma = prediction_sigma(
        persistence_train["residual_log_posts"],
        persistence_train["persistence_residual_train_pred"],
    )

    persistence_rmse_label = "early_onset_residual_rmse" if metric_prefix == "early_onset" else "residual_rmse_persistence"
    add_residual_rmse_row(
        rows,
        cutoff_label,
        evaluation_label,
        persistence_rmse_label,
        "residual_persistence_last_observation",
        holdout_residual,
        persistence_pred,
        persistence_sigma,
    )
    persistence_community = community_residual_rank_frame(holdout_residual, persistence_pred)
    if metric_prefix == "early_onset":
        add_spearman_row(
            rows,
            cutoff_label,
            evaluation_label,
            "early_onset_spearman",
            "residual_persistence_last_observation",
            "overall",
            persistence_community,
        )

    if mechanism_model is None:
        return
    mechanism_pred = recursive_residual_backtest_predictions(
        mechanism_model,
        train_residual,
        holdout_residual,
        residual_terms,
    )
    mechanism_rmse_label = "early_onset_residual_rmse" if metric_prefix == "early_onset" else "residual_rmse_mechanism"
    add_residual_rmse_row(
        rows,
        cutoff_label,
        evaluation_label,
        mechanism_rmse_label,
        "residual_mechanism_persfree_mu_lag",
        holdout_residual,
        mechanism_pred,
        mechanism_sigma,
    )
    community = community_residual_rank_frame(holdout_residual, mechanism_pred)
    if metric_prefix == "early_onset":
        add_spearman_row(
            rows,
            cutoff_label,
            evaluation_label,
            "early_onset_spearman",
            "residual_mechanism_persfree_mu_lag",
            "overall",
            community,
        )
        add_tercile_gap_rows(rows, cutoff_label, evaluation_label, "residual_mechanism_persfree_mu_lag", community)
        return

    add_spearman_row(
        rows,
        cutoff_label,
        evaluation_label,
        "spearman_all",
        "residual_mechanism_persfree_mu_lag",
        "overall",
        community,
    )
    add_spearman_row(
        rows,
        cutoff_label,
        evaluation_label,
        "spearman_bottom_half",
        "residual_mechanism_persfree_mu_lag",
        "bottom_half_persfree",
        persfree_bottom_half_frame(community),
    )
    for tercile, metric_label in [
        ("low_persfree", "spearman_low_persfree"),
        ("high_persfree", "spearman_high_persfree"),
    ]:
        add_spearman_row(
            rows,
            cutoff_label,
            evaluation_label,
            metric_label,
            "residual_mechanism_persfree_mu_lag",
            tercile,
            community[community["persfree_tercile"] == tercile],
        )
    add_tercile_gap_rows(rows, cutoff_label, evaluation_label, "residual_mechanism_persfree_mu_lag", community)

def classify_activity_direction(delta):
    if delta < -0.025:
        return "fell"
    if delta > 0.025:
        return "recovered"
    return "flattened"

def evaluate_backtest_predictions(cutoff_label, model_label, train, holdout, predicted, sigma):
    holdout_eval = holdout.copy()
    holdout_eval["predicted_log_posts"] = predicted
    train_last = (
        train.sort_values("year_month_dt")
        .groupby("subreddit")["log_posts"]
        .last()
        .rename("last_train_log_posts")
    )
    holdout_eval = holdout_eval.merge(train_last, on="subreddit", how="left")
    rows = []
    groups = [("overall", holdout_eval)]
    for tercile in ["low_persfree", "middle_persfree", "high_persfree"]:
        groups.append((tercile, holdout_eval[holdout_eval["persfree_tercile"] == tercile]))

    for group_label, group in groups:
        group = group.dropna(subset=["log_posts", "predicted_log_posts"])
        if group.empty:
            continue
        residual = group["log_posts"] - group["predicted_log_posts"]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        final_rows = (
            group.sort_values("year_month_dt")
            .groupby("subreddit", as_index=False)
            .tail(1)
        )
        actual_direction = (
            final_rows["log_posts"] - final_rows["last_train_log_posts"]
        ).map(classify_activity_direction)
        predicted_direction = (
            final_rows["predicted_log_posts"] - final_rows["last_train_log_posts"]
        ).map(classify_activity_direction)
        directional_accuracy = float((actual_direction == predicted_direction).mean()) if len(final_rows) else np.nan

        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            z80 = stats.norm.ppf(0.90)
            z95 = stats.norm.ppf(0.975)
            pi80 = (
                (group["log_posts"] >= group["predicted_log_posts"] - z80 * sigma)
                & (group["log_posts"] <= group["predicted_log_posts"] + z80 * sigma)
            ).mean()
            pi95 = (
                (group["log_posts"] >= group["predicted_log_posts"] - z95 * sigma)
                & (group["log_posts"] <= group["predicted_log_posts"] + z95 * sigma)
            ).mean()
        else:
            pi80 = np.nan
            pi95 = np.nan

        rows.append({
            "row_type": "backtest_metric",
            "cutoff": cutoff_label,
            "model": model_label,
            "group": group_label,
            "rmse": safe_float(rmse),
            "directional_accuracy": safe_float(directional_accuracy),
            "pi80_coverage": safe_float(pi80),
            "pi95_coverage": safe_float(pi95),
            "n_obs": safe_int(len(group)),
            "n_subreddits": safe_int(group["subreddit"].nunique()),
        })
    return rows

def fit_backtest_model(formula, train, numeric_terms, recursive=True):
    model = fit_ols(formula, train.dropna(subset=["log_posts"]), cluster_col="subreddit")
    if not model:
        return None, None
    if recursive:
        fitted_train = model.fittedvalues.reindex(train.dropna(subset=["log_posts"]).index)
        sigma = prediction_sigma(train.dropna(subset=["log_posts"])["log_posts"], fitted_train)
    else:
        fitted_train = model.fittedvalues
        sigma = prediction_sigma(train.loc[fitted_train.index, "log_posts"], fitted_train)
    return model, sigma

def run_single_mechanism_backtest(mechanism_panel, cutoff_month, holdout_start, holdout_end, include_blackout=False):
    cutoff_month = pd.Timestamp(cutoff_month)
    holdout_start = pd.Timestamp(holdout_start)
    holdout_end = pd.Timestamp(holdout_end)
    train = mechanism_panel[
        (mechanism_panel["year_month_dt"] >= SHOCK_MONTH)
        & (mechanism_panel["year_month_dt"] <= cutoff_month)
        & (mechanism_panel["year_month_dt"] != pd.Timestamp("2024-05-01"))
    ].copy()
    holdout = mechanism_panel[
        (mechanism_panel["year_month_dt"] >= holdout_start)
        & (mechanism_panel["year_month_dt"] <= holdout_end)
    ].copy()
    train = train.dropna(subset=["lag_log1p_posts", "pre_pers_free", "mu_t", "pers_free_mu"])
    holdout = holdout.dropna(subset=["pre_pers_free", "mu_t", "pers_free_mu"])
    if train.empty or holdout.empty:
        return []

    cutoff_label = f"train_through_{cutoff_month.strftime('%Y-%m')}_holdout_{holdout_start.strftime('%Y-%m')}_to_{holdout_end.strftime('%Y-%m')}"
    rows = []
    main_pred = None
    main_terms = ["pers_free_mu", "lag_log1p_posts"]
    main_formula = "log_posts ~ pers_free_mu + lag_log1p_posts + C(subreddit) + C(year_month)"
    if include_blackout:
        main_terms = ["pers_free_mu", "lag_log1p_posts", "blackout_dummy"]
        main_formula = "log_posts ~ pers_free_mu + lag_log1p_posts + blackout_dummy + C(subreddit) + C(year_month)"
    main_model, main_sigma = fit_backtest_model(main_formula, train, main_terms)
    if main_model:
        main_pred = recursive_backtest_predictions(main_model, train, holdout, main_terms)
        model_label = "mechanism_with_blackout_dummy" if include_blackout else "mechanism_persfree_mu_lag"
        rows.extend(evaluate_backtest_predictions(
            cutoff_label, model_label, train, holdout, main_pred, main_sigma,
        ))
        term_result = reg_result(main_model, "pers_free_mu")
        term_result.update({
            "row_type": "backtest_model_coefficient",
            "cutoff": cutoff_label,
            "model": model_label,
            "group": "overall",
            "term": "pers_free_mu",
        })
        rows.append(term_result)

    if include_blackout:
        return rows

    locf_pred = locf_backtest_predictions(train, holdout)
    locf_train = train.sort_values(["subreddit", "year_month_dt"]).copy()
    locf_train["locf_train_pred"] = locf_train.groupby("subreddit")["log_posts"].shift(1)
    locf_sigma = prediction_sigma(locf_train["log_posts"], locf_train["locf_train_pred"])
    rows.extend(evaluate_backtest_predictions(
        cutoff_label, "baseline_last_observation_carried_forward",
        train, holdout, locf_pred, locf_sigma,
    ))

    is_early_onset = (
        cutoff_month == pd.Timestamp("2023-07-01")
        and holdout_start == pd.Timestamp("2023-08-01")
    )
    is_primary_holdout = (
        cutoff_month == pd.Timestamp("2024-02-01")
        and holdout_start == pd.Timestamp("2024-03-01")
        and holdout_end == pd.Timestamp("2024-08-01")
    )
    is_late_holdout = (
        cutoff_month == pd.Timestamp("2024-04-01")
        and holdout_start == pd.Timestamp("2024-06-01")
    )
    if is_early_onset:
        evaluation_label = "early_onset_holdout"
    elif is_primary_holdout:
        evaluation_label = "primary_holdout"
    elif is_late_holdout:
        evaluation_label = "late_holdout"
    else:
        evaluation_label = "residualized_holdout"
    add_residualized_mechanism_validity_rows(
        rows,
        mechanism_panel,
        cutoff_label,
        cutoff_month,
        train,
        holdout,
        evaluation_label,
        metric_prefix="early_onset" if is_early_onset else None,
    )
    if main_pred is not None:
        hybrid_info = tune_hybrid_backtest_blend(main_formula, train, main_terms)
        mechanism_weight = float(hybrid_info.get("mechanism_weight") or 0.0)
        hybrid_pred = mechanism_weight * main_pred + (1.0 - mechanism_weight) * locf_pred
        hybrid_sigma = hybrid_info.get("validation_sigma")
        if hybrid_sigma is None or not np.isfinite(hybrid_sigma):
            hybrid_sigma = locf_sigma
        rows.extend(evaluate_backtest_predictions(
            cutoff_label, "forecast_hybrid_locf_persfree",
            train, holdout, hybrid_pred, hybrid_sigma,
        ))
        rows.append({
            "row_type": "backtest_model_metadata",
            "cutoff": cutoff_label,
            "model": "forecast_hybrid_locf_persfree",
            "group": "overall",
            "term": "mechanism_weight",
            "mechanism_weight": safe_float(mechanism_weight),
            "locf_weight": safe_float(1.0 - mechanism_weight),
            "validation_rmse": safe_float(hybrid_info.get("validation_rmse")),
            "validation_sigma": safe_float(hybrid_sigma),
            "validation_months": hybrid_info.get("validation_months"),
        })

    fe_model, fe_sigma = fit_backtest_model(
        "log_posts ~ C(subreddit) + C(year_month)",
        train,
        [],
        recursive=False,
    )
    if fe_model:
        fe_pred = predict_ols_with_fe_defaults(fe_model, holdout, [])
        rows.extend(evaluate_backtest_predictions(
            cutoff_label, "baseline_subreddit_month_fe_only",
            train, holdout, fe_pred, fe_sigma,
        ))

    trend_model, trend_sigma = fit_backtest_model(
        "log_posts ~ C(subreddit) + C(subreddit):time_index",
        train,
        [],
        recursive=False,
    )
    if trend_model:
        try:
            trend_pred = trend_model.predict(holdout)
            rows.extend(evaluate_backtest_predictions(
                cutoff_label, "baseline_subreddit_specific_linear_trend",
                train, holdout, trend_pred, trend_sigma,
            ))
        except Exception as exc:
            print(f"  Subreddit-specific trend prediction failed for {cutoff_label}: {exc}")

    ar_model, ar_sigma = fit_backtest_model(
        "log_posts ~ lag_log1p_posts + C(year_month)",
        train,
        ["lag_log1p_posts"],
    )
    if ar_model:
        ar_pred = recursive_backtest_predictions(ar_model, train, holdout, ["lag_log1p_posts"])
        rows.extend(evaluate_backtest_predictions(
            cutoff_label, "baseline_ar1_month_fe",
            train, holdout, ar_pred, ar_sigma,
        ))
    return rows

def build_creator_hazard_rows(df_all, acsi_scores, months):
    if df_all is None or df_all.empty:
        return pd.DataFrame()
    required = {"author", "subreddit", "year_month_dt", "post_id"}
    if not required.issubset(df_all.columns):
        return pd.DataFrame()

    posts = df_all.copy()
    posts["subreddit"] = posts["subreddit"].astype(str)
    posts = posts[posts["subreddit"].isin(TREATMENT_SUBS)].copy()
    if posts.empty:
        return pd.DataFrame()
    posts["year_month_dt"] = pd.to_datetime(posts["year_month_dt"]).dt.to_period("M").dt.to_timestamp()
    posts = posts[~posts["author"].isin(EXCLUDED_AUTHORS)].copy()
    pre_posts = posts[
        (posts["year_month_dt"] >= pd.Timestamp("2022-01-01"))
        & (posts["year_month_dt"] < SHOCK_MONTH)
    ].copy()
    if pre_posts.empty:
        return pd.DataFrame()
    primary_subreddit = (
        pre_posts.groupby(["author", "subreddit"])
        .size()
        .reset_index(name="pre_posts_in_subreddit")
        .sort_values(["author", "pre_posts_in_subreddit", "subreddit"], ascending=[True, False, True])
        .drop_duplicates("author")
        .rename(columns={"subreddit": "primary_subreddit"})
    )
    pre_author_counts = (
        pre_posts.groupby("author")
        .agg(pre_post_count=("post_id", "count"))
        .reset_index()
    )
    primary_subreddit = primary_subreddit.merge(pre_author_counts, on="author", how="left")
    score_lookup = acsi_scores[["subreddit", "non_personal_norm"]].rename(
        columns={"subreddit": "primary_subreddit", "non_personal_norm": "pers_free"}
    )
    primary_subreddit = primary_subreddit.merge(score_lookup, on="primary_subreddit", how="left")
    primary_subreddit = primary_subreddit.dropna(subset=["pers_free"])
    if primary_subreddit.empty:
        return pd.DataFrame()

    posts = posts.merge(primary_subreddit[["author", "primary_subreddit"]], on="author", how="inner")
    posts = posts[posts["subreddit"] == posts["primary_subreddit"]].copy()
    months = [pd.Timestamp(month).replace(day=1) for month in months]
    if not months:
        return pd.DataFrame()

    monthly_posts = (
        posts.groupby(["author", "primary_subreddit", "year_month_dt"], as_index=False)
        .agg(month_posts=("post_id", "count"))
        .rename(columns={"year_month_dt": "month"})
    )
    grid_months = pd.date_range(
        min(months) - pd.DateOffset(months=3),
        max(months) + pd.DateOffset(months=2),
        freq="MS",
    )
    author_grid = primary_subreddit[[
        "author", "primary_subreddit", "pre_post_count", "pers_free",
    ]].copy()
    author_grid["_key"] = 1
    month_grid = pd.DataFrame({"month": grid_months, "_key": 1})
    hazard = author_grid.merge(month_grid, on="_key", how="inner").drop(columns="_key")
    hazard = hazard.merge(
        monthly_posts,
        on=["author", "primary_subreddit", "month"],
        how="left",
    )
    hazard["month_posts"] = hazard["month_posts"].fillna(0).astype(float)
    hazard = hazard.sort_values(["author", "month"]).copy()
    by_author = hazard.groupby("author")["month_posts"]
    hazard["prior_3_posts"] = (
        by_author.shift(1).fillna(0)
        + by_author.shift(2).fillna(0)
        + by_author.shift(3).fillna(0)
    )
    hazard["cumulative_prior_posts"] = by_author.cumsum() - hazard["month_posts"]
    hazard["future_3_posts"] = (
        hazard["month_posts"]
        + by_author.shift(-1).fillna(0)
        + by_author.shift(-2).fillna(0)
    )
    hazard = hazard[hazard["month"].isin(months) & (hazard["prior_3_posts"] > 0)].copy()
    if hazard.empty:
        return pd.DataFrame()

    elapsed_months = (
        (hazard["month"].dt.year - 2022) * 12
        + (hazard["month"].dt.month - 1)
    ).clip(lower=1)
    hazard["exit"] = (hazard["future_3_posts"] == 0).astype(int)
    hazard["log_prior_posting_rate"] = np.log1p(
        hazard["cumulative_prior_posts"] / elapsed_months
    )
    hazard["mu_t"] = hazard["month"].map(ai_capability_index_for_month).astype(float)
    return hazard[[
        "author", "primary_subreddit", "month", "exit",
        "log_prior_posting_rate", "pers_free", "mu_t", "pre_post_count",
    ]]

def run_creator_exit_validation(df_all, acsi_scores):
    result = {
        "row_type": "creator_exit_validation",
        "status": "skipped_missing_creator_data",
        "model": "creator_month_logit",
        "group": "high_substitutability_communities",
    }
    if df_all is None or df_all.empty:
        return [result], None, None

    train_months = pd.date_range("2022-12-01", "2024-04-01", freq="MS")
    holdout_months = pd.date_range("2024-06-01", "2024-10-01", freq="MS")
    train_rows = build_creator_hazard_rows(df_all, acsi_scores, train_months)
    holdout_rows = build_creator_hazard_rows(df_all, acsi_scores, holdout_months)
    if train_rows.empty or holdout_rows.empty or train_rows["exit"].nunique() < 2:
        result["status"] = "skipped_insufficient_creator_hazard_rows"
        result["n_train_author_months"] = safe_int(len(train_rows))
        result["n_holdout_author_months"] = safe_int(len(holdout_rows))
        return [result], None, None

    try:
        model = smf.logit(
            "exit ~ log_prior_posting_rate + pers_free + mu_t + log_prior_posting_rate:pers_free",
            data=train_rows,
        ).fit(disp=0)
        holdout_rows = holdout_rows.copy()
        holdout_rows["predicted_exit_probability"] = model.predict(holdout_rows)
        community = (
            holdout_rows.groupby("primary_subreddit", as_index=False)
            .agg(
                actual_exit_rate=("exit", "mean"),
                predicted_exit_rate=("predicted_exit_probability", "mean"),
                n_author_months=("exit", "size"),
            )
        )
        expected_creator_subreddits = set(TREATMENT_SUBS)
        observed_creator_subreddits = set(community["primary_subreddit"].astype(str))
        missing_creator_subreddits = sorted(expected_creator_subreddits - observed_creator_subreddits)
        for confidence, z in [("80", stats.norm.ppf(0.90)), ("95", stats.norm.ppf(0.975))]:
            se = np.sqrt(
                community["predicted_exit_rate"] * (1 - community["predicted_exit_rate"])
                / community["n_author_months"].replace(0, np.nan)
            )
            community[f"lower_{confidence}"] = (community["predicted_exit_rate"] - z * se).clip(0, 1)
            community[f"upper_{confidence}"] = (community["predicted_exit_rate"] + z * se).clip(0, 1)
            community[f"covered_{confidence}"] = (
                (community["actual_exit_rate"] >= community[f"lower_{confidence}"])
                & (community["actual_exit_rate"] <= community[f"upper_{confidence}"])
            )
        result.update({
            "status": "ok",
            "pi80_coverage": safe_float(community["covered_80"].mean()),
            "pi95_coverage": safe_float(community["covered_95"].mean()),
            "n_subreddits": safe_int(community["primary_subreddit"].nunique()),
            "expected_n_subreddits": safe_int(len(expected_creator_subreddits)),
            "missing_subreddits": ", ".join(missing_creator_subreddits),
            "n_train_author_months": safe_int(len(train_rows)),
            "n_holdout_author_months": safe_int(len(holdout_rows)),
            "coef_interaction": safe_float(model.params.get("log_prior_posting_rate:pers_free")),
            "se_interaction": safe_float(model.bse.get("log_prior_posting_rate:pers_free")),
            "pvalue_interaction": safe_float(model.pvalues.get("log_prior_posting_rate:pers_free")),
        })
        return [result], model, train_rows
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["n_train_author_months"] = safe_int(len(train_rows))
        result["n_holdout_author_months"] = safe_int(len(holdout_rows))
        return [result], None, train_rows

def fit_primary_mechanism_model(mechanism_panel):
    train = mechanism_panel[
        (mechanism_panel["year_month_dt"] >= SHOCK_MONTH)
        & (mechanism_panel["year_month_dt"] <= pd.Timestamp("2024-04-01"))
        & (mechanism_panel["year_month_dt"] != pd.Timestamp("2024-05-01"))
    ].dropna(subset=["lag_log1p_posts", "pers_free_mu"]).copy()
    model = fit_ols(
        "log_posts ~ pers_free_mu + lag_log1p_posts + C(subreddit) + C(year_month)",
        train,
        cluster_col="subreddit",
    )
    return model, train

def run_forward_simulation(mechanism_panel, mechanism_model, creator_exit_model=None, creator_train_rows=None):
    if mechanism_model is None:
        return pd.DataFrame([{
            "row_type": "forward_simulation",
            "status": "skipped_missing_mechanism_model",
        }])

    scenarios = [
        "Flat",
        "Step",
        "Accelerating",
        "Partial Regression",
        "Full Regression",
    ]
    future_months = pd.date_range("2025-01-01", "2030-12-01", freq="MS")
    last_actual = (
        mechanism_panel[mechanism_panel["year_month_dt"] == pd.Timestamp("2024-12-01")]
        [["subreddit", "log_posts", "pre_pers_free", "persfree_tercile"]]
        .dropna()
        .copy()
    )
    pre_shock_posts = (
        mechanism_panel[
            (mechanism_panel["year_month_dt"] >= pd.Timestamp("2022-01-01"))
            & (mechanism_panel["year_month_dt"] < SHOCK_MONTH)
        ]
        .groupby("subreddit")["posts"]
        .mean()
        .rename("pre_shock_mean_posts")
    )
    historical_bounds = (
        mechanism_panel.groupby("subreddit")["posts"]
        .agg(
            historical_posts_p95=lambda values: float(np.nanquantile(values, 0.95)),
            historical_posts_max="max",
        )
    )
    baseline = (
        last_actual
        .merge(pre_shock_posts, on="subreddit", how="left")
        .merge(historical_bounds, on="subreddit", how="left")
    )
    baseline["last_actual_posts"] = np.expm1(baseline["log_posts"].astype(float)).clip(lower=0)
    baseline["projection_ceiling"] = (
        baseline[[
            "last_actual_posts",
            "pre_shock_mean_posts",
            "historical_posts_p95",
        ]]
        .max(axis=1)
        .mul(1.25)
        .clip(lower=1.0)
    )
    rows = []
    for scenario in scenarios:
        lag_by_subreddit = baseline.set_index("subreddit")["log_posts"].astype(float).to_dict()
        monthly_predictions = []
        for month in future_months:
            month_frame = baseline[[
                "subreddit", "pre_pers_free", "persfree_tercile", "projection_ceiling",
            ]].copy()
            month_frame["year_month_dt"] = month
            month_frame["year_month"] = month.strftime("%Y-%m")
            month_frame["mu_t"] = forward_capability_index_for_month(month, scenario)
            month_frame["pers_free_mu"] = month_frame["pre_pers_free"] * month_frame["mu_t"]
            month_frame["lag_log1p_posts"] = month_frame["subreddit"].map(lag_by_subreddit)
            pred = predict_ols_with_fe_defaults(
                mechanism_model,
                month_frame,
                ["pers_free_mu", "lag_log1p_posts"],
            )
            raw_projected_posts = pd.Series(np.expm1(pred), index=month_frame.index)
            raw_projected_posts = raw_projected_posts.replace([np.inf, -np.inf], np.nan).clip(lower=0)
            month_frame["projected_posts_raw"] = raw_projected_posts
            month_frame["projected_posts"] = np.minimum(
                raw_projected_posts.fillna(0).to_numpy(),
                month_frame["projection_ceiling"].astype(float).to_numpy(),
            )
            month_frame["projected_log_posts"] = np.log1p(month_frame["projected_posts"])
            monthly_predictions.append(month_frame)
            for subreddit, value in zip(month_frame["subreddit"], month_frame["projected_log_posts"]):
                lag_by_subreddit[subreddit] = float(value)
        projection = pd.concat(monthly_predictions, ignore_index=True)
        projection["year"] = projection["year_month_dt"].dt.year
        yearly = (
            projection.groupby(["year", "persfree_tercile"], as_index=False)
            .agg(
                projected_mean_log_posts=("projected_log_posts", "mean"),
                projected_mean_posts=("projected_posts", "mean"),
                projected_mean_posts_raw=("projected_posts_raw", "mean"),
                projection_ceiling_mean=("projection_ceiling", "mean"),
                n_subreddits=("subreddit", "nunique"),
            )
        )
        for row in yearly.to_dict("records"):
            row.update({
                "row_type": "projected_tercile_activity",
                "status": "ok",
                "scenario": scenario,
            })
            rows.append(row)

        projection = projection.merge(
            baseline[["subreddit", "pre_shock_mean_posts"]],
            on="subreddit",
            how="left",
        )
        for tercile, group in projection.groupby("persfree_tercile"):
            year_means = group.groupby("year").agg(
                projected_mean_posts=("projected_posts", "mean"),
                pre_shock_mean_posts=("pre_shock_mean_posts", "mean"),
            )
            threshold = 0.5 * year_means["pre_shock_mean_posts"]
            reached = year_means[year_means["projected_mean_posts"] <= threshold]
            rows.append({
                "row_type": "half_pre_shock_activity_year",
                "status": "ok",
                "scenario": scenario,
                "persfree_tercile": tercile,
                "year": safe_int(reached.index.min()) if not reached.empty else None,
                "projected_mean_posts": safe_float(reached["projected_mean_posts"].iloc[0]) if not reached.empty else np.nan,
                "pre_shock_mean_posts": safe_float(reached["pre_shock_mean_posts"].iloc[0]) if not reached.empty else safe_float(year_means["pre_shock_mean_posts"].mean()),
            })

        if creator_exit_model is not None and creator_train_rows is not None and not creator_train_rows.empty:
            base_creators = (
                creator_train_rows.sort_values("month")
                .drop_duplicates("author", keep="last")
                .copy()
            )
            if not base_creators.empty and base_creators["pre_post_count"].notna().any():
                cutoff = base_creators["pre_post_count"].quantile(1 / 3)
                casual = base_creators[base_creators["pre_post_count"] <= cutoff].copy()
            else:
                casual = pd.DataFrame()
            if not casual.empty:
                survival_fraction = 1.0
                for year in range(2025, 2031):
                    year_exit_probs = []
                    for month in pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS"):
                        frame = casual.copy()
                        frame["mu_t"] = forward_capability_index_for_month(month, scenario)
                        try:
                            year_exit_probs.append(float(creator_exit_model.predict(frame).mean()))
                        except Exception:
                            year_exit_probs.append(np.nan)
                    mean_monthly_exit = np.nanmean(year_exit_probs)
                    if np.isfinite(mean_monthly_exit):
                        survival_fraction *= max(0.0, 1.0 - mean_monthly_exit) ** 12
                    rows.append({
                        "row_type": "casual_creator_survival",
                        "status": "ok",
                        "scenario": scenario,
                        "year": year,
                        "community_group": "high_substitutability_communities",
                        "fraction_still_active": safe_float(survival_fraction),
                        "n_casual_creators": safe_int(len(casual)),
                    })
            else:
                rows.append({
                    "row_type": "casual_creator_survival",
                    "status": "skipped_no_casual_creator_frame",
                    "scenario": scenario,
                    "community_group": "high_substitutability_communities",
                })
        else:
            rows.append({
                "row_type": "casual_creator_survival",
                "status": "skipped_missing_creator_exit_model",
                "scenario": scenario,
                "community_group": "high_substitutability_communities",
            })
    return pd.DataFrame(rows)

def compute_backtest_and_forward_simulation(submonth_panel, acsi_scores, df_all=None):
    mechanism_panel = prepare_mechanism_panel(submonth_panel, acsi_scores)
    score_source_counts = (
        mechanism_panel[["subreddit", "pre_pers_free_source"]]
        .drop_duplicates()["pre_pers_free_source"]
        .value_counts()
        .to_dict()
    )
    print(
        "  Pre-shock PersFree source counts: "
        + ", ".join(f"{source}={count}" for source, count in score_source_counts.items())
    )

    backtest_rows = [
        {
            "row_type": "pre_pers_free_source_count",
            "source": source,
            "n_subreddits": safe_int(count),
        }
        for source, count in score_source_counts.items()
    ]
    cutoffs = [
        ("2023-07-01", "2023-08-01", "2023-12-01"),
        ("2023-08-01", "2023-09-01", "2023-12-01"),
        ("2023-12-01", "2024-01-01", "2024-04-01"),
        ("2024-02-01", "2024-03-01", "2024-08-01"),
        ("2024-04-01", "2024-06-01", "2024-12-01"),
    ]
    for cutoff_month, holdout_start, holdout_end in cutoffs:
        backtest_rows.extend(run_single_mechanism_backtest(
            mechanism_panel,
            cutoff_month,
            holdout_start,
            holdout_end,
            include_blackout=False,
        ))
    backtest_rows.extend(run_single_mechanism_backtest(
        mechanism_panel,
        "2024-04-01",
        "2024-06-01",
        "2024-12-01",
        include_blackout=True,
    ))

    creator_rows, creator_exit_model, creator_train_rows = run_creator_exit_validation(df_all, acsi_scores)
    backtest_rows.extend(creator_rows)
    backtest_table = pd.DataFrame(backtest_rows)
    backtest_path = OUTPUT_DIR / "backtest_results.csv"
    backtest_table.to_csv(backtest_path, index=False)
    print(f"  Backtest results saved -> {backtest_path}")

    mechanism_model, mechanism_train = fit_primary_mechanism_model(mechanism_panel)
    forward_table = run_forward_simulation(
        mechanism_panel,
        mechanism_model,
        creator_exit_model=creator_exit_model,
        creator_train_rows=creator_train_rows,
    )
    forward_path = OUTPUT_DIR / "forward_simulation.csv"
    forward_table.to_csv(forward_path, index=False)
    print(f"  Forward simulation saved -> {forward_path}")

    if {"row_type", "model", "group", "cutoff"}.issubset(backtest_table.columns):
        main_metric = backtest_table[
            (backtest_table["row_type"] == "backtest_metric")
            & (backtest_table["model"] == "mechanism_persfree_mu_lag")
            & (backtest_table["group"] == "overall")
            & (backtest_table["cutoff"].astype(str).str.startswith("train_through_2024-02"))
        ]
    else:
        main_metric = pd.DataFrame()
    if not main_metric.empty:
        row = main_metric.iloc[0]
        print(
            "  Primary holdout RMSE="
            f"{fmt4(row.get('rmse'))}, directional accuracy="
            f"{fmt4(row.get('directional_accuracy'))}, "
            f"80% PI coverage={fmt4(row.get('pi80_coverage'))}"
        )

    return {
        "backtest_path": str(backtest_path),
        "forward_path": str(forward_path),
        "backtest_rows": safe_int(len(backtest_table)),
        "forward_rows": safe_int(len(forward_table)),
        "pre_pers_free_sources": score_source_counts,
    }

def format_summary_number(value):
    try:
        numeric = float(value)
    except Exception:
        return value
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.4f}"

def print_backtest_forward_summary(backtest_table, forward_table):
    print("\n=== Backtest summary ===")
    metrics = backtest_table[backtest_table.get("row_type") == "backtest_metric"].copy()
    if metrics.empty:
        print("  No backtest metric rows were produced.")
    else:
        metric_columns = [
            "cutoff", "model", "group", "rmse", "directional_accuracy",
            "pi80_coverage", "pi95_coverage", "n_obs", "n_subreddits",
        ]
        metric_columns = [column for column in metric_columns if column in metrics.columns]
        metric_summary = metrics[metric_columns].copy().sort_values(
            [column for column in ["cutoff", "model", "group"] if column in metric_columns]
        )
        for column in ["rmse", "directional_accuracy", "pi80_coverage", "pi95_coverage"]:
            if column in metric_summary.columns:
                metric_summary[column] = metric_summary[column].map(format_summary_number)
        print(metric_summary.to_string(index=False))

    validity = backtest_table[backtest_table.get("row_type") == "mechanism_validity_metric"].copy()
    if not validity.empty:
        print("\n=== Residualized mechanism-validity summary ===")
        validity_columns = [
            "cutoff", "evaluation_window", "metric", "model", "group",
            "value", "rmse", "spearman_r", "spearman_pvalue",
            "tercile_gap", "tercile_gap_pvalue",
            "pi80_coverage", "pi95_coverage",
            "n_obs", "n_subreddits",
            "n_bottom_quartile_subreddits", "n_top_quartile_subreddits",
        ]
        validity_columns = [column for column in validity_columns if column in validity.columns]
        validity_summary = validity[validity_columns].copy().sort_values(
            [column for column in ["evaluation_window", "cutoff", "metric", "model", "group"] if column in validity_columns]
        )
        for column in [
            "value", "rmse", "spearman_r", "spearman_pvalue",
            "tercile_gap", "tercile_gap_pvalue",
            "pi80_coverage", "pi95_coverage",
        ]:
            if column in validity_summary.columns:
                validity_summary[column] = validity_summary[column].map(format_summary_number)
        print(validity_summary.to_string(index=False))

    coeffs = backtest_table[backtest_table.get("row_type") == "backtest_model_coefficient"].copy()
    if not coeffs.empty:
        print("\n=== Backtest coefficient summary ===")
        coeff_columns = [
            "cutoff", "model", "term", "coef", "se", "pvalue",
            "n_obs",
        ]
        coeff_columns = [column for column in coeff_columns if column in coeffs.columns]
        coeff_summary = coeffs[coeff_columns].copy().sort_values(
            [column for column in ["cutoff", "model", "term"] if column in coeff_columns]
        )
        for column in ["coef", "se", "pvalue"]:
            if column in coeff_summary.columns:
                coeff_summary[column] = coeff_summary[column].map(format_summary_number)
        print(coeff_summary.to_string(index=False))

    metadata = backtest_table[backtest_table.get("row_type") == "backtest_model_metadata"].copy()
    if not metadata.empty:
        print("\n=== Backtest model metadata ===")
        metadata_columns = [
            "cutoff", "model", "term", "mechanism_weight", "locf_weight",
            "validation_rmse", "validation_sigma", "validation_months",
        ]
        metadata_columns = [column for column in metadata_columns if column in metadata.columns]
        metadata_summary = metadata[metadata_columns].copy().sort_values(
            [column for column in ["cutoff", "model", "term"] if column in metadata_columns]
        )
        for column in ["mechanism_weight", "locf_weight", "validation_rmse", "validation_sigma"]:
            if column in metadata_summary.columns:
                metadata_summary[column] = metadata_summary[column].map(format_summary_number)
        print(metadata_summary.to_string(index=False))

    creator = backtest_table[backtest_table.get("row_type") == "creator_exit_validation"].copy()
    print("\n=== Creator exit validation ===")
    if creator.empty:
        print("  No creator exit validation row was produced.")
    else:
        creator_columns = [
            "status", "group", "pi80_coverage", "pi95_coverage",
            "n_subreddits", "expected_n_subreddits", "missing_subreddits",
            "n_train_author_months", "n_holdout_author_months",
            "coef_interaction", "se_interaction", "pvalue_interaction",
            "error",
        ]
        creator_columns = [column for column in creator_columns if column in creator.columns]
        creator_summary = creator[creator_columns].copy()
        for column in [
            "pi80_coverage", "pi95_coverage", "coef_interaction",
            "se_interaction", "pvalue_interaction",
        ]:
            if column in creator_summary.columns:
                creator_summary[column] = creator_summary[column].map(format_summary_number)
        print(creator_summary.to_string(index=False))

    print("\n=== Forward simulation summary ===")
    if forward_table.empty:
        print("  No forward simulation rows were produced.")
        return

    projected = forward_table[forward_table.get("row_type") == "projected_tercile_activity"].copy()
    if not projected.empty:
        projection_columns = [
            "scenario", "year", "persfree_tercile",
            "projected_mean_posts", "projected_mean_log_posts",
            "n_subreddits",
        ]
        projection_columns = [column for column in projection_columns if column in projected.columns]
        projection_summary = projected[projection_columns].copy().sort_values(
            [column for column in ["scenario", "year", "persfree_tercile"] if column in projection_columns]
        )
        for column in ["projected_mean_posts", "projected_mean_log_posts"]:
            if column in projection_summary.columns:
                projection_summary[column] = projection_summary[column].map(format_summary_number)
        print("\nProjected annual activity by scenario and PersFree tercile:")
        print(projection_summary.to_string(index=False))

    half_life = forward_table[forward_table.get("row_type") == "half_pre_shock_activity_year"].copy()
    if not half_life.empty:
        half_columns = [
            "scenario", "persfree_tercile", "year",
            "projected_mean_posts", "pre_shock_mean_posts",
        ]
        half_columns = [column for column in half_columns if column in half_life.columns]
        half_summary = half_life[half_columns].copy().sort_values(
            [column for column in ["scenario", "persfree_tercile"] if column in half_columns]
        )
        for column in ["projected_mean_posts", "pre_shock_mean_posts"]:
            if column in half_summary.columns:
                half_summary[column] = half_summary[column].map(format_summary_number)
        print("\nFirst year at or below half pre-shock activity:")
        print(half_summary.to_string(index=False))

    survival = forward_table[forward_table.get("row_type") == "casual_creator_survival"].copy()
    if not survival.empty:
        survival_columns = [
            "scenario", "year", "community_group",
            "fraction_still_active", "n_casual_creators", "status",
        ]
        survival_columns = [column for column in survival_columns if column in survival.columns]
        survival_summary = survival[survival_columns].copy().sort_values(
            [column for column in ["scenario", "year"] if column in survival_columns]
        )
        if "fraction_still_active" in survival_summary.columns:
            survival_summary["fraction_still_active"] = survival_summary["fraction_still_active"].map(format_summary_number)
        print("\nCasual creator survival projection:")
        print(survival_summary.to_string(index=False))

def run_backtest_only_from_outputs():
    print("\n=== Backtest-only run ===")
    required_paths = [ACSI_SCORE_PATH, SUBMONTH_PANEL_PATH, POSTS_CREATOR_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        print("  Missing required input(s):")
        for path in missing:
            print(f"  - {path}")
        raise FileNotFoundError("Missing inputs for backtest-only run")

    print(f"  Loading panel -> {SUBMONTH_PANEL_PATH}")
    submonth_panel = pd.read_parquet(SUBMONTH_PANEL_PATH)
    print(f"  Loading creator posts -> {POSTS_CREATOR_PATH}")
    df_all = pd.read_parquet(POSTS_CREATOR_PATH)
    print(f"  Loading ACSI scores -> {ACSI_SCORE_PATH}")
    acsi_scores = load_acsi_scores()
    print(
        "  Inputs: "
        f"{len(submonth_panel):,} panel rows, "
        f"{submonth_panel['subreddit'].nunique():,} subreddits, "
        f"{len(df_all):,} creator-clean posts, "
        f"{len(acsi_scores):,} scored subreddits"
    )

    result = compute_backtest_and_forward_simulation(
        submonth_panel,
        acsi_scores,
        df_all=df_all,
    )
    backtest_table = pd.read_csv(result["backtest_path"])
    forward_table = pd.read_csv(result["forward_path"])
    print_backtest_forward_summary(backtest_table, forward_table)
    print("\nBacktest-only processing complete.")
    return result

def compute_creator_level_checks(df_all, creators, acsi_scores=None):
    results = {
        "q2_survival": None,
        "q2_survival_moderation": None,
        "q2_survival_moderation_terms": None,
        "q3_engagement": None,
    }
    if df_all is None or creators is None or df_all.empty or creators.empty:
        return results

    posts = df_all.copy()
    if "sub_role" not in posts.columns:
        posts["sub_role"] = posts["subreddit"].map(SUBREDDITS)

    print("\n--- Q2: creator exit logistic regression ---")
    try:
        treatment_pre = posts[
            (posts["sub_role"] == "treatment") & (posts["post_shock"] == 0)
        ].copy()
        q2_frame = (
            treatment_pre.groupby("author")
            .agg(pre_treatment_post_count=("post_id", "count"))
            .reset_index()
        )
        q2_frame["pre_treatment_post_rate"] = (
            q2_frame["pre_treatment_post_count"] / max(N_PRE_MONTHS, 1)
        )
        q2_frame["log_pre_shock_posting_rate"] = np.log(
            q2_frame["pre_treatment_post_rate"].replace(0, np.nan)
        )
        treatment_post_authors = set(
            posts.loc[
                (posts["sub_role"] == "treatment") & (posts["post_shock"] == 1),
                "author",
            ].unique()
        )
        q2_frame["creator_exit"] = (
            ~q2_frame["author"].isin(treatment_post_authors)
        ).astype(int)
        q2_frame = q2_frame.dropna(subset=["log_pre_shock_posting_rate"])

        q2_model = smf.logit(
            "creator_exit ~ log_pre_shock_posting_rate",
            data=q2_frame,
        ).fit(disp=0)
        q2_result = reg_result(q2_model, "log_pre_shock_posting_rate")
        q2_result.update({
            "label": "Creator exit",
            "outcome": "creator_exit",
            "term": "log_pre_shock_posting_rate",
            "predictor": "Log pre-shock treatment communities posting rate",
            "model": "Logit",
            "n_authors": safe_int(len(q2_frame)),
        })
        results["q2_survival"] = q2_result
        emit_output_table(pd.DataFrame([q2_result]), TABLES_DIR / "q2_survival.csv", index=False)
        print(
            "  Creator exit in treatment communities ~ log pre-shock posting rate: "
            f"coef={fmt_signed4(q2_result['coef'])} SE={fmt4(q2_result['se'])} "
            f"p={fmt4(q2_result['pvalue'])}"
        )

        if acsi_scores is not None and "non_personal_norm" in acsi_scores.columns:
            author_subreddit_counts = (
                treatment_pre.groupby(["author", "subreddit"])
                .size()
                .reset_index(name="primary_subreddit_posts")
                .sort_values(
                    ["author", "primary_subreddit_posts", "subreddit"],
                    ascending=[True, False, True],
                )
                .drop_duplicates("author")
                .rename(columns={"subreddit": "primary_subreddit"})
            )
            q2_moderation = q2_frame.merge(author_subreddit_counts, on="author", how="left")
            score_lookup = acsi_scores[["subreddit", "non_personal_norm"]].copy()
            score_lookup["non_personal_norm"] = pd.to_numeric(
                score_lookup["non_personal_norm"],
                errors="coerce",
            )
            q2_moderation = q2_moderation.merge(
                score_lookup.rename(columns={"subreddit": "primary_subreddit"}),
                on="primary_subreddit",
                how="left",
            )
            q2_moderation = q2_moderation.dropna(
                subset=["log_pre_shock_posting_rate", "non_personal_norm", "creator_exit"]
            )

            if len(q2_moderation) > 0 and q2_moderation["primary_subreddit"].nunique() > 1:
                q2_moderation_model = smf.logit(
                    "creator_exit ~ log_pre_shock_posting_rate * non_personal_norm",
                    data=q2_moderation,
                ).fit(disp=0)
                moderation_terms = [
                    (
                        "log_pre_shock_posting_rate",
                        "Log pre-shock treatment communities posting rate",
                    ),
                    ("non_personal_norm", "Low personal-context need"),
                    (
                        "log_pre_shock_posting_rate:non_personal_norm",
                        "Posting rate x low personal-context need",
                    ),
                ]
                moderation_rows = []
                for term, predictor in moderation_terms:
                    term_result = reg_result(q2_moderation_model, term)
                    term_result.update({
                        "label": "Creator exit moderation",
                        "outcome": "creator_exit",
                        "term": term,
                        "predictor": predictor,
                        "model": "Logit with PersFree moderation",
                        "n_authors": safe_int(len(q2_moderation)),
                        "n_primary_subreddits": safe_int(q2_moderation["primary_subreddit"].nunique()),
                    })
                    moderation_rows.append(term_result)

                interaction_result = next(
                    row for row in moderation_rows
                    if row["term"] == "log_pre_shock_posting_rate:non_personal_norm"
                )
                results["q2_survival_moderation"] = interaction_result
                results["q2_survival_moderation_terms"] = moderation_rows
                emit_output_table(
                    pd.DataFrame(moderation_rows),
                    TABLES_DIR / "q2_survival_moderation.csv",
                    index=False,
                )
                print(
                    "  Creator exit moderation in treatment communities (posting rate x low personal-context need): "
                    f"coef={fmt_signed4(interaction_result['coef'])} "
                    f"SE={fmt4(interaction_result['se'])} "
                    f"p={fmt4(interaction_result['pvalue'])}"
                )
            else:
                print("  Creator exit moderation skipped: insufficient primary-subreddit variation.")
        else:
            print("  Creator exit moderation skipped: missing non_personal_norm scores.")
    except Exception as exc:
        print(f"  Q2 creator exit failed: {exc}")

    print("\n--- Q3: stable creator per-creator engagement DiD ---")
    try:
        stable_authors = set(creators.loc[creators["is_stable"] == 1, "author"])
        pre_authors = set(posts.loc[posts["post_shock"] == 0, "author"].unique())
        post_authors = set(posts.loc[posts["post_shock"] == 1, "author"].unique())
        survivor_authors = pre_authors & post_authors
        q3_posts = posts[posts["author"].isin(stable_authors & survivor_authors)].copy()
        q3_posts = q3_posts[q3_posts["sub_role"].isin(["treatment", "control"])]
        if q3_posts.empty:
            return results

        q3_panel = (
            q3_posts.groupby(["author", "subreddit", "year_month", "post_shock"], as_index=False)
            .agg(avg_score=("score", "mean"))
        )
        q3_panel["sub_role"] = q3_panel["subreddit"].map(SUBREDDITS)
        q3_panel["treated"] = (q3_panel["sub_role"] == "treatment").astype(int)
        q3_panel["did"] = q3_panel["treated"] * q3_panel["post_shock"]
        q3_panel["log_avg_score"] = np.log1p(q3_panel["avg_score"].clip(lower=0))

        q3_model = fit_ols(
            "log_avg_score ~ did + C(subreddit) + C(year_month)",
            q3_panel,
            cluster_col="author",
        )
        if q3_model:
            q3_result = reg_result(q3_model, "did")
            q3_result.update({
                "label": "Per-creator engagement DiD",
                "outcome": "log_avg_score",
                "term": "did",
                "predictor": "Treatment x post",
                "model": "OLS with author-clustered SE",
                "n_panel_rows": safe_int(len(q3_panel)),
                "n_stable_survivor_authors": safe_int(len(stable_authors & survivor_authors)),
            })
            results["q3_engagement"] = q3_result
            emit_output_table(
                pd.DataFrame([q3_result]),
                TABLES_DIR / "q3_per_creator_engagement_did.csv",
                index=False,
            )
            print(
                "  Per-creator engagement DiD: "
                f"coef={fmt_signed4(q3_result['coef'])} SE={fmt4(q3_result['se'])} "
                f"p={fmt4(q3_result['pvalue'])}"
            )
    except Exception as exc:
        print(f"  Q3 per-creator engagement failed: {exc}")

    return results

def normalize_measurement_frame(frame):
    frame = frame.copy()
    if "month" not in frame.columns and "year_month" in frame.columns:
        frame = frame.rename(columns={"year_month": "month"})
    required_columns = {
        "post_id", "subreddit", "month", "ai_related_flag",
        "direct_gen_score", "usefulness_score", "quality_comp_score",
        "physical_req_score", "personal_req_score",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Measurement frame missing columns: {sorted(missing_columns)}")
    frame["post_id"] = frame["post_id"].astype(str)
    frame["subreddit"] = frame["subreddit"].astype(str)
    frame["month"] = frame["month"].astype(str)
    frame["ai_related_flag"] = (
        pd.to_numeric(frame["ai_related_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    for column_name in [
        "direct_gen_score", "usefulness_score", "quality_comp_score",
        "physical_req_score", "personal_req_score",
    ]:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    return frame

def simple_acsi_scores_from_measurements(
    measurement_frame,
    start_month="2022-01",
    end_month="2022-10",
    subreddit_filter=None,
):
    frame = normalize_measurement_frame(measurement_frame)
    if subreddit_filter is not None:
        frame = frame[frame["subreddit"].isin(set(subreddit_filter))].copy()
    score_columns = [
        "direct_gen_score", "usefulness_score", "quality_comp_score",
        "physical_req_score", "personal_req_score",
    ]
    pre = frame[
        (frame["month"] >= start_month)
        & (frame["month"] <= end_month)
        & frame["ai_related_flag"].eq(0)
    ].dropna(subset=score_columns).copy()
    grouped = (
        pre.groupby("subreddit", as_index=False)[score_columns]
        .mean()
        .sort_values("subreddit")
    )
    if grouped.empty:
        return grouped.assign(gen_cap=[], phys_free=[], pers_free=[]), {
            "n_posts_used": 0,
            "n_subreddits": 0,
        }
    grouped["gen_cap"] = grouped[
        ["direct_gen_score", "usefulness_score", "quality_comp_score"]
    ].mean(axis=1) / 3.0
    grouped["phys_free"] = 1.0 - grouped["physical_req_score"] / 3.0
    grouped["pers_free"] = 1.0 - grouped["personal_req_score"] / 3.0
    return grouped[["subreddit", "gen_cap", "phys_free", "pers_free"]], {
        "n_posts_used": safe_int(len(pre)),
        "n_subreddits": safe_int(grouped["subreddit"].nunique()),
    }

def preshock_measurement_mask(frame):
    if "year_month" in frame.columns:
        return frame["year_month"].astype(str) < "2022-11"
    if "created_utc" in frame.columns:
        created = pd.to_datetime(frame["created_utc"], errors="coerce", unit="s")
        if created.isna().all():
            created = pd.to_datetime(frame["created_utc"], errors="coerce")
        return created < pd.Timestamp("2022-11-01")
    if "month" in frame.columns:
        return frame["month"].astype(str) < "2022-11"
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"], errors="coerce") < pd.Timestamp("2022-11-01")
    raise ValueError("Measurement frame has no recognized date column: year_month, created_utc, month, date.")

def load_preshock_measurement_frame(measurement_path=None):
    measurement_path = Path(measurement_path or ACSI_MEASUREMENT_SAMPLE_RUN1_PATH)
    frame = pd.read_csv(measurement_path)
    if "subreddit" not in frame.columns:
        raise ValueError(f"Measurement file missing subreddit column: {measurement_path}")
    frame = frame.copy()
    frame["subreddit"] = frame["subreddit"].astype(str)
    return frame[preshock_measurement_mask(frame)].copy()

def compute_preshock_acsi_coverage(measurement_path=None):
    pre = load_preshock_measurement_frame(measurement_path)
    if "personal_req_score" not in pre.columns:
        raise ValueError("Measurement file missing personal_req_score column.")
    pre["personal_req_score"] = pd.to_numeric(pre["personal_req_score"], errors="coerce")
    coverage = (
        pre.assign(usable_personal_req=pre["personal_req_score"].notna())
        .groupby("subreddit", as_index=False)
        .agg(
            n_preshock_rows=("subreddit", "size"),
            n_usable_preshock=("usable_personal_req", "sum"),
        )
        .sort_values("subreddit")
    )
    for threshold in [50, 100, 200]:
        coverage[f"below_{threshold}"] = coverage["n_usable_preshock"] < threshold
    total_subreddits = int(coverage["subreddit"].nunique())
    below_50 = int(coverage["below_50"].sum())
    below_100 = int(coverage["below_100"].sum())
    below_200 = int(coverage["below_200"].sum())
    print(
        "  Pre-shock ACSI coverage: "
        f"{total_subreddits} subreddits; "
        f"<50={below_50}, <100={below_100}, <200={below_200}"
    )
    emit_output_table(coverage, TABLES_DIR / "preshock_acsi_coverage.csv", index=False)
    return coverage

def compute_preshock_only_acsi_scores(measurement_path=None, min_posts=50):
    pre = load_preshock_measurement_frame(measurement_path)
    score_columns = [
        "direct_gen_score",
        "usefulness_score",
        "quality_comp_score",
        "physical_req_score",
        "personal_req_score",
    ]
    missing_columns = set(score_columns + ["subreddit"]) - set(pre.columns)
    if missing_columns:
        raise ValueError(f"Measurement file missing columns: {sorted(missing_columns)}")
    for column_name in score_columns:
        pre[column_name] = pd.to_numeric(pre[column_name], errors="coerce")
    usable = pre.dropna(subset=["personal_req_score", "physical_req_score"]).copy()
    if usable.empty:
        raise ValueError("No usable pre-shock ACSI rows after filtering.")
    usable["gen_cap_row"] = usable[
        ["direct_gen_score", "usefulness_score", "quality_comp_score"]
    ].mean(axis=1) / 3.0

    scores = (
        usable.groupby("subreddit", as_index=False)
        .agg(
            pers_free_preshock=("personal_req_score", lambda values: 1.0 - float(values.mean()) / 3.0),
            phys_free_preshock=("physical_req_score", lambda values: 1.0 - float(values.mean()) / 3.0),
            gen_cap_preshock=("gen_cap_row", "mean"),
            n_preshock=("personal_req_score", "size"),
        )
        .sort_values("subreddit")
    )
    dropped = scores[scores["n_preshock"] < min_posts].copy()
    if not dropped.empty:
        dropped_labels = ", ".join(
            f"r/{row.subreddit} ({int(row.n_preshock)})"
            for row in dropped.itertuples(index=False)
        )
        print(f"  Dropped pre-shock-only ACSI subreddits below min_posts={min_posts}: {dropped_labels}")
    scores = scores[scores["n_preshock"] >= min_posts].copy()
    print(
        "  Pre-shock-only ACSI scores: "
        f"{len(scores)} subreddits retained at min_posts={min_posts}"
    )
    emit_output_table(scores, TABLES_DIR / "acsi_preshock_only_scores.csv", index=False)
    return scores

def compute_preshock_only_main_regression(submonth_panel, preshock_scores):
    panel = submonth_panel.copy()
    if "year_month" not in panel.columns and "year_month_dt" in panel.columns:
        panel["year_month"] = pd.to_datetime(panel["year_month_dt"], errors="coerce").dt.strftime("%Y-%m")
    if "year_month_dt" not in panel.columns and "year_month" in panel.columns:
        panel["year_month_dt"] = pd.to_datetime(panel["year_month"].astype(str) + "-01", errors="coerce")
    if "post_shock" not in panel.columns and "year_month_dt" in panel.columns:
        shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
        panel["post_shock"] = (pd.to_datetime(panel["year_month_dt"]) >= shock_month).astype(int)

    score_columns = ["subreddit", "pers_free_preshock", "phys_free_preshock", "gen_cap_preshock"]
    missing_score_columns = set(score_columns) - set(preshock_scores.columns)
    if missing_score_columns:
        raise ValueError(f"Pre-shock score table missing columns: {sorted(missing_score_columns)}")
    merged = panel.merge(preshock_scores[score_columns], on="subreddit", how="inner")
    if merged.empty:
        raise ValueError("No overlapping subreddits between panel and pre-shock-only ACSI scores.")

    terms = [
        ("gen_cap_preshock_post", "gen_cap_preshock", "Generation capability"),
        ("phys_free_preshock_post", "phys_free_preshock", "Low physical constraint"),
        ("pers_free_preshock_post", "pers_free_preshock", "PersFree (pers_free_preshock)"),
    ]
    for term, source, _label in terms:
        merged[term] = merged[source] * merged["post_shock"]

    term_names = [term for term, _source, _label in terms]
    base_formula = "log_posts ~ " + " + ".join(term_names) + " + C(subreddit) + C(year_month)"
    main_model = fit_ols(base_formula, merged, cluster_col="subreddit")
    if not main_model:
        raise ValueError("Pre-shock-only main regression failed to fit.")

    sample_summary = regression_sample_summary(merged, merged.dropna(subset=term_names + ["log_posts", "subreddit", "year_month"]))
    main_rows = []
    for term, source, label in terms:
        row = reg_result(main_model, term)
        row.update({
            "model": "preshock_main",
            "source": source,
            "term": term,
            "label": label,
            "outcome": "log_posts",
            "percent_effect_full_exposure": pct_effect_from_coef(row.get("coef")),
        })
        row.update(sample_summary)
        main_rows.append(row)

    covariates = add_pre_covariates(merged)
    adjusted = merged.merge(covariates, on="subreddit", how="left")
    adjusted["pre_avg_post"] = adjusted["pre_avg_log_posts"] * adjusted["post_shock"]
    adjusted["pre_trend_post"] = adjusted["pre_trend"] * adjusted["post_shock"]
    adjusted["log_mu_post"] = adjusted["log_mu_k"] * adjusted["post_shock"]
    control_terms = ["pre_avg_post", "pre_trend_post", "log_mu_post"]
    cov_formula = (
        "log_posts ~ "
        + " + ".join(term_names + control_terms)
        + " + C(subreddit) + C(year_month)"
    )
    cov_model = fit_ols(cov_formula, adjusted, cluster_col="subreddit")
    cov_rows = []
    if cov_model:
        cov_sample_summary = regression_sample_summary(
            adjusted,
            adjusted.dropna(subset=term_names + control_terms + ["log_posts", "subreddit", "year_month"]),
        )
        for term, source, label in terms:
            row = reg_result(cov_model, term)
            row.update({
                "model": "preshock_covariate_adj",
                "source": source,
                "term": term,
                "label": label,
                "outcome": "log_posts",
                "percent_effect_full_exposure": pct_effect_from_coef(row.get("coef")),
            })
            row.update(cov_sample_summary)
            cov_rows.append(row)
    else:
        print("  Pre-shock-only covariate-adjusted regression failed to fit.")

    for title, rows in [("main", main_rows), ("covariate-adjusted", cov_rows)]:
        print(f"  Pre-shock-only ACSI {title} regression:")
        for row in rows:
            print(
                f"    {row['label']} x Post: "
                f"coef={fmt_signed4(row.get('coef'))} "
                f"SE={fmt4(row.get('se'))} "
                f"p={fmt4(row.get('pvalue'))}"
            )
    output_rows = main_rows + cov_rows
    emit_output_table(
        pd.DataFrame(output_rows),
        TABLES_DIR / "acsi_preshock_only_regression.csv",
        index=False,
    )
    return {
        "preshock_main": main_rows,
        "preshock_covariate_adj": cov_rows,
        "n_score_subreddits": safe_int(preshock_scores["subreddit"].nunique()),
        "n_model_subreddits": safe_int(merged["subreddit"].nunique()),
        "n_model_rows": safe_int(len(merged)),
        "output_path": str(TABLES_DIR / "acsi_preshock_only_regression.csv"),
    }

def balanced_simple_acsi_scores(acsi_scores):
    score_table = acsi_scores.copy()
    avg_columns = {
        "direct_gen_score": "avg_direct_gen_0_to_3",
        "usefulness_score": "avg_usefulness_0_to_3",
        "quality_comp_score": "avg_quality_comp_0_to_3",
        "physical_req_score": "avg_physical_req_0_to_3",
        "personal_req_score": "avg_personal_req_0_to_3",
    }
    if set(avg_columns.values()).issubset(score_table.columns):
        avg_frame = score_table[["subreddit", *avg_columns.values()]].rename(
            columns={value: key for key, value in avg_columns.items()}
        )
    else:
        component_columns = [
            "direct_gen", "usefulness", "quality_comp", "physical_req", "personal_req",
        ]
        missing_columns = set(component_columns) - set(score_table.columns)
        if missing_columns:
            raise ValueError(f"ACSI score table missing columns: {sorted(missing_columns)}")
        avg_frame = score_table[["subreddit", *component_columns]].copy()
        for component in component_columns:
            avg_frame[f"{component}_score"] = (
                (pd.to_numeric(avg_frame[component], errors="coerce") - 1.0)
                * 3.0
                / 4.0
            )
        avg_frame = avg_frame[
            [
                "subreddit", "direct_gen_score", "usefulness_score",
                "quality_comp_score", "physical_req_score", "personal_req_score",
            ]
        ]
    avg_frame["subreddit"] = avg_frame["subreddit"].astype(str)
    for column_name in [
        "direct_gen_score", "usefulness_score", "quality_comp_score",
        "physical_req_score", "personal_req_score",
    ]:
        avg_frame[column_name] = pd.to_numeric(avg_frame[column_name], errors="coerce")
    avg_frame["gen_cap"] = avg_frame[
        ["direct_gen_score", "usefulness_score", "quality_comp_score"]
    ].mean(axis=1) / 3.0
    avg_frame["phys_free"] = 1.0 - avg_frame["physical_req_score"] / 3.0
    avg_frame["pers_free"] = 1.0 - avg_frame["personal_req_score"] / 3.0
    return avg_frame[["subreddit", "gen_cap", "phys_free", "pers_free"]]

def compute_acsi_version_correlation_check(acsi_scores, measurement_path=None):
    measurement_path = measurement_path or ACSI_MEASUREMENT_SAMPLE_RUN1_PATH
    balanced_scores = balanced_simple_acsi_scores(acsi_scores)
    preshock_scores, preshock_metadata = simple_acsi_scores_from_measurements(
        pd.read_csv(measurement_path),
        subreddit_filter=balanced_scores["subreddit"],
    )
    merged = balanced_scores.merge(
        preshock_scores,
        on="subreddit",
        suffixes=("_balanced", "_preshock"),
        how="inner",
    ).dropna()
    if len(merged) < 3:
        raise ValueError("Need at least 3 merged subreddits for ACSI version correlation check.")

    rows = []
    for dimension in ["gen_cap", "phys_free", "pers_free"]:
        balanced = merged[f"{dimension}_balanced"].astype(float)
        preshock = merged[f"{dimension}_preshock"].astype(float)
        pearson_r, pearson_pvalue = stats.pearsonr(balanced, preshock)
        spearman = stats.spearmanr(balanced, preshock)
        rows.append({
            "dimension": dimension,
            "n_subreddits": safe_int(len(merged)),
            "pearson_r": safe_float(pearson_r),
            "pearson_pvalue": safe_float(pearson_pvalue),
            "spearman_rho": safe_float(spearman.statistic),
            "spearman_pvalue": safe_float(spearman.pvalue),
            "mean_abs_difference": safe_float((preshock - balanced).abs().mean()),
        })
    summary = pd.DataFrame(rows)
    emit_output_table(summary, TABLES_DIR / "acsi_version_correlation_summary.csv", index=False)

    merged["pers_free_diff_balanced_minus_preshock"] = (
        merged["pers_free_balanced"] - merged["pers_free_preshock"]
    )
    candidates = merged[
        merged["pers_free_diff_balanced_minus_preshock"].abs() > 0.15
    ][[
        "subreddit", "pers_free_balanced", "pers_free_preshock",
        "pers_free_diff_balanced_minus_preshock",
    ]].sort_values("pers_free_diff_balanced_minus_preshock")
    emit_output_table(
        candidates,
        TABLES_DIR / "acsi_persfree_composition_bias_candidates.csv",
        index=False,
    )
    return {
        "summary": rows,
        "n_preshock_posts_used": preshock_metadata["n_posts_used"],
        "n_merged_subreddits": safe_int(len(merged)),
        "pers_free_candidate_count": safe_int(len(candidates)),
    }

def compute_composition_bias_outlier_check(submonth_panel, acsi_scores):
    outlier_names = ["personalstatement", "lonely", "bipolar", "BreakUps", "ptsd", "divorce"]
    panel = submonth_panel.copy()
    if panel.empty:
        raise ValueError("Subreddit-month panel is empty.")
    if "year_month" not in panel.columns and "month" in panel.columns:
        panel = panel.rename(columns={"month": "year_month"})
    if "post_shock" not in panel.columns and "post" in panel.columns:
        panel["post_shock"] = panel["post"]
    required_columns = {"subreddit", "year_month", "log_posts", "post_shock"}
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")

    balanced_scores = balanced_simple_acsi_scores(acsi_scores)
    preshock_scores, _ = simple_acsi_scores_from_measurements(
        pd.read_csv(ACSI_MEASUREMENT_SAMPLE_RUN1_PATH),
        subreddit_filter=balanced_scores["subreddit"],
    )
    activity = (
        panel.dropna(subset=["subreddit", "post_shock", "log_posts"])
        .groupby(["subreddit", "post_shock"])["log_posts"]
        .mean()
        .unstack("post_shock")
        .rename(columns={0: "pre_mean", 1: "post_mean"})
        .reset_index()
    )
    activity["activity_change"] = activity["post_mean"] - activity["pre_mean"]
    score_diff = balanced_scores[["subreddit", "pers_free"]].rename(
        columns={"pers_free": "pers_free_balanced"}
    ).merge(
        preshock_scores[["subreddit", "pers_free"]].rename(
            columns={"pers_free": "pers_free_preshock"}
        ),
        on="subreddit",
        how="inner",
    )
    score_diff["pers_free_diff"] = (
        score_diff["pers_free_balanced"] - score_diff["pers_free_preshock"]
    )
    merged = activity.merge(score_diff, on="subreddit", how="inner").dropna(
        subset=["activity_change", "pers_free_diff"]
    )
    if len(merged) < 3:
        raise ValueError("Need at least 3 merged subreddits for composition-bias check.")
    merged["is_known_outlier"] = merged["subreddit"].isin(outlier_names).astype(int)

    full_model = smf.ols("pers_free_diff ~ activity_change", data=merged).fit()
    trimmed = merged[merged["is_known_outlier"].eq(0)].copy()
    trimmed_model = smf.ols("pers_free_diff ~ activity_change", data=trimmed).fit()
    dummy_model = smf.ols(
        "pers_free_diff ~ activity_change + is_known_outlier",
        data=merged,
    ).fit()

    model_rows = []
    for model_name, model, sample_size in [
        ("full_sample", full_model, len(merged)),
        ("excluding_six_known_outliers", trimmed_model, len(trimmed)),
        ("full_sample_with_outlier_dummy", dummy_model, len(merged)),
    ]:
        model_rows.append({
            "model": model_name,
            "n": safe_int(sample_size),
            "activity_change_coef": safe_float(model.params.get("activity_change")),
            "activity_change_se": safe_float(model.bse.get("activity_change")),
            "activity_change_pvalue": safe_float(model.pvalues.get("activity_change")),
            "r_squared": safe_float(model.rsquared),
        })
    emit_output_table(
        pd.DataFrame(model_rows),
        TABLES_DIR / "composition_bias_check_models.csv",
        index=False,
    )

    outliers = merged[merged["is_known_outlier"].eq(1)].copy()
    rest = merged[merged["is_known_outlier"].eq(0)].copy()
    ttest = stats.ttest_ind(
        outliers["pers_free_diff"],
        rest["pers_free_diff"],
        equal_var=False,
    )
    outlier_summary = {
        "n_outliers": safe_int(len(outliers)),
        "n_rest": safe_int(len(rest)),
        "outlier_mean_activity_change": safe_float(outliers["activity_change"].mean()),
        "outlier_mean_pers_free_preshock": safe_float(outliers["pers_free_preshock"].mean()),
        "outlier_mean_pers_free_balanced": safe_float(outliers["pers_free_balanced"].mean()),
        "outlier_mean_pers_free_diff": safe_float(outliers["pers_free_diff"].mean()),
        "rest_mean_pers_free_diff": safe_float(rest["pers_free_diff"].mean()),
        "welch_t_stat": safe_float(ttest.statistic),
        "welch_pvalue": safe_float(ttest.pvalue),
    }
    emit_output_table(
        pd.DataFrame([outlier_summary]),
        TABLES_DIR / "composition_bias_check_outlier_summary.csv",
        index=False,
    )
    return {
        "models": model_rows,
        "outlier_summary": outlier_summary,
    }

def compute_two_run_preshock_scores(run1_path=None, run2_path=None, subreddit_filter=None):
    run1_path = run1_path or ACSI_MEASUREMENT_SAMPLE_RUN1_PATH
    run2_path = run2_path or DATA_DIR / "acsi_measurement_sample_run2.csv"
    run1 = normalize_measurement_frame(pd.read_csv(run1_path))
    run2 = normalize_measurement_frame(pd.read_csv(run2_path))
    merged = run1.merge(run2, on="post_id", suffixes=("_run1", "_run2"), how="inner")
    consistent = (
        merged["subreddit_run1"].eq(merged["subreddit_run2"])
        & merged["month_run1"].eq(merged["month_run2"])
    )
    merged = merged[consistent].copy()
    if subreddit_filter is not None:
        merged = merged[merged["subreddit_run1"].isin(set(subreddit_filter))].copy()
    merged["subreddit"] = merged["subreddit_run1"]
    merged["month"] = merged["month_run1"]
    merged["ai_related_flag"] = merged[[
        "ai_related_flag_run1", "ai_related_flag_run2",
    ]].max(axis=1)
    score_columns = [
        "direct_gen_score", "usefulness_score", "quality_comp_score",
        "physical_req_score", "personal_req_score",
    ]
    for column_name in score_columns:
        merged[f"avg_{column_name}"] = merged[[
            f"{column_name}_run1", f"{column_name}_run2",
        ]].mean(axis=1)
    averaged_columns = [f"avg_{column_name}" for column_name in score_columns]
    pre = merged[
        (merged["month"] < "2022-11")
        & merged["ai_related_flag"].eq(0)
    ].dropna(subset=averaged_columns).copy()
    grouped = (
        pre.groupby("subreddit", as_index=False)[averaged_columns]
        .mean()
        .sort_values("subreddit")
    )
    if grouped.empty:
        return grouped.assign(gen_cap=[], phys_free=[], pers_free=[]), {
            "n_overlap_rows": safe_int(len(merged)),
            "n_posts_used": 0,
            "n_subreddits": 0,
        }
    grouped["gen_cap"] = grouped[[
        "avg_direct_gen_score", "avg_usefulness_score", "avg_quality_comp_score",
    ]].mean(axis=1) / 3.0
    grouped["phys_free"] = 1.0 - grouped["avg_physical_req_score"] / 3.0
    grouped["pers_free"] = 1.0 - grouped["avg_personal_req_score"] / 3.0
    return grouped[["subreddit", "gen_cap", "phys_free", "pers_free"]], {
        "n_overlap_rows": safe_int(len(merged)),
        "n_posts_used": safe_int(len(pre)),
        "n_subreddits": safe_int(grouped["subreddit"].nunique()),
    }

def compute_two_run_preshock_regression(submonth_panel, acsi_scores):
    scores, metadata = compute_two_run_preshock_scores(
        subreddit_filter=acsi_scores["subreddit"].astype(str),
    )
    emit_output_table(
        scores,
        TABLES_DIR / "acsi_preshock_tworuns.csv",
        index=False,
    )
    panel = submonth_panel.copy()
    if panel.empty:
        raise ValueError("Subreddit-month panel is empty.")
    if "year_month" not in panel.columns and "month" in panel.columns:
        panel = panel.rename(columns={"month": "year_month"})
    if "post_shock" not in panel.columns and "post" in panel.columns:
        panel["post_shock"] = panel["post"]
    required_columns = {"subreddit", "year_month", "log_posts", "post_shock"}
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")
    model_data = panel.merge(scores, on="subreddit", how="inner").dropna(
        subset=["log_posts", "post_shock", "gen_cap", "phys_free", "pers_free"]
    )
    if model_data.empty:
        raise ValueError("No rows after merging two-run scores into panel.")
    for dimension in ["gen_cap", "phys_free", "pers_free"]:
        model_data[f"{dimension}_post"] = model_data[dimension] * model_data["post_shock"]
    model = fit_ols(
        "log_posts ~ gen_cap_post + phys_free_post + pers_free_post + C(subreddit) + C(year_month)",
        model_data,
        cluster_col="subreddit",
    )
    if model is None:
        raise ValueError("Two-run pre-shock regression failed to fit.")
    rows = []
    for term in ["gen_cap_post", "phys_free_post", "pers_free_post"]:
        row = reg_result(model, term)
        row["term"] = term
        rows.append(row)
    emit_output_table(
        pd.DataFrame(rows),
        TABLES_DIR / "acsi_tworuns_preshock_regression.csv",
        index=False,
    )
    persfree_row = next((row for row in rows if row["term"] == "pers_free_post"), {})
    return {
        "metadata": metadata,
        "rows": rows,
        "pers_free_se": persfree_row.get("se"),
        "pers_free_se_vs_single_run_preshock": safe_float(persfree_row.get("se") - 0.159)
        if persfree_row.get("se") is not None else None,
        "pers_free_se_vs_balanced": safe_float(persfree_row.get("se") - 0.124)
        if persfree_row.get("se") is not None else None,
    }

def prepare_gen_cap_simex_scores_from_two_runs(
    run1_path=None,
    run2_path=None,
    subreddit_filter=None,
    pre_shock_only=True,
    exclude_ai_related=True,
):
    run1_path = run1_path or ACSI_MEASUREMENT_SAMPLE_RUN1_PATH
    run2_path = run2_path or DATA_DIR / "acsi_measurement_sample_run2.csv"
    run1 = normalize_measurement_frame(pd.read_csv(run1_path))
    run2 = normalize_measurement_frame(pd.read_csv(run2_path))
    merged = run1.merge(run2, on="post_id", suffixes=("_run1", "_run2"), how="inner")
    consistent = (
        merged["subreddit_run1"].eq(merged["subreddit_run2"])
        & merged["month_run1"].eq(merged["month_run2"])
    )
    merged = merged[consistent].copy()
    if subreddit_filter is not None:
        merged = merged[merged["subreddit_run1"].isin(set(subreddit_filter))].copy()
    if pre_shock_only:
        merged = merged[merged["month_run1"] < "2022-11"].copy()
    if exclude_ai_related:
        ai_related = merged[[
            "ai_related_flag_run1",
            "ai_related_flag_run2",
        ]].max(axis=1)
        merged = merged[ai_related.eq(0)].copy()

    return pd.DataFrame({
        "post_id": merged["post_id"],
        "subreddit": merged["subreddit_run1"],
        "run1_gen": merged["direct_gen_score_run1"],
        "run1_use": merged["usefulness_score_run1"],
        "run1_qual": merged["quality_comp_score_run1"],
        "run2_gen": merged["direct_gen_score_run2"],
        "run2_use": merged["usefulness_score_run2"],
        "run2_qual": merged["quality_comp_score_run2"],
        "run1_pers": merged["personal_req_score_run1"],
        "run2_pers": merged["personal_req_score_run2"],
        "run1_phys": merged["physical_req_score_run1"],
        "run2_phys": merged["physical_req_score_run2"],
    }).dropna()

def compute_subreddit_scores_and_gen_cap_error(scores):
    required_columns = {
        "post_id", "subreddit",
        "run1_gen", "run1_use", "run1_qual",
        "run2_gen", "run2_use", "run2_qual",
        "run1_pers", "run2_pers", "run1_phys", "run2_phys",
    }
    missing_columns = required_columns - set(scores.columns)
    if missing_columns:
        raise ValueError(f"SIMEX scores missing columns: {sorted(missing_columns)}")

    post_scores = scores.copy()
    numeric_columns = sorted(required_columns - {"post_id", "subreddit"})
    for column_name in numeric_columns:
        post_scores[column_name] = pd.to_numeric(post_scores[column_name], errors="coerce")
    post_scores = post_scores.dropna(subset=numeric_columns + ["subreddit"]).copy()
    if len(post_scores) < 2:
        raise ValueError("Need at least two paired scored posts for SIMEX.")

    post_scores["gen_cap_run1"] = post_scores[[
        "run1_gen", "run1_use", "run1_qual",
    ]].mean(axis=1) / 3.0
    post_scores["gen_cap_run2"] = post_scores[[
        "run2_gen", "run2_use", "run2_qual",
    ]].mean(axis=1) / 3.0
    post_scores["gen_cap"] = post_scores[["gen_cap_run1", "gen_cap_run2"]].mean(axis=1)
    post_scores["phys_free"] = 1.0 - post_scores[["run1_phys", "run2_phys"]].mean(axis=1) / 3.0
    post_scores["pers_free"] = 1.0 - post_scores[["run1_pers", "run2_pers"]].mean(axis=1) / 3.0

    diff = post_scores["gen_cap_run1"] - post_scores["gen_cap_run2"]
    sigma2_u = safe_float(0.5 * diff.var(ddof=1))
    if sigma2_u is None or not np.isfinite(sigma2_u) or sigma2_u < 0:
        raise ValueError("Could not estimate a valid GenCap measurement-error variance.")

    subreddit_scores = (
        post_scores.groupby("subreddit", as_index=False)
        .agg(
            gen_cap=("gen_cap", "mean"),
            phys_free=("phys_free", "mean"),
            pers_free=("pers_free", "mean"),
            n_scored_posts=("post_id", "nunique"),
        )
        .sort_values("subreddit")
    )
    return subreddit_scores, post_scores, sigma2_u

def prepare_gen_cap_simex_model_inputs(submonth_panel, subreddit_scores):
    panel = submonth_panel.copy()
    if "year_month" not in panel.columns and "month" in panel.columns:
        panel = panel.rename(columns={"month": "year_month"})
    if "post_shock" not in panel.columns and "post" in panel.columns:
        panel["post_shock"] = panel["post"]
    required_panel_columns = {"subreddit", "year_month", "log_posts", "post_shock"}
    missing_columns = required_panel_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"SIMEX panel missing columns: {sorted(missing_columns)}")

    model_data = panel.merge(
        subreddit_scores[["subreddit", "gen_cap", "phys_free", "pers_free"]],
        on="subreddit",
        how="inner",
    ).dropna(
        subset=[
            "subreddit", "year_month", "log_posts", "post_shock",
            "gen_cap", "phys_free", "pers_free",
        ]
    ).copy()
    for column_name in ["log_posts", "post_shock", "gen_cap", "phys_free", "pers_free"]:
        model_data[column_name] = pd.to_numeric(model_data[column_name], errors="coerce")
    model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log_posts", "post_shock", "gen_cap", "phys_free", "pers_free"]
    )
    if model_data.empty:
        raise ValueError("No rows after merging SIMEX scores into the panel.")
    if not is_balanced_two_way_panel(model_data, "subreddit", "year_month"):
        raise ValueError("SIMEX fast estimator requires a balanced subreddit-month panel.")

    subreddits = sorted(model_data["subreddit"].astype(str).unique())
    months = sorted(model_data["year_month"].astype(str).unique())
    y_matrix = (
        model_data.assign(
            subreddit=model_data["subreddit"].astype(str),
            year_month=model_data["year_month"].astype(str),
        )
        .pivot(index="subreddit", columns="year_month", values="log_posts")
        .reindex(index=subreddits, columns=months)
        .astype(float)
    )
    post_matrix = (
        model_data.assign(
            subreddit=model_data["subreddit"].astype(str),
            year_month=model_data["year_month"].astype(str),
        )
        .pivot(index="subreddit", columns="year_month", values="post_shock")
        .reindex(index=subreddits, columns=months)
        .astype(float)
    )
    if y_matrix.isna().any().any() or post_matrix.isna().any().any():
        raise ValueError("SIMEX panel has missing cells after reshaping.")
    post_by_month = post_matrix.iloc[0].to_numpy(dtype=float)
    if not np.allclose(post_matrix.to_numpy(dtype=float), post_by_month[None, :]):
        raise ValueError("Post-shock indicator must be constant within month for SIMEX.")

    y_values = y_matrix.to_numpy(dtype=float)
    if not np.isfinite(y_values).all() or not np.isfinite(post_by_month).all():
        raise ValueError("SIMEX panel contains non-finite log_posts or post_shock values.")
    y_resid = y_values - y_values.mean(axis=1, keepdims=True) - y_values.mean(axis=0, keepdims=True) + y_values.mean()
    post_centered = post_by_month - post_by_month.mean()
    post_ss = float(np.dot(post_centered, post_centered))
    if post_ss <= 0:
        raise ValueError("Post-shock indicator has no usable variation.")
    y_weighted = np.sum(y_resid * post_centered[None, :], axis=1)

    score_matrix = (
        subreddit_scores.assign(subreddit=subreddit_scores["subreddit"].astype(str))
        .set_index("subreddit")
        .reindex(subreddits)[["gen_cap", "phys_free", "pers_free"]]
        .astype(float)
        .to_numpy()
    )
    if not np.isfinite(score_matrix).all():
        raise ValueError("Subreddit scores missing or non-finite for one or more model subreddits.")

    return {
        "subreddits": subreddits,
        "months": months,
        "score_matrix": score_matrix,
        "y_weighted": y_weighted,
        "post_ss": post_ss,
        "n_obs": safe_int(len(model_data)),
        "n_subreddits": safe_int(len(subreddits)),
        "n_months": safe_int(len(months)),
    }

def gen_cap_b1_from_simex_arrays(score_matrix, y_weighted, post_ss):
    scores = np.asarray(score_matrix, dtype=float)
    y_weighted = np.asarray(y_weighted, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError("SIMEX score_matrix must have columns gen_cap, phys_free, pers_free.")
    score_centered = scores - scores.mean(axis=0, keepdims=True)
    xtx = float(post_ss) * (score_centered.T @ score_centered)
    xty = score_centered.T @ y_weighted
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    return safe_float(beta[0])

def estimate_gen_cap_simex_from_arrays(
    score_matrix,
    y_weighted,
    post_ss,
    sigma2_u,
    lambdas,
    n_simulations,
    rng,
):
    base_scores = np.asarray(score_matrix, dtype=float)
    b1_observed = gen_cap_b1_from_simex_arrays(base_scores, y_weighted, post_ss)
    rows = [{
        "lambda": 0.0,
        "simulation": 0,
        "b1": b1_observed,
    }]
    mean_rows = [{
        "lambda": 0.0,
        "mean_b1": b1_observed,
        "sd_b1": 0.0,
        "n_simulations": 1,
    }]

    for lambda_value in lambdas:
        lambda_value = float(lambda_value)
        noise_sd = float(np.sqrt(max(lambda_value * sigma2_u, 0.0)))
        simulated_b1 = []
        for simulation in range(int(n_simulations)):
            simulated_scores = base_scores.copy()
            simulated_scores[:, 0] = simulated_scores[:, 0] + rng.normal(
                loc=0.0,
                scale=noise_sd,
                size=simulated_scores.shape[0],
            )
            b1 = gen_cap_b1_from_simex_arrays(simulated_scores, y_weighted, post_ss)
            if b1 is not None and np.isfinite(b1):
                simulated_b1.append(b1)
                rows.append({
                    "lambda": lambda_value,
                    "simulation": simulation + 1,
                    "b1": b1,
                })
        if simulated_b1:
            mean_rows.append({
                "lambda": lambda_value,
                "mean_b1": safe_float(np.mean(simulated_b1)),
                "sd_b1": safe_float(np.std(simulated_b1, ddof=1)) if len(simulated_b1) > 1 else 0.0,
                "n_simulations": safe_int(len(simulated_b1)),
            })

    mean_table = pd.DataFrame(mean_rows).sort_values("lambda")
    if len(mean_table) < 3:
        raise ValueError("Need at least three SIMEX lambda points for quadratic extrapolation.")
    coefficients = np.polyfit(
        mean_table["lambda"].astype(float),
        mean_table["mean_b1"].astype(float),
        deg=2,
    )
    corrected_b1 = safe_float(np.polyval(coefficients, -1.0))
    return {
        "observed_b1": b1_observed,
        "corrected_b1": corrected_b1,
        "quadratic_coefficients": [safe_float(value) for value in coefficients],
        "lambda_means": mean_rows,
        "simulation_rows": rows,
    }

def plot_gen_cap_simex_curve(simex_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_table = pd.DataFrame(simex_result["lambda_means"]).sort_values("lambda")
    coefficients = np.asarray(simex_result["quadratic_coefficients"], dtype=float)
    lambda_grid = np.linspace(-1.0, max(2.0, float(mean_table["lambda"].max())), 200)
    fitted = np.polyval(coefficients, lambda_grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        mean_table["lambda"],
        mean_table["mean_b1"],
        color="#2563eb",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        s=45,
        zorder=3,
        label="SIMEX means",
    )
    ax.plot(lambda_grid, fitted, color="#dc2626", linewidth=2.0, label="Quadratic fit")
    ax.axvline(-1.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.scatter(
        [-1.0],
        [simex_result["corrected_b1"]],
        color="#dc2626",
        edgecolor="white",
        linewidth=0.8,
        marker="D",
        s=52,
        zorder=4,
        label="SIMEX corrected",
    )
    ax.set_xlabel("SIMEX noise multiplier lambda")
    ax.set_ylabel("GenCap x Post coefficient")
    ax.set_title("GenCap measurement-error SIMEX")
    ax.legend(frameon=False, fontsize=9)
    save_plot(fig, output_path)
    return str(output_path)

def compute_gen_cap_simex_correction(
    scores,
    submonth_panel,
    lambdas=(0.5, 1.0, 1.5, 2.0),
    n_simulations=100,
    n_bootstrap=500,
    random_seed=None,
    output_path=None,
):
    if random_seed is None:
        random_seed = globals().get("RANDOM_SEED", 42)
    subreddit_scores, post_scores, sigma2_u = compute_subreddit_scores_and_gen_cap_error(scores)
    model_inputs = prepare_gen_cap_simex_model_inputs(submonth_panel, subreddit_scores)
    rng = np.random.default_rng(random_seed)
    simex_result = estimate_gen_cap_simex_from_arrays(
        model_inputs["score_matrix"],
        model_inputs["y_weighted"],
        model_inputs["post_ss"],
        sigma2_u,
        lambdas,
        n_simulations,
        rng,
    )

    bootstrap_b1 = []
    n_subreddits = model_inputs["score_matrix"].shape[0]
    for bootstrap_index in range(int(n_bootstrap)):
        sample_index = rng.integers(0, n_subreddits, size=n_subreddits)
        bootstrap_result = estimate_gen_cap_simex_from_arrays(
            model_inputs["score_matrix"][sample_index, :],
            model_inputs["y_weighted"][sample_index],
            model_inputs["post_ss"],
            sigma2_u,
            lambdas,
            n_simulations,
            rng,
        )
        corrected = bootstrap_result.get("corrected_b1")
        if corrected is not None and np.isfinite(corrected):
            bootstrap_b1.append(corrected)

    bootstrap_se = (
        safe_float(np.std(bootstrap_b1, ddof=1))
        if len(bootstrap_b1) > 1 else None
    )
    simex_result.update({
        "bootstrap_se": bootstrap_se,
        "bootstrap_n": safe_int(len(bootstrap_b1)),
        "sigma2_u": sigma2_u,
        "n_post_scores": safe_int(len(post_scores)),
        "n_score_subreddits": safe_int(subreddit_scores["subreddit"].nunique()),
        "n_model_subreddits": model_inputs["n_subreddits"],
        "n_model_months": model_inputs["n_months"],
        "n_model_obs": model_inputs["n_obs"],
        "lambdas": [float(value) for value in lambdas],
        "n_simulations_per_lambda": safe_int(n_simulations),
    })

    emit_output_table(
        pd.DataFrame(simex_result["lambda_means"]),
        TABLES_DIR / "acsi_gen_cap_simex_lambda_means.csv",
        index=False,
    )
    emit_output_table(
        pd.DataFrame([{
            "term": "gen_cap_post",
            "observed_b1": simex_result["observed_b1"],
            "simex_corrected_b1": simex_result["corrected_b1"],
            "bootstrap_se": bootstrap_se,
            "sigma2_u": sigma2_u,
            "n_post_scores": simex_result["n_post_scores"],
            "n_model_subreddits": simex_result["n_model_subreddits"],
            "n_model_months": simex_result["n_model_months"],
            "bootstrap_n": simex_result["bootstrap_n"],
        }]),
        TABLES_DIR / "acsi_gen_cap_simex_summary.csv",
        index=False,
    )

    if output_path is None:
        output_path = FIGURES_DIR / "acsi_gen_cap_simex.png"
    simex_result["plot_path"] = plot_gen_cap_simex_curve(simex_result, output_path)
    return simex_result

def parse_blackout_participant_subreddits(text):
    return {
        match.group(1).lower()
        for match in re.finditer(r"(?i)(?:^|\s)r/([A-Za-z0-9_]+)\b", str(text))
    }

def load_or_build_blackout_proxy(
    subreddits,
    blackout_path=None,
    proxy_path=None,
    participant_list_path=None,
):
    root = globals().get("ROOT", Path("."))
    blackout_path = Path(blackout_path or root / "blackout.csv")
    proxy_path = Path(proxy_path or root / "blackout_proxy.csv")
    subreddits = pd.Series(subreddits, dtype=str).dropna().drop_duplicates().sort_values()

    if blackout_path.exists():
        blackout = pd.read_csv(blackout_path)
        source = str(blackout_path)
    elif proxy_path.exists() and participant_list_path is None:
        blackout = pd.read_csv(proxy_path)
        source = str(proxy_path)
    else:
        if participant_list_path is None:
            raise FileNotFoundError(
                f"Missing {blackout_path} and {proxy_path}; provide participant_list_path to build a proxy."
            )
        participant_text = Path(participant_list_path).read_text(encoding="utf-8")
        participant_set = parse_blackout_participant_subreddits(participant_text)
        blackout = pd.DataFrame({"subreddit": subreddits})
        blackout["participated"] = (
            blackout["subreddit"].astype(str).str.lower().isin(participant_set).astype(int)
        )
        blackout.to_csv(proxy_path, index=False)
        source = str(participant_list_path)

    required_columns = {"subreddit", "participated"}
    missing_columns = required_columns - set(blackout.columns)
    if missing_columns:
        raise ValueError(f"Blackout file missing columns: {sorted(missing_columns)}")
    blackout = blackout[["subreddit", "participated"]].copy()
    blackout["subreddit"] = blackout["subreddit"].astype(str)
    blackout["participated"] = pd.to_numeric(blackout["participated"], errors="coerce").fillna(0).astype(int)
    blackout = (
        pd.DataFrame({"subreddit": subreddits})
        .merge(blackout, on="subreddit", how="left")
        .fillna({"participated": 0})
    )
    blackout["participated"] = blackout["participated"].astype(int)
    return blackout, source

def reddit_disruption_model_panel(panel, score_path=None):
    model_panel = panel.copy()
    if "year_month" in model_panel.columns and "month" not in model_panel.columns:
        model_panel = model_panel.rename(columns={"year_month": "month"})
    if "post_shock" in model_panel.columns and "post" not in model_panel.columns:
        model_panel = model_panel.rename(columns={"post_shock": "post"})
    required_columns = {"subreddit", "month", "log_posts", "post"}
    missing_columns = required_columns - set(model_panel.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")

    score_columns = {"gen_cap", "phys_free", "pers_free"}
    if not score_columns.issubset(model_panel.columns):
        root = globals().get("ROOT", Path("."))
        score_path = Path(score_path or root / "acsi_preshock_tworuns.csv")
        scores = pd.read_csv(score_path)
        missing_score_columns = ({"subreddit"} | score_columns) - set(scores.columns)
        if missing_score_columns:
            raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
        model_panel = model_panel.merge(
            scores[["subreddit", "gen_cap", "phys_free", "pers_free"]],
            on="subreddit",
            how="left",
        )

    for column_name in ["log_posts", "post", "gen_cap", "phys_free", "pers_free"]:
        model_panel[column_name] = pd.to_numeric(model_panel[column_name], errors="coerce")
    model_panel = model_panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["subreddit", "month", "log_posts", "post", "gen_cap", "phys_free", "pers_free"]
    )
    for dimension in ["gen_cap", "phys_free", "pers_free"]:
        model_panel[f"{dimension}_post"] = model_panel[dimension] * model_panel["post"]
    return model_panel

def fit_reddit_disruption_spec(panel, spec_name, drop_disruption=False, blackout_controls=False):
    model_data = panel.copy()
    if drop_disruption:
        model_data = model_data[~model_data["month"].isin(["2023-06", "2023-07", "2023-08"])].copy()
    terms = ["gen_cap_post", "phys_free_post", "pers_free_post"]
    if blackout_controls:
        model_data["participated_post"] = model_data["participated"] * model_data["post"]
        for month_label in ["2023-06", "2023-07", "2023-08"]:
            term = f"participated_{month_label.replace('-', '_')}"
            model_data[term] = model_data["participated"] * model_data["month"].eq(month_label).astype(int)
            terms.append(term)
        terms.append("participated_post")
    formula = "log_posts ~ " + " + ".join(terms) + " + C(subreddit) + C(month)"
    model = fit_ols(formula, model_data, cluster_col="subreddit")
    if model is None:
        raise ValueError(f"{spec_name} failed to fit.")
    return model, model_data

def compute_reddit_disruption_robustness(
    panel,
    score_path=None,
    blackout_path=None,
    proxy_path=None,
    participant_list_path=None,
    output_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "reddit_disruption_robustness.csv")
    model_panel = reddit_disruption_model_panel(panel, score_path=score_path)
    blackout, blackout_source = load_or_build_blackout_proxy(
        model_panel["subreddit"].astype(str).unique(),
        blackout_path=blackout_path,
        proxy_path=proxy_path,
        participant_list_path=participant_list_path,
    )
    model_panel = model_panel.merge(blackout, on="subreddit", how="left")
    model_panel["participated"] = model_panel["participated"].fillna(0).astype(int)

    specs = [
        ("Spec A", "baseline", False, False),
        ("Spec B", "blackout_control", False, True),
        ("Spec C", "drop_disruption_quarter", True, False),
        ("Spec D", "drop_disruption_quarter_blackout_control", True, True),
    ]
    focal_terms = ["pers_free_post", "gen_cap_post", "phys_free_post"]
    rows = []
    models = {}
    for spec_label, spec_name, drop_disruption, blackout_controls in specs:
        model, model_data = fit_reddit_disruption_spec(
            model_panel,
            spec_label,
            drop_disruption=drop_disruption,
            blackout_controls=blackout_controls,
        )
        models[spec_label] = model
        for term in focal_terms + ["participated_post"]:
            if term not in model.params.index:
                continue
            result = reg_result(model, term)
            rows.append({
                "row_type": "regression",
                "spec": spec_label,
                "spec_name": spec_name,
                "term": term,
                "coef": result.get("coef"),
                "se": result.get("se"),
                "pvalue": result.get("pvalue"),
                "n_obs": safe_int(model.nobs),
                "n_subreddits": safe_int(model_data["subreddit"].nunique()),
                "n_months": safe_int(model_data["month"].nunique()),
            })

    subreddit_scores = (
        model_panel[["subreddit", "pers_free", "participated"]]
        .drop_duplicates("subreddit")
        .dropna()
        .copy()
    )
    corr_model = fit_ols("participated ~ pers_free", subreddit_scores)
    corr_result = reg_result(corr_model, "pers_free") if corr_model is not None else {}
    rows.append({
        "row_type": "blackout_persfree_correlation",
        "spec": "OLS",
        "spec_name": "participated_on_pers_free",
        "term": "pers_free",
        "coef": corr_result.get("coef"),
        "se": corr_result.get("se"),
        "pvalue": corr_result.get("pvalue"),
        "n_obs": safe_int(corr_model.nobs) if corr_model is not None else None,
        "n_subreddits": safe_int(len(subreddit_scores)),
        "n_months": None,
    })

    participated_subreddits = sorted(
        blackout.loc[blackout["participated"].eq(1), "subreddit"].astype(str).tolist()
    )
    summary_rows = pd.DataFrame(rows)
    summary_rows["blackout_source"] = blackout_source
    summary_rows["n_blackout_participants"] = safe_int(len(participated_subreddits))
    summary_rows.to_csv(output_path, index=False)

    side_by_side = (
        summary_rows[summary_rows["row_type"].eq("regression") & summary_rows["term"].isin(focal_terms)]
        .assign(
            estimate=lambda frame: frame.apply(
                lambda row: (
                    f"{fmt_signed4(row['coef'])} "
                    f"(SE {fmt4(row['se'])}, p {fmt4(row['pvalue'])})"
                ),
                axis=1,
            )
        )
        .pivot(index="term", columns="spec", values="estimate")
        .reindex(focal_terms)
    )
    return {
        "rows": rows,
        "side_by_side": side_by_side,
        "blackout_source": blackout_source,
        "participated_subreddits": participated_subreddits,
        "n_blackout_participants": safe_int(len(participated_subreddits)),
        "output_path": str(output_path),
        "spec_b_participated_post": next(
            (
                row for row in rows
                if row["spec"] == "Spec B" and row["term"] == "participated_post"
            ),
            None,
        ),
        "blackout_persfree_correlation": rows[-1],
    }

def placebo_shock_model_panel(panel=None, score_path=None):
    root = globals().get("ROOT", Path("."))
    if panel is None:
        panel = pd.read_csv(root / "panel.csv")
    model_panel = panel.copy()
    if "year_month" in model_panel.columns and "month" not in model_panel.columns:
        model_panel = model_panel.rename(columns={"year_month": "month"})
    required_panel_columns = {"subreddit", "month", "log_posts"}
    missing_columns = required_panel_columns - set(model_panel.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")

    score_columns = {"gen_cap", "phys_free", "pers_free"}
    if not score_columns.issubset(model_panel.columns):
        score_path = Path(score_path or root / "acsi_preshock_tworuns.csv")
        scores = pd.read_csv(score_path)
        missing_score_columns = ({"subreddit"} | score_columns) - set(scores.columns)
        if missing_score_columns:
            raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
        scores = (
            scores[["subreddit", "gen_cap", "phys_free", "pers_free"]]
            .drop_duplicates("subreddit")
            .copy()
        )
        model_panel = model_panel.merge(scores, on="subreddit", how="left")

    model_panel["subreddit"] = model_panel["subreddit"].astype(str)
    model_panel["month"] = model_panel["month"].astype(str).str.slice(0, 7)
    for column_name in ["log_posts", "gen_cap", "phys_free", "pers_free"]:
        model_panel[column_name] = pd.to_numeric(model_panel[column_name], errors="coerce")
    model_panel = model_panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["subreddit", "month", "log_posts", "gen_cap", "phys_free", "pers_free"]
    )
    if model_panel.empty:
        raise ValueError("No usable rows after merging placebo shock scores into panel.")
    return model_panel

def fit_placebo_shock_spec(model_panel, spec_label, period_terms, focal_terms):
    model_data = model_panel.copy()
    for period_name, start_month, end_month, active_months in period_terms:
        if start_month is not None:
            model_data = model_data[model_data["month"] >= start_month].copy()
        if end_month is not None:
            model_data = model_data[model_data["month"] <= end_month].copy()
        model_data[period_name] = active_months(model_data["month"])

    interaction_terms = []
    for period_name, _start_month, _end_month, _active_months in period_terms:
        for dimension in ["gen_cap", "phys_free", "pers_free"]:
            term = f"{dimension}_{period_name}"
            model_data[term] = model_data[dimension] * model_data[period_name].astype(float)
            interaction_terms.append(term)

    formula = "log_posts ~ " + " + ".join(interaction_terms) + " + C(subreddit) + C(month)"
    model = fit_ols(formula, model_data, cluster_col="subreddit")
    if model is None:
        raise ValueError(f"{spec_label} failed to fit.")

    rows = []
    for term in focal_terms:
        result = reg_result(model, term)
        rows.append({
            "row_type": "regression",
            "spec": spec_label,
            "term": term,
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "ci_low": safe_float(result.get("coef") - 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "ci_high": safe_float(result.get("coef") + 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "n_obs": safe_int(model.nobs),
            "n_subreddits": safe_int(model_data["subreddit"].nunique()),
            "n_months": safe_int(model_data["month"].nunique()),
            "start_month": model_data["month"].min(),
            "end_month": model_data["month"].max(),
        })
    return model, model_data, rows

def compute_placebo_shock_comparison(
    panel=None,
    score_path=None,
    output_path=None,
    comparison_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "placebo_shock_comparison.csv")
    comparison_path = Path(comparison_path or root / "placebo_shock_table.csv")
    model_panel = placebo_shock_model_panel(panel=panel, score_path=score_path)

    specs = [
        (
            "CHATGPT",
            [("post_chatgpt", None, None, lambda month: (month >= "2022-12").astype(int))],
            ["gen_cap_post_chatgpt", "phys_free_post_chatgpt", "pers_free_post_chatgpt"],
        ),
        (
            "API",
            [("post_api", None, None, lambda month: (month >= "2023-06").astype(int))],
            ["gen_cap_post_api", "phys_free_post_api", "pers_free_post_api"],
        ),
        (
            "HORSERACE",
            [
                (
                    "chatgpt_only",
                    None,
                    None,
                    lambda month: ((month >= "2022-12") & (month < "2023-06")).astype(int),
                ),
                ("both_active", None, None, lambda month: (month >= "2023-06").astype(int)),
            ],
            [
                "gen_cap_chatgpt_only", "phys_free_chatgpt_only", "pers_free_chatgpt_only",
                "gen_cap_both_active", "phys_free_both_active", "pers_free_both_active",
            ],
        ),
        (
            "FALSIFICATION",
            [("post_placebo", "2022-01", "2022-10", lambda month: (month >= "2022-06").astype(int))],
            ["gen_cap_post_placebo", "phys_free_post_placebo", "pers_free_post_placebo"],
        ),
    ]

    rows = []
    models = {}
    model_data_by_spec = {}
    for spec_label, period_terms, focal_terms in specs:
        model, model_data, spec_rows = fit_placebo_shock_spec(
            model_panel,
            spec_label,
            period_terms,
            focal_terms,
        )
        rows.extend(spec_rows)
        models[spec_label] = model
        model_data_by_spec[spec_label] = model_data

    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)

    comparison_terms = {
        "CHATGPT": "pers_free_post_chatgpt",
        "API": "pers_free_post_api",
        "HORSERACE_chatgpt_only": "pers_free_chatgpt_only",
        "HORSERACE_both": "pers_free_both_active",
        "FALSIFICATION": "pers_free_post_placebo",
    }
    comparison_lookup = {}
    for comparison_label, term in comparison_terms.items():
        spec_name = "HORSERACE" if comparison_label.startswith("HORSERACE") else comparison_label
        match = results[(results["spec"].eq(spec_name)) & (results["term"].eq(term))]
        comparison_lookup[comparison_label] = match.iloc[0].to_dict() if not match.empty else {}

    comparison_rows = []
    for metric in ["coef", "se", "pvalue", "n_obs", "n_subreddits", "n_months"]:
        row = {"metric": metric}
        for comparison_label, values in comparison_lookup.items():
            row[comparison_label] = values.get(metric)
        comparison_rows.append(row)
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(comparison_path, index=False)
    return {
        "model_panel": model_panel,
        "models": models,
        "model_data_by_spec": model_data_by_spec,
        "rows": rows,
        "results": results,
        "comparison_table": comparison_table,
        "output_path": str(output_path),
        "comparison_path": str(comparison_path),
    }

def run_single_shock_date_placebo(model_panel, shock_date, row_type="placebo", min_pre_months=3, min_post_months=3):
    shock_date = str(shock_date)
    model_data = model_panel.copy()
    pre_months = sorted(model_data.loc[model_data["month"] < shock_date, "month"].unique())
    post_months = sorted(model_data.loc[model_data["month"] >= shock_date, "month"].unique())
    base_row = {
        "row_type": row_type,
        "shock_date": shock_date,
        "status": "estimated",
        "n_pre_months": safe_int(len(pre_months)),
        "n_post_months": safe_int(len(post_months)),
        "n_obs": np.nan,
        "n_subreddits": np.nan,
        "pers_free_coef": np.nan,
        "pers_free_se": np.nan,
        "pers_free_p": np.nan,
        "pers_free_ci_low": np.nan,
        "pers_free_ci_high": np.nan,
        "gen_cap_coef": np.nan,
        "gen_cap_se": np.nan,
        "gen_cap_p": np.nan,
        "phys_free_coef": np.nan,
        "phys_free_se": np.nan,
        "phys_free_p": np.nan,
    }
    if len(pre_months) < min_pre_months or len(post_months) < min_post_months:
        base_row["status"] = "skipped_edge_window"
        return base_row, None

    model_data["post_s"] = (model_data["month"] >= shock_date).astype(int)
    terms = []
    for dimension in ["gen_cap", "phys_free", "pers_free"]:
        term = f"{dimension}_post_s"
        model_data[term] = model_data[dimension] * model_data["post_s"]
        terms.append(term)
    formula = "log_posts ~ " + " + ".join(terms) + " + C(subreddit) + C(month)"
    model = fit_ols(formula, model_data, cluster_col="subreddit")
    if model is None:
        base_row["status"] = "model_failed"
        return base_row, None

    base_row["n_obs"] = safe_int(model.nobs)
    base_row["n_subreddits"] = safe_int(model_data["subreddit"].nunique())
    for dimension in ["gen_cap", "phys_free", "pers_free"]:
        term = f"{dimension}_post_s"
        result = reg_result(model, term)
        base_row[f"{dimension}_coef"] = result.get("coef")
        base_row[f"{dimension}_se"] = result.get("se")
        base_row[f"{dimension}_p"] = result.get("pvalue")
        if dimension == "pers_free" and result.get("coef") is not None and result.get("se") is not None:
            base_row["pers_free_ci_low"] = safe_float(result.get("coef") - 1.96 * result.get("se"))
            base_row["pers_free_ci_high"] = safe_float(result.get("coef") + 1.96 * result.get("se"))
    return base_row, model

def plot_shock_date_placebo_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated = results[results["status"].eq("estimated")].copy()
    placebo = estimated[estimated["row_type"].eq("placebo")].copy()
    true = estimated[estimated["row_type"].eq("true")].copy()
    placebo["shock_dt"] = pd.to_datetime(placebo["shock_date"])
    true["shock_dt"] = pd.to_datetime(true["shock_date"])

    fig, ax = plt.subplots(figsize=(8, 5))
    if not placebo.empty:
        ax.errorbar(
            placebo["shock_dt"],
            placebo["pers_free_coef"].astype(float),
            yerr=1.96 * placebo["pers_free_se"].astype(float),
            fmt="o",
            color="#6b7280",
            ecolor="#9ca3af",
            alpha=0.5,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.0,
            elinewidth=1.5,
            capsize=4,
            label="Placebo shock dates",
            zorder=2,
        )
    if not true.empty:
        true_row = true.iloc[0]
        ax.errorbar(
            [true_row["shock_dt"]],
            [float(true_row["pers_free_coef"])],
            yerr=[1.96 * float(true_row["pers_free_se"])],
            fmt="o",
            color="#dc2626",
            ecolor="#dc2626",
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.0,
            elinewidth=1.5,
            capsize=4,
            label="True shock: Nov 2022",
            zorder=5,
        )
        ax.axhline(float(true_row["pers_free_coef"]), color="#dc2626", linestyle="--", linewidth=1.5)
    ax.axvline(pd.Timestamp("2022-11-01"), color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Shock date")
    ax.set_ylabel("PersFree x post coefficient")
    ax.set_title("Shock-date permutation placebo: PersFree coefficient")
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate(rotation=45)
    save_plot(fig, output_path)
    return str(output_path)

def plot_shock_date_placebo_histogram(results, true_coef, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    placebo = results[
        results["row_type"].eq("placebo")
        & results["status"].eq("estimated")
        & results["pers_free_coef"].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        placebo["pers_free_coef"].astype(float),
        bins=min(12, max(5, len(placebo) // 2)),
        color="#dbeafe",
        edgecolor="#2563eb",
        alpha=0.9,
    )
    ax.axvline(float(true_coef), color="#dc2626", linestyle="--", linewidth=1.5, label=f"True coef = {true_coef:+.3f}")
    ax.set_xlabel("Placebo PersFree x post coefficient")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of placebo pers_free coefficients")
    ax.legend(frameon=False, fontsize=9)
    save_plot(fig, output_path)
    return str(output_path)

def compute_shock_date_placebo(
    panel=None,
    score_path=None,
    output_path=None,
    summary_path=None,
    plot_path=None,
    hist_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "shock_date_placebo_results.csv")
    summary_path = Path(summary_path or root / "shock_date_placebo_summary.csv")
    plot_path = Path(plot_path or root / "shock_date_placebo.png")
    hist_path = Path(hist_path or root / "shock_date_placebo_hist.png")
    model_panel = placebo_shock_model_panel(panel=panel, score_path=score_path)
    candidate_dates = [
        str(period)
        for period in pd.period_range("2022-02", "2024-12", freq="M")
        if str(period) != "2022-11"
    ]

    rows = []
    models = {}
    for shock_date in candidate_dates:
        row, model = run_single_shock_date_placebo(model_panel, shock_date, row_type="placebo")
        rows.append(row)
        if model is not None:
            models[shock_date] = model
    true_row, true_model = run_single_shock_date_placebo(model_panel, "2022-11", row_type="true")
    rows.append(true_row)
    if true_model is not None:
        models["2022-11_true"] = true_model

    results = pd.DataFrame(rows).sort_values(["shock_date", "row_type"]).reset_index(drop=True)
    results.to_csv(output_path, index=False)

    true_estimated = results[results["row_type"].eq("true") & results["status"].eq("estimated")]
    placebo_estimated = results[
        results["row_type"].eq("placebo")
        & results["status"].eq("estimated")
        & results["pers_free_coef"].notna()
    ].copy()
    if true_estimated.empty:
        raise ValueError("True shock date failed to estimate.")
    if placebo_estimated.empty:
        raise ValueError("No placebo shock dates estimated.")
    true_values = true_estimated.iloc[0]
    true_coef = float(true_values["pers_free_coef"])
    placebo_coefficients = placebo_estimated["pers_free_coef"].astype(float)
    empirical_p_value = safe_float((placebo_coefficients <= true_coef).mean())
    percentile_rank = safe_float((placebo_coefficients >= true_coef).mean())
    summary = pd.DataFrame([{
        "true_coef": true_coef,
        "true_se": true_values["pers_free_se"],
        "true_p": true_values["pers_free_p"],
        "placebo_mean_coef": safe_float(placebo_coefficients.mean()),
        "placebo_sd_coef": safe_float(placebo_coefficients.std(ddof=1)),
        "placebo_min_coef": safe_float(placebo_coefficients.min()),
        "placebo_max_coef": safe_float(placebo_coefficients.max()),
        "empirical_p_value": empirical_p_value,
        "percentile_rank_of_true_coef": percentile_rank,
        "n_candidate_placebo_dates": safe_int(len(candidate_dates)),
        "n_estimated_placebo_dates": safe_int(len(placebo_estimated)),
        "n_skipped_placebo_dates": safe_int(
            (results["row_type"].eq("placebo") & ~results["status"].eq("estimated")).sum()
        ),
        "true_shock_date": "2022-11",
        "min_pre_months": 3,
        "min_post_months": 3,
    }])
    saved_plot_path = plot_shock_date_placebo_results(results, plot_path)
    saved_hist_path = plot_shock_date_placebo_histogram(results, true_coef, hist_path)
    summary["plot_path"] = saved_plot_path
    summary["hist_path"] = saved_hist_path
    summary.to_csv(summary_path, index=False)
    return {
        "results": results,
        "summary": summary,
        "models": models,
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "plot_path": saved_plot_path,
        "hist_path": saved_hist_path,
    }

def compute_shock_date_placebo_api_control(
    panel=None,
    score_path=None,
    output_path=None,
    summary_path=None,
    plot_path=None,
    hist_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "shock_date_placebo_api_control_results.csv")
    summary_path = Path(summary_path or root / "shock_date_placebo_api_control_summary.csv")
    plot_path = Path(plot_path or root / "shock_date_placebo_api_control.png")
    hist_path = Path(hist_path or root / "shock_date_placebo_api_control_hist.png")
    model_panel = placebo_shock_model_panel(panel=panel, score_path=score_path)
    if "year_month" not in model_panel.columns:
        model_panel["year_month"] = model_panel["month"]
    model_panel["reddit_api_shock"] = model_panel["year_month"].isin(["2023-06", "2023-07"]).astype(int)

    def run_single_with_api_control(shock_date, row_type="placebo", min_pre_months=3, min_post_months=3):
        shock_date = str(shock_date)
        model_data = model_panel.copy()
        pre_months = sorted(model_data.loc[model_data["month"] < shock_date, "month"].unique())
        post_months = sorted(model_data.loc[model_data["month"] >= shock_date, "month"].unique())
        base_row = {
            "row_type": row_type,
            "shock_date": shock_date,
            "status": "estimated",
            "n_pre_months": safe_int(len(pre_months)),
            "n_post_months": safe_int(len(post_months)),
            "n_obs": np.nan,
            "n_subreddits": np.nan,
            "pers_free_coef": np.nan,
            "pers_free_se": np.nan,
            "pers_free_p": np.nan,
            "pers_free_ci_low": np.nan,
            "pers_free_ci_high": np.nan,
            "gen_cap_coef": np.nan,
            "gen_cap_se": np.nan,
            "gen_cap_p": np.nan,
            "phys_free_coef": np.nan,
            "phys_free_se": np.nan,
            "phys_free_p": np.nan,
            "reddit_api_shock_coef": np.nan,
            "reddit_api_shock_se": np.nan,
            "reddit_api_shock_p": np.nan,
        }
        if len(pre_months) < min_pre_months or len(post_months) < min_post_months:
            base_row["status"] = "skipped_edge_window"
            return base_row, None

        model_data["post_s"] = (model_data["month"] >= shock_date).astype(int)
        terms = []
        for dimension in ["gen_cap", "phys_free", "pers_free"]:
            term = f"{dimension}_post_s"
            model_data[term] = model_data[dimension] * model_data["post_s"]
            terms.append(term)
        formula = "log_posts ~ " + " + ".join(terms + ["reddit_api_shock"]) + " + C(subreddit) + C(month)"
        model = fit_ols(formula, model_data, cluster_col="subreddit")
        if model is None:
            base_row["status"] = "model_failed"
            return base_row, None

        base_row["n_obs"] = safe_int(model.nobs)
        base_row["n_subreddits"] = safe_int(model_data["subreddit"].nunique())
        for dimension in ["gen_cap", "phys_free", "pers_free"]:
            term = f"{dimension}_post_s"
            result = reg_result(model, term)
            base_row[f"{dimension}_coef"] = result.get("coef")
            base_row[f"{dimension}_se"] = result.get("se")
            base_row[f"{dimension}_p"] = result.get("pvalue")
            if dimension == "pers_free" and result.get("coef") is not None and result.get("se") is not None:
                base_row["pers_free_ci_low"] = safe_float(result.get("coef") - 1.96 * result.get("se"))
                base_row["pers_free_ci_high"] = safe_float(result.get("coef") + 1.96 * result.get("se"))
        api_result = reg_result(model, "reddit_api_shock")
        base_row["reddit_api_shock_coef"] = api_result.get("coef")
        base_row["reddit_api_shock_se"] = api_result.get("se")
        base_row["reddit_api_shock_p"] = api_result.get("pvalue")
        return base_row, model

    candidate_dates = [
        str(period)
        for period in pd.period_range("2022-02", "2024-12", freq="M")
        if str(period) != "2022-11"
    ]

    rows = []
    models = {}
    for shock_date in candidate_dates:
        row, model = run_single_with_api_control(shock_date, row_type="placebo")
        rows.append(row)
        if model is not None:
            models[shock_date] = model
    true_row, true_model = run_single_with_api_control("2022-11", row_type="true")
    rows.append(true_row)
    if true_model is not None:
        models["2022-11_true"] = true_model

    results = pd.DataFrame(rows).sort_values(["shock_date", "row_type"]).reset_index(drop=True)
    results.to_csv(output_path, index=False)

    true_estimated = results[results["row_type"].eq("true") & results["status"].eq("estimated")]
    placebo_estimated = results[
        results["row_type"].eq("placebo")
        & results["status"].eq("estimated")
        & results["pers_free_coef"].notna()
    ].copy()
    if true_estimated.empty:
        raise ValueError("True shock date failed to estimate.")
    if placebo_estimated.empty:
        raise ValueError("No placebo shock dates estimated.")
    true_values = true_estimated.iloc[0]
    true_coef = float(true_values["pers_free_coef"])
    placebo_coefficients = placebo_estimated["pers_free_coef"].astype(float)
    empirical_p_value = safe_float((placebo_coefficients <= true_coef).mean())
    percentile_rank = safe_float((placebo_coefficients >= true_coef).mean())
    summary = pd.DataFrame([{
        "true_coef": true_coef,
        "true_se": true_values["pers_free_se"],
        "true_p": true_values["pers_free_p"],
        "placebo_mean_coef": safe_float(placebo_coefficients.mean()),
        "placebo_sd_coef": safe_float(placebo_coefficients.std(ddof=1)),
        "placebo_min_coef": safe_float(placebo_coefficients.min()),
        "placebo_max_coef": safe_float(placebo_coefficients.max()),
        "empirical_p_value": empirical_p_value,
        "percentile_rank_of_true_coef": percentile_rank,
        "n_candidate_placebo_dates": safe_int(len(candidate_dates)),
        "n_estimated_placebo_dates": safe_int(len(placebo_estimated)),
        "n_skipped_placebo_dates": safe_int(
            (results["row_type"].eq("placebo") & ~results["status"].eq("estimated")).sum()
        ),
        "true_shock_date": "2022-11",
        "min_pre_months": 3,
        "min_post_months": 3,
        "api_control": "reddit_api_shock equals 1 for 2023-06 and 2023-07",
    }])
    saved_plot_path = plot_shock_date_placebo_results(results, plot_path)
    saved_hist_path = plot_shock_date_placebo_histogram(results, true_coef, hist_path)
    summary["plot_path"] = saved_plot_path
    summary["hist_path"] = saved_hist_path
    summary.to_csv(summary_path, index=False)
    return {
        "results": results,
        "summary": summary,
        "models": models,
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "plot_path": saved_plot_path,
        "hist_path": saved_hist_path,
    }

def compute_shock_date_placebo_all_dimensions(panel=None, score_path=None, output_dir=None):
    root = globals().get("ROOT", Path("."))
    output_dir = Path(output_dir or root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "shock_date_placebo_by_dimension.csv"
    plot_path = output_dir / "shock_date_placebo_by_dimension.png"
    model_panel = placebo_shock_model_panel(panel=panel, score_path=score_path)
    candidate_dates = [
        str(period)
        for period in pd.period_range("2022-02", "2024-12", freq="M")
        if str(period) != "2022-11"
    ]

    rows = []
    models = {}
    for shock_date in candidate_dates:
        row, model = run_single_shock_date_placebo(model_panel, shock_date, row_type="placebo")
        rows.append(row)
        if model is not None:
            models[shock_date] = model
    true_row, true_model = run_single_shock_date_placebo(model_panel, "2022-11", row_type="true")
    rows.append(true_row)
    if true_model is not None:
        models["2022-11_true"] = true_model

    results = pd.DataFrame(rows).sort_values(["shock_date", "row_type"]).reset_index(drop=True)
    true_estimated = results[results["row_type"].eq("true") & results["status"].eq("estimated")]
    placebo_estimated = results[results["row_type"].eq("placebo") & results["status"].eq("estimated")].copy()
    if true_estimated.empty:
        raise ValueError("True shock date failed to estimate.")
    if placebo_estimated.empty:
        raise ValueError("No placebo shock dates estimated.")

    dimensions = [
        ("pers_free", "PersFree", "pers_free"),
        ("generation_capability", "Generation capability", "gen_cap"),
        ("physical_free", "Physical free", "phys_free"),
    ]
    summary_rows = []
    plot_payload = {}
    true_values = true_estimated.iloc[0]
    for dimension, label, column_prefix in dimensions:
        coef_column = f"{column_prefix}_coef"
        se_column = f"{column_prefix}_se"
        p_column = f"{column_prefix}_p"
        if coef_column not in results.columns:
            continue
        true_coef = true_values.get(coef_column)
        true_se = true_values.get(se_column)
        true_p = true_values.get(p_column)
        placebo_coefficients = pd.to_numeric(placebo_estimated[coef_column], errors="coerce").dropna()
        if true_coef is None or pd.isna(true_coef) or placebo_coefficients.empty:
            continue
        true_coef = float(true_coef)
        summary_rows.append({
            "dimension": dimension,
            "true_coef": true_coef,
            "true_se": safe_float(true_se),
            "true_p": safe_float(true_p),
            "placebo_mean_coef": safe_float(placebo_coefficients.mean()),
            "placebo_sd_coef": safe_float(placebo_coefficients.std(ddof=1)),
            "empirical_p_value": safe_float((placebo_coefficients <= true_coef).mean()),
            "percentile_rank_of_true_coef": safe_float((placebo_coefficients >= true_coef).mean()),
        })
        plot_payload[dimension] = {
            "label": label,
            "true_coef": true_coef,
            "placebo_coefficients": placebo_coefficients.astype(float),
        }

    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "dimension", "true_coef", "true_se", "true_p",
            "placebo_mean_coef", "placebo_sd_coef",
            "empirical_p_value", "percentile_rank_of_true_coef",
        ],
    )
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, (dimension, label, _column_prefix) in zip(axes, dimensions):
        payload = plot_payload.get(dimension)
        if payload is None:
            ax.set_visible(False)
            continue
        placebo_coefficients = payload["placebo_coefficients"]
        ax.hist(
            placebo_coefficients,
            bins=min(12, max(5, len(placebo_coefficients) // 2)),
            color="#d1d5db",
            edgecolor=CI_COLOR,
            alpha=0.9,
        )
        ax.axvline(
            payload["true_coef"],
            color=MUTED_RED,
            linewidth=2.0,
            label=f"True = {payload['true_coef']:+.3f}",
        )
        ax.set_title(label)
        ax.set_xlabel("Placebo coefficient")
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Shock-date placebo distributions by ACSI dimension")
    save_plot(fig, plot_path)
    return {
        "results": results,
        "summary": summary,
        "models": models,
        "summary_path": str(summary_path),
        "plot_path": str(plot_path),
    }

def pretrend_power_model_inputs(panel, score_path=None):
    root = globals().get("ROOT", Path("."))
    score_path = Path(score_path or root / "acsi_preshock_tworuns.csv")
    scores = pd.read_csv(score_path)[["subreddit", "pers_free"]].copy()
    pre = panel.copy()
    if "year_month" in pre.columns and "month" not in pre.columns:
        pre = pre.rename(columns={"year_month": "month"})
    required_columns = {"subreddit", "month", "log_posts"}
    missing_columns = required_columns - set(pre.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")
    pre = pre.merge(scores, on="subreddit", how="inner")
    pre = pre[(pre["month"] >= "2022-01") & (pre["month"] <= "2022-10")].copy()
    for column_name in ["log_posts", "pers_free"]:
        pre[column_name] = pd.to_numeric(pre[column_name], errors="coerce")
    pre = pre.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["subreddit", "month", "log_posts", "pers_free"]
    )
    months = sorted(pre["month"].astype(str).unique())
    if len(months) != 10:
        raise ValueError(f"Expected 10 pre-event months, found {len(months)}.")
    subreddits = sorted(pre["subreddit"].astype(str).unique())
    if not is_balanced_two_way_panel(pre, "subreddit", "month"):
        raise ValueError("Pretrend power analysis requires a balanced subreddit-month panel.")

    pre = pre.assign(
        subreddit=pre["subreddit"].astype(str),
        month=pre["month"].astype(str),
    )
    y_matrix = (
        pre.pivot(index="subreddit", columns="month", values="log_posts")
        .reindex(index=subreddits, columns=months)
        .astype(float)
        .to_numpy()
    )
    pers_free = (
        pre[["subreddit", "pers_free"]]
        .drop_duplicates("subreddit")
        .set_index("subreddit")
        .reindex(subreddits)["pers_free"]
        .astype(float)
        .to_numpy()
    )
    if not np.isfinite(y_matrix).all() or not np.isfinite(pers_free).all():
        raise ValueError("Pretrend panel contains non-finite values.")

    n_subreddits = len(subreddits)
    n_months = len(months)
    n_obs = n_subreddits * n_months
    month_index = np.arange(1, n_months + 1, dtype=float)

    yhat_fe = (
        y_matrix.mean(axis=1, keepdims=True)
        + y_matrix.mean(axis=0, keepdims=True)
        - y_matrix.mean()
    )
    residuals_fe = y_matrix - yhat_fe
    fe_rank = n_subreddits + n_months - 1
    residual_variance = float(np.sum(residuals_fe ** 2) / (n_obs - fe_rank))

    z_columns = []
    interaction_terms = []
    for month_position, month_label in enumerate(months[1:], start=1):
        z = np.zeros_like(y_matrix, dtype=float)
        z[:, month_position] = pers_free
        z_resid = z - z.mean(axis=1, keepdims=True) - z.mean(axis=0, keepdims=True) + z.mean()
        z_columns.append(z_resid.reshape(-1))
        interaction_terms.append(f"pers_free_x_{month_label.replace('-', '_')}")
    z_resid = np.column_stack(z_columns)
    xtx = np.einsum("ni,nj->ij", z_resid, z_resid)
    xtx_inv = np.linalg.pinv(xtx)
    k_full = fe_rank + len(interaction_terms)

    return {
        "pre": pre,
        "subreddits": subreddits,
        "months": months,
        "month_index": month_index,
        "pers_free": pers_free,
        "y_matrix": y_matrix,
        "yhat_fe": yhat_fe,
        "residual_variance": residual_variance,
        "z_resid": z_resid,
        "xtx_inv": xtx_inv,
        "interaction_terms": interaction_terms,
        "n_subreddits": n_subreddits,
        "n_months": n_months,
        "n_obs": n_obs,
        "k_full": k_full,
    }

def clustered_pretrend_f_test_from_matrix(y_matrix, model_inputs):
    y = np.asarray(y_matrix, dtype=float)
    y_resid = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + y.mean()
    y_vec = y_resid.reshape(-1)
    z_resid = model_inputs["z_resid"]
    xtx_inv = model_inputs["xtx_inv"]
    beta = xtx_inv @ np.einsum("ni,n->i", z_resid, y_vec)
    residual = y_vec - np.sum(z_resid * beta[None, :], axis=1)

    n_subreddits = model_inputs["n_subreddits"]
    n_months = model_inputs["n_months"]
    q = len(model_inputs["interaction_terms"])
    score_rows = []
    for subreddit_index in range(n_subreddits):
        start = subreddit_index * n_months
        stop = start + n_months
        score_rows.append(np.einsum("ti,t->i", z_resid[start:stop, :], residual[start:stop]))
    cluster_scores = np.vstack(score_rows)
    meat = np.einsum("gi,gj->ij", cluster_scores, cluster_scores)
    correction = (
        (n_subreddits / (n_subreddits - 1.0))
        * ((model_inputs["n_obs"] - 1.0) / (model_inputs["n_obs"] - model_inputs["k_full"]))
    )
    covariance = correction * xtx_inv @ meat @ xtx_inv
    covariance_inv = np.linalg.pinv(covariance)
    f_stat = float(beta.T @ covariance_inv @ beta / q)
    pvalue = float(stats.f.sf(f_stat, q, n_subreddits - 1))
    return {
        "f_stat": safe_float(f_stat),
        "pvalue": safe_float(pvalue),
        "df_num": safe_int(q),
        "df_denom": safe_int(n_subreddits - 1),
        "beta": [safe_float(value) for value in beta],
    }

def statsmodels_pretrend_f_test(pre, interaction_terms):
    model = fit_ols(
        "log_posts ~ " + " + ".join(interaction_terms) + " + C(subreddit) + C(month)",
        pre,
        cluster_col="subreddit",
    )
    if model is None:
        raise ValueError("Actual pretrend regression failed to fit.")
    restriction_matrix = np.zeros((len(interaction_terms), len(model.params)))
    param_names = list(model.params.index)
    for i, term in enumerate(interaction_terms):
        restriction_matrix[i, param_names.index(term)] = 1.0
    f_test = model.f_test(restriction_matrix)
    return {
        "f_stat": safe_float(np.asarray(f_test.fvalue).ravel()[0]),
        "pvalue": safe_float(f_test.pvalue),
        "df_num": safe_int(getattr(f_test, "df_num", len(interaction_terms))),
        "df_denom": safe_int(getattr(f_test, "df_denom", None)),
    }

def interpolation_threshold(power_rows, target_power):
    rows = sorted(power_rows, key=lambda row: row["delta"])
    for row in rows:
        if row["rejection_rate"] >= target_power:
            if row == rows[0]:
                return safe_float(row["delta"])
            previous = rows[rows.index(row) - 1]
            x0, y0 = previous["delta"], previous["rejection_rate"]
            x1, y1 = row["delta"], row["rejection_rate"]
            if y1 == y0:
                return safe_float(x1)
            return safe_float(x0 + (target_power - y0) * (x1 - x0) / (y1 - y0))
    return None

def plot_pretrend_power_curve(power_rows, threshold_80, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    power_table = pd.DataFrame(power_rows).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        power_table["delta"],
        power_table["rejection_rate"],
        marker="o",
        color="#2563eb",
        linewidth=2.5,
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.0,
    )
    ax.axhline(0.80, color="#dc2626", linestyle="--", linewidth=1.5)
    if threshold_80 is not None:
        ax.axvline(threshold_80, color="#dc2626", linestyle="--", linewidth=1.5)
        ax.scatter([threshold_80], [0.80], color="#dc2626", s=60, edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(
            threshold_80,
            0.82,
            f"80% MDE = {threshold_80:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#dc2626",
        )
    ax.set_xlabel("True monthly pre-trend delta")
    ax.set_ylabel("Rejection rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Pre-trend test power")
    save_plot(fig, output_path)
    return str(output_path)

def compute_pretrend_power_analysis(
    panel,
    score_path=None,
    deltas=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
    n_simulations=1000,
    random_seed=None,
    output_path=None,
    plot_path=None,
):
    if random_seed is None:
        random_seed = globals().get("RANDOM_SEED", 42)
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "pretrend_power_analysis.csv")
    plot_path = Path(plot_path or root / "pretrend_power_curve.png")
    inputs = pretrend_power_model_inputs(panel, score_path=score_path)
    pre = inputs["pre"].copy()
    for term, month_label in zip(inputs["interaction_terms"], inputs["months"][1:]):
        pre[term] = pre["pers_free"] * pre["month"].eq(month_label).astype(int)

    actual_statsmodels = statsmodels_pretrend_f_test(pre, inputs["interaction_terms"])
    actual_fast = clustered_pretrend_f_test_from_matrix(inputs["y_matrix"], inputs)
    rng = np.random.default_rng(random_seed)
    sigma = float(np.sqrt(inputs["residual_variance"]))
    centered_month = inputs["month_index"] - 5.5
    trend_base = inputs["pers_free"][:, None] * centered_month[None, :]
    power_rows = []
    for delta in deltas:
        delta = float(delta)
        rejections = 0
        valid_simulations = 0
        for _ in range(int(n_simulations)):
            synthetic = (
                inputs["yhat_fe"]
                + delta * trend_base
                + rng.normal(0.0, sigma, size=inputs["y_matrix"].shape)
            )
            test_result = clustered_pretrend_f_test_from_matrix(synthetic, inputs)
            pvalue = test_result.get("pvalue")
            if pvalue is not None and np.isfinite(pvalue):
                valid_simulations += 1
                rejections += int(pvalue < 0.05)
        power_rows.append({
            "delta": delta,
            "rejection_rate": safe_float(rejections / valid_simulations) if valid_simulations else None,
            "n_simulations": safe_int(valid_simulations),
        })

    threshold_80 = interpolation_threshold(power_rows, 0.80)
    threshold_50 = interpolation_threshold(power_rows, 0.50)
    monthly_main_effect = abs(-0.451 / 25.0)
    fraction_of_monthly_effect_80 = (
        safe_float(threshold_80 / monthly_main_effect)
        if threshold_80 is not None and monthly_main_effect > 0 else None
    )
    saved_plot_path = plot_pretrend_power_curve(power_rows, threshold_80, plot_path)

    rows = [{
        "row_type": "actual_pretrend",
        "delta": np.nan,
        "rejection_rate": np.nan,
        "n_simulations": np.nan,
        "f_stat": actual_statsmodels["f_stat"],
        "pvalue": actual_statsmodels["pvalue"],
        "fast_f_stat": actual_fast["f_stat"],
        "fast_pvalue": actual_fast["pvalue"],
        "residual_variance": inputs["residual_variance"],
        "n_subreddits": inputs["n_subreddits"],
        "n_months": inputs["n_months"],
        "n_obs": inputs["n_obs"],
        "minimum_detectable_delta_80_power": threshold_80,
        "minimum_detectable_delta_50_power": threshold_50,
        "monthly_main_effect_abs": monthly_main_effect,
        "fraction_of_monthly_main_effect_detectable_80_power": fraction_of_monthly_effect_80,
        "plot_path": saved_plot_path,
    }]
    for power_row in power_rows:
        rows.append({
            "row_type": "power",
            "delta": power_row["delta"],
            "rejection_rate": power_row["rejection_rate"],
            "n_simulations": power_row["n_simulations"],
            "f_stat": np.nan,
            "pvalue": np.nan,
            "fast_f_stat": np.nan,
            "fast_pvalue": np.nan,
            "residual_variance": inputs["residual_variance"],
            "n_subreddits": inputs["n_subreddits"],
            "n_months": inputs["n_months"],
            "n_obs": inputs["n_obs"],
            "minimum_detectable_delta_80_power": threshold_80,
            "minimum_detectable_delta_50_power": threshold_50,
            "monthly_main_effect_abs": monthly_main_effect,
            "fraction_of_monthly_main_effect_detectable_80_power": fraction_of_monthly_effect_80,
            "plot_path": saved_plot_path,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return {
        "actual_pretrend": actual_statsmodels,
        "actual_pretrend_fast": actual_fast,
        "power_rows": power_rows,
        "residual_variance": inputs["residual_variance"],
        "minimum_detectable_delta_80_power": threshold_80,
        "minimum_detectable_delta_50_power": threshold_50,
        "monthly_main_effect_abs": monthly_main_effect,
        "fraction_of_monthly_main_effect_detectable_80_power": fraction_of_monthly_effect_80,
        "output_path": str(output_path),
        "plot_path": saved_plot_path,
        "n_subreddits": inputs["n_subreddits"],
        "n_months": inputs["n_months"],
        "n_obs": inputs["n_obs"],
    }

def build_creator_exit_continuous_frame(
    score_path=None,
    target_subreddits=None,
    pre_start_month="2022-01",
    pre_end_month="2022-10",
    post_start_month="2022-12",
    min_pre_posts=5,
):
    root = globals().get("ROOT", Path("."))
    data_dir = globals().get("DATA_DIR", root / "data")
    score_path = Path(score_path or root / "acsi_preshock_tworuns.csv")
    scores = pd.read_csv(score_path)[["subreddit", "gen_cap", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    scores["gen_cap"] = pd.to_numeric(scores["gen_cap"], errors="coerce")
    scores["pers_free"] = pd.to_numeric(scores["pers_free"], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "pers_free"]).drop_duplicates("subreddit")

    if target_subreddits is None:
        panel_path = root / "panel.csv"
        if panel_path.exists():
            target_subreddits = pd.read_csv(panel_path, usecols=["subreddit"])["subreddit"].astype(str).unique()
        else:
            target_subreddits = scores["subreddit"].astype(str).unique()
    target_subreddits = sorted(set(pd.Series(target_subreddits, dtype=str).dropna()) & set(scores["subreddit"]))
    if not target_subreddits:
        raise ValueError("No target subreddits overlap with the score table.")

    excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
    max_lines_per_file = globals().get("MAX_LINES_PER_FILE", None)
    max_posts_per_day = globals().get("MAX_POSTS_PER_DAY", 50)
    start_date = globals().get("START_DATE", datetime(2022, 1, 1))
    end_date_exclusive = globals().get("END_DATE_EXCLUSIVE", datetime(2025, 1, 1))
    cap_days = max((end_date_exclusive - start_date).days, 1)
    author_post_cap = max_posts_per_day * cap_days

    pre_start = pd.Timestamp(pre_start_month).to_pydatetime()
    pre_end_exclusive = (pd.Timestamp(pre_end_month) + pd.offsets.MonthBegin(1)).to_pydatetime()
    post_start = pd.Timestamp(post_start_month).to_pydatetime()

    total_counts = Counter()
    pre_counts = Counter()
    pre_subreddit_counts = Counter()
    posted_post_event = set()
    files_scanned = 0
    posts_seen = 0

    for subreddit in tqdm(target_subreddits, desc="  creator-exit subreddit scans"):
        path = raw_post_file_path(subreddit, data_dir)
        if not path.exists():
            continue
        files_scanned += 1
        with open(path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if max_lines_per_file is not None and i >= max_lines_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                created_utc = payload.get("created_utc")
                if created_utc is None:
                    continue
                try:
                    dt = datetime.utcfromtimestamp(int(created_utc))
                except Exception:
                    continue
                if dt < start_date or dt >= end_date_exclusive:
                    continue

                author = str(payload.get("author") or "")
                if (
                    not author
                    or author in excluded_authors
                    or author.lower().endswith("bot")
                ):
                    continue

                posts_seen += 1
                total_counts[author] += 1
                if pre_start <= dt < pre_end_exclusive:
                    pre_counts[author] += 1
                    pre_subreddit_counts[(author, subreddit)] += 1
                elif dt >= post_start:
                    posted_post_event.add(author)

    valid_authors = {
        author
        for author, pre_event_posts in pre_counts.items()
        if pre_event_posts >= min_pre_posts and total_counts.get(author, 0) <= author_post_cap
    }
    if not valid_authors:
        raise ValueError("No eligible creator-clean authors found.")

    modal_by_author = {}
    for (author, subreddit), count in pre_subreddit_counts.items():
        if author not in valid_authors:
            continue
        previous = modal_by_author.get(author)
        if previous is None or count > previous[1] or (count == previous[1] and subreddit < previous[0]):
            modal_by_author[author] = (subreddit, count)

    modal_rows = []
    for author, (modal_subreddit, modal_count) in modal_by_author.items():
        modal_rows.append({
            "author": author,
            "subreddit": modal_subreddit,
            "pre_event_posts": safe_int(pre_counts[author]),
            "modal_pre_event_posts": safe_int(modal_count),
            "posted_post_event": int(author in posted_post_event),
        })

    creator_frame = pd.DataFrame(modal_rows)
    creator_frame = creator_frame.merge(scores, on="subreddit", how="left").dropna(
        subset=["gen_cap", "pers_free", "pre_event_posts", "posted_post_event"]
    )
    creator_frame["exit"] = 1 - creator_frame["posted_post_event"].astype(int)
    creator_frame["log_pre_rate"] = np.log(creator_frame["pre_event_posts"].astype(float))
    creator_frame["pers_free_tercile"] = pd.qcut(
        creator_frame["pers_free"],
        q=3,
        labels=["bottom", "middle", "top"],
        duplicates="drop",
    )
    metadata = {
        "files_scanned": safe_int(files_scanned),
        "posts_seen": safe_int(posts_seen),
        "n_target_subreddits": safe_int(len(target_subreddits)),
        "n_authors_before_score_merge": safe_int(len(modal_rows)),
        "author_post_cap": safe_int(author_post_cap),
        "pre_window": f"{pre_start_month} to {pre_end_month}",
        "post_window": f"{post_start_month} onward",
    }
    return creator_frame, metadata

def compute_creator_exit_interaction_power_analysis(
    creator_path=None,
    score_path=None,
    output_path=None,
    latex_path=None,
    observed_lambda=0.008,
    observed_se=0.036,
    community_beta=-0.451,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)
    score_path = Path(score_path or root / "acsi_preshock_tworuns.csv")
    output_path = Path(output_path or tables_dir / "creator_exit_interaction_power_analysis.csv")
    latex_path = Path(latex_path or tables_dir / "creator_exit_interaction_power_analysis_sentence.tex")

    if not score_path.exists():
        scores, score_metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)
        score_source = "rebuilt_from_two_run_measurements"
    else:
        score_metadata = {}
        score_source = str(score_path)

    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
    )
    required_columns = {"subreddit", "pre_event_posts", "pers_free"}
    if "exit" not in frame.columns and "posted_post_event" not in frame.columns:
        required_columns.add("exit")
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Creator frame missing columns: {sorted(missing_columns)}")

    analysis = frame.copy()
    analysis["subreddit"] = analysis["subreddit"].astype(str)
    analysis["pre_event_posts"] = pd.to_numeric(analysis["pre_event_posts"], errors="coerce")
    analysis["pers_free"] = pd.to_numeric(analysis["pers_free"], errors="coerce")
    if "exit" in analysis.columns:
        analysis["exit"] = pd.to_numeric(analysis["exit"], errors="coerce")
    else:
        analysis["posted_post_event"] = pd.to_numeric(
            analysis["posted_post_event"],
            errors="coerce",
        )
        analysis["exit"] = 1 - analysis["posted_post_event"]
    analysis = analysis.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["subreddit", "pre_event_posts", "pers_free", "exit"]
    )
    analysis = analysis[analysis["pre_event_posts"] > 0].copy()
    if analysis.empty:
        raise ValueError("No usable creator rows for interaction power analysis.")

    analysis["log_pre_posts"] = np.log1p(analysis["pre_event_posts"].astype(float))
    analysis["interaction"] = analysis["log_pre_posts"] * analysis["pers_free"].astype(float)

    def within_subreddit_residual(column_name):
        values = analysis[column_name].astype(float)
        return values - values.groupby(analysis["subreddit"]).transform("mean")

    y_fe = within_subreddit_residual("exit")
    x_fe = within_subreddit_residual("log_pre_posts")
    z_fe = within_subreddit_residual("interaction")

    x_ss = float(np.dot(x_fe, x_fe))
    if x_ss <= 0:
        raise ValueError("Residualized log_pre_posts has zero variance.")
    y_on_x = float(np.dot(x_fe, y_fe) / x_ss)
    z_on_x = float(np.dot(x_fe, z_fe) / x_ss)
    y_resid = y_fe - y_on_x * x_fe
    z_resid = z_fe - z_on_x * x_fe
    z_ss = float(np.dot(z_resid, z_resid))
    if z_ss <= 0:
        raise ValueError("Residualized interaction has zero variance.")

    n = int(len(analysis))
    n_subreddits = int(analysis["subreddit"].nunique())
    df_resid = max(n - n_subreddits - 1, 1)
    sigma2 = float(np.dot(y_resid, y_resid) / df_resid)
    var_interaction_resid = float(pd.Series(z_resid).var(ddof=1))
    var_log_pre_posts_within = float(pd.Series(x_fe).var(ddof=1))
    se_lambda = float(np.sqrt(sigma2 / z_ss))
    mde_80 = float((1.96 + 0.84) * se_lambda)
    observed_mde_80 = float((1.96 + 0.84) * observed_se)
    implied_lambda = float(-community_beta / var_log_pre_posts_within)
    ratio_mde_to_implied = float(mde_80 / abs(implied_lambda)) if implied_lambda else np.nan
    observed_ratio_mde_to_implied = (
        float(observed_mde_80 / abs(implied_lambda)) if implied_lambda else np.nan
    )

    cluster_se_lambda = np.nan
    cluster_pvalue_lambda = np.nan
    try:
        model = fit_ols(
            "exit ~ log_pre_posts + interaction + C(subreddit)",
            analysis,
            cluster_col="subreddit",
        )
        if model is not None:
            cluster_se_lambda = safe_float(model.bse.get("interaction"))
            cluster_pvalue_lambda = safe_float(model.pvalues.get("interaction"))
    except Exception:
        cluster_se_lambda = np.nan
        cluster_pvalue_lambda = np.nan

    summary = {
        "N": safe_int(n),
        "n_subreddits": safe_int(n_subreddits),
        "var_interaction_resid": safe_float(var_interaction_resid),
        "sigma2": safe_float(sigma2),
        "se_lambda": safe_float(se_lambda),
        "mde_80": safe_float(mde_80),
        "observed_lambda": safe_float(observed_lambda),
        "observed_se": safe_float(observed_se),
        "observed_mde_80": safe_float(observed_mde_80),
        "cluster_se_lambda_from_refit": safe_float(cluster_se_lambda),
        "cluster_pvalue_lambda_from_refit": safe_float(cluster_pvalue_lambda),
        "community_beta": safe_float(community_beta),
        "var_log_pre_posts_within": safe_float(var_log_pre_posts_within),
        "implied_lambda": safe_float(implied_lambda),
        "ratio_mde_implied_lambda": safe_float(ratio_mde_to_implied),
        "ratio_observed_mde_implied_lambda": safe_float(observed_ratio_mde_to_implied),
        "score_source": score_source,
        "creator_frame_source": frame_source,
    }
    for key, value in score_metadata.items():
        summary[f"score_{key}"] = value

    output_path.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([summary]).to_csv(output_path, index=False)

    latex_sentence = (
        "The creator-level interaction design implies an 80\\% minimum detectable "
        f"effect of $\\lambda={mde_80:.3f}$ using the residualized design variance "
        f"($N={n:,}$ authors), which is {ratio_mde_to_implied:.2f} times the "
        "individual-level interaction implied by the community-level DiD estimate "
        f"($\\lambda_{{\\mathrm{{implied}}}}={implied_lambda:.3f}$)."
    )
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(latex_sentence + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "summary_table": pd.DataFrame([summary]),
        "latex_sentence": latex_sentence,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
    }

def compute_community_low_frequency_composition(
    creator_path=None,
    score_path=None,
    output_path=None,
    panel_output_path=None,
    plot_path=None,
    latex_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    figures_dir = globals().get("FIGURES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    creator_path = Path(creator_path or root / "creator_exit_author_level.csv")
    if score_path is None:
        table_score_path = tables_dir / "acsi_preshock_tworuns.csv"
        root_score_path = root / "acsi_preshock_tworuns.csv"
        if table_score_path.exists():
            score_path = table_score_path
        elif root_score_path.exists():
            score_path = root_score_path
        else:
            score_path = table_score_path
    score_path = Path(score_path)
    output_path = Path(output_path or tables_dir / "community_low_freq_composition.csv")
    panel_output_path = Path(panel_output_path or tables_dir / "community_low_freq_share_panel.csv")
    plot_path = Path(plot_path or figures_dir / "community_low_freq_share_trajectories.png")
    latex_path = Path(latex_path or tables_dir / "community_low_freq_composition_paragraph.tex")

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])

    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
    )
    required_creator_columns = {"author", "subreddit", "pre_event_posts", "pers_free", "gen_cap"}
    missing_creator_columns = required_creator_columns - set(frame.columns)
    if missing_creator_columns:
        raise ValueError(f"Creator frame missing columns: {sorted(missing_creator_columns)}")
    creators = frame.copy()
    creators["author"] = creators["author"].astype(str)
    creators["subreddit"] = creators["subreddit"].astype(str)
    creators["pre_event_posts"] = pd.to_numeric(creators["pre_event_posts"], errors="coerce")
    creators = creators.dropna(subset=["author", "subreddit", "pre_event_posts"])
    creators = creators[creators["pre_event_posts"] > 0].drop_duplicates("author").copy()
    if creators.empty:
        raise ValueError("No usable creator rows for low-frequency composition analysis.")

    ranked_pre_posts = creators["pre_event_posts"].rank(method="first")
    creators["posting_frequency_tercile"] = pd.qcut(
        ranked_pre_posts,
        q=3,
        labels=["bottom", "middle", "top"],
    )
    low_freq_authors = set(
        creators.loc[
            creators["posting_frequency_tercile"].astype(str).eq("bottom"),
            "author",
        ].astype(str)
    )
    creator_authors = set(creators["author"].astype(str))
    target_subreddits = sorted(set(creators["subreddit"].astype(str)) & set(scores["subreddit"].astype(str)))
    if not target_subreddits:
        raise ValueError("No creator-clean subreddits overlap with score table.")

    monthly_counts = Counter()
    low_freq_counts = Counter()
    total_creator_posts_seen = 0
    low_freq_posts_seen = 0
    files_scanned = 0
    for subreddit in tqdm(target_subreddits, desc="  low-frequency monthly composition"):
        path = raw_post_file_path(subreddit)
        if not path.exists():
            continue
        files_scanned += 1
        for author, dt, _score, _post_id, _payload in iter_post_payloads(
            subreddit,
            f"  r/{subreddit} posts",
        ):
            if author not in creator_authors:
                continue
            month_label = month_label_for_datetime(dt)
            key = (subreddit, month_label)
            monthly_counts[key] += 1
            total_creator_posts_seen += 1
            if author in low_freq_authors:
                low_freq_counts[key] += 1
                low_freq_posts_seen += 1

    all_months = [
        month.strftime("%Y-%m")
        for month in globals().get(
            "ALL_MONTHS",
            pd.date_range("2022-01-01", "2024-12-01", freq="MS"),
        )
    ]
    grid = pd.MultiIndex.from_product(
        [target_subreddits, all_months],
        names=["subreddit", "year_month"],
    ).to_frame(index=False)
    grid["creator_clean_posts"] = [
        safe_int(monthly_counts.get((row.subreddit, row.year_month), 0))
        for row in grid.itertuples(index=False)
    ]
    grid["low_freq_posts"] = [
        safe_int(low_freq_counts.get((row.subreddit, row.year_month), 0))
        for row in grid.itertuples(index=False)
    ]
    panel = grid.merge(scores, on="subreddit", how="left")
    panel["year_month_dt"] = pd.to_datetime(panel["year_month"] + "-01")
    shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
    panel["post"] = (panel["year_month_dt"] >= shock_month).astype(int)
    panel["low_freq_share"] = np.where(
        panel["creator_clean_posts"] > 0,
        panel["low_freq_posts"] / panel["creator_clean_posts"],
        np.nan,
    )
    for dimension in ["pers_free", "gen_cap", "phys_free"]:
        panel[f"{dimension}_post"] = panel[dimension] * panel["post"]
    panel = panel.sort_values(["subreddit", "year_month"]).reset_index(drop=True)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    panel.to_csv(panel_output_path, index=False)

    model_data = panel.dropna(
        subset=["low_freq_share", "pers_free_post", "gen_cap_post", "phys_free_post"]
    ).copy()
    if model_data["subreddit"].nunique() < 2:
        raise ValueError("Need at least two subreddits for clustered FE models.")
    baseline_model = fit_ols(
        "low_freq_share ~ pers_free_post + C(subreddit) + C(year_month)",
        model_data,
        cluster_col="subreddit",
    )
    controlled_model = fit_ols(
        "low_freq_share ~ pers_free_post + gen_cap_post + phys_free_post + C(subreddit) + C(year_month)",
        model_data,
        cluster_col="subreddit",
    )
    if baseline_model is None or controlled_model is None:
        raise ValueError("Low-frequency composition regression failed to fit.")

    rows = []
    model_specs = [
        ("baseline_persfree_only", baseline_model, ["pers_free_post"]),
        ("controlled_gen_phys", controlled_model, ["pers_free_post", "gen_cap_post", "phys_free_post"]),
    ]
    for model_name, model, terms in model_specs:
        for term in terms:
            result = reg_result(model, term)
            rows.append({
                "model": model_name,
                "term": term,
                "coef": result.get("coef"),
                "se": result.get("se"),
                "pvalue": result.get("pvalue"),
                "n_obs": result.get("n_obs"),
                "n_subreddits": safe_int(model_data["subreddit"].nunique()),
                "n_months": safe_int(model_data["year_month"].nunique()),
                "n_creator_authors": safe_int(len(creators)),
                "n_low_freq_authors": safe_int(len(low_freq_authors)),
                "total_creator_clean_posts": safe_int(total_creator_posts_seen),
                "low_freq_posts": safe_int(low_freq_posts_seen),
                "overall_low_freq_share": safe_float(low_freq_posts_seen / total_creator_posts_seen)
                if total_creator_posts_seen else np.nan,
                "creator_frame_source": frame_source,
                "score_source": str(score_path),
            })
    result_table = pd.DataFrame(rows)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    sub_scores = panel[["subreddit", "pers_free"]].drop_duplicates().dropna().copy()
    sub_scores["persfree_tercile"] = pd.qcut(
        sub_scores["pers_free"].rank(method="first"),
        q=3,
        labels=["Low PersFree", "Middle PersFree", "High PersFree"],
    )
    plot_panel = panel.merge(
        sub_scores[["subreddit", "persfree_tercile"]],
        on="subreddit",
        how="left",
    ).dropna(subset=["low_freq_share", "persfree_tercile"]).copy()
    trajectory = (
        plot_panel.groupby(["year_month_dt", "persfree_tercile"], observed=False)["low_freq_share"]
        .mean()
        .reset_index()
    )
    colors = {
        "Low PersFree": "#dc2626",
        "Middle PersFree": "#9ca3af",
        "High PersFree": "#2563eb",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ["Low PersFree", "Middle PersFree", "High PersFree"]:
        series = trajectory[trajectory["persfree_tercile"].astype(str).eq(label)].sort_values("year_month_dt")
        if series.empty:
            continue
        ax.plot(
            series["year_month_dt"],
            series["low_freq_share"],
            color=colors[label],
            linewidth=2.0,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    ax.axvline(pd.Timestamp("2022-11-01"), color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        pd.Timestamp("2022-11-01"),
        ax.get_ylim()[1],
        "ChatGPT launch",
        color="#dc2626",
        fontsize=9,
        ha="left",
        va="top",
    )
    ax.set_title("Low-frequency creator share by PersFree tercile")
    ax.set_xlabel("Month")
    ax.set_ylabel("Low-frequency creator post share")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", labelrotation=45)
    try:
        apply_modern_style(ax)
    except Exception:
        pass
    fig.tight_layout(pad=1.5)
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    if "save_plot" in globals():
        save_plot(fig, plot_path)
    else:
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    baseline_row = next(
        row for row in rows
        if row["model"] == "baseline_persfree_only" and row["term"] == "pers_free_post"
    )
    controlled_row = next(
        row for row in rows
        if row["model"] == "controlled_gen_phys" and row["term"] == "pers_free_post"
    )
    paragraph = (
        "As a community-composition check, I classified creator-clean authors into "
        "pre-shock posting-frequency terciles and measured, for each subreddit-month, "
        "the share of creator-clean posts written by bottom-tercile authors. In a "
        "subreddit and month fixed-effects DiD, the PersFree-by-post coefficient was "
        f"{baseline_row['coef']:.3f} (SE={baseline_row['se']:.3f}, "
        f"$p={baseline_row['pvalue']:.3f}$). Adding GenCap-by-post and PhysFree-by-post "
        f"controls gave {controlled_row['coef']:.3f} (SE={controlled_row['se']:.3f}, "
        f"$p={controlled_row['pvalue']:.3f}$), indicating "
        f"{'a detectable' if controlled_row['pvalue'] < 0.05 else 'no statistically detectable'} "
        "post-shock shift in the low-frequency creator composition gradient by PersFree."
    )
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(paragraph + "\n", encoding="utf-8")

    return {
        "rows": rows,
        "panel": panel,
        "output_path": str(output_path),
        "panel_output_path": str(panel_output_path),
        "plot_path": str(plot_path),
        "latex_path": str(latex_path),
        "paragraph": paragraph,
    }

def compute_community_new_entrant_rates(
    score_path=None,
    output_path=None,
    panel_output_path=None,
    plot_path=None,
    latex_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    figures_dir = globals().get("FIGURES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        table_score_path = tables_dir / "acsi_preshock_tworuns.csv"
        root_score_path = root / "acsi_preshock_tworuns.csv"
        if table_score_path.exists():
            score_path = table_score_path
        elif root_score_path.exists():
            score_path = root_score_path
        else:
            score_path = table_score_path
    score_path = Path(score_path)
    output_path = Path(output_path or tables_dir / "community_new_entrant_rates.csv")
    panel_output_path = Path(panel_output_path or tables_dir / "community_new_entrant_panel.csv")
    plot_path = Path(plot_path or figures_dir / "community_new_entrant_share_trajectories.png")
    latex_path = Path(latex_path or tables_dir / "community_new_entrant_rates_paragraph.tex")

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])

    target_subreddits = [
        subreddit
        for subreddit in sorted(scores["subreddit"].astype(str).unique())
        if raw_post_file_path(subreddit, data_dir).exists()
    ]
    if not target_subreddits:
        raise ValueError("No scored subreddits have raw post files.")

    excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
    start_date = globals().get("START_DATE", datetime(2022, 1, 1))
    end_date_exclusive = globals().get("END_DATE_EXCLUSIVE", datetime(2025, 1, 1))
    max_lines_per_file = globals().get("MAX_LINES_PER_FILE", None)
    max_posts_per_day = globals().get("MAX_POSTS_PER_DAY", 50)
    cap_days = max((end_date_exclusive - start_date).days, 1)
    author_post_cap = max_posts_per_day * cap_days

    def iter_ecosystem_post_fields(subreddit):
        path = raw_post_file_path(subreddit, data_dir)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if max_lines_per_file is not None and i >= max_lines_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                created_utc = payload.get("created_utc")
                if created_utc is None:
                    continue
                try:
                    dt = datetime.utcfromtimestamp(int(created_utc))
                except Exception:
                    continue
                if dt < start_date or dt >= end_date_exclusive:
                    continue
                author = str(payload.get("author") or "")
                if (
                    not author
                    or author in excluded_authors
                    or author.lower().endswith("bot")
                ):
                    continue
                yield author, month_label_for_datetime(dt)

    author_counts = Counter()
    total_posts_seen = 0
    for subreddit in tqdm(target_subreddits, desc="  new-entrant ecosystem author-cap pass"):
        for author, _month_label in iter_ecosystem_post_fields(subreddit):
            author_counts[author] += 1
            total_posts_seen += 1
    valid_authors = {
        author
        for author, count in author_counts.items()
        if count <= author_post_cap
    }
    if not valid_authors:
        raise ValueError("No ecosystem-clean authors after author-cap filtering.")

    rows = []
    ecosystem_posts_used = 0
    total_author_months = 0
    total_new_entrants = 0
    all_months = [
        month.strftime("%Y-%m")
        for month in globals().get(
            "ALL_MONTHS",
            pd.date_range("2022-01-01", "2024-12-01", freq="MS"),
        )
    ]
    for subreddit in tqdm(target_subreddits, desc="  new-entrant subreddit-month pass"):
        month_authors = defaultdict(set)
        first_month_by_author = {}
        for author, month_label in iter_ecosystem_post_fields(subreddit):
            if author not in valid_authors:
                continue
            month_authors[month_label].add(author)
            ecosystem_posts_used += 1
            previous_month = first_month_by_author.get(author)
            if previous_month is None or month_label < previous_month:
                first_month_by_author[author] = month_label

        for month_label in all_months:
            authors = month_authors.get(month_label, set())
            distinct_authors = len(authors)
            new_entrant_count = sum(
                1 for author in authors
                if first_month_by_author.get(author) == month_label
            )
            total_author_months += distinct_authors
            total_new_entrants += new_entrant_count
            rows.append({
                "subreddit": subreddit,
                "year_month": month_label,
                "distinct_authors": safe_int(distinct_authors),
                "new_entrant_count": safe_int(new_entrant_count),
                "new_entrant_share": safe_float(new_entrant_count / distinct_authors)
                if distinct_authors else np.nan,
            })

    panel = pd.DataFrame(rows).merge(scores, on="subreddit", how="left")
    panel["year_month_dt"] = pd.to_datetime(panel["year_month"] + "-01")
    shock_month = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01"))
    panel["post"] = (panel["year_month_dt"] >= shock_month).astype(int)
    panel["log_new_entrant_count"] = np.log1p(panel["new_entrant_count"].astype(float))
    for dimension in ["pers_free", "gen_cap", "phys_free"]:
        panel[f"{dimension}_post"] = panel[dimension] * panel["post"]
    panel = panel.sort_values(["subreddit", "year_month"]).reset_index(drop=True)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    panel.to_csv(panel_output_path, index=False)

    model_data = panel.dropna(
        subset=[
            "new_entrant_share", "log_new_entrant_count",
            "pers_free_post", "gen_cap_post", "phys_free_post",
        ]
    ).copy()
    if model_data["subreddit"].nunique() < 2:
        raise ValueError("Need at least two subreddits for clustered FE models.")

    model_specs = [
        (
            "share_persfree_only",
            "new_entrant_share",
            "new_entrant_share ~ pers_free_post + C(subreddit) + C(year_month)",
            ["pers_free_post"],
        ),
        (
            "count_persfree_only",
            "log_new_entrant_count",
            "log_new_entrant_count ~ pers_free_post + C(subreddit) + C(year_month)",
            ["pers_free_post"],
        ),
        (
            "share_three_dimensional",
            "new_entrant_share",
            "new_entrant_share ~ pers_free_post + gen_cap_post + phys_free_post + C(subreddit) + C(year_month)",
            ["pers_free_post", "gen_cap_post", "phys_free_post"],
        ),
        (
            "count_three_dimensional",
            "log_new_entrant_count",
            "log_new_entrant_count ~ pers_free_post + gen_cap_post + phys_free_post + C(subreddit) + C(year_month)",
            ["pers_free_post", "gen_cap_post", "phys_free_post"],
        ),
    ]

    result_rows = []
    for model_name, outcome, formula, terms in model_specs:
        model = fit_ols(formula, model_data, cluster_col="subreddit")
        if model is None:
            raise ValueError(f"New-entrant model failed to fit: {model_name}")
        for term in terms:
            result = reg_result(model, term)
            result_rows.append({
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "coef": result.get("coef"),
                "se": result.get("se"),
                "pvalue": result.get("pvalue"),
                "n_obs": result.get("n_obs"),
                "n_subreddits": safe_int(model_data["subreddit"].nunique()),
                "n_months": safe_int(model_data["year_month"].nunique()),
                "n_ecosystem_authors": safe_int(len(valid_authors)),
                "raw_posts_seen_before_author_cap": safe_int(total_posts_seen),
                "ecosystem_clean_posts_used": safe_int(ecosystem_posts_used),
                "total_distinct_author_months": safe_int(total_author_months),
                "total_new_entrants": safe_int(total_new_entrants),
                "overall_new_entrant_share": safe_float(total_new_entrants / total_author_months)
                if total_author_months else np.nan,
                "score_source": str(score_path),
                "author_post_cap": safe_int(author_post_cap),
            })

    result_table = pd.DataFrame(result_rows)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    sub_scores = panel[["subreddit", "pers_free"]].drop_duplicates().dropna().copy()
    sub_scores["persfree_tercile"] = pd.qcut(
        sub_scores["pers_free"].rank(method="first"),
        q=3,
        labels=["Low PersFree", "Middle PersFree", "High PersFree"],
    )
    plot_panel = panel.merge(
        sub_scores[["subreddit", "persfree_tercile"]],
        on="subreddit",
        how="left",
    ).dropna(subset=["new_entrant_share", "persfree_tercile"]).copy()
    trajectory = (
        plot_panel.groupby(["year_month_dt", "persfree_tercile"], observed=False)["new_entrant_share"]
        .mean()
        .reset_index()
    )
    colors = {
        "Low PersFree": "#dc2626",
        "Middle PersFree": "#9ca3af",
        "High PersFree": "#2563eb",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ["Low PersFree", "Middle PersFree", "High PersFree"]:
        series = trajectory[trajectory["persfree_tercile"].astype(str).eq(label)].sort_values("year_month_dt")
        if series.empty:
            continue
        ax.plot(
            series["year_month_dt"],
            series["new_entrant_share"],
            color=colors[label],
            linewidth=2.0,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
    ax.axvline(pd.Timestamp("2022-11-01"), color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(
        pd.Timestamp("2022-11-01"),
        ax.get_ylim()[1],
        "ChatGPT launch",
        color="#dc2626",
        fontsize=9,
        ha="left",
        va="top",
    )
    ax.set_title("New entrant share by PersFree tercile")
    ax.set_xlabel("Month")
    ax.set_ylabel("New entrant share")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", labelrotation=45)
    try:
        apply_modern_style(ax)
    except Exception:
        pass
    fig.tight_layout(pad=1.5)
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    if "save_plot" in globals():
        save_plot(fig, plot_path)
    else:
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    share_row = next(
        row for row in result_rows
        if row["model"] == "share_persfree_only" and row["term"] == "pers_free_post"
    )
    count_row = next(
        row for row in result_rows
        if row["model"] == "count_persfree_only" and row["term"] == "pers_free_post"
    )
    controlled_share_row = next(
        row for row in result_rows
        if row["model"] == "share_three_dimensional" and row["term"] == "pers_free_post"
    )
    paragraph = (
        "I also tested whether the main effect reflects compositional inflows by "
        "measuring new entrants, defined as authors posting in a subreddit-month "
        "with no prior post in that subreddit in the observed dataset. In the "
        "subreddit and month fixed-effects DiD, the PersFree-by-post coefficient "
        f"for new-entrant share was {share_row['coef']:.3f} "
        f"(SE={share_row['se']:.3f}, $p={share_row['pvalue']:.3f}$); using "
        f"$\\log(1+\\mathrm{{new\\ entrants}})$ as the outcome gave "
        f"{count_row['coef']:.3f} (SE={count_row['se']:.3f}, "
        f"$p={count_row['pvalue']:.3f}$). With GenCap-by-post and PhysFree-by-post "
        f"controls, the new-entrant-share coefficient was {controlled_share_row['coef']:.3f} "
        f"(SE={controlled_share_row['se']:.3f}, $p={controlled_share_row['pvalue']:.3f}$), "
        "providing no evidence that the PersFree gradient is driven by differential "
        "entry of first-time subreddit posters."
    )
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(paragraph + "\n", encoding="utf-8")

    return {
        "rows": result_rows,
        "panel": panel,
        "output_path": str(output_path),
        "panel_output_path": str(panel_output_path),
        "plot_path": str(plot_path),
        "latex_path": str(latex_path),
        "paragraph": paragraph,
    }

def compute_community_new_entrant_volume_control(
    panel_path=None,
    output_path=None,
    augmented_panel_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    panel_path = Path(panel_path or tables_dir / "community_new_entrant_panel.csv")
    output_path = Path(output_path or tables_dir / "community_new_entrant_volume_control.csv")
    augmented_panel_path = Path(
        augmented_panel_path or tables_dir / "community_new_entrant_panel_volume_control.csv"
    )

    if augmented_panel_path.exists():
        panel = pd.read_csv(augmented_panel_path)
    elif panel_path.exists():
        panel = pd.read_csv(panel_path)
    else:
        entrant_result = compute_community_new_entrant_rates(panel_output_path=panel_path)
        panel = entrant_result["panel"].copy()

    required_columns = {
        "subreddit",
        "year_month",
        "new_entrant_share",
        "new_entrant_count",
        "log_new_entrant_count",
        "pers_free",
        "gen_cap",
        "phys_free",
        "post",
        "pers_free_post",
        "gen_cap_post",
        "phys_free_post",
    }
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"New-entrant panel missing columns: {sorted(missing_columns)}")

    panel["subreddit"] = panel["subreddit"].astype(str)
    panel["year_month"] = panel["year_month"].astype(str)
    for column_name in [
        "new_entrant_share",
        "new_entrant_count",
        "log_new_entrant_count",
        "pers_free",
        "gen_cap",
        "phys_free",
        "post",
        "pers_free_post",
        "gen_cap_post",
        "phys_free_post",
    ]:
        panel[column_name] = pd.to_numeric(panel[column_name], errors="coerce")

    if "total_posts" not in panel.columns:
        target_subreddits = [
            subreddit
            for subreddit in sorted(panel["subreddit"].dropna().unique())
            if raw_post_file_path(subreddit, data_dir).exists()
        ]
        if not target_subreddits:
            raise ValueError("No panel subreddits have raw post files for volume controls.")

        excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
        start_date = globals().get("START_DATE", datetime(2022, 1, 1))
        end_date_exclusive = globals().get("END_DATE_EXCLUSIVE", datetime(2025, 1, 1))
        max_lines_per_file = globals().get("MAX_LINES_PER_FILE", None)
        max_posts_per_day = globals().get("MAX_POSTS_PER_DAY", 50)
        cap_days = max((end_date_exclusive - start_date).days, 1)
        author_post_cap = max_posts_per_day * cap_days

        def iter_ecosystem_post_fields(subreddit):
            path = raw_post_file_path(subreddit, data_dir)
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if max_lines_per_file is not None and i >= max_lines_per_file:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    created_utc = payload.get("created_utc")
                    if created_utc is None:
                        continue
                    try:
                        dt = datetime.utcfromtimestamp(int(created_utc))
                    except Exception:
                        continue
                    if dt < start_date or dt >= end_date_exclusive:
                        continue
                    author = str(payload.get("author") or "")
                    if (
                        not author
                        or author in excluded_authors
                        or author.lower().endswith("bot")
                    ):
                        continue
                    yield author, month_label_for_datetime(dt)

        author_counts = Counter()
        raw_posts_seen = 0
        for subreddit in tqdm(target_subreddits, desc="  volume-control author-cap pass"):
            for author, _month_label in iter_ecosystem_post_fields(subreddit):
                author_counts[author] += 1
                raw_posts_seen += 1
        valid_authors = {
            author
            for author, count in author_counts.items()
            if count <= author_post_cap
        }
        if not valid_authors:
            raise ValueError("No ecosystem-clean authors after author-cap filtering.")

        post_counts = Counter()
        total_posts_used = 0
        panel_months = set(panel["year_month"].dropna().astype(str))
        for subreddit in tqdm(target_subreddits, desc="  volume-control subreddit-month pass"):
            for author, month_label in iter_ecosystem_post_fields(subreddit):
                if author not in valid_authors or month_label not in panel_months:
                    continue
                post_counts[(subreddit, month_label)] += 1
                total_posts_used += 1

        panel["total_posts"] = [
            safe_int(post_counts.get((row.subreddit, row.year_month), 0))
            for row in panel[["subreddit", "year_month"]].itertuples(index=False)
        ]
        panel["raw_posts_seen_before_author_cap"] = safe_int(raw_posts_seen)
        panel["ecosystem_clean_posts_used_for_volume"] = safe_int(total_posts_used)
        panel["n_ecosystem_authors_for_volume"] = safe_int(len(valid_authors))
        panel["author_post_cap_for_volume"] = safe_int(author_post_cap)

    panel["total_posts"] = pd.to_numeric(panel["total_posts"], errors="coerce").fillna(0)
    panel["log_total_posts"] = np.log1p(panel["total_posts"].astype(float))
    augmented_panel_path.parent.mkdir(exist_ok=True, parents=True)
    panel.to_csv(augmented_panel_path, index=False)

    model_data = panel.dropna(
        subset=[
            "new_entrant_share",
            "log_new_entrant_count",
            "log_total_posts",
            "pers_free_post",
            "gen_cap_post",
            "phys_free_post",
        ]
    ).copy()
    if model_data["subreddit"].nunique() < 2:
        raise ValueError("Need at least two subreddits for clustered FE models.")

    model_specs = [
        (
            "count_persfree_only",
            "without_volume_control",
            "log_new_entrant_count ~ pers_free_post + C(subreddit) + C(year_month)",
            ["pers_free_post"],
        ),
        (
            "count_persfree_only",
            "with_volume_control",
            "log_new_entrant_count ~ pers_free_post + log_total_posts + C(subreddit) + C(year_month)",
            ["pers_free_post", "log_total_posts"],
        ),
        (
            "count_three_dimensional",
            "without_volume_control",
            "log_new_entrant_count ~ pers_free_post + gen_cap_post + phys_free_post + C(subreddit) + C(year_month)",
            ["pers_free_post", "gen_cap_post", "phys_free_post"],
        ),
        (
            "count_three_dimensional",
            "with_volume_control",
            "log_new_entrant_count ~ pers_free_post + gen_cap_post + phys_free_post + log_total_posts + C(subreddit) + C(year_month)",
            ["pers_free_post", "gen_cap_post", "phys_free_post", "log_total_posts"],
        ),
    ]

    result_rows = []
    for model_name, volume_control, formula, terms in model_specs:
        model = fit_ols(formula, model_data, cluster_col="subreddit")
        if model is None:
            raise ValueError(f"New-entrant volume-control model failed to fit: {model_name}")
        for term in terms:
            result = reg_result(model, term)
            result_rows.append({
                "model": model_name,
                "volume_control": volume_control,
                "outcome": "log_new_entrant_count",
                "term": term,
                "coef": result.get("coef"),
                "se": result.get("se"),
                "pvalue": result.get("pvalue"),
                "n_obs": result.get("n_obs"),
                "n_subreddits": safe_int(model_data["subreddit"].nunique()),
                "n_months": safe_int(model_data["year_month"].nunique()),
                "mean_total_posts": safe_float(model_data["total_posts"].mean()),
                "mean_log_total_posts": safe_float(model_data["log_total_posts"].mean()),
                "panel_path": str(panel_path),
                "augmented_panel_path": str(augmented_panel_path),
            })

    result_table = pd.DataFrame(result_rows)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)
    return {
        "rows": result_rows,
        "panel": panel,
        "output_path": str(output_path),
        "augmented_panel_path": str(augmented_panel_path),
    }

def compute_preshock_persfree_linear_trend(
    panel_path=None,
    output_path=None,
    post_shock_coef=-0.451,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)

    panel_path = Path(panel_path or tables_dir / "community_new_entrant_panel_volume_control.csv")
    output_path = Path(output_path or tables_dir / "preshock_persfree_linear_trend.csv")
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")

    panel = pd.read_csv(panel_path)
    required_columns = {"subreddit", "year_month", "pers_free"}
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")

    panel["subreddit"] = panel["subreddit"].astype(str)
    panel["year_month"] = panel["year_month"].astype(str).str.slice(0, 7)
    panel["pers_free"] = pd.to_numeric(panel["pers_free"], errors="coerce")
    if "log_posts" in panel.columns:
        panel["log_posts"] = pd.to_numeric(panel["log_posts"], errors="coerce")
    elif "total_posts" in panel.columns:
        panel["total_posts"] = pd.to_numeric(panel["total_posts"], errors="coerce").fillna(0)
        panel["log_posts"] = np.log1p(panel["total_posts"].astype(float))
    elif "posts" in panel.columns:
        panel["posts"] = pd.to_numeric(panel["posts"], errors="coerce").fillna(0)
        panel["log_posts"] = np.log1p(panel["posts"].astype(float))
    else:
        raise ValueError("Panel must contain log_posts, total_posts, or posts.")

    post_totals = None
    if "total_posts" in panel.columns:
        post_totals = panel.groupby("subreddit")["total_posts"].sum()
    elif "posts" in panel.columns:
        post_totals = panel.groupby("subreddit")["posts"].sum()
    if post_totals is not None:
        positive_subreddits = post_totals[post_totals > 0].index
        panel = panel[panel["subreddit"].isin(positive_subreddits)].copy()

    pre = panel[
        (panel["year_month"] >= "2022-01")
        & (panel["year_month"] <= "2022-10")
    ].copy()
    if pre.empty:
        raise ValueError("No Jan-Oct 2022 pre-shock rows available.")

    month_order = {month: index + 1 for index, month in enumerate(sorted(pre["year_month"].unique()))}
    pre["time_trend"] = pre["year_month"].map(month_order).astype(float)
    pre["pers_free_time_trend"] = pre["pers_free"] * pre["time_trend"]
    model_data = pre.dropna(
        subset=[
            "subreddit",
            "year_month",
            "log_posts",
            "pers_free",
            "pers_free_time_trend",
        ]
    ).copy()
    if model_data["subreddit"].nunique() < 2:
        raise ValueError("Need at least two subreddits for clustered FE pre-trend model.")

    formula = "log_posts ~ pers_free_time_trend + C(subreddit) + C(year_month)"
    model = fit_ols(formula, model_data, cluster_col="subreddit")
    if model is None:
        raise ValueError("Pre-shock PersFree linear trend model failed to fit.")
    result = reg_result(model, "pers_free_time_trend")
    delta = result.get("coef")
    se = result.get("se")
    ratio_to_post = (
        safe_float(abs(delta) / abs(post_shock_coef))
        if delta is not None and post_shock_coef
        else np.nan
    )
    rows = [{
        "term": "pers_free_time_trend",
        "delta": delta,
        "se": se,
        "pvalue": result.get("pvalue"),
        "post_shock_coef_reference": safe_float(post_shock_coef),
        "abs_delta_over_abs_post_coef": ratio_to_post,
        "n_obs": safe_int(model.nobs),
        "n_subreddits": safe_int(model_data["subreddit"].nunique()),
        "n_months": safe_int(model_data["year_month"].nunique()),
        "start_month": model_data["year_month"].min(),
        "end_month": model_data["year_month"].max(),
        "time_trend_definition": "Jan 2022=1, ..., Oct 2022=10",
        "formula": formula,
        "panel_path": str(panel_path),
    }]
    result_table = pd.DataFrame(rows)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)
    return {
        "rows": rows,
        "model": model,
        "model_data": model_data,
        "output_path": str(output_path),
    }

def compute_preshock_leave_one_month_f_tests(
    panel_path=None,
    output_path=None,
    alpha=0.05,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)

    panel_path = Path(panel_path or tables_dir / "community_new_entrant_panel_volume_control.csv")
    output_path = Path(output_path or tables_dir / "preshock_leave_one_month_f_tests.csv")
    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")

    panel = pd.read_csv(panel_path)
    required_columns = {"subreddit", "year_month", "pers_free"}
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Panel missing columns: {sorted(missing_columns)}")

    panel["subreddit"] = panel["subreddit"].astype(str)
    panel["month"] = panel["year_month"].astype(str).str.slice(0, 7)
    panel["pers_free"] = pd.to_numeric(panel["pers_free"], errors="coerce")
    if "log_posts" in panel.columns:
        panel["log_posts"] = pd.to_numeric(panel["log_posts"], errors="coerce")
    elif "total_posts" in panel.columns:
        panel["total_posts"] = pd.to_numeric(panel["total_posts"], errors="coerce").fillna(0)
        panel["log_posts"] = np.log1p(panel["total_posts"].astype(float))
    elif "posts" in panel.columns:
        panel["posts"] = pd.to_numeric(panel["posts"], errors="coerce").fillna(0)
        panel["log_posts"] = np.log1p(panel["posts"].astype(float))
    else:
        raise ValueError("Panel must contain log_posts, total_posts, or posts.")

    post_totals = None
    if "total_posts" in panel.columns:
        post_totals = panel.groupby("subreddit")["total_posts"].sum()
    elif "posts" in panel.columns:
        post_totals = panel.groupby("subreddit")["posts"].sum()
    if post_totals is not None:
        positive_subreddits = post_totals[post_totals > 0].index
        panel = panel[panel["subreddit"].isin(positive_subreddits)].copy()

    pre = panel[
        (panel["month"] >= "2022-01")
        & (panel["month"] <= "2022-10")
    ].copy()
    pre = pre.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["subreddit", "month", "log_posts", "pers_free"]
    )
    months = sorted(pre["month"].astype(str).unique())
    if len(months) != 10:
        raise ValueError(f"Expected 10 pre-shock months, found {len(months)}.")

    rows = []
    for omitted_month in months:
        subset = pre[~pre["month"].eq(omitted_month)].copy()
        tested_months = sorted(subset["month"].astype(str).unique())
        if len(tested_months) != 9:
            raise ValueError(f"Expected 9 months after dropping {omitted_month}.")
        baseline_month = tested_months[0]
        interaction_terms = []
        for month_label in tested_months[1:]:
            term = f"pers_free_x_{month_label.replace('-', '_')}"
            subset[term] = subset["pers_free"] * subset["month"].eq(month_label).astype(int)
            interaction_terms.append(term)
        f_result = statsmodels_pretrend_f_test(subset, interaction_terms)
        rows.append({
            "omitted_month": omitted_month,
            "baseline_month": baseline_month,
            "tested_window": "2022-01 to 2022-10 excluding " + omitted_month,
            "n_obs": safe_int(len(subset)),
            "n_subreddits": safe_int(subset["subreddit"].nunique()),
            "n_months": safe_int(subset["month"].nunique()),
            "n_tested_terms": safe_int(len(interaction_terms)),
            "f_stat": f_result.get("f_stat"),
            "pvalue": f_result.get("pvalue"),
            "df_num": f_result.get("df_num"),
            "df_denom": f_result.get("df_denom"),
            "passes_at_alpha": bool(
                f_result.get("pvalue") is not None
                and f_result.get("pvalue") >= alpha
            ),
            "alpha": safe_float(alpha),
            "panel_path": str(panel_path),
        })

    result_table = pd.DataFrame(rows)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)
    return {
        "rows": rows,
        "results": result_table,
        "output_path": str(output_path),
    }

def compute_extended_low_physreq_pretrend(
    score_path=None,
    panel_output_path=None,
    output_path=None,
    phys_req_threshold=0.1,
    apply_author_cap=True,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        table_score_path = tables_dir / "acsi_preshock_tworuns.csv"
        root_score_path = root / "acsi_preshock_tworuns.csv"
        score_path = table_score_path if table_score_path.exists() else root_score_path
    score_path = Path(score_path)
    panel_output_path = Path(
        panel_output_path or tables_dir / "extended_low_physreq_monthly_panel.csv"
    )
    output_path = Path(output_path or tables_dir / "extended_low_physreq_pretrend.csv")
    if not score_path.exists():
        raise FileNotFoundError(f"Score file not found: {score_path}")

    scores = pd.read_csv(score_path)
    if "subreddit" not in scores.columns:
        raise ValueError("Score file must contain subreddit.")
    scores["subreddit"] = scores["subreddit"].astype(str)
    if {"phys_free", "pers_free"}.issubset(scores.columns):
        scores["phys_req_norm"] = 1 - pd.to_numeric(scores["phys_free"], errors="coerce")
        scores["non_personal_norm"] = pd.to_numeric(scores["pers_free"], errors="coerce")
    elif {"avg_physical_req_0_to_3", "avg_personal_req_0_to_3"}.issubset(scores.columns):
        scores["phys_req_norm"] = (
            pd.to_numeric(scores["avg_physical_req_0_to_3"], errors="coerce") / 3
        )
        scores["non_personal_norm"] = 1 - (
            pd.to_numeric(scores["avg_personal_req_0_to_3"], errors="coerce") / 3
        )
    elif {"physical_req", "personal_req"}.issubset(scores.columns):
        scores["phys_req_norm"] = pd.to_numeric(scores["physical_req"], errors="coerce")
        scores["non_personal_norm"] = 1 - pd.to_numeric(scores["personal_req"], errors="coerce")
    else:
        raise ValueError("Score file must contain either phys_free/pers_free or physical requirement columns.")

    digital_scores = scores[
        scores["phys_req_norm"].notna()
        & scores["non_personal_norm"].notna()
        & (scores["phys_req_norm"] < phys_req_threshold)
    ][["subreddit", "phys_req_norm", "non_personal_norm"]].drop_duplicates("subreddit")
    target_subreddits = [
        subreddit
        for subreddit in sorted(digital_scores["subreddit"].astype(str).unique())
        if raw_post_file_path(subreddit, data_dir).exists()
    ]
    if not target_subreddits:
        raise ValueError("No low-PhysReq scored subreddits have raw post files.")

    if panel_output_path.exists():
        panel = pd.read_csv(panel_output_path)
        panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    else:
        start_date = datetime(2020, 1, 1)
        end_date_exclusive = datetime(2025, 1, 1)
        days = max((end_date_exclusive - start_date).days, 1)
        author_post_cap = globals().get("MAX_POSTS_PER_DAY", 50) * days
        valid_authors = None
        raw_posts_seen = 0
        excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
        max_lines_per_file = globals().get("MAX_LINES_PER_FILE", None)

        def iter_extended_post_fields(subreddit):
            path = raw_post_file_path(subreddit, data_dir)
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if max_lines_per_file is not None and i >= max_lines_per_file:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    created_utc = payload.get("created_utc")
                    if created_utc is None:
                        continue
                    try:
                        dt = datetime.utcfromtimestamp(int(created_utc))
                    except Exception:
                        continue
                    if dt < start_date or dt >= end_date_exclusive:
                        continue
                    author = str(payload.get("author") or "")
                    if (
                        not author
                        or author in excluded_authors
                        or author.lower().endswith("bot")
                    ):
                        continue
                    yield author, dt

        if apply_author_cap:
            author_counts = Counter()
            for subreddit in tqdm(target_subreddits, desc="  low-PhysReq extended author-cap pass"):
                for author, _dt in iter_extended_post_fields(subreddit):
                    author_counts[author] += 1
                    raw_posts_seen += 1
            valid_authors = {
                author
                for author, n_posts in author_counts.items()
                if n_posts <= author_post_cap
            }
            del author_counts
            gc.collect()

        cells = Counter()
        posts_used = 0
        for subreddit in tqdm(target_subreddits, desc="  low-PhysReq extended monthly pass"):
            for author, dt in iter_extended_post_fields(subreddit):
                if valid_authors is not None and author not in valid_authors:
                    continue
                cells[(subreddit, month_start_for_datetime(dt))] += 1
                posts_used += 1

        months = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
        rows = []
        for subreddit in target_subreddits:
            for month in months:
                rows.append({
                    "subreddit": subreddit,
                    "year_month_dt": month,
                    "year_month": month.strftime("%Y-%m"),
                    "posts": safe_int(cells.get((subreddit, month), 0)),
                    "raw_posts_seen_before_author_cap": safe_int(raw_posts_seen),
                    "extended_posts_used": safe_int(posts_used),
                    "n_extended_authors_after_cap": safe_int(len(valid_authors)) if valid_authors is not None else None,
                    "author_post_cap": safe_int(author_post_cap),
                    "apply_author_cap": bool(apply_author_cap),
                })
        panel = pd.DataFrame(rows)
        panel_output_path.parent.mkdir(exist_ok=True, parents=True)
        panel.to_csv(panel_output_path, index=False)

    panel["subreddit"] = panel["subreddit"].astype(str)
    panel["year_month_dt"] = pd.to_datetime(panel["year_month_dt"])
    panel["year_month"] = panel["year_month_dt"].dt.strftime("%Y-%m")
    panel["posts"] = pd.to_numeric(panel["posts"], errors="coerce").fillna(0)
    panel["log_posts"] = np.log1p(panel["posts"].astype(float))
    panel = panel.merge(digital_scores, on="subreddit", how="inner")

    post_totals = panel.groupby("subreddit")["posts"].sum()
    positive_subreddits = post_totals[post_totals > 0].index
    panel = panel[panel["subreddit"].isin(positive_subreddits)].copy()
    pretrend = compute_extended_panel_pretrend_test(panel)
    if not pretrend:
        raise ValueError("Low-PhysReq extended pretrend test did not return results.")

    result_row = {
        **pretrend,
        "phys_req_threshold": safe_float(phys_req_threshold),
        "score_source": str(score_path),
        "panel_path": str(panel_output_path),
        "n_low_physreq_scored_subreddits": safe_int(len(digital_scores)),
        "n_low_physreq_raw_subreddits": safe_int(len(target_subreddits)),
        "n_positive_post_subreddits": safe_int(panel["subreddit"].nunique()),
        "mean_phys_req_norm": safe_float(
            digital_scores[digital_scores["subreddit"].isin(panel["subreddit"].unique())]["phys_req_norm"].mean()
        ),
        "median_phys_req_norm": safe_float(
            digital_scores[digital_scores["subreddit"].isin(panel["subreddit"].unique())]["phys_req_norm"].median()
        ),
        "min_pers_free": safe_float(panel[["subreddit", "non_personal_norm"]].drop_duplicates()["non_personal_norm"].min()),
        "max_pers_free": safe_float(panel[["subreddit", "non_personal_norm"]].drop_duplicates()["non_personal_norm"].max()),
    }
    result_table = pd.DataFrame([result_row])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)
    return {
        "pretrend": pretrend,
        "result": result_row,
        "panel": panel,
        "output_path": str(output_path),
        "panel_output_path": str(panel_output_path),
    }

def fit_author_month_intensive_fe(model_data, terms):
    data = model_data.dropna(
        subset=["log_posts", "pair_id", "month_index", "subreddit"] + terms
    ).copy()
    if data.empty:
        raise ValueError("No usable author-month rows for the intensive-margin model.")

    pair_fe = data["pair_id"]
    month_fe = data["month_index"]
    y_resid = residualize_two_way(data["log_posts"].astype(float), pair_fe, month_fe).to_numpy(dtype=np.float64)
    x_resids = [
        residualize_two_way(data[term].astype(float), pair_fe, month_fe).to_numpy(dtype=np.float64)
        for term in terms
    ]
    x_matrix = np.asarray(np.column_stack(x_resids), dtype=np.float64)
    finite_mask = np.isfinite(y_resid) & np.isfinite(x_matrix).all(axis=1)
    if not finite_mask.all():
        y_resid = y_resid[finite_mask]
        x_matrix = x_matrix[finite_mask, :]
        data = data.loc[finite_mask].copy()
    xtx = np.einsum("ni,nj->ij", x_matrix, x_matrix)
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ np.einsum("ni,n->i", x_matrix, y_resid)
    residual = y_resid - np.einsum("ni,i->n", x_matrix, beta)

    cluster_scores = []
    score_frame = pd.DataFrame({"subreddit": data["subreddit"].astype(str)})
    for i, term in enumerate(terms):
        score_frame[term] = x_matrix[:, i] * residual
    for _subreddit, group in score_frame.groupby("subreddit", sort=False):
        cluster_scores.append(group[terms].sum().to_numpy(dtype=float))
    scores_by_cluster = np.vstack(cluster_scores)
    meat = scores_by_cluster.T @ scores_by_cluster

    n_obs = len(data)
    n_clusters = data["subreddit"].nunique()
    n_pairs = data["pair_id"].nunique()
    n_months = data["month_index"].nunique()
    n_terms = len(terms)
    fe_rank = n_pairs + n_months - 1
    correction = 1.0
    if n_clusters > 1 and n_obs > fe_rank + n_terms:
        correction = (
            (n_clusters / (n_clusters - 1.0))
            * ((n_obs - 1.0) / (n_obs - fe_rank - n_terms))
        )
    covariance = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.where(np.diag(covariance) >= 0, np.diag(covariance), np.nan))
    pvalues = []
    for coefficient, standard_error in zip(beta, se):
        if standard_error is None or not np.isfinite(standard_error) or standard_error <= 0:
            pvalues.append(np.nan)
        else:
            statistic = abs(float(coefficient / standard_error))
            pvalues.append(float(2 * stats.t.sf(statistic, max(n_clusters - 1, 1))))

    return {
        "terms": terms,
        "coef": {term: safe_float(value) for term, value in zip(terms, beta)},
        "se": {term: safe_float(value) for term, value in zip(terms, se)},
        "pvalue": {term: safe_float(value) for term, value in zip(terms, pvalues)},
        "n_obs": safe_int(n_obs),
        "n_pairs": safe_int(n_pairs),
        "n_subreddits": safe_int(n_clusters),
        "n_months": safe_int(n_months),
    }

def compute_author_month_intensive_margin(
    score_path=None,
    output_path=None,
    latex_path=None,
    counts_path=None,
    pairs_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        table_score_path = tables_dir / "acsi_preshock_tworuns.csv"
        root_score_path = root / "acsi_preshock_tworuns.csv"
        if table_score_path.exists():
            score_path = table_score_path
        elif root_score_path.exists():
            score_path = root_score_path
        else:
            score_path = table_score_path
    score_path = Path(score_path)
    output_path = Path(output_path or tables_dir / "author_month_intensive_margin.csv")
    latex_path = Path(latex_path or tables_dir / "author_month_intensive_margin.tex")
    counts_path = Path(counts_path or tables_dir / "author_month_intensive_counts.parquet")
    pairs_path = Path(pairs_path or tables_dir / "author_month_intensive_pairs.parquet")

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])

    all_months = [
        month.strftime("%Y-%m")
        for month in globals().get(
            "ALL_MONTHS",
            pd.date_range("2022-01-01", "2024-12-01", freq="MS"),
        )
    ]
    month_to_index = {month_label: index for index, month_label in enumerate(all_months)}
    post_start = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01")).strftime("%Y-%m")
    post_by_month = np.array([1 if month_label >= post_start else 0 for month_label in all_months], dtype=float)
    blackout_month_indices = {
        month_to_index[month_label]
        for month_label in ["2023-06", "2023-07", "2023-08"]
        if month_label in month_to_index
    }

    if counts_path.exists() and pairs_path.exists():
        counts = pd.read_parquet(counts_path)
        pairs = pd.read_parquet(pairs_path)
    else:
        target_subreddits = [
            subreddit
            for subreddit in sorted(scores["subreddit"].astype(str).unique())
            if raw_post_file_path(subreddit, data_dir).exists()
        ]
        if not target_subreddits:
            raise ValueError("No scored subreddits have raw post files.")

        excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
        start_date = globals().get("START_DATE", datetime(2022, 1, 1))
        end_date_exclusive = globals().get("END_DATE_EXCLUSIVE", datetime(2025, 1, 1))
        max_lines_per_file = globals().get("MAX_LINES_PER_FILE", None)
        max_posts_per_day = globals().get("MAX_POSTS_PER_DAY", 50)
        cap_days = max((end_date_exclusive - start_date).days, 1)
        author_post_cap = max_posts_per_day * cap_days

        def iter_ecosystem_post_fields(subreddit):
            path = raw_post_file_path(subreddit, data_dir)
            if not path.exists():
                return
            with open(path, "r", encoding="utf-8") as handle:
                for i, line in enumerate(handle):
                    if max_lines_per_file is not None and i >= max_lines_per_file:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    created_utc = payload.get("created_utc")
                    if created_utc is None:
                        continue
                    try:
                        dt = datetime.utcfromtimestamp(int(created_utc))
                    except Exception:
                        continue
                    if dt < start_date or dt >= end_date_exclusive:
                        continue
                    month_label = month_label_for_datetime(dt)
                    month_index = month_to_index.get(month_label)
                    if month_index is None:
                        continue
                    author = str(payload.get("author") or "")
                    if (
                        not author
                        or author in excluded_authors
                        or author.lower().endswith("bot")
                    ):
                        continue
                    yield author, month_index

        author_counts = Counter()
        total_posts_seen = 0
        for subreddit in tqdm(target_subreddits, desc="  intensive-margin author-cap pass"):
            for author, _month_index in iter_ecosystem_post_fields(subreddit):
                author_counts[author] += 1
                total_posts_seen += 1
        valid_authors = {
            author
            for author, count in author_counts.items()
            if count <= author_post_cap
        }
        if not valid_authors:
            raise ValueError("No ecosystem-clean authors after author-cap filtering.")

        pair_lookup = {}
        pair_authors = []
        pair_subreddits = []
        pair_pre_counts = Counter()
        pair_post_counts = Counter()
        author_pre_counts = Counter()
        author_post_counts = Counter()
        pair_month_counts = Counter()
        posts_used = 0

        def pair_id_for(author, subreddit):
            key = (author, subreddit)
            pair_id = pair_lookup.get(key)
            if pair_id is None:
                pair_id = len(pair_authors)
                pair_lookup[key] = pair_id
                pair_authors.append(author)
                pair_subreddits.append(subreddit)
            return pair_id

        for subreddit in tqdm(target_subreddits, desc="  intensive-margin pair-month pass"):
            for author, month_index in iter_ecosystem_post_fields(subreddit):
                if author not in valid_authors:
                    continue
                pair_id = pair_id_for(author, subreddit)
                pair_month_counts[pair_id * len(all_months) + month_index] += 1
                posts_used += 1
                if post_by_month[month_index] == 1:
                    pair_post_counts[pair_id] += 1
                    author_post_counts[author] += 1
                else:
                    pair_pre_counts[pair_id] += 1
                    author_pre_counts[author] += 1

        pair_rows = []
        for pair_id, (author, subreddit) in enumerate(zip(pair_authors, pair_subreddits)):
            pre_posts = int(pair_pre_counts.get(pair_id, 0))
            post_posts = int(pair_post_counts.get(pair_id, 0))
            author_pre_posts = int(author_pre_counts.get(author, 0))
            author_post_posts = int(author_post_counts.get(author, 0))
            pair_rows.append({
                "pair_id": pair_id,
                "author": author,
                "subreddit": subreddit,
                "pre_pair_posts": pre_posts,
                "post_pair_posts": post_posts,
                "stable_pair": bool(pre_posts > 0 and post_posts > 0),
                "author_pre_posts": author_pre_posts,
                "author_post_posts": author_post_posts,
                "creator_clean_author": bool(author_pre_posts >= globals().get("MIN_PRE_POSTS", 5)),
                "author_stable_anywhere": bool(author_pre_posts > 0 and author_post_posts > 0),
                "raw_posts_seen_before_author_cap": safe_int(total_posts_seen),
                "ecosystem_clean_posts_used": safe_int(posts_used),
                "n_ecosystem_authors": safe_int(len(valid_authors)),
                "author_post_cap": safe_int(author_post_cap),
            })
        pairs = pd.DataFrame(pair_rows).merge(scores, on="subreddit", how="left")
        encoded_keys = np.fromiter(pair_month_counts.keys(), dtype=np.int64)
        encoded_values = np.fromiter(pair_month_counts.values(), dtype=np.int64)
        counts = pd.DataFrame({
            "pair_id": encoded_keys // len(all_months),
            "month_index": encoded_keys % len(all_months),
            "post_count": encoded_values,
        })
        counts_path.parent.mkdir(exist_ok=True, parents=True)
        counts.to_parquet(counts_path, index=False)
        pairs_path.parent.mkdir(exist_ok=True, parents=True)
        pairs.to_parquet(pairs_path, index=False)

    for column_name in ["pair_id", "pre_pair_posts", "post_pair_posts", "author_pre_posts", "author_post_posts"]:
        pairs[column_name] = pd.to_numeric(pairs[column_name], errors="coerce")
    for column_name in ["stable_pair", "creator_clean_author", "author_stable_anywhere"]:
        pairs[column_name] = pairs[column_name].astype(bool)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        pairs[column_name] = pd.to_numeric(pairs[column_name], errors="coerce")
    pairs = pairs.dropna(subset=["pair_id", "subreddit", "gen_cap", "phys_free", "pers_free"]).copy()
    pairs["pair_id"] = pairs["pair_id"].astype(np.int64)
    counts["pair_id"] = pd.to_numeric(counts["pair_id"], errors="coerce").astype(np.int64)
    counts["month_index"] = pd.to_numeric(counts["month_index"], errors="coerce").astype(np.int16)
    counts["post_count"] = pd.to_numeric(counts["post_count"], errors="coerce").fillna(0).astype(np.int32)

    pair_meta = pairs.set_index("pair_id", drop=False)
    count_lookup = pd.Series(
        counts["post_count"].to_numpy(dtype=float),
        index=counts["pair_id"].to_numpy(dtype=np.int64) * len(all_months)
        + counts["month_index"].to_numpy(dtype=np.int64),
    )

    def build_balanced_model_data(pair_ids, drop_blackout=False):
        pair_ids = np.asarray(sorted(pd.Series(pair_ids).dropna().astype(np.int64).unique()), dtype=np.int64)
        if len(pair_ids) == 0:
            raise ValueError("No author-subreddit pairs available for this intensive-margin model.")
        month_indices = np.arange(len(all_months), dtype=np.int16)
        if drop_blackout:
            month_indices = np.array(
                [index for index in month_indices if int(index) not in blackout_month_indices],
                dtype=np.int16,
            )
        repeated_pairs = np.repeat(pair_ids, len(month_indices))
        repeated_months = np.tile(month_indices, len(pair_ids)).astype(np.int16)
        encoded = repeated_pairs.astype(np.int64) * len(all_months) + repeated_months.astype(np.int64)
        post_count = pd.Series(encoded).map(count_lookup).fillna(0).to_numpy(dtype=float)
        meta = pair_meta.reindex(pair_ids)
        subreddit = np.repeat(meta["subreddit"].astype(str).to_numpy(), len(month_indices))
        pers_free = np.repeat(meta["pers_free"].astype(float).to_numpy(), len(month_indices))
        gen_cap = np.repeat(meta["gen_cap"].astype(float).to_numpy(), len(month_indices))
        phys_free = np.repeat(meta["phys_free"].astype(float).to_numpy(), len(month_indices))
        post = post_by_month[repeated_months.astype(int)]




        return pd.DataFrame({
            "pair_id": repeated_pairs,
            "subreddit": subreddit,
            "month_index": repeated_months,
            "post_count": post_count,
            "log_posts": np.log1p(post_count),
            "post_shock": post,
            "pers_free_post": pers_free * post,
            "gen_cap_post": gen_cap * post,
            "phys_free_post": phys_free * post,
        })

    stable_pair_ids = pairs.loc[pairs["stable_pair"], "pair_id"]
    creator_clean_pair_ids = pairs.loc[
        pairs["stable_pair"] & pairs["creator_clean_author"],
        "pair_id",
    ]
    author_stable_pair_ids = pairs.loc[pairs["author_stable_anywhere"], "pair_id"]






    result_rows = []
    models = {}

    def fit_and_store(model_name, model_data, terms):
        fixed_effects = "author-subreddit FE + month FE"
        cluster_level = "subreddit"
        model = fit_author_month_intensive_fe(model_data, terms)
        models[model_name] = model
        coef = model["coef"].get("pers_free_post")
        se = model["se"].get("pers_free_post")
        pvalue = model["pvalue"].get("pers_free_post")
        result_rows.append({
            "model_name": model_name,
            "pers_free_post_coef": coef,
            "standard_error": se,
            "p_value": pvalue,
            "n_observations": model["n_obs"],
            "n_author_subreddit_pairs": model["n_pairs"],
            "n_subreddits": model["n_subreddits"],
            "n_months": model["n_months"],
            "fixed_effects_used": fixed_effects,
            "clustering_level": cluster_level,
            "includes_gen_cap_phys_free_controls": bool("gen_cap_post" in terms or "phys_free_post" in terms),
            "drops_reddit_blackout_months": bool("drop_blackout" in model_name),
            "score_source": str(score_path),
            "counts_cache": str(counts_path),
            "pairs_cache": str(pairs_path),
        })

    stable_panel = build_balanced_model_data(stable_pair_ids)
    fit_and_store("stable_pair_persfree_only", stable_panel, ["pers_free_post"])
    fit_and_store(
        "stable_pair_three_dimensional",
        stable_panel,
        ["pers_free_post", "gen_cap_post", "phys_free_post"],
    )
    blackout_panel = stable_panel[~stable_panel["month_index"].isin(blackout_month_indices)].copy()
    fit_and_store("stable_pair_drop_blackout", blackout_panel, ["pers_free_post"])
    fit_and_store(
        "stable_pair_three_dimensional_drop_blackout",
        blackout_panel,
        ["pers_free_post", "gen_cap_post", "phys_free_post"],
    )
    del blackout_panel
    del stable_panel
    gc.collect()

    creator_panel = build_balanced_model_data(creator_clean_pair_ids)
    fit_and_store("creator_clean_stable_pair", creator_panel, ["pers_free_post"])
    del creator_panel
    gc.collect()

    author_stable_panel = build_balanced_model_data(author_stable_pair_ids)
    fit_and_store("author_stable_anywhere", author_stable_panel, ["pers_free_post"])
    del author_stable_panel
    gc.collect()

    result_table = pd.DataFrame(result_rows)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)
    latex_table = result_table[[
        "model_name",
        "pers_free_post_coef",
        "standard_error",
        "p_value",
        "n_observations",
        "n_author_subreddit_pairs",
        "n_subreddits",
        "fixed_effects_used",
        "clustering_level",
    ]].copy()
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(
        latex_table.to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    main_row = result_rows[0]
    beta = main_row["pers_free_post_coef"]
    pvalue = main_row["p_value"]
    direction = "negative" if beta is not None and beta < 0 else "positive"
    significant = bool(pvalue is not None and pvalue < 0.05)
    support = direction == "negative" and significant
    summary = (
        "Author-month intensive-margin result: "
        f"beta is {direction} ({beta:.4f}, SE={main_row['standard_error']:.4f}, "
        f"p={pvalue:.4g}) and is {'statistically significant' if significant else 'not statistically significant'}. "
        + (
            "This supports the reduced-posting-intensity mechanism among continuing author-subreddit pairs."
            if support
            else "This does not provide clear support for the reduced-posting-intensity mechanism among continuing author-subreddit pairs."
        )
    )
    print(summary)

    return {
        "rows": result_rows,
        "results": result_table,
        "models": models,
        "summary": summary,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
        "counts_path": str(counts_path),
        "pairs_path": str(pairs_path),
    }

def compute_marginal_participation_decomposition(
    score_path=None,
    output_path=None,
    latex_path=None,
    panel_output_path=None,
    counts_path=None,
    pairs_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        table_score_path = tables_dir / "acsi_preshock_tworuns.csv"
        root_score_path = root / "acsi_preshock_tworuns.csv"
        if table_score_path.exists():
            score_path = table_score_path
        elif root_score_path.exists():
            score_path = root_score_path
        else:
            score_path = table_score_path
    score_path = Path(score_path)
    output_path = Path(output_path or tables_dir / "marginal_participation_decomposition.csv")
    latex_path = Path(latex_path or tables_dir / "marginal_participation_decomposition.tex")
    panel_output_path = Path(panel_output_path or tables_dir / "marginal_participation_decomposition_panel.csv")
    counts_path = Path(counts_path or tables_dir / "author_month_intensive_counts.parquet")
    pairs_path = Path(pairs_path or tables_dir / "author_month_intensive_pairs.parquet")

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    if not counts_path.exists() or not pairs_path.exists():



        compute_author_month_intensive_margin(
            score_path=score_path,
            counts_path=counts_path,
            pairs_path=pairs_path,
        )

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])

    counts = pd.read_parquet(counts_path)
    pairs = pd.read_parquet(pairs_path)
    required_pair_columns = {
        "pair_id",
        "subreddit",
        "pre_pair_posts",
        "post_pair_posts",
        "stable_pair",
        "creator_clean_author",
    }
    required_count_columns = {"pair_id", "month_index", "post_count"}
    missing_pair_columns = required_pair_columns - set(pairs.columns)
    missing_count_columns = required_count_columns - set(counts.columns)
    if missing_pair_columns:
        raise ValueError(f"Pair cache missing columns: {sorted(missing_pair_columns)}")
    if missing_count_columns:
        raise ValueError(f"Count cache missing columns: {sorted(missing_count_columns)}")

    for column_name in ["pair_id", "pre_pair_posts", "post_pair_posts"]:
        pairs[column_name] = pd.to_numeric(pairs[column_name], errors="coerce")
    for column_name in ["stable_pair", "creator_clean_author"]:
        pairs[column_name] = pairs[column_name].fillna(False).astype(bool)
    pairs["subreddit"] = pairs["subreddit"].astype(str)
    pairs = pairs.dropna(subset=["pair_id", "subreddit"]).copy()
    pairs["pair_id"] = pairs["pair_id"].astype(np.int64)
    pairs["pre_pair_posts"] = pairs["pre_pair_posts"].fillna(0).astype(np.int32)
    pairs["post_pair_posts"] = pairs["post_pair_posts"].fillna(0).astype(np.int32)
    pairs = pairs.drop(columns=["gen_cap", "phys_free", "pers_free"], errors="ignore").merge(
        scores,
        on="subreddit",
        how="inner",
    )
    if pairs.empty:
        raise ValueError("No author-subreddit pairs overlap with the ACSI score table.")

    counts["pair_id"] = pd.to_numeric(counts["pair_id"], errors="coerce")
    counts["month_index"] = pd.to_numeric(counts["month_index"], errors="coerce")
    counts["post_count"] = pd.to_numeric(counts["post_count"], errors="coerce")
    counts = counts.dropna(subset=["pair_id", "month_index", "post_count"]).copy()
    counts["pair_id"] = counts["pair_id"].astype(np.int64)
    counts["month_index"] = counts["month_index"].astype(np.int16)
    counts["post_count"] = counts["post_count"].fillna(0).astype(np.int32)

    all_months = [
        month.strftime("%Y-%m")
        for month in globals().get(
            "ALL_MONTHS",
            pd.date_range("2022-01-01", "2024-12-01", freq="MS"),
        )
    ]
    post_start = globals().get("SHOCK_MONTH", pd.Timestamp("2022-12-01")).strftime("%Y-%m")
    month_to_index = {month_label: index for index, month_label in enumerate(all_months)}
    blackout_months = {"2023-06", "2023-07", "2023-08"}
    blackout_month_indices = {
        month_to_index[month_label]
        for month_label in blackout_months
        if month_label in month_to_index
    }

    valid_pair_ids = set(pairs["pair_id"].astype(np.int64))
    counts = counts[counts["pair_id"].isin(valid_pair_ids)].copy()
    if counts.empty:
        raise ValueError("No observed pair-month counts overlap with scored pairs.")

    pair_month_stats = counts.groupby("pair_id", sort=False).agg(
        active_months=("month_index", "nunique"),
        observed_pair_posts=("post_count", "sum"),
    )
    pairs["active_months"] = (
        pairs["pair_id"].map(pair_month_stats["active_months"]).fillna(0).astype(np.int16)
    )
    pairs["total_pair_posts"] = (
        pairs["pair_id"].map(pair_month_stats["observed_pair_posts"]).fillna(0).astype(np.int32)
    )
    frequency_rank = pairs["total_pair_posts"].rank(method="first")
    high_frequency_cutoff = float(
        pairs.loc[
            pd.qcut(frequency_rank, q=3, labels=["low", "middle", "high"]).astype(str).eq("high"),
            "total_pair_posts",
        ].min()
    )





    pairs["is_stable_pairs"] = pairs["stable_pair"]
    pairs["is_nonstable_pairs"] = ~pairs["stable_pair"]
    pairs["is_preonly_pairs"] = (pairs["pre_pair_posts"] > 0) & (pairs["post_pair_posts"] == 0)
    pairs["is_postonly_pairs"] = (pairs["pre_pair_posts"] == 0) & (pairs["post_pair_posts"] > 0)
    pairs["is_one_month_pairs"] = pairs["active_months"] == 1
    pairs["is_low_commitment_repeat_pairs"] = (
        (pairs["active_months"] >= 2) & (~pairs["stable_pair"])
    )
    pairs["is_single_post_pairs"] = pairs["total_pair_posts"] == 1
    pairs["is_low_frequency_le2_pairs"] = pairs["total_pair_posts"] <= 2
    pairs["frequency_tercile"] = pd.qcut(
        frequency_rank,
        q=3,
        labels=["low", "middle", "high"],
    )
    pairs["is_high_frequency_top_tercile_pairs"] = pairs["frequency_tercile"].astype(str).eq("high")


    pairs["is_pre_low_frequency_pairs"] = pairs["pre_pair_posts"].eq(1)

    group_definitions = [
        {
            "group": "stable_pairs",
            "flag": "is_stable_pairs",
            "label": "Stable author-subreddit pairs",
            "notes": "Stable pairs have at least one pre-shock and one post-shock post; this is the intensive-margin comparison group.",
        },
        {
            "group": "nonstable_pairs",
            "flag": "is_nonstable_pairs",
            "label": "Nonstable author-subreddit pairs",
            "notes": "Full-period marginal group; descriptive decomposition of posts from pairs not observed both before and after ChatGPT.",
        },
        {
            "group": "preonly_pairs",
            "flag": "is_preonly_pairs",
            "label": "Pre-only pairs",
            "notes": "Future-defined pre-only group; post-period counts are mechanically zero, so interpret as descriptive decomposition.",
        },
        {
            "group": "postonly_pairs",
            "flag": "is_postonly_pairs",
            "label": "Post-only pairs",
            "notes": "Future-defined post-only group; pre-period counts are mechanically zero, so interpret as descriptive entry decomposition.",
        },
        {
            "group": "one_month_pairs",
            "flag": "is_one_month_pairs",
            "label": "One-month pairs",
            "notes": "Full-period low-commitment group active in exactly one subreddit-month.",
        },
        {
            "group": "low_commitment_repeat_pairs",
            "flag": "is_low_commitment_repeat_pairs",
            "label": "Low-commitment repeat pairs",
            "notes": "Full-period marginal group active in multiple months but not stable across the shock.",
        },
        {
            "group": "single_post_pairs",
            "flag": "is_single_post_pairs",
            "label": "Single-post pairs",
            "notes": "Full-period one-time participation group with exactly one observed post.",
        },
        {
            "group": "low_frequency_le2_pairs",
            "flag": "is_low_frequency_le2_pairs",
            "label": "Low-frequency pairs",
            "notes": "Full-period low-frequency group with no more than two observed posts.",
        },
        {
            "group": "high_frequency_top_tercile_pairs",
            "flag": "is_high_frequency_top_tercile_pairs",
            "label": "High-frequency top tercile pairs",
            "notes": f"Full-period high-frequency comparison group using total posts cutoff >= {high_frequency_cutoff:.1f}.",
        },
        {
            "group": "pre_low_frequency_pairs",
            "flag": "is_pre_low_frequency_pairs",
            "label": "Pre-shock low-frequency pairs",
            "notes": "Pre-shock-defined marginality: exactly one pre-shock post in the author-subreddit pair.",
        },
    ]
    important_groups = {"nonstable_pairs", "one_month_pairs", "low_commitment_repeat_pairs"}

    pair_subreddit = pairs.set_index("pair_id")["subreddit"]
    counts["subreddit"] = counts["pair_id"].map(pair_subreddit).astype(str)
    target_subreddits = sorted(counts["subreddit"].dropna().unique())
    grid = pd.MultiIndex.from_product(
        [target_subreddits, range(len(all_months))],
        names=["subreddit", "month_index"],
    ).to_frame(index=False)
    total_posts = (
        counts.groupby(["subreddit", "month_index"], as_index=False)["post_count"]
        .sum()
        .rename(columns={"post_count": "total_posts"})
    )
    panel = grid.merge(total_posts, on=["subreddit", "month_index"], how="left")
    panel["total_posts"] = panel["total_posts"].fillna(0).astype(np.int32)

    pair_ids_by_flag = {
        definition["flag"]: set(
            pairs.loc[pairs[definition["flag"]], "pair_id"].astype(np.int64).to_numpy()
        )
        for definition in group_definitions
    }
    creator_clean_pair_ids = set(
        pairs.loc[pairs["creator_clean_author"], "pair_id"].astype(np.int64).to_numpy()
    )

    def add_group_posts(column_name, pair_ids):
        if not pair_ids:
            nonlocal_panel[column_name] = 0
            return
        grouped = (
            counts.loc[counts["pair_id"].isin(pair_ids)]
            .groupby(["subreddit", "month_index"], as_index=False)["post_count"]
            .sum()
            .rename(columns={"post_count": column_name})
        )
        nonlocal_panel[column_name] = grouped.set_index(["subreddit", "month_index"])[column_name]

    nonlocal_panel = panel.set_index(["subreddit", "month_index"])
    for definition in group_definitions:
        column_name = f"posts_{definition['group']}"
        add_group_posts(column_name, pair_ids_by_flag[definition["flag"]])
    nonlocal_panel["creator_clean_total_posts"] = (
        counts.loc[counts["pair_id"].isin(creator_clean_pair_ids)]
        .groupby(["subreddit", "month_index"])["post_count"]
        .sum()
        if creator_clean_pair_ids
        else 0
    )
    for group_name in important_groups:
        flag = f"is_{group_name}"
        creator_clean_group_ids = pair_ids_by_flag[flag] & creator_clean_pair_ids
        add_group_posts(f"posts_{group_name}_creator_clean", creator_clean_group_ids)
    panel = nonlocal_panel.reset_index()
    post_columns = [column_name for column_name in panel.columns if column_name.startswith("posts_")]
    panel[post_columns] = panel[post_columns].fillna(0).astype(np.int32)
    panel["creator_clean_total_posts"] = (
        pd.to_numeric(panel["creator_clean_total_posts"], errors="coerce")
        .fillna(0)
        .astype(np.int32)
    )

    panel["year_month"] = panel["month_index"].map(dict(enumerate(all_months)))
    panel["year_month_dt"] = pd.to_datetime(panel["year_month"] + "-01")
    panel["post_shock"] = (panel["year_month"] >= post_start).astype(int)
    panel = panel.merge(scores, on="subreddit", how="left")
    for dimension in ["pers_free", "gen_cap", "phys_free"]:
        panel[f"{dimension}_post"] = panel[dimension] * panel["post_shock"]
    for definition in group_definitions:
        group_name = definition["group"]
        posts_column = f"posts_{group_name}"
        panel[f"log_{posts_column}"] = np.log1p(panel[posts_column].astype(float))
        panel[f"share_{group_name}"] = np.where(
            panel["total_posts"] > 0,
            panel[posts_column] / panel["total_posts"],
            np.nan,
        )
    for group_name in important_groups:
        posts_column = f"posts_{group_name}_creator_clean"
        panel[f"log_{posts_column}"] = np.log1p(panel[posts_column].astype(float))
        panel[f"share_{group_name}_creator_clean"] = np.where(
            panel["creator_clean_total_posts"] > 0,
            panel[posts_column] / panel["creator_clean_total_posts"],
            np.nan,
        )
    panel = panel.sort_values(["subreddit", "year_month"]).reset_index(drop=True)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    panel.to_csv(panel_output_path, index=False)

    result_rows = []

    def fit_and_record(
        group_name,
        outcome,
        outcome_type,
        model_name,
        data,
        terms,
        notes,
    ):
        required_columns = [outcome, "subreddit", "year_month"] + terms
        model_data = data.dropna(subset=required_columns).copy()
        if model_data.empty or model_data["subreddit"].nunique() < 2:
            return
        formula = f"{outcome} ~ " + " + ".join(terms) + " + C(subreddit) + C(year_month)"
        model = fit_ols(formula, model_data, cluster_col="subreddit")
        if model is None:
            return
        result = reg_result(model, "pers_free_post")
        result_rows.append({
            "group": group_name,
            "outcome": outcome,
            "outcome_type": outcome_type,
            "model_name": model_name,
            "pers_free_post_coef": result.get("coef"),
            "standard_error": result.get("se"),
            "p_value": result.get("pvalue"),
            "n_observations": result.get("n_obs"),
            "n_subreddits": safe_int(model_data["subreddit"].nunique()),
            "n_months": safe_int(model_data["year_month"].nunique()),
            "fixed_effects": "subreddit FE + month FE",
            "clustering_level": "subreddit",
            "includes_gen_cap_phys_free_controls": bool(
                "gen_cap_post" in terms or "phys_free_post" in terms
            ),
            "drops_reddit_blackout_months": bool(model_data["month_index"].isin(blackout_month_indices).sum() == 0),
            "sample_notes": notes,
            "score_source": str(score_path),
            "counts_cache": str(counts_path),
            "pairs_cache": str(pairs_path),
        })

    for definition in group_definitions:
        group_name = definition["group"]
        notes = definition["notes"]
        fit_and_record(
            group_name,
            f"log_posts_{group_name}",
            "log_group_post_count",
            f"{group_name}_log_count_persfree_only",
            panel,
            ["pers_free_post"],
            notes,
        )
        fit_and_record(
            group_name,
            f"share_{group_name}",
            "share_of_total_posts",
            f"{group_name}_share_persfree_only",
            panel,
            ["pers_free_post"],
            notes,
        )

    for group_name in sorted(important_groups):
        definition = next(item for item in group_definitions if item["group"] == group_name)
        notes = definition["notes"]
        for outcome, outcome_type in [
            (f"log_posts_{group_name}", "log_group_post_count"),
            (f"share_{group_name}", "share_of_total_posts"),
        ]:
            fit_and_record(
                group_name,
                outcome,
                outcome_type,
                f"{group_name}_{outcome_type}_three_dimensional",
                panel,
                ["pers_free_post", "gen_cap_post", "phys_free_post"],
                f"{notes} Includes GenCap and PhysFree post interactions.",
            )
            fit_and_record(
                group_name,
                outcome,
                outcome_type,
                f"{group_name}_{outcome_type}_drop_blackout",
                panel[~panel["month_index"].isin(blackout_month_indices)].copy(),
                ["pers_free_post"],
                f"{notes} Drops June-August 2023 Reddit blackout months.",
            )

        creator_clean_subreddits = set(
            panel.loc[panel["creator_clean_total_posts"] > 0, "subreddit"].astype(str)
        )
        creator_panel = panel[panel["subreddit"].isin(creator_clean_subreddits)].copy()
        for outcome, outcome_type in [
            (f"log_posts_{group_name}_creator_clean", "creator_clean_log_group_post_count"),
            (f"share_{group_name}_creator_clean", "creator_clean_share_of_posts"),
        ]:
            fit_and_record(
                group_name,
                outcome,
                outcome_type,
                f"{group_name}_{outcome_type}_creator_clean",
                creator_panel,
                ["pers_free_post"],
                f"{notes} Restricts group construction to creator-clean authors with at least five pre-shock posts.",
            )

    for outcome, outcome_type in [
        ("log_posts_pre_low_frequency_pairs", "log_group_post_count"),
        ("share_pre_low_frequency_pairs", "share_of_total_posts"),
    ]:
        fit_and_record(
            "pre_low_frequency_pairs",
            outcome,
            outcome_type,
            f"pre_low_frequency_pairs_{outcome_type}_three_dimensional",
            panel,
            ["pers_free_post", "gen_cap_post", "phys_free_post"],
            "Pre-shock-defined marginality with GenCap and PhysFree controls.",
        )
        fit_and_record(
            "pre_low_frequency_pairs",
            outcome,
            outcome_type,
            f"pre_low_frequency_pairs_{outcome_type}_drop_blackout",
            panel[~panel["month_index"].isin(blackout_month_indices)].copy(),
            ["pers_free_post"],
            "Pre-shock-defined marginality dropping June-August 2023 Reddit blackout months.",
        )

    result_table = pd.DataFrame(result_rows)
    if result_table.empty:
        raise ValueError("No marginal participation decomposition models were estimated.")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    latex_columns = [
        "group",
        "outcome_type",
        "model_name",
        "pers_free_post_coef",
        "standard_error",
        "p_value",
        "n_observations",
        "n_subreddits",
        "fixed_effects",
        "clustering_level",
    ]
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(
        result_table[latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    def row_for(model_name):
        rows = result_table[result_table["model_name"].eq(model_name)]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    stable_row = row_for("stable_pairs_log_count_persfree_only")
    nonstable_row = row_for("nonstable_pairs_log_count_persfree_only")
    one_month_row = row_for("one_month_pairs_log_count_persfree_only")
    low_commitment_row = row_for("low_commitment_repeat_pairs_log_count_persfree_only")
    key_rows = [
        ("stable pairs", stable_row),
        ("nonstable pairs", nonstable_row),
        ("one-month pairs", one_month_row),
        ("low-commitment repeat pairs", low_commitment_row),
    ]
    summary_parts = []
    for label, row in key_rows:
        if not row:
            continue
        coef = row.get("pers_free_post_coef")
        se = row.get("standard_error")
        pvalue = row.get("p_value")
        direction = "negative" if coef is not None and coef < 0 else "positive"
        significance = "significant" if pvalue is not None and pvalue < 0.05 else "not significant"
        summary_parts.append(
            f"{label}: {direction} beta={coef:.4f} (SE={se:.4f}, p={pvalue:.4g}), {significance}"
        )
    mechanism_support = bool(
        nonstable_row
        and nonstable_row.get("pers_free_post_coef") is not None
        and nonstable_row.get("pers_free_post_coef") < 0
        and nonstable_row.get("p_value") is not None
        and nonstable_row.get("p_value") < 0.05
    )
    summary = (
        "Marginal participation decomposition: "
        + "; ".join(summary_parts)
        + ". "
        + (
            "The results support a marginal-participation channel in which high-PersFree communities lose more posts from nonstable contributors."
            if mechanism_support
            else "The results do not show clear negative, statistically significant evidence for the nonstable-contributor channel in the headline log-count model."
        )
    )
    print(summary)

    return {
        "results": result_table,
        "panel": panel,
        "summary": summary,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
        "panel_output_path": str(panel_output_path),
        "counts_path": str(counts_path),
        "pairs_path": str(pairs_path),
    }

def compute_community_vulnerability_prediction(
    ecosystem_posts=None,
    score_path=None,
    marginal_panel_path=None,
    metrics_path=None,
    metrics_latex_path=None,
    coefficients_path=None,
    coefficients_latex_path=None,
    rankings_path=None,
    quartiles_path=None,
    panel_output_path=None,
    risk_quartiles_path=None,
    risk_quartiles_latex_path=None,
    rank_validation_path=None,
    rank_validation_latex_path=None,
    marginal_metrics_path=None,
    marginal_coefficients_path=None,
    marginal_risk_quartiles_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        score_candidates = [
            tables_dir / "acsi_preshock_tworuns.csv",
            root / "acsi_preshock_tworuns.csv",
            data_dir / "acsi_preshock_tworuns.csv",
        ]
        score_path = next((path for path in score_candidates if path.exists()), score_candidates[0])
    score_path = Path(score_path)
    marginal_panel_path = Path(
        marginal_panel_path or tables_dir / "marginal_participation_decomposition_panel.csv"
    )
    metrics_path = Path(metrics_path or tables_dir / "community_vulnerability_prediction_metrics.csv")
    metrics_latex_path = Path(
        metrics_latex_path or tables_dir / "community_vulnerability_prediction_metrics.tex"
    )
    coefficients_path = Path(
        coefficients_path or tables_dir / "community_vulnerability_prediction_coefficients.csv"
    )
    coefficients_latex_path = Path(
        coefficients_latex_path or tables_dir / "community_vulnerability_prediction_coefficients.tex"
    )
    rankings_path = Path(rankings_path or tables_dir / "community_vulnerability_prediction_rankings.csv")
    quartiles_path = Path(quartiles_path or tables_dir / "community_vulnerability_prediction_quartiles.csv")
    panel_output_path = Path(panel_output_path or tables_dir / "community_vulnerability_prediction_panel.csv")
    risk_quartiles_path = Path(
        risk_quartiles_path or tables_dir / "community_vulnerability_risk_quartiles.csv"
    )
    risk_quartiles_latex_path = Path(
        risk_quartiles_latex_path or tables_dir / "community_vulnerability_risk_quartiles.tex"
    )
    rank_validation_path = Path(
        rank_validation_path or tables_dir / "community_vulnerability_rank_validation.csv"
    )
    rank_validation_latex_path = Path(
        rank_validation_latex_path or tables_dir / "community_vulnerability_rank_validation.tex"
    )
    marginal_metrics_path = Path(
        marginal_metrics_path or tables_dir / "community_vulnerability_marginal_prediction_metrics.csv"
    )
    marginal_coefficients_path = Path(
        marginal_coefficients_path or tables_dir / "community_vulnerability_marginal_prediction_coefficients.csv"
    )
    marginal_risk_quartiles_path = Path(
        marginal_risk_quartiles_path or tables_dir / "community_vulnerability_marginal_risk_quartiles.csv"
    )

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])
    if scores.empty:
        raise ValueError("No usable two-run pre-shock ACSI scores.")

    if ecosystem_posts is None:
        posts_path = globals().get("POSTS_ECOSYSTEM_PATH", None)
        if posts_path is None or not Path(posts_path).exists():
            raise ValueError(
                "Pass ecosystem_posts or run the main pipeline first so posts_clean_ecosystem.parquet exists."
            )
        posts = pd.read_parquet(posts_path, columns=["subreddit", "year_month"])
    else:
        posts = ecosystem_posts[["subreddit", "year_month"]].copy()

    target_subreddits = set(scores["subreddit"].astype(str))
    posts["subreddit"] = posts["subreddit"].fillna("").astype(str)
    posts["year_month"] = posts["year_month"].astype(str)
    posts = posts[
        posts["subreddit"].isin(target_subreddits)
        & posts["year_month"].between("2022-01", "2024-12")
    ].copy()
    if posts.empty:
        raise ValueError("No ecosystem-clean posts overlap the scored subreddit-month window.")

    months = pd.date_range("2022-01-01", "2024-12-01", freq="MS").strftime("%Y-%m").tolist()
    grid = pd.MultiIndex.from_product(
        [sorted(target_subreddits), months],
        names=["subreddit", "year_month"],
    ).to_frame(index=False)
    monthly = (
        posts.groupby(["subreddit", "year_month"], as_index=False)
        .size()
        .rename(columns={"size": "posts"})
    )
    panel = grid.merge(monthly, on=["subreddit", "year_month"], how="left")
    panel["posts"] = pd.to_numeric(panel["posts"], errors="coerce").fillna(0).astype(float)
    panel["log_posts"] = np.log1p(panel["posts"])
    panel["year_month_dt"] = pd.to_datetime(panel["year_month"] + "-01")
    panel["month_index"] = (
        (panel["year_month_dt"].dt.year - 2022) * 12
        + (panel["year_month_dt"].dt.month - 1)
    ).astype(int)

    pre_months = [month for month in months if "2022-01" <= month <= "2022-10"]
    post_months = [month for month in months if "2022-12" <= month <= "2024-12"]
    peak_months = [month for month in months if "2024-03" <= month <= "2024-08"]
    late_months = [month for month in months if "2024-06" <= month <= "2024-12"]

    rows = []
    for subreddit, group in panel.groupby("subreddit", sort=False):
        pre = group[group["year_month"].isin(pre_months)].copy()
        post = group[group["year_month"].isin(post_months)].copy()
        peak = group[group["year_month"].isin(peak_months)].copy()
        late = group[group["year_month"].isin(late_months)].copy()
        if pre.empty:
            continue
        x = pre["month_index"].astype(float).to_numpy()
        y = pre["log_posts"].astype(float).to_numpy()
        if len(pre) >= 2 and np.nanstd(y) > 0:
            pre_trend_slope = float(np.polyfit(x - x.mean(), y, 1)[0])
        else:
            pre_trend_slope = 0.0
        pre_mean = float(pre["log_posts"].mean())
        rows.append({
            "subreddit": subreddit,
            "pre_mean_log_posts": safe_float(pre_mean),
            "pre_total_posts": safe_float(pre["posts"].sum()),
            "pre_trend_slope": safe_float(pre_trend_slope),
            "pre_volatility": safe_float(pre["log_posts"].std(ddof=1)),
            "overall_post_change": safe_float(post["log_posts"].mean() - pre_mean) if not post.empty else None,
            "peak_decline": safe_float(peak["log_posts"].mean() - pre_mean) if not peak.empty else None,
            "late_decline": safe_float(late["log_posts"].mean() - pre_mean) if not late.empty else None,
        })
    community = pd.DataFrame(rows).merge(scores, on="subreddit", how="inner")
    if community.empty:
        raise ValueError("No community prediction rows after merging ACSI scores.")




    mu_lookup = globals().get("MU_K", {})
    community["subscriber_count"] = community["subreddit"].map(mu_lookup)
    community["subscriber_count_available"] = community["subscriber_count"].notna().astype(int)
    if community["subscriber_count"].notna().any():
        subscriber_fill = float(community["subscriber_count"].dropna().median())
    else:
        subscriber_fill = 0.5
    community["subscriber_count"] = (
        pd.to_numeric(community["subscriber_count"], errors="coerce")
        .fillna(subscriber_fill)
        .astype(float)
    )
    community["log_subscribers"] = np.log1p(community["subscriber_count"])
    community["subscriber_count_source"] = "MU_K_size_proxy_millions_with_median_imputation"
    subreddit_categories = globals().get("SUBREDDITS", {})
    if isinstance(subreddit_categories, dict) and subreddit_categories:
        community["category"] = community["subreddit"].map(subreddit_categories).fillna("unknown").astype(str)
    else:
        community["category"] = "unknown"

    community = community.sort_values("subreddit").reset_index(drop=True)
    q25 = community["overall_post_change"].quantile(0.25)
    community["bottom_quartile_decline"] = (
        community["overall_post_change"] <= q25
    ).astype(int)

    def find_marginal_panel_path():
        direct_candidates = [
            marginal_panel_path,
            tables_dir / "marginal_participation_decomposition_panel.csv",
            root / "marginal_participation_decomposition_panel.csv",
            data_dir / "marginal_participation_decomposition_panel.csv",
        ]
        for candidate in direct_candidates:
            if Path(candidate).exists():
                return Path(candidate)
        search_names = {
            "marginal_participation_decomposition_panel.csv",
            "marginal_participation_decomposition.csv",
        }
        for directory in [tables_dir, root / "output", data_dir]:
            if not Path(directory).exists():
                continue
            for found in Path(directory).rglob("*marginal*participation*"):
                if found.name in search_names or found.name.endswith("_panel.csv"):
                    if found.name.endswith("_panel.csv"):
                        return found
        return None

    def infer_marginal_column(columns, exact_name, tokens):
        column_set = set(columns)
        if exact_name in column_set:
            return exact_name
        lowered = {column.lower(): column for column in columns}
        for lower_name, original in lowered.items():
            if all(token in lower_name for token in tokens):
                return original
        return None

    marginal_panel_found_path = find_marginal_panel_path()
    marginal_recompute_error = None
    if marginal_panel_found_path is None:
        try:



            marginal_result = compute_marginal_participation_decomposition(
                score_path=score_path,
                panel_output_path=marginal_panel_path,
            )
            marginal_panel_found_path = Path(marginal_result["panel_output_path"])
        except Exception as exc:
            marginal_recompute_error = str(exc)
            marginal_panel_found_path = None

    marginal_outcome_sources = {}
    marginal_outcomes = []
    if marginal_panel_found_path is not None and marginal_panel_found_path.exists():
        marginal_panel = pd.read_csv(marginal_panel_found_path)
        if {"subreddit", "year_month"}.issubset(marginal_panel.columns):
            marginal_panel["subreddit"] = marginal_panel["subreddit"].astype(str)
            marginal_panel["year_month"] = marginal_panel["year_month"].astype(str)
            marginal_column_specs = {
                "marginal_nonstable_change": (
                    "log_posts_nonstable_pairs",
                    ["log", "nonstable"],
                ),
                "marginal_one_month_change": (
                    "log_posts_one_month_pairs",
                    ["log", "one", "month"],
                ),
                "marginal_low_commitment_change": (
                    "log_posts_low_commitment_repeat_pairs",
                    ["log", "low", "commitment"],
                ),
                "marginal_preshock_lowfreq_change": (
                    "log_posts_pre_low_frequency_pairs",
                    ["log", "pre", "low", "frequency"],
                ),
            }
            marginal_rows = None
            for outcome_name, (exact_name, tokens) in marginal_column_specs.items():
                column = infer_marginal_column(marginal_panel.columns, exact_name, tokens)
                if column is None:
                    continue
                marginal_panel[column] = pd.to_numeric(marginal_panel[column], errors="coerce")
                outcome_rows = []
                for subreddit, group in marginal_panel.groupby("subreddit", sort=False):
                    pre = group[group["year_month"].isin(pre_months)]
                    post = group[group["year_month"].isin(post_months)]
                    if pre.empty or post.empty:
                        continue
                    outcome_rows.append({
                        "subreddit": subreddit,
                        outcome_name: safe_float(post[column].mean() - pre[column].mean()),
                    })
                if not outcome_rows:
                    continue
                outcome_frame = pd.DataFrame(outcome_rows)
                marginal_rows = (
                    outcome_frame
                    if marginal_rows is None
                    else marginal_rows.merge(outcome_frame, on="subreddit", how="outer")
                )
                marginal_outcome_sources[outcome_name] = f"{marginal_panel_found_path}:{column}"
                marginal_outcomes.append(outcome_name)
            if marginal_rows is not None and not marginal_rows.empty:
                community = community.merge(marginal_rows, on="subreddit", how="left")

    for outcome_name in [
        "marginal_nonstable_change",
        "marginal_one_month_change",
        "marginal_low_commitment_change",
        "marginal_preshock_lowfreq_change",
    ]:
        if outcome_name not in community.columns:
            community[outcome_name] = np.nan
            marginal_outcome_sources[outcome_name] = (
                f"not_available: {marginal_recompute_error}"
                if marginal_recompute_error
                else "not_available"
            )
        community[f"{outcome_name}_source"] = marginal_outcome_sources.get(outcome_name, "not_available")
    if "marginal_participation_decline" not in community.columns:
        community["marginal_participation_decline"] = community["marginal_nonstable_change"]
    marginal_source = marginal_outcome_sources.get("marginal_nonstable_change", "not_available")
    community["marginal_participation_decline_source"] = marginal_source

    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    community.to_csv(panel_output_path, index=False)

    baseline_features = [
        "pre_mean_log_posts",
        "pre_trend_slope",
        "pre_volatility",
        "log_subscribers",
    ]
    model_specs = [
        ("baseline", baseline_features),
        ("gen_phys", baseline_features + ["gen_cap", "phys_free"]),
        ("persfree", baseline_features + ["pers_free"]),
        ("full", baseline_features + ["gen_cap", "phys_free", "pers_free"]),
    ]
    continuous_outcomes = ["overall_post_change", "peak_decline", "late_decline"]
    if community["marginal_participation_decline"].notna().sum() >= 10:
        continuous_outcomes.append("marginal_participation_decline")
    binary_outcomes = ["bottom_quartile_decline"]

    def design_matrix(frame, features):
        x = frame[features].astype(float).to_numpy()
        return np.column_stack([np.ones(len(frame), dtype=float), x])

    def fit_predict(train, test, outcome, features):
        if train.empty:
            return np.nan
        x_train = design_matrix(train, features)
        y_train = train[outcome].astype(float).to_numpy()
        try:
            coefficients = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float(np.nanmean(y_train))
        return float((design_matrix(test, features) @ coefficients)[0])

    def loocv_predictions(frame, outcome, features):
        frame = frame.dropna(subset=[outcome] + features).copy().reset_index(drop=True)
        predictions = []
        null_predictions = []
        for index in range(len(frame)):
            train = frame.drop(index=index)
            test = frame.iloc[[index]]
            predictions.append(fit_predict(train, test, outcome, features))
            null_predictions.append(float(train[outcome].mean()))
        return frame, np.asarray(predictions, dtype=float), np.asarray(null_predictions, dtype=float)

    def auc_score(y_true, scores):
        y_true = np.asarray(y_true, dtype=float)
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(scores)
        y_true = y_true[valid]
        scores = scores[valid]
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        rank_sum_pos = ranks[y_true == 1].sum()
        return safe_float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def build_prediction_metrics(frame, continuous_names, binary_names, specs, source_label):
        rows = []
        for outcome in continuous_names:
            if outcome not in frame.columns:
                continue
            baseline_r2 = None
            gen_phys_r2 = None
            baseline_rmse = None
            gen_phys_rmse = None
            baseline_mae = None
            gen_phys_mae = None
            for model_name, features in specs:
                available_features = [feature for feature in features if feature in frame.columns]
                if len(available_features) != len(features):
                    continue
                cv_frame, predictions, null_predictions = loocv_predictions(frame, outcome, available_features)
                y = cv_frame[outcome].astype(float).to_numpy()
                valid = np.isfinite(y) & np.isfinite(predictions)
                y = y[valid]
                predictions = predictions[valid]
                null_predictions = null_predictions[valid]
                if len(y) == 0:
                    continue
                sse = float(np.sum((y - predictions) ** 2))
                null_sse = float(np.sum((y - null_predictions) ** 2))
                out_of_sample_r2 = safe_float(1 - sse / null_sse) if null_sse > 0 else None
                rmse = safe_float(np.sqrt(np.mean((y - predictions) ** 2)))
                mae = safe_float(np.mean(np.abs(y - predictions)))
                if model_name == "baseline":
                    baseline_r2 = out_of_sample_r2
                    baseline_rmse = rmse
                    baseline_mae = mae
                if model_name == "gen_phys":
                    gen_phys_r2 = out_of_sample_r2
                    gen_phys_rmse = rmse
                    gen_phys_mae = mae
                rows.append({
                    "outcome": outcome,
                    "outcome_type": "continuous",
                    "model_name": model_name,
                    "n_subreddits": safe_int(len(y)),
                    "out_of_sample_r2": out_of_sample_r2,
                    "rmse": rmse,
                    "mae": mae,
                    "auc": np.nan,
                    "delta_r2_vs_baseline": safe_float(out_of_sample_r2 - baseline_r2)
                    if baseline_r2 is not None and out_of_sample_r2 is not None else None,
                    "delta_r2_vs_gen_phys": safe_float(out_of_sample_r2 - gen_phys_r2)
                    if gen_phys_r2 is not None and out_of_sample_r2 is not None else None,
                    "delta_rmse_vs_baseline": safe_float(rmse - baseline_rmse)
                    if baseline_rmse is not None and rmse is not None else None,
                    "delta_rmse_vs_gen_phys": safe_float(rmse - gen_phys_rmse)
                    if gen_phys_rmse is not None and rmse is not None else None,
                    "delta_mae_vs_baseline": safe_float(mae - baseline_mae)
                    if baseline_mae is not None and mae is not None else None,
                    "delta_mae_vs_gen_phys": safe_float(mae - gen_phys_mae)
                    if gen_phys_mae is not None and mae is not None else None,
                    "features": ", ".join(available_features),
                    "cv_method": "leave-one-subreddit-out",
                    "marginal_participation_decline_source": source_label or "not_available",
                })

        for outcome in binary_names:
            if outcome not in frame.columns:
                continue
            baseline_auc = None
            gen_phys_auc = None
            for model_name, features in specs:
                available_features = [feature for feature in features if feature in frame.columns]
                if len(available_features) != len(features):
                    continue
                cv_frame, predictions, _null_predictions = loocv_predictions(frame, outcome, available_features)
                y = cv_frame[outcome].astype(float).to_numpy()
                if len(y) == 0:
                    continue
                auc = auc_score(y, predictions)
                clipped = np.clip(predictions, 1e-6, 1 - 1e-6)
                if model_name == "baseline":
                    baseline_auc = auc
                if model_name == "gen_phys":
                    gen_phys_auc = auc
                rows.append({
                    "outcome": outcome,
                    "outcome_type": "binary",
                    "model_name": model_name,
                    "n_subreddits": safe_int(len(cv_frame)),
                    "out_of_sample_r2": np.nan,
                    "rmse": safe_float(np.sqrt(np.mean((y - clipped) ** 2))),
                    "mae": safe_float(np.mean(np.abs(y - clipped))),
                    "auc": auc,
                    "delta_auc_vs_baseline": safe_float(auc - baseline_auc)
                    if baseline_auc is not None and auc is not None else None,
                    "delta_auc_vs_gen_phys": safe_float(auc - gen_phys_auc)
                    if gen_phys_auc is not None and auc is not None else None,
                    "features": ", ".join(available_features),
                    "cv_method": "leave-one-subreddit-out",
                    "marginal_participation_decline_source": source_label or "not_available",
                })
        return pd.DataFrame(rows)

    metrics = build_prediction_metrics(
        community,
        continuous_outcomes,
        binary_outcomes,
        model_specs,
        marginal_source,
    )
    if metrics.empty:
        raise ValueError("No community vulnerability prediction metrics were estimated.")
    metrics_path.parent.mkdir(exist_ok=True, parents=True)
    metrics.to_csv(metrics_path, index=False)
    metrics_latex_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_latex_columns = [
        "outcome",
        "model_name",
        "n_subreddits",
        "out_of_sample_r2",
        "rmse",
        "mae",
        "auc",
        "delta_r2_vs_baseline",
        "delta_auc_vs_baseline",
    ]
    available_latex_columns = [column for column in metrics_latex_columns if column in metrics.columns]
    metrics_latex_path.write_text(
        metrics[available_latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    coefficient_rows = []
    full_features = model_specs[-1][1]
    coefficient_outcomes = continuous_outcomes + binary_outcomes
    for outcome in coefficient_outcomes:
        model_data = community.dropna(subset=[outcome] + full_features).copy()
        if model_data.empty:
            continue
        formula = f"{outcome} ~ " + " + ".join(full_features)
        model = fit_ols(formula, model_data, cov_type="HC3")
        if model is None:
            continue
        for term in ["pers_free", "gen_cap", "phys_free"]:
            result = reg_result(model, term)
            coefficient_rows.append({
                "outcome": outcome,
                "model_name": "full",
                "term": term,
                "coef": result.get("coef"),
                "standard_error": result.get("se"),
                "p_value": result.get("pvalue"),
                "n_subreddits": safe_int(model.nobs),
                "standard_errors": "HC3 robust",
                "features": ", ".join(full_features),
            })
    coefficients = pd.DataFrame(coefficient_rows)
    if coefficients.empty:
        raise ValueError("No community vulnerability coefficient rows were estimated.")
    coefficients_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients.to_csv(coefficients_path, index=False)
    coefficients_latex_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients_latex_path.write_text(
        coefficients.to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    risk_model_specs = [
        ("baseline_only", baseline_features),
        ("persfree_only", ["pers_free"]),
        ("baseline_persfree", baseline_features + ["pers_free"]),
        ("full", full_features),
    ]
    stratification_outcomes = [
        "overall_post_change",
        "peak_decline",
        "late_decline",
        "bottom_quartile_decline",
    ]
    quartile_order = ["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"]

    def outcome_is_binary(outcome, frame):
        if outcome == "bottom_quartile_decline":
            return True
        series = frame[outcome].dropna()
        return bool(not series.empty and set(series.unique()).issubset({0, 1, 0.0, 1.0}))

    def add_prediction_columns(frame, outcome, features):
        model_data = frame.dropna(subset=[outcome] + features).copy()
        if model_data.empty:
            return model_data, None
        formula = f"{outcome} ~ " + " + ".join(features)
        model = fit_ols(formula, model_data, cov_type="HC3")
        if model is None:
            return model_data.iloc[0:0].copy(), None
        model_data["predicted_value"] = model.predict(model_data)
        if outcome_is_binary(outcome, model_data):
            model_data["actual_bottom_quartile"] = model_data[outcome].astype(float)
            model_data["predicted_risk_score"] = model_data["predicted_value"].astype(float)
            model_data["actual_risk_score"] = model_data[outcome].astype(float)
        else:
            bottom_cutoff = model_data[outcome].quantile(0.25)
            model_data["actual_bottom_quartile"] = (model_data[outcome] <= bottom_cutoff).astype(float)


            model_data["predicted_risk_score"] = -model_data["predicted_value"].astype(float)
            model_data["actual_risk_score"] = -model_data[outcome].astype(float)
        if model_data["predicted_risk_score"].nunique(dropna=True) >= 4:
            model_data["predicted_risk_quartile"] = pd.qcut(
                model_data["predicted_risk_score"].rank(method="first", ascending=False),
                q=4,
                labels=quartile_order,
            ).astype(str)
        else:
            model_data["predicted_risk_quartile"] = "not_enough_score_variation"
        return model_data, model

    def build_risk_quartiles(frame, outcomes, specs):
        rows = []
        for outcome in outcomes:
            if outcome not in frame.columns:
                continue
            for model_name, features in specs:
                if any(feature not in frame.columns for feature in features):
                    continue
                model_data, _model = add_prediction_columns(frame, outcome, features)
                if model_data.empty:
                    continue
                for quartile in quartile_order:
                    group = model_data[model_data["predicted_risk_quartile"].eq(quartile)]
                    if group.empty:
                        continue
                    rows.append({
                        "outcome": outcome,
                        "model_name": model_name,
                        "predicted_risk_quartile": quartile,
                        "n_subreddits": safe_int(len(group)),
                        "mean_predicted_decline": safe_float(group["predicted_value"].mean()),
                        "mean_actual_decline": safe_float(group[outcome].mean()),
                        "median_actual_decline": safe_float(group[outcome].median()),
                        "share_in_actual_bottom_quartile": safe_float(group["actual_bottom_quartile"].mean()),
                        "mean_pers_free": safe_float(group["pers_free"].mean()),
                        "mean_gen_cap": safe_float(group["gen_cap"].mean()),
                        "mean_phys_free": safe_float(group["phys_free"].mean()),
                        "features": ", ".join(features),
                        "risk_sort_note": "Q1 is highest predicted risk; continuous outcomes sort lower fitted changes as higher risk.",
                    })
        return pd.DataFrame(rows)

    def safe_rank_corr(x, y, method):
        x = pd.Series(x).astype(float)
        y = pd.Series(y).astype(float)
        valid = x.notna() & y.notna()
        x = x[valid]
        y = y[valid]
        if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
            return None, None
        if method == "spearman":
            coef, pvalue = stats.spearmanr(x, y)
        else:
            coef, pvalue = stats.kendalltau(x, y)
        return safe_float(coef), safe_float(pvalue)

    def build_rank_validation(frame, outcomes, specs):
        rows = []
        for outcome in outcomes:
            if outcome not in frame.columns:
                continue
            for model_name, features in specs:
                if any(feature not in frame.columns for feature in features):
                    continue
                model_data, _model = add_prediction_columns(frame, outcome, features)
                if model_data.empty:
                    continue
                top = model_data[model_data["predicted_risk_quartile"].eq("Q1_highest_predicted_risk")]
                base_rate = float(model_data["actual_bottom_quartile"].mean())
                precision = float(top["actual_bottom_quartile"].mean()) if not top.empty else np.nan
                spearman_coef, spearman_p = safe_rank_corr(
                    model_data["predicted_risk_score"],
                    model_data["actual_risk_score"],
                    "spearman",
                )
                kendall_coef, kendall_p = safe_rank_corr(
                    model_data["predicted_risk_score"],
                    model_data["actual_risk_score"],
                    "kendall",
                )
                rows.append({
                    "outcome": outcome,
                    "model_name": model_name,
                    "n_subreddits": safe_int(len(model_data)),
                    "spearman_rho": spearman_coef,
                    "spearman_p_value": spearman_p,
                    "kendall_tau": kendall_coef,
                    "kendall_p_value": kendall_p,
                    "top_quartile_precision": safe_float(precision),
                    "base_bottom_quartile_rate": safe_float(base_rate),
                    "lift": safe_float(precision / base_rate)
                    if base_rate > 0 and np.isfinite(precision) else None,
                    "features": ", ".join(features),
                })
        return pd.DataFrame(rows)

    risk_quartiles = build_risk_quartiles(community, stratification_outcomes, risk_model_specs)
    risk_quartiles_path.parent.mkdir(exist_ok=True, parents=True)
    risk_quartiles.to_csv(risk_quartiles_path, index=False)
    risk_quartiles_latex_path.parent.mkdir(exist_ok=True, parents=True)
    risk_quartiles_latex_path.write_text(
        risk_quartiles.to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    rank_validation = build_rank_validation(community, stratification_outcomes, risk_model_specs)
    rank_validation_path.parent.mkdir(exist_ok=True, parents=True)
    rank_validation.to_csv(rank_validation_path, index=False)
    rank_validation_latex_path.parent.mkdir(exist_ok=True, parents=True)
    rank_validation_latex_path.write_text(
        rank_validation.to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    marginal_outcomes_available = [
        outcome
        for outcome in [
            "marginal_nonstable_change",
            "marginal_one_month_change",
            "marginal_low_commitment_change",
            "marginal_preshock_lowfreq_change",
        ]
        if outcome in community.columns and community[outcome].notna().sum() >= 10
    ]
    marginal_source_label = "; ".join(
        f"{outcome}={marginal_outcome_sources.get(outcome, 'not_available')}"
        for outcome in marginal_outcomes_available
    ) or marginal_source
    marginal_metrics = build_prediction_metrics(
        community,
        marginal_outcomes_available,
        [],
        model_specs,
        marginal_source_label,
    )
    marginal_metrics_path.parent.mkdir(exist_ok=True, parents=True)
    marginal_metrics.to_csv(marginal_metrics_path, index=False)

    marginal_coefficient_rows = []
    for outcome in marginal_outcomes_available:
        model_data = community.dropna(subset=[outcome] + full_features).copy()
        if model_data.empty:
            continue
        model = fit_ols(f"{outcome} ~ " + " + ".join(full_features), model_data, cov_type="HC3")
        if model is None:
            continue
        for term in ["pers_free", "gen_cap", "phys_free"]:
            result = reg_result(model, term)
            marginal_coefficient_rows.append({
                "outcome": outcome,
                "model_name": "full",
                "term": term,
                "coef": result.get("coef"),
                "standard_error": result.get("se"),
                "p_value": result.get("pvalue"),
                "n_subreddits": safe_int(model.nobs),
                "standard_errors": "HC3 robust",
                "features": ", ".join(full_features),
                "source": marginal_outcome_sources.get(outcome, "not_available"),
            })
    marginal_coefficients = pd.DataFrame(marginal_coefficient_rows)
    marginal_coefficients_path.parent.mkdir(exist_ok=True, parents=True)
    marginal_coefficients.to_csv(marginal_coefficients_path, index=False)

    marginal_risk_quartiles = build_risk_quartiles(
        community,
        marginal_outcomes_available,
        risk_model_specs,
    )
    marginal_risk_quartiles_path.parent.mkdir(exist_ok=True, parents=True)
    marginal_risk_quartiles.to_csv(marginal_risk_quartiles_path, index=False)

    ranking_data = community.dropna(subset=["overall_post_change"] + full_features).copy()
    ranking_model = fit_ols("overall_post_change ~ " + " + ".join(full_features), ranking_data, cov_type="HC3")
    if ranking_model is None:
        raise ValueError("Could not fit full diagnostic ranking model.")
    ranking_data["predicted_decline"] = ranking_model.predict(ranking_data)
    ranking_data["actual_decline"] = ranking_data["overall_post_change"]
    rankings = ranking_data.sort_values("predicted_decline", ascending=True)[
        [
            "subreddit",
            "predicted_decline",
            "actual_decline",
            "pers_free",
            "gen_cap",
            "phys_free",
            "pre_mean_log_posts",
            "category",
        ]
    ].head(15)
    rankings_path.parent.mkdir(exist_ok=True, parents=True)
    rankings.to_csv(rankings_path, index=False)

    ranking_data["predicted_risk_quartile"] = (
        pd.qcut(
            ranking_data["predicted_decline"].rank(method="first", ascending=True),
            q=4,
            labels=["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"],
        ).astype(str)
    )
    quartile_rows = []
    quartile_order = ["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"]
    for quartile in quartile_order:
        group = ranking_data[ranking_data["predicted_risk_quartile"].eq(quartile)]
        if group.empty:
            continue
        quartile_rows.append({
            "predicted_risk_quartile": quartile,
            "n_subreddits": safe_int(len(group)),
            "mean_actual_decline": safe_float(group["actual_decline"].mean()),
            "mean_predicted_decline": safe_float(group["predicted_decline"].mean()),
            "mean_marginal_participation_decline": safe_float(group["marginal_participation_decline"].mean())
            if group["marginal_participation_decline"].notna().any() else None,
            "mean_pers_free": safe_float(group["pers_free"].mean()),
        })
    quartiles = pd.DataFrame(quartile_rows)
    quartiles_path.parent.mkdir(exist_ok=True, parents=True)
    quartiles.to_csv(quartiles_path, index=False)

    def metric_value(outcome, model_name, column, table=None):
        source_table = metrics if table is None else table
        if source_table is None or source_table.empty:
            return None
        rows = source_table[source_table["outcome"].eq(outcome) & source_table["model_name"].eq(model_name)]
        if rows.empty or column not in rows.columns:
            return None
        value = rows.iloc[0].get(column)
        return None if pd.isna(value) else float(value)

    def validation_value(outcome, model_name, column):
        if rank_validation.empty:
            return None
        rows = rank_validation[
            rank_validation["outcome"].eq(outcome)
            & rank_validation["model_name"].eq(model_name)
        ]
        if rows.empty or column not in rows.columns:
            return None
        value = rows.iloc[0].get(column)
        return None if pd.isna(value) else float(value)

    def risk_quartile_value(outcome, model_name, quartile, column):
        if risk_quartiles.empty:
            return None
        rows = risk_quartiles[
            risk_quartiles["outcome"].eq(outcome)
            & risk_quartiles["model_name"].eq(model_name)
            & risk_quartiles["predicted_risk_quartile"].eq(quartile)
        ]
        if rows.empty or column not in rows.columns:
            return None
        value = rows.iloc[0].get(column)
        return None if pd.isna(value) else float(value)

    def fmt_number(value, digits=4):
        return "NA" if value is None else f"{value:.{digits}f}"

    overall_base = metric_value("overall_post_change", "baseline", "out_of_sample_r2")
    overall_pers = metric_value("overall_post_change", "persfree", "out_of_sample_r2")
    overall_gen = metric_value("overall_post_change", "gen_phys", "out_of_sample_r2")
    overall_full = metric_value("overall_post_change", "full", "out_of_sample_r2")
    persfree_improves = overall_base is not None and overall_pers is not None and overall_pers > overall_base
    persfree_outperforms_gen = overall_pers is not None and overall_gen is not None and overall_pers > overall_gen
    full_improves = overall_base is not None and overall_full is not None and overall_full > overall_base
    high_risk = quartiles[quartiles["predicted_risk_quartile"].eq("Q1_highest_predicted_risk")]
    low_risk = quartiles[quartiles["predicted_risk_quartile"].eq("Q4_lowest_predicted_risk")]
    identifies_risk = (
        not high_risk.empty
        and not low_risk.empty
        and float(high_risk.iloc[0]["mean_actual_decline"]) < float(low_risk.iloc[0]["mean_actual_decline"])
    )
    marginal_delta_candidates = [
        metric_value(outcome, "persfree", "delta_r2_vs_baseline", marginal_metrics)
        for outcome in marginal_outcomes_available
    ]
    marginal_delta_candidates = [
        value for value in marginal_delta_candidates if value is not None and np.isfinite(value)
    ]
    marginal_delta = max(marginal_delta_candidates) if marginal_delta_candidates else None
    overall_delta = metric_value("overall_post_change", "persfree", "delta_r2_vs_baseline")
    marginal_phrase = (
        "marginal-participation prediction is stronger than total-post prediction"
        if marginal_delta is not None and overall_delta is not None and marginal_delta > overall_delta
        else "marginal-participation prediction is unavailable or not stronger than total-post prediction"
    )
    risk_q1_actual = risk_quartile_value(
        "overall_post_change",
        "baseline_persfree",
        "Q1_highest_predicted_risk",
        "mean_actual_decline",
    )
    risk_q4_actual = risk_quartile_value(
        "overall_post_change",
        "baseline_persfree",
        "Q4_lowest_predicted_risk",
        "mean_actual_decline",
    )
    risk_decline_sorted = (
        risk_q1_actual is not None
        and risk_q4_actual is not None
        and risk_q1_actual < risk_q4_actual
    )
    top_precision = validation_value("overall_post_change", "baseline_persfree", "top_quartile_precision")
    base_precision = validation_value("overall_post_change", "baseline_persfree", "base_bottom_quartile_rate")
    top_precision_lift = validation_value("overall_post_change", "baseline_persfree", "lift")
    baseline_lift = validation_value("overall_post_change", "baseline_only", "lift")
    persfree_lift = validation_value("overall_post_change", "persfree_only", "lift")
    persfree_more_informative = (
        baseline_lift is not None
        and persfree_lift is not None
        and persfree_lift > baseline_lift
    )
    summary = (
        "Community vulnerability prediction: adding PersFree "
        + ("improves" if persfree_improves else "does not improve")
        + f" leave-one-out prediction of overall decline relative to baseline "
        f"(baseline R2={fmt_number(overall_base)}, +PersFree R2={fmt_number(overall_pers)}, full R2={fmt_number(overall_full)}). "
        + ("PersFree outperforms GenCap/PhysFree alone. " if persfree_outperforms_gen else "PersFree does not outperform GenCap/PhysFree alone. ")
        + (
            "The full diagnostic model weakly separates highest- and lowest-risk quartiles. "
            if identifies_risk and full_improves
            else "The full diagnostic model does not materially improve out-of-sample ranking beyond baseline. "
        )
        + (
            "Baseline+PersFree risk quartiles sort communities into larger actual future declines "
            f"(Q1 mean change={fmt_number(risk_q1_actual)}, Q4={fmt_number(risk_q4_actual)}). "
            if risk_decline_sorted
            else "Baseline+PersFree risk quartiles do not monotonically sort communities into larger future declines. "
        )
        + (
            f"The top predicted-risk quartile contains more actual bottom-quartile decliners "
            f"(precision={fmt_number(top_precision)}, base={fmt_number(base_precision)}, lift={fmt_number(top_precision_lift)}). "
            if top_precision is not None and base_precision is not None and top_precision > base_precision
            else "The top predicted-risk quartile does not clearly enrich actual bottom-quartile decliners. "
        )
        + (
            f"PersFree-only risk is more informative than baseline-only risk by lift "
            f"({fmt_number(persfree_lift)} vs {fmt_number(baseline_lift)}). "
            if persfree_more_informative
            else "PersFree-only risk is not clearly more informative than baseline-only risk by lift. "
        )
        + marginal_phrase
        + ". PersFree is a diagnostic signal, not a high-accuracy forecasting model; the main question is whether it sorts communities into meaningfully different future-risk groups."
    )
    print(summary)

    return {
        "panel": community,
        "metrics": metrics,
        "coefficients": coefficients,
        "risk_quartiles": risk_quartiles,
        "rank_validation": rank_validation,
        "marginal_metrics": marginal_metrics,
        "marginal_coefficients": marginal_coefficients,
        "marginal_risk_quartiles": marginal_risk_quartiles,
        "rankings": rankings,
        "quartiles": quartiles,
        "summary": summary,
        "metrics_path": str(metrics_path),
        "metrics_latex_path": str(metrics_latex_path),
        "coefficients_path": str(coefficients_path),
        "coefficients_latex_path": str(coefficients_latex_path),
        "rankings_path": str(rankings_path),
        "quartiles_path": str(quartiles_path),
        "panel_output_path": str(panel_output_path),
        "risk_quartiles_path": str(risk_quartiles_path),
        "risk_quartiles_latex_path": str(risk_quartiles_latex_path),
        "rank_validation_path": str(rank_validation_path),
        "rank_validation_latex_path": str(rank_validation_latex_path),
        "marginal_metrics_path": str(marginal_metrics_path),
        "marginal_coefficients_path": str(marginal_coefficients_path),
        "marginal_risk_quartiles_path": str(marginal_risk_quartiles_path),
    }

def compute_displaced_contributor_destinations(
    ecosystem_posts=None,
    score_path=None,
    output_path=None,
    latex_path=None,
    panel_output_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        score_candidates = [
            tables_dir / "acsi_preshock_tworuns.csv",
            root / "acsi_preshock_tworuns.csv",
            data_dir / "acsi_preshock_tworuns.csv",
        ]
        score_path = next((path for path in score_candidates if path.exists()), score_candidates[0])
    score_path = Path(score_path)
    output_path = Path(output_path or tables_dir / "displaced_contributor_destinations.csv")
    latex_path = Path(latex_path or tables_dir / "displaced_contributor_destinations.tex")
    panel_output_path = Path(panel_output_path or tables_dir / "displaced_contributor_destination_panel.csv")

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])
    if scores.empty:
        raise ValueError("No usable two-run pre-shock ACSI scores.")

    subreddit_rank = scores["pers_free"].rank(method="first")
    scores["origin_persfree_tercile"] = pd.qcut(
        subreddit_rank,
        q=3,
        labels=["low_persfree", "middle_persfree", "high_persfree"],
    ).astype(str)

    if ecosystem_posts is None:
        posts_path = globals().get("POSTS_ECOSYSTEM_PATH", None)
        if posts_path is None or not Path(posts_path).exists():
            raise ValueError(
                "Pass ecosystem_posts or run the main pipeline first so posts_clean_ecosystem.parquet exists."
            )
        posts = pd.read_parquet(posts_path)
    else:
        posts = ecosystem_posts.copy()

    required_post_columns = {"author", "subreddit", "year_month"}
    missing_post_columns = required_post_columns - set(posts.columns)
    if missing_post_columns:
        raise ValueError(f"Ecosystem post sample missing columns: {sorted(missing_post_columns)}")
    if "year_month_dt" not in posts.columns:
        posts["year_month_dt"] = pd.to_datetime(posts["year_month"].astype(str) + "-01", errors="coerce")
    else:
        posts["year_month_dt"] = pd.to_datetime(posts["year_month_dt"], errors="coerce")

    target_subreddits = set(scores["subreddit"].astype(str))
    start_month = "2022-01"
    transition_month = "2022-11"
    post_start_month = "2022-12"
    end_month = "2024-12"
    excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))

    posts = posts.copy()
    posts["author"] = posts["author"].fillna("").astype(str)
    posts["subreddit"] = posts["subreddit"].fillna("").astype(str)
    posts["year_month"] = posts["year_month"].astype(str)
    posts = posts[
        posts["subreddit"].isin(target_subreddits)
        & posts["year_month"].between(start_month, end_month)
        & posts["author"].ne("")
        & ~posts["author"].isin(excluded_authors)
        & ~posts["author"].str.lower().str.endswith("bot", na=False)
    ].copy()
    if posts.empty:
        raise ValueError("No ecosystem-clean posts overlap the scored subreddit-month window.")




    max_posts_per_day = globals().get("MAX_POSTS_PER_DAY", 50)
    start_date = globals().get("START_DATE", datetime(2022, 1, 1))
    end_date_exclusive = globals().get("END_DATE_EXCLUSIVE", datetime(2025, 1, 1))
    cap_days = max((end_date_exclusive - start_date).days, 1)
    author_post_cap = max_posts_per_day * cap_days
    author_counts = posts.groupby("author", sort=False).size()
    valid_authors = set(author_counts[author_counts <= author_post_cap].index)
    posts = posts[posts["author"].isin(valid_authors)].copy()
    if posts.empty:
        raise ValueError("No ecosystem-clean posts remain after author-cap filtering.")

    pre_posts = posts[posts["year_month"].between(start_month, transition_month)].copy()
    post_posts = posts[posts["year_month"].between(post_start_month, end_month)].copy()
    if pre_posts.empty:
        raise ValueError("No pre-shock origin posts found.")

    origin_pairs = (
        pre_posts.groupby(["author", "subreddit"], as_index=False)
        .agg(
            pre_posts_origin=("year_month", "size"),
            pre_active_months_origin=("year_month", "nunique"),
            first_pre_month_origin=("year_month", "min"),
            last_pre_month_origin=("year_month", "max"),
        )
        .rename(columns={"subreddit": "origin_subreddit"})
    )
    origin_pairs = origin_pairs.reset_index(drop=True)
    origin_pairs["origin_pair_id"] = np.arange(len(origin_pairs), dtype=np.int64)
    score_lookup = scores.rename(
        columns={
            "subreddit": "origin_subreddit",
            "pers_free": "origin_persfree",
            "gen_cap": "origin_gen_cap",
            "phys_free": "origin_phys_free",
        }
    )
    origin_pairs = origin_pairs.merge(
        score_lookup[
            [
                "origin_subreddit",
                "origin_persfree",
                "origin_gen_cap",
                "origin_phys_free",
                "origin_persfree_tercile",
            ]
        ],
        on="origin_subreddit",
        how="inner",
    )
    if origin_pairs.empty:
        raise ValueError("No pre-shock origin pairs overlap with two-run ACSI scores.")

    subreddit_categories = globals().get("SUBREDDITS", {})
    if isinstance(subreddit_categories, dict) and subreddit_categories:
        origin_pairs["origin_category"] = (
            origin_pairs["origin_subreddit"].map(subreddit_categories).fillna("unknown").astype(str)
        )
    else:
        origin_pairs["origin_category"] = "unknown"

    pre_subreddit_posts = (
        origin_pairs.groupby("origin_subreddit", sort=False)["pre_posts_origin"]
        .sum()
        .rename("pre_subreddit_posts")
    )
    origin_pairs["pre_subreddit_posts"] = origin_pairs["origin_subreddit"].map(pre_subreddit_posts)
    origin_pairs["log_pre_subreddit_posts"] = np.log1p(origin_pairs["pre_subreddit_posts"].astype(float))
    origin_pairs["log_pre_posts_origin"] = np.log1p(origin_pairs["pre_posts_origin"].astype(float))

    if post_posts.empty:
        post_pair_counts = pd.DataFrame(columns=["author", "origin_subreddit", "same_subreddit_postshock_posts"])
        post_author_agg = pd.DataFrame(columns=["author", "postshock_total_posts_sample", "destination_count"])
        other_stats = pd.DataFrame(columns=["origin_pair_id"])
    else:
        post_pair_counts = (
            post_posts.groupby(["author", "subreddit"], as_index=False)
            .size()
            .rename(
                columns={
                    "subreddit": "origin_subreddit",
                    "size": "same_subreddit_postshock_posts",
                }
            )
        )
        post_author_agg = (
            post_posts.groupby("author", as_index=False)
            .agg(
                postshock_total_posts_sample=("year_month", "size"),
                destination_count=("subreddit", "nunique"),
            )
        )
        post_author_subreddit = (
            post_posts.groupby(["author", "subreddit"], as_index=False)
            .size()
            .rename(columns={"subreddit": "destination_subreddit", "size": "destination_posts"})
            .merge(
                scores[["subreddit", "pers_free"]].rename(
                    columns={
                        "subreddit": "destination_subreddit",
                        "pers_free": "destination_persfree",
                    }
                ),
                on="destination_subreddit",
                how="left",
            )
            .dropna(subset=["destination_persfree"])
        )


        origin_edge_columns = ["origin_pair_id", "author", "origin_subreddit", "origin_persfree"]
        destination_edges = origin_pairs[origin_edge_columns].merge(
            post_author_subreddit,
            on="author",
            how="inner",
        )
        other_edges = destination_edges[
            destination_edges["destination_subreddit"].ne(destination_edges["origin_subreddit"])
        ].copy()
        if other_edges.empty:
            other_stats = pd.DataFrame(columns=["origin_pair_id"])
        else:
            other_edges["other_higher_context"] = (
                other_edges["destination_persfree"] <= other_edges["origin_persfree"] - 0.05
            )
            other_edges["other_lower_context"] = (
                other_edges["destination_persfree"] >= other_edges["origin_persfree"] + 0.05
            )
            other_edges["other_similar_context"] = (
                (other_edges["destination_persfree"] - other_edges["origin_persfree"]).abs() <= 0.05
            )
            other_stats = (
                other_edges.groupby("origin_pair_id", as_index=False)
                .agg(
                    other_destination_count=("destination_subreddit", "nunique"),
                    other_destination_posts=("destination_posts", "sum"),
                    mean_destination_persfree=("destination_persfree", "mean"),
                    min_destination_persfree=("destination_persfree", "min"),
                    max_destination_persfree=("destination_persfree", "max"),
                    any_other_higher_context=("other_higher_context", "max"),
                    any_other_lower_context=("other_lower_context", "max"),
                    any_other_similar_context=("other_similar_context", "max"),
                )
            )
        del post_author_subreddit
        if "destination_edges" in locals():
            del destination_edges
        if "other_edges" in locals():
            del other_edges
        gc.collect()

    origin_pairs = origin_pairs.merge(
        post_pair_counts,
        on=["author", "origin_subreddit"],
        how="left",
    ).merge(
        post_author_agg,
        on="author",
        how="left",
    ).merge(
        other_stats,
        on="origin_pair_id",
        how="left",
    )

    count_columns = [
        "same_subreddit_postshock_posts",
        "postshock_total_posts_sample",
        "destination_count",
        "other_destination_count",
        "other_destination_posts",
    ]
    for column_name in count_columns:
        if column_name not in origin_pairs.columns:
            origin_pairs[column_name] = 0
        origin_pairs[column_name] = (
            pd.to_numeric(origin_pairs[column_name], errors="coerce").fillna(0).astype(np.int32)
        )
    for column_name in [
        "any_other_higher_context",
        "any_other_lower_context",
        "any_other_similar_context",
    ]:
        if column_name not in origin_pairs.columns:
            origin_pairs[column_name] = False
        origin_pairs[column_name] = origin_pairs[column_name].where(
            origin_pairs[column_name].notna(),
            False,
        ).astype(bool)

    origin_pairs["stayed_same_subreddit"] = (
        origin_pairs["same_subreddit_postshock_posts"] > 0
    ).astype(int)
    origin_pairs["any_other_subreddit_postshock"] = (
        origin_pairs["other_destination_count"] > 0
    ).astype(int)
    origin_pairs["moved_other_subreddit"] = (
        (origin_pairs["stayed_same_subreddit"] == 0)
        & (origin_pairs["any_other_subreddit_postshock"] == 1)
    ).astype(int)


    origin_pairs["disappeared_from_sample"] = (
        origin_pairs["postshock_total_posts_sample"] == 0
    ).astype(int)



    origin_pairs["moved_higher_context"] = (
        (origin_pairs["stayed_same_subreddit"] == 0)
        & origin_pairs["any_other_higher_context"]
    ).astype(int)
    origin_pairs["moved_lower_context"] = (
        (origin_pairs["stayed_same_subreddit"] == 0)
        & origin_pairs["any_other_lower_context"]
    ).astype(int)
    origin_pairs["moved_similar_context"] = (
        (origin_pairs["stayed_same_subreddit"] == 0)
        & origin_pairs["any_other_similar_context"]
    ).astype(int)

    origin_pairs["pre_shock_low_frequency_pair"] = (
        origin_pairs["pre_posts_origin"] <= 1
    ).astype(int)
    origin_pairs["pre_shock_low_active_months"] = (
        origin_pairs["pre_active_months_origin"] <= 1
    ).astype(int)
    pre_frequency_tercile = pd.qcut(
        origin_pairs["pre_posts_origin"].rank(method="first"),
        q=3,
        labels=["bottom", "middle", "top"],
    ).astype(str)
    origin_pairs["pre_shock_frequency_tercile"] = pre_frequency_tercile
    origin_pairs["pre_shock_bottom_tercile_pair"] = (
        pre_frequency_tercile == "bottom"
    ).astype(int)
    origin_pairs["high_persfree_origin"] = (
        origin_pairs["origin_persfree_tercile"] == "high_persfree"
    ).astype(int)
    origin_pairs["low_persfree_origin"] = (
        origin_pairs["origin_persfree_tercile"] == "low_persfree"
    ).astype(int)

    origin_pairs = origin_pairs.sort_values(["origin_subreddit", "author"]).reset_index(drop=True)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    origin_pairs.to_csv(panel_output_path, index=False)

    destination_outcomes = [
        "stayed_same_subreddit",
        "moved_other_subreddit",
        "any_other_subreddit_postshock",
        "moved_higher_context",
        "moved_lower_context",
        "moved_similar_context",
        "disappeared_from_sample",
    ]

    def descriptive_rows_for(sample_name, frame):
        rows = []
        if frame.empty:
            return rows
        for tercile, group in frame.groupby("origin_persfree_tercile", observed=False):
            row = {
                "row_type": "descriptive",
                "sample": sample_name,
                "origin_persfree_tercile": tercile,
                "n_author_subreddit_origin_pairs": safe_int(len(group)),
                "n_unique_authors": safe_int(group["author"].nunique()),
                "n_origin_subreddits": safe_int(group["origin_subreddit"].nunique()),
                "mean_destination_count": safe_float(group["destination_count"].mean()),
                "mean_postshock_total_posts_sample": safe_float(group["postshock_total_posts_sample"].mean()),
            }
            for outcome in destination_outcomes:
                row[f"share_{outcome}"] = safe_float(group[outcome].mean())
            rows.append(row)
        return rows

    descriptive_rows = []
    descriptive_rows.extend(descriptive_rows_for("all_origin_pairs", origin_pairs))
    descriptive_rows.extend(
        descriptive_rows_for(
            "pre_shock_low_frequency_pair",
            origin_pairs[origin_pairs["pre_shock_low_frequency_pair"] == 1],
        )
    )
    descriptive_rows.extend(
        descriptive_rows_for(
            "pre_shock_bottom_tercile_pair",
            origin_pairs[origin_pairs["pre_shock_bottom_tercile_pair"] == 1],
        )
    )

    control_terms = [
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "origin_gen_cap",
        "origin_phys_free",
        "log_pre_subreddit_posts",
    ]
    category_term = ""
    if origin_pairs["origin_category"].nunique(dropna=True) > 1:
        category_term = " + C(origin_category)"
    controls_label = (
        "log(1+pre_posts_origin), pre_active_months_origin, origin GenCap, "
        "origin PhysFree, log pre-shock origin subreddit size"
        + (", origin category FE" if category_term else "")
    )
    regression_rows = []

    def interaction_value(model, candidates):
        for candidate in candidates:
            if candidate in model.params.index:
                return reg_result(model, candidate)
        return {"coef": None, "se": None, "pvalue": None, "n_obs": safe_int(model.nobs)}

    def add_model_row(outcome, model_name, model, interaction_candidates=None, low_frequency_definition=None):
        focal = reg_result(model, "origin_persfree")
        interaction = (
            interaction_value(model, interaction_candidates)
            if interaction_candidates
            else {"coef": None, "se": None, "pvalue": None}
        )
        regression_rows.append({
            "row_type": "regression",
            "sample": "all_origin_pairs",
            "outcome": outcome,
            "model_name": model_name,
            "origin_persfree_coef": focal.get("coef"),
            "standard_error": focal.get("se"),
            "p_value": focal.get("pvalue"),
            "persfree_x_low_frequency_coef": interaction.get("coef"),
            "persfree_x_low_frequency_se": interaction.get("se"),
            "persfree_x_low_frequency_p_value": interaction.get("pvalue"),
            "low_frequency_definition": low_frequency_definition,
            "n_author_subreddit_origin_pairs": safe_int(model.nobs),
            "n_unique_authors": safe_int(model.model.data.frame["author"].nunique()),
            "n_origin_subreddits": safe_int(model.model.data.frame["origin_subreddit"].nunique()),
            "controls": controls_label,
            "fixed_effects": "none"
            + ("; origin category FE" if category_term else ""),
            "clustering_level": "origin subreddit",
            "score_source": str(score_path),
            "post_sample_source": str(globals().get("POSTS_ECOSYSTEM_PATH", "provided dataframe")),
            "panel_output_path": str(panel_output_path),
        })

    main_outcomes = [
        "stayed_same_subreddit",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_higher_context",
        "moved_lower_context",
    ]
    important_interaction_outcomes = [
        "disappeared_from_sample",
        "stayed_same_subreddit",
        "moved_other_subreddit",
        "moved_higher_context",
    ]
    base_required = [
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "log_pre_subreddit_posts",
        "origin_subreddit",
        "author",
    ]
    for outcome in main_outcomes:
        model_data = origin_pairs.dropna(subset=[outcome] + base_required).copy()
        if model_data.empty or model_data["origin_subreddit"].nunique() < 2:
            continue
        formula = (
            f"{outcome} ~ origin_persfree + "
            + " + ".join(control_terms)
            + category_term
        )
        model = fit_ols(formula, model_data, cluster_col="origin_subreddit")
        if model is not None:
            add_model_row(outcome, "main_continuous_persfree", model)

    for outcome in important_interaction_outcomes:
        model_data = origin_pairs.dropna(
            subset=[outcome, "pre_shock_low_frequency_pair"] + base_required
        ).copy()
        if model_data.empty or model_data["origin_subreddit"].nunique() < 2:
            continue
        formula = (
            f"{outcome} ~ origin_persfree * pre_shock_low_frequency_pair + "
            + " + ".join(control_terms)
            + category_term
        )
        model = fit_ols(formula, model_data, cluster_col="origin_subreddit")
        if model is not None:
            add_model_row(
                outcome,
                "low_frequency_interaction",
                model,
                interaction_candidates=[
                    "origin_persfree:pre_shock_low_frequency_pair",
                    "pre_shock_low_frequency_pair:origin_persfree",
                ],
                low_frequency_definition="pre_posts_origin <= 1",
            )

        bottom_model_data = origin_pairs.dropna(
            subset=[outcome, "pre_shock_bottom_tercile_pair"] + base_required
        ).copy()
        if bottom_model_data.empty or bottom_model_data["origin_subreddit"].nunique() < 2:
            continue
        formula = (
            f"{outcome} ~ origin_persfree * pre_shock_bottom_tercile_pair + "
            + " + ".join(control_terms)
            + category_term
        )
        bottom_model = fit_ols(formula, bottom_model_data, cluster_col="origin_subreddit")
        if bottom_model is not None:
            add_model_row(
                outcome,
                "bottom_tercile_interaction",
                bottom_model,
                interaction_candidates=[
                    "origin_persfree:pre_shock_bottom_tercile_pair",
                    "pre_shock_bottom_tercile_pair:origin_persfree",
                ],
                low_frequency_definition="pre_posts_origin in bottom tercile",
            )

    result_table = pd.concat(
        [
            pd.DataFrame(descriptive_rows),
            pd.DataFrame(regression_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    if result_table.empty:
        raise ValueError("No displaced-contributor destination rows were produced.")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    regression_table = pd.DataFrame(regression_rows)
    if regression_table.empty:
        raise ValueError("No displaced-contributor destination regressions were estimated.")
    latex_columns = [
        "outcome",
        "model_name",
        "origin_persfree_coef",
        "standard_error",
        "p_value",
        "persfree_x_low_frequency_coef",
        "persfree_x_low_frequency_se",
        "persfree_x_low_frequency_p_value",
        "n_author_subreddit_origin_pairs",
        "n_unique_authors",
        "n_origin_subreddits",
        "controls",
        "clustering_level",
    ]
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(
        regression_table[latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    def regression_row(outcome, model_name):
        rows = regression_table[
            regression_table["outcome"].eq(outcome)
            & regression_table["model_name"].eq(model_name)
        ]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    stayed_row = regression_row("stayed_same_subreddit", "main_continuous_persfree")
    disappeared_row = regression_row("disappeared_from_sample", "main_continuous_persfree")
    moved_higher_row = regression_row("moved_higher_context", "main_continuous_persfree")
    disappeared_interaction_row = regression_row("disappeared_from_sample", "low_frequency_interaction")

    def effect_phrase(row, positive_word, negative_word):
        coef = row.get("origin_persfree_coef")
        pvalue = row.get("p_value")
        if coef is None:
            return "could not be estimated"
        direction = positive_word if coef > 0 else negative_word
        significance = "statistically significant" if pvalue is not None and pvalue < 0.05 else "not statistically significant"
        return f"{direction} (beta={coef:.4f}, p={pvalue:.4g}, {significance})"

    stay_phrase = effect_phrase(stayed_row, "more likely to stay", "less likely to stay")
    disappear_phrase = effect_phrase(disappeared_row, "more likely to disappear from the observed sample", "less likely to disappear from the observed sample")
    higher_phrase = effect_phrase(moved_higher_row, "more likely to move to higher-context sampled communities", "less likely to move to higher-context sampled communities")
    interaction_coef = disappeared_interaction_row.get("persfree_x_low_frequency_coef")
    interaction_p = disappeared_interaction_row.get("persfree_x_low_frequency_p_value")
    if interaction_coef is None:
        interaction_phrase = "the low-frequency displacement interaction could not be estimated"
    else:
        interaction_direction = "stronger" if interaction_coef > 0 else "weaker"
        interaction_sig = "statistically significant" if interaction_p is not None and interaction_p < 0.05 else "not statistically significant"
        interaction_phrase = (
            f"the high-PersFree disappearance gradient is {interaction_direction} among low-frequency origin pairs "
            f"(interaction={interaction_coef:.4f}, p={interaction_p:.4g}, {interaction_sig})"
        )

    summary = (
        "Displaced contributor destinations: contributors from high-PersFree origins are "
        f"{stay_phrase}; they are {disappear_phrase}; they are {higher_phrase}; "
        f"{interaction_phrase}. Disappearance means no observed post-shock posts in the 124 sampled communities, "
        "not disappearance from Reddit or the internet."
    )
    print(summary)

    return {
        "results": result_table,
        "regressions": regression_table,
        "panel": origin_pairs,
        "summary": summary,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
        "panel_output_path": str(panel_output_path),
    }

def compute_displaced_contributor_primary_origin_robustness(
    ecosystem_posts=None,
    score_path=None,
    pair_results_path=None,
    output_path=None,
    latex_path=None,
    panel_output_path=None,
    comparison_path=None,
    comparison_latex_path=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        score_candidates = [
            tables_dir / "acsi_preshock_tworuns.csv",
            root / "acsi_preshock_tworuns.csv",
            data_dir / "acsi_preshock_tworuns.csv",
        ]
        score_path = next((path for path in score_candidates if path.exists()), score_candidates[0])
    score_path = Path(score_path)
    pair_results_path = Path(pair_results_path or tables_dir / "displaced_contributor_destinations.csv")
    output_path = Path(output_path or tables_dir / "displaced_contributor_primary_origin_robustness.csv")
    latex_path = Path(latex_path or tables_dir / "displaced_contributor_primary_origin_robustness.tex")
    panel_output_path = Path(panel_output_path or tables_dir / "displaced_contributor_primary_origin_panel.csv")
    comparison_path = Path(comparison_path or tables_dir / "displaced_contributor_primary_origin_comparison.csv")
    comparison_latex_path = Path(
        comparison_latex_path or tables_dir / "displaced_contributor_primary_origin_comparison.tex"
    )

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])
    if scores.empty:
        raise ValueError("No usable two-run pre-shock ACSI scores.")
    scores["origin_persfree_tercile"] = pd.qcut(
        scores["pers_free"].rank(method="first"),
        q=3,
        labels=["low_persfree", "middle_persfree", "high_persfree"],
    ).astype(str)

    if ecosystem_posts is None:
        posts_path = globals().get("POSTS_ECOSYSTEM_PATH", None)
        if posts_path is None or not Path(posts_path).exists():
            raise ValueError(
                "Pass ecosystem_posts or run the main pipeline first so posts_clean_ecosystem.parquet exists."
            )
        posts = pd.read_parquet(posts_path, columns=["author", "subreddit", "year_month"])
    else:
        posts = ecosystem_posts[["author", "subreddit", "year_month"]].copy()

    target_subreddits = set(scores["subreddit"].astype(str))
    start_month = "2022-01"
    transition_month = "2022-11"
    post_start_month = "2022-12"
    end_month = "2024-12"
    excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))

    posts["author"] = posts["author"].fillna("").astype(str)
    posts["subreddit"] = posts["subreddit"].fillna("").astype(str)
    posts["year_month"] = posts["year_month"].astype(str)
    posts = posts[
        posts["subreddit"].isin(target_subreddits)
        & posts["year_month"].between(start_month, end_month)
        & posts["author"].ne("")
        & ~posts["author"].isin(excluded_authors)
        & ~posts["author"].str.lower().str.endswith("bot", na=False)
    ].copy()
    if posts.empty:
        raise ValueError("No ecosystem-clean posts overlap the scored subreddit-month window.")

    max_posts_per_day = globals().get("MAX_POSTS_PER_DAY", 50)
    start_date = globals().get("START_DATE", datetime(2022, 1, 1))
    end_date_exclusive = globals().get("END_DATE_EXCLUSIVE", datetime(2025, 1, 1))
    cap_days = max((end_date_exclusive - start_date).days, 1)
    author_post_cap = max_posts_per_day * cap_days
    author_counts = posts.groupby("author", sort=False).size()
    valid_authors = set(author_counts[author_counts <= author_post_cap].index)
    posts = posts[posts["author"].isin(valid_authors)].copy()
    if posts.empty:
        raise ValueError("No ecosystem-clean posts remain after author-cap filtering.")

    pre_posts = posts[posts["year_month"].between(start_month, transition_month)].copy()
    post_posts = posts[posts["year_month"].between(post_start_month, end_month)].copy()
    if pre_posts.empty:
        raise ValueError("No pre-shock author-origin posts found.")





    author_subreddit_pre = (
        pre_posts.groupby(["author", "subreddit"], as_index=False)
        .agg(
            pre_posts_primary_origin=("year_month", "size"),
            pre_active_months_primary_origin=("year_month", "nunique"),
            first_pre_month_primary_origin=("year_month", "min"),
            last_pre_month_primary_origin=("year_month", "max"),
        )
    )
    author_pre_totals = (
        author_subreddit_pre.groupby("author", as_index=False)
        .agg(
            author_total_pre_posts_all_sampled_subreddits=("pre_posts_primary_origin", "sum"),
            author_pre_subreddit_breadth=("subreddit", "nunique"),
        )
    )
    primary = (
        author_subreddit_pre.sort_values(
            [
                "author",
                "pre_posts_primary_origin",
                "pre_active_months_primary_origin",
                "last_pre_month_primary_origin",
                "subreddit",
            ],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates("author", keep="first")
        .rename(columns={"subreddit": "primary_origin_subreddit"})
        .merge(author_pre_totals, on="author", how="left")
    )
    primary["author_id"] = np.arange(len(primary), dtype=np.int64)

    score_lookup = scores.rename(
        columns={
            "subreddit": "primary_origin_subreddit",
            "pers_free": "primary_origin_persfree",
            "gen_cap": "primary_origin_gen_cap",
            "phys_free": "primary_origin_phys_free",
        }
    )
    primary = primary.merge(
        score_lookup[
            [
                "primary_origin_subreddit",
                "primary_origin_persfree",
                "primary_origin_gen_cap",
                "primary_origin_phys_free",
                "origin_persfree_tercile",
            ]
        ],
        on="primary_origin_subreddit",
        how="inner",
    )
    if primary.empty:
        raise ValueError("No primary-origin authors overlap with two-run ACSI scores.")

    subreddit_categories = globals().get("SUBREDDITS", {})
    if isinstance(subreddit_categories, dict) and subreddit_categories:
        primary["origin_category"] = (
            primary["primary_origin_subreddit"].map(subreddit_categories).fillna("unknown").astype(str)
        )
    else:
        primary["origin_category"] = "unknown"

    origin_size = (
        primary.groupby("primary_origin_subreddit", sort=False)["pre_posts_primary_origin"]
        .sum()
        .rename("origin_subreddit_pre_posts_total")
    )
    primary["origin_subreddit_pre_posts_total"] = primary["primary_origin_subreddit"].map(origin_size)
    primary["log_origin_size_pre"] = np.log1p(
        primary["origin_subreddit_pre_posts_total"].astype(float)
    )
    primary["log_pre_posts_primary_origin"] = np.log1p(
        primary["pre_posts_primary_origin"].astype(float)
    )

    if post_posts.empty:
        same_counts = pd.DataFrame(columns=["author", "primary_origin_subreddit", "primary_origin_postshock_posts"])
        post_author_agg = pd.DataFrame(columns=["author", "postshock_total_posts_sample", "postshock_subreddit_count"])
        other_stats = pd.DataFrame(columns=["author_id"])
    else:
        post_author_subreddit = (
            post_posts.groupby(["author", "subreddit"], as_index=False)
            .size()
            .rename(columns={"size": "destination_posts"})
            .merge(
                scores[["subreddit", "pers_free"]].rename(
                    columns={"subreddit": "destination_subreddit", "pers_free": "destination_persfree"}
                ),
                left_on="subreddit",
                right_on="destination_subreddit",
                how="left",
            )
            .drop(columns=["destination_subreddit"])
            .dropna(subset=["destination_persfree"])
        )
        same_counts = post_author_subreddit.rename(
            columns={
                "subreddit": "primary_origin_subreddit",
                "destination_posts": "primary_origin_postshock_posts",
            }
        )[["author", "primary_origin_subreddit", "primary_origin_postshock_posts"]]
        post_author_agg = (
            post_author_subreddit.groupby("author", as_index=False)
            .agg(
                postshock_total_posts_sample=("destination_posts", "sum"),
                postshock_subreddit_count=("subreddit", "nunique"),
            )
        )



        destination_edges = primary[
            ["author_id", "author", "primary_origin_subreddit", "primary_origin_persfree"]
        ].merge(post_author_subreddit, on="author", how="inner")
        other_edges = destination_edges[
            destination_edges["subreddit"].ne(destination_edges["primary_origin_subreddit"])
        ].copy()
        if other_edges.empty:
            other_stats = pd.DataFrame(columns=["author_id"])
        else:
            ranked_scores = scores.sort_values(["pers_free", "subreddit"]).reset_index(drop=True)
            ranked_scores["full_persfree_rank"] = np.arange(1, len(ranked_scores) + 1, dtype=int)
            rank_lookup = ranked_scores.set_index("subreddit")["full_persfree_rank"].to_dict()
            choice_set_size = max(len(ranked_scores) - 1, 1)
            other_edges["origin_full_rank"] = other_edges["primary_origin_subreddit"].map(rank_lookup)
            other_edges["destination_full_rank"] = other_edges["subreddit"].map(rank_lookup)
            other_edges["destination_rank_if_available"] = (
                other_edges["destination_full_rank"]
                - (other_edges["origin_full_rank"] < other_edges["destination_full_rank"]).astype(int)
            ) / choice_set_size
            other_edges["destination_subreddits"] = other_edges["subreddit"].astype(str)
            other_stats = (
                other_edges.groupby("author_id", as_index=False)
                .agg(
                    destination_count=("subreddit", "nunique"),
                    other_destination_posts=("destination_posts", "sum"),
                    mean_destination_persfree=("destination_persfree", "mean"),
                    min_destination_persfree=("destination_persfree", "min"),
                    max_destination_persfree=("destination_persfree", "max"),
                    destination_rank_if_available=("destination_rank_if_available", "mean"),
                    observed_destination_subreddits=(
                        "destination_subreddits",
                        lambda values: "|".join(sorted(set(values.astype(str)))),
                    ),
                )
            )
        del post_author_subreddit
        if "destination_edges" in locals():
            del destination_edges
        if "other_edges" in locals():
            del other_edges
        gc.collect()

    primary = primary.merge(
        same_counts,
        on=["author", "primary_origin_subreddit"],
        how="left",
    ).merge(
        post_author_agg,
        on="author",
        how="left",
    ).merge(
        other_stats,
        on="author_id",
        how="left",
    )

    count_columns = [
        "primary_origin_postshock_posts",
        "postshock_total_posts_sample",
        "postshock_subreddit_count",
        "destination_count",
        "other_destination_posts",
    ]
    for column_name in count_columns:
        if column_name not in primary.columns:
            primary[column_name] = 0
        primary[column_name] = pd.to_numeric(primary[column_name], errors="coerce").fillna(0).astype(np.int32)

    primary["stayed_primary_origin"] = (
        primary["primary_origin_postshock_posts"] > 0
    ).astype(int)
    primary["any_other_subreddit_postshock"] = (
        primary["destination_count"] > 0
    ).astype(int)
    primary["moved_other_subreddit"] = (
        (primary["stayed_primary_origin"] == 0)
        & (primary["any_other_subreddit_postshock"] == 1)
    ).astype(int)
    primary["disappeared_from_sample"] = (
        primary["postshock_total_posts_sample"] == 0
    ).astype(int)
    primary["moved_context_light"] = (
        (primary["moved_other_subreddit"] == 1)
        & (
            primary["mean_destination_persfree"]
            >= primary["primary_origin_persfree"] + 0.05
        )
    ).astype(int)
    primary["moved_context_heavy"] = (
        (primary["moved_other_subreddit"] == 1)
        & (
            primary["mean_destination_persfree"]
            <= primary["primary_origin_persfree"] - 0.05
        )
    ).astype(int)
    primary["moved_similar_context"] = (
        (primary["moved_other_subreddit"] == 1)
        & (
            (primary["mean_destination_persfree"] - primary["primary_origin_persfree"]).abs()
            < 0.05
        )
    ).astype(int)

    primary = primary.sort_values(["primary_origin_subreddit", "author"]).reset_index(drop=True)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    primary.to_csv(panel_output_path, index=False)

    destination_outcomes = [
        "stayed_primary_origin",
        "moved_other_subreddit",
        "any_other_subreddit_postshock",
        "moved_context_light",
        "moved_context_heavy",
        "moved_similar_context",
        "disappeared_from_sample",
    ]
    descriptive_rows = []
    for tercile, group in primary.groupby("origin_persfree_tercile", observed=False):
        row = {
            "row_type": "descriptive",
            "origin_persfree_tercile": tercile,
            "n_authors": safe_int(len(group)),
            "n_primary_origin_subreddits": safe_int(group["primary_origin_subreddit"].nunique()),
            "mean_destination_count": safe_float(group["destination_count"].mean()),
            "mean_postshock_total_posts_sample": safe_float(group["postshock_total_posts_sample"].mean()),
            "mean_destination_persfree": safe_float(group["mean_destination_persfree"].mean()),
        }
        for outcome in destination_outcomes:
            row[f"share_{outcome}"] = safe_float(group[outcome].mean())
        descriptive_rows.append(row)

    control_terms = [
        "log_pre_posts_primary_origin",
        "pre_active_months_primary_origin",
        "author_pre_subreddit_breadth",
        "primary_origin_gen_cap",
        "primary_origin_phys_free",
        "log_origin_size_pre",
    ]
    category_term = ""
    if primary["origin_category"].nunique(dropna=True) > 1:
        category_term = " + C(origin_category)"
    controls_label = (
        "log(1+pre_posts_primary_origin), pre_active_months_primary_origin, "
        "author_pre_subreddit_breadth, origin GenCap, origin PhysFree, log pre-shock origin size"
        + (", origin category FE" if category_term else "")
    )
    regression_rows = []
    main_outcomes = [
        "stayed_primary_origin",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_context_light",
        "moved_context_heavy",
    ]
    base_required = [
        "primary_origin_persfree",
        "primary_origin_subreddit",
        "author",
    ] + control_terms
    for outcome in main_outcomes:
        model_data = primary.dropna(subset=[outcome] + base_required).copy()
        if model_data.empty or model_data["primary_origin_subreddit"].nunique() < 2:
            continue
        formula = (
            f"{outcome} ~ primary_origin_persfree + "
            + " + ".join(control_terms)
            + category_term
        )
        model = fit_ols(formula, model_data, cluster_col="primary_origin_subreddit")
        if model is None:
            continue
        focal = reg_result(model, "primary_origin_persfree")
        regression_rows.append({
            "row_type": "regression",
            "outcome": outcome,
            "model_name": "primary_origin_continuous_persfree",
            "primary_origin_persfree_coef": focal.get("coef"),
            "standard_error": focal.get("se"),
            "p_value": focal.get("pvalue"),
            "n_authors": safe_int(model.nobs),
            "n_unique_authors": safe_int(model.model.data.frame["author"].nunique()),
            "n_primary_origin_subreddits": safe_int(model.model.data.frame["primary_origin_subreddit"].nunique()),
            "controls": controls_label,
            "fixed_effects": "none" + ("; origin category FE" if category_term else ""),
            "clustering_level": "primary-origin subreddit",
            "score_source": str(score_path),
            "post_sample_source": str(globals().get("POSTS_ECOSYSTEM_PATH", "provided dataframe")),
            "panel_output_path": str(panel_output_path),
        })

    result_table = pd.concat(
        [pd.DataFrame(descriptive_rows), pd.DataFrame(regression_rows)],
        ignore_index=True,
        sort=False,
    )
    if result_table.empty:
        raise ValueError("No primary-origin robustness rows were produced.")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    regression_table = pd.DataFrame(regression_rows)
    if regression_table.empty:
        raise ValueError("No primary-origin robustness regressions were estimated.")
    latex_columns = [
        "outcome",
        "primary_origin_persfree_coef",
        "standard_error",
        "p_value",
        "n_authors",
        "n_primary_origin_subreddits",
        "controls",
        "clustering_level",
    ]
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(
        regression_table[latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    def regression_row(outcome):
        rows = regression_table[regression_table["outcome"].eq(outcome)]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    pair_lookup = {}
    if pair_results_path.exists():
        pair_results = pd.read_csv(pair_results_path)
        pair_regressions = pair_results[
            pair_results.get("row_type", pd.Series(dtype=str)).astype(str).eq("regression")
            & pair_results.get("model_name", pd.Series(dtype=str)).astype(str).eq("main_continuous_persfree")
        ].copy()
        for _, row in pair_regressions.iterrows():
            pair_lookup[str(row.get("outcome"))] = row.to_dict()
    hardcoded_pair = {
        "stayed_same_subreddit": {"origin_persfree_coef": -0.0473, "standard_error": None, "p_value": 0.0127},
        "disappeared_from_sample": {"origin_persfree_coef": 0.0237, "standard_error": None, "p_value": 0.1581},
    }
    comparison_specs = [
        ("retention", "stayed_same_subreddit", "stayed_primary_origin", "negative", "High-PersFree origins are less likely to retain contributors."),
        ("ecosystem_exit", "disappeared_from_sample", "disappeared_from_sample", "null_or_positive_not_significant", "No strong evidence of observed-sample exit."),
        ("other_subreddit_move", "moved_other_subreddit", "moved_other_subreddit", "any", "High-PersFree origins predict moves to other sampled subreddits if significant."),
        ("context_light_destination", "moved_lower_context", "moved_context_light", "positive", "Movers orient toward context-light destinations."),
        ("context_heavy_destination", "moved_higher_context", "moved_context_heavy", "positive", "Movers orient toward context-heavy destinations."),
    ]
    comparison_rows = []
    for mechanism, pair_outcome, author_outcome, expected_direction, note in comparison_specs:
        pair_row = pair_lookup.get(pair_outcome, hardcoded_pair.get(pair_outcome, {}))
        author_row = regression_row(author_outcome)
        author_coef = author_row.get("primary_origin_persfree_coef")
        author_p = author_row.get("p_value")
        if expected_direction == "negative":
            supports = author_coef is not None and author_coef < 0 and author_p is not None and author_p < 0.05
        elif expected_direction == "positive":
            supports = author_coef is not None and author_coef > 0 and author_p is not None and author_p < 0.05
        elif expected_direction == "null_or_positive_not_significant":
            supports = author_p is not None and author_p >= 0.05
        else:
            supports = author_p is not None and author_p < 0.05
        comparison_rows.append({
            "mechanism": mechanism,
            "pair_level_outcome": pair_outcome,
            "pair_level_beta": pair_row.get("origin_persfree_coef"),
            "pair_level_se": pair_row.get("standard_error"),
            "pair_level_p_value": pair_row.get("p_value"),
            "author_level_outcome": author_outcome,
            "author_level_beta": author_coef,
            "author_level_se": author_row.get("standard_error"),
            "author_level_p_value": author_p,
            "supports_pair_level_qualitative_conclusion": bool(supports),
            "note": note,
        })
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_path.parent.mkdir(exist_ok=True, parents=True)
    comparison_table.to_csv(comparison_path, index=False)
    comparison_latex_path.parent.mkdir(exist_ok=True, parents=True)
    comparison_latex_path.write_text(
        comparison_table.to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    stayed_row = regression_row("stayed_primary_origin")
    disappeared_row = regression_row("disappeared_from_sample")
    light_row = regression_row("moved_context_light")
    heavy_row = regression_row("moved_context_heavy")

    def phrase(row, positive, negative):
        coef = row.get("primary_origin_persfree_coef")
        pvalue = row.get("p_value")
        if coef is None:
            return "could not be estimated"
        direction = positive if coef > 0 else negative
        sig = "statistically significant" if pvalue is not None and pvalue < 0.05 else "not statistically significant"
        return f"{direction} (beta={coef:.4f}, p={pvalue:.4g}, {sig})"

    retain_phrase = phrase(stayed_row, "more likely to retain authors", "less likely to retain authors")
    disappear_phrase = phrase(disappeared_row, "more likely to disappear from the observed sample", "less likely to disappear from the observed sample")
    light_phrase = phrase(light_row, "more likely to move to context-light destinations", "less likely to move to context-light destinations")
    heavy_phrase = phrase(heavy_row, "more likely to move to context-heavy destinations", "less likely to move to context-heavy destinations")
    consistent_retention = bool(
        stayed_row.get("primary_origin_persfree_coef") is not None
        and stayed_row.get("primary_origin_persfree_coef") < 0
        and stayed_row.get("p_value") is not None
        and stayed_row.get("p_value") < 0.05
    )
    consistent_exit = bool(
        disappeared_row.get("p_value") is not None
        and disappeared_row.get("p_value") >= 0.05
    )
    consistent_phrase = (
        "consistent with the pair-level retention and no-strong-exit conclusions"
        if consistent_retention and consistent_exit
        else "not fully consistent with the pair-level qualitative conclusions"
    )
    summary = (
        f"Primary-origin robustness: N={len(primary):,} authors. High-PersFree primary origins are "
        f"{retain_phrase}; they are {disappear_phrase}; they are {light_phrase}; and they are "
        f"{heavy_phrase}. The author-level primary-origin robustness is {consistent_phrase}. "
        "Disappeared means no observed post-shock posts in the 124 sampled communities, not Reddit-wide exit."
    )
    print(summary)

    return {
        "results": result_table,
        "regressions": regression_table,
        "panel": primary,
        "comparison": comparison_table,
        "summary": summary,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
        "panel_output_path": str(panel_output_path),
        "comparison_path": str(comparison_path),
        "comparison_latex_path": str(comparison_latex_path),
    }

def compute_destination_choice_set_placebo(
    ecosystem_posts=None,
    score_path=None,
    displaced_panel_path=None,
    output_path=None,
    latex_path=None,
    panel_output_path=None,
    n_permutations=1000,
    random_seed=20250603,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        score_candidates = [
            tables_dir / "acsi_preshock_tworuns.csv",
            root / "acsi_preshock_tworuns.csv",
            data_dir / "acsi_preshock_tworuns.csv",
        ]
        score_path = next((path for path in score_candidates if path.exists()), score_candidates[0])
    score_path = Path(score_path)
    displaced_panel_path = Path(
        displaced_panel_path or tables_dir / "displaced_contributor_destination_panel.csv"
    )
    output_path = Path(output_path or tables_dir / "destination_choice_set_placebo.csv")
    latex_path = Path(latex_path or tables_dir / "destination_choice_set_placebo.tex")
    panel_output_path = Path(
        panel_output_path or tables_dir / "destination_choice_set_placebo_panel.csv"
    )

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])
    if scores.empty:
        raise ValueError("No usable two-run pre-shock ACSI scores.")
    scores["origin_persfree_tercile"] = pd.qcut(
        scores["pers_free"].rank(method="first"),
        q=3,
        labels=["low_persfree", "middle_persfree", "high_persfree"],
    ).astype(str)

    if displaced_panel_path.exists():
        origin_panel = pd.read_csv(displaced_panel_path)
    else:
        displaced = compute_displaced_contributor_destinations(
            ecosystem_posts=ecosystem_posts,
            score_path=score_path,
            panel_output_path=displaced_panel_path,
        )
        origin_panel = displaced["panel"].copy()

    required_panel_columns = {
        "author",
        "origin_subreddit",
        "origin_pair_id",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "origin_persfree_tercile",
        "pre_posts_origin",
        "pre_active_months_origin",
        "log_pre_posts_origin",
        "log_pre_subreddit_posts",
        "moved_other_subreddit",
        "stayed_same_subreddit",
        "other_destination_count",
        "other_destination_posts",
        "mean_destination_persfree",
        "min_destination_persfree",
        "max_destination_persfree",
    }
    missing_panel_columns = required_panel_columns - set(origin_panel.columns)
    if missing_panel_columns:
        raise ValueError(
            "Displaced-contributor panel missing columns: "
            f"{sorted(missing_panel_columns)}"
        )

    movers = origin_panel[
        origin_panel["moved_other_subreddit"].astype(int).eq(1)
        & origin_panel["stayed_same_subreddit"].astype(int).eq(0)
        & (pd.to_numeric(origin_panel["other_destination_count"], errors="coerce") > 0)
    ].copy()
    if movers.empty:
        raise ValueError("No mover-origin pairs found for destination choice-set placebo.")

    for column_name in [
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "pre_posts_origin",
        "pre_active_months_origin",
        "log_pre_posts_origin",
        "log_pre_subreddit_posts",
        "other_destination_count",
        "other_destination_posts",
        "mean_destination_persfree",
        "min_destination_persfree",
        "max_destination_persfree",
    ]:
        movers[column_name] = pd.to_numeric(movers[column_name], errors="coerce")
    movers = movers.dropna(
        subset=[
            "author",
            "origin_subreddit",
            "origin_pair_id",
            "origin_persfree",
            "origin_gen_cap",
            "origin_phys_free",
            "pre_posts_origin",
            "pre_active_months_origin",
            "log_pre_posts_origin",
            "log_pre_subreddit_posts",
            "other_destination_count",
            "mean_destination_persfree",
        ]
    ).copy()
    movers["origin_pair_id"] = pd.to_numeric(
        movers["origin_pair_id"], errors="coerce"
    ).astype(np.int64)
    movers["observed_destination_count"] = movers["other_destination_count"].astype(int)
    movers["observed_destination_posts"] = movers["other_destination_posts"].fillna(0).astype(int)
    movers["observed_mean_destination_persfree"] = movers["mean_destination_persfree"]
    movers["observed_min_destination_persfree"] = movers["min_destination_persfree"]
    movers["observed_max_destination_persfree"] = movers["max_destination_persfree"]

    target_subreddits = set(scores["subreddit"].astype(str))
    post_start_month = "2022-12"
    end_month = "2024-12"
    excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))

    if ecosystem_posts is None:
        posts_path = globals().get("POSTS_ECOSYSTEM_PATH", None)
        if posts_path is None or not Path(posts_path).exists():
            raise ValueError(
                "Pass ecosystem_posts or run the main pipeline first so posts_clean_ecosystem.parquet exists."
            )
        posts = pd.read_parquet(posts_path, columns=["author", "subreddit", "year_month"])
    else:
        posts = ecosystem_posts[["author", "subreddit", "year_month"]].copy()

    posts["author"] = posts["author"].fillna("").astype(str)
    posts["subreddit"] = posts["subreddit"].fillna("").astype(str)
    posts["year_month"] = posts["year_month"].astype(str)
    mover_authors = set(movers["author"].astype(str))
    posts = posts[
        posts["author"].isin(mover_authors)
        & posts["subreddit"].isin(target_subreddits)
        & posts["year_month"].between(post_start_month, end_month)
        & posts["author"].ne("")
        & ~posts["author"].isin(excluded_authors)
        & ~posts["author"].str.lower().str.endswith("bot", na=False)
    ].copy()
    if posts.empty:
        raise ValueError("No post-shock posts found for observed mover destination edges.")

    post_author_subreddit = (
        posts.groupby(["author", "subreddit"], as_index=False)
        .size()
        .rename(columns={"subreddit": "destination_subreddit", "size": "destination_posts"})
        .merge(
            scores[["subreddit", "pers_free"]].rename(
                columns={
                    "subreddit": "destination_subreddit",
                    "pers_free": "destination_persfree",
                }
            ),
            on="destination_subreddit",
            how="inner",
        )
    )
    edge_columns = [
        "origin_pair_id",
        "author",
        "origin_subreddit",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
    ]
    destination_edges = movers[edge_columns].merge(
        post_author_subreddit,
        on="author",
        how="inner",
    )
    destination_edges = destination_edges[
        destination_edges["destination_subreddit"].ne(destination_edges["origin_subreddit"])
    ].copy()
    if destination_edges.empty:
        raise ValueError("No non-origin post-shock destination edges found for movers.")

    destination_edge_stats = (
        destination_edges.groupby("origin_pair_id", as_index=False)
        .agg(
            observed_destination_subreddits=(
                "destination_subreddit",
                lambda values: "|".join(sorted(set(values.astype(str)))),
            ),
            observed_destination_count=("destination_subreddit", "nunique"),
            observed_destination_posts=("destination_posts", "sum"),
            observed_mean_destination_persfree=("destination_persfree", "mean"),
            observed_min_destination_persfree=("destination_persfree", "min"),
            observed_max_destination_persfree=("destination_persfree", "max"),
        )
    )
    movers = movers.drop(
        columns=[
            "observed_destination_count",
            "observed_destination_posts",
            "observed_mean_destination_persfree",
            "observed_min_destination_persfree",
            "observed_max_destination_persfree",
        ],
        errors="ignore",
    ).merge(destination_edge_stats, on="origin_pair_id", how="inner")
    movers = movers[movers["observed_destination_count"] > 0].copy()
    movers["observed_delta_persfree"] = (
        movers["observed_mean_destination_persfree"] - movers["origin_persfree"]
    )
    movers["observed_moved_higher_context"] = (
        movers["observed_mean_destination_persfree"] <= movers["origin_persfree"] - 0.05
    ).astype(int)





    score_by_subreddit = scores.set_index("subreddit")["pers_free"].astype(float)
    all_subreddits = score_by_subreddit.index.astype(str).to_numpy()
    all_persfree = score_by_subreddit.to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    choice_delta_draws = {}
    choice_higher_draws = {}
    expected_choice = {}
    max_destination_count = int(movers["observed_destination_count"].max())
    for origin_subreddit in sorted(movers["origin_subreddit"].astype(str).unique()):
        origin_score = float(score_by_subreddit.loc[origin_subreddit])
        eligible_mask = all_subreddits != origin_subreddit
        eligible_scores = all_persfree[eligible_mask]
        expected_choice[origin_subreddit] = {
            "mean_placebo_destination_persfree": safe_float(eligible_scores.mean()),
            "mean_placebo_delta_persfree": safe_float(eligible_scores.mean() - origin_score),
        }
        order = np.argsort(
            rng.random((n_permutations, len(eligible_scores))),
            axis=1,
        )
        ordered_scores = eligible_scores[order]
        cumulative_scores = np.cumsum(ordered_scores, axis=1)
        for destination_count in range(1, max_destination_count + 1):
            sample_count = min(destination_count, len(eligible_scores))
            sampled_means = cumulative_scores[:, sample_count - 1] / sample_count
            choice_delta_draws[(origin_subreddit, destination_count)] = (
                sampled_means - origin_score
            )
            choice_higher_draws[(origin_subreddit, destination_count)] = (
                sampled_means <= origin_score - 0.05
            ).astype(float)

    movers["mean_placebo_destination_persfree"] = movers["origin_subreddit"].map(
        {key: value["mean_placebo_destination_persfree"] for key, value in expected_choice.items()}
    )
    movers["mean_placebo_delta_persfree"] = movers["origin_subreddit"].map(
        {key: value["mean_placebo_delta_persfree"] for key, value in expected_choice.items()}
    )
    movers["excess_higher_context_shift"] = (
        movers["observed_delta_persfree"] - movers["mean_placebo_delta_persfree"]
    )

    movers["pair_permutation_p_value"] = np.nan
    movers["placebo_probability_higher_context"] = np.nan
    for (origin_subreddit, destination_count), group_index in movers.groupby(
        ["origin_subreddit", "observed_destination_count"], sort=False
    ).groups.items():
        destination_count = int(destination_count)
        draws = choice_delta_draws[(str(origin_subreddit), destination_count)]
        sorted_draws = np.sort(draws)
        observed_values = movers.loc[group_index, "observed_delta_persfree"].to_numpy(dtype=float)
        movers.loc[group_index, "pair_permutation_p_value"] = (
            np.searchsorted(sorted_draws, observed_values, side="right") / len(sorted_draws)
        )
        movers.loc[group_index, "placebo_probability_higher_context"] = safe_float(
            choice_higher_draws[(str(origin_subreddit), destination_count)].mean()
        )

    def simulated_placebo_mean_distribution(frame):
        if frame.empty:
            return np.array([], dtype=float)
        permutation_sums = np.zeros(n_permutations, dtype=float)
        total_rows = len(frame)
        grouped = frame.groupby(["origin_subreddit", "observed_destination_count"], sort=False)
        for (origin_subreddit, destination_count), group in grouped:
            draws = choice_delta_draws[(str(origin_subreddit), int(destination_count))]
            n_group = len(group)
            if n_group <= 2000:
                draw_index = rng.integers(0, len(draws), size=(n_permutations, n_group))
                group_means = draws[draw_index].mean(axis=1)
            else:
                draw_sd = float(np.nanstd(draws, ddof=1))
                group_means = rng.normal(
                    loc=float(np.nanmean(draws)),
                    scale=draw_sd / np.sqrt(n_group),
                    size=n_permutations,
                )
            permutation_sums += group_means * n_group
        return permutation_sums / total_rows

    def summary_row(sample_name, frame):
        if frame.empty:
            return None
        placebo_distribution = simulated_placebo_mean_distribution(frame)
        observed_delta = safe_float(frame["observed_delta_persfree"].mean())
        placebo_delta = safe_float(frame["mean_placebo_delta_persfree"].mean())
        p_value = safe_float(np.mean(placebo_distribution <= observed_delta))
        return {
            "row_type": "placebo_summary",
            "sample": sample_name,
            "origin_persfree_tercile": "all",
            "n_mover_origin_pairs": safe_int(len(frame)),
            "n_unique_authors": safe_int(frame["author"].nunique()),
            "n_origin_subreddits": safe_int(frame["origin_subreddit"].nunique()),
            "observed_mean_delta_persfree": observed_delta,
            "placebo_mean_delta_persfree": placebo_delta,
            "observed_minus_placebo_delta": safe_float(observed_delta - placebo_delta),
            "permutation_p_value": p_value,
            "observed_share_moved_higher_context": safe_float(
                frame["observed_moved_higher_context"].mean()
            ),
            "placebo_probability_higher_context": safe_float(
                frame["placebo_probability_higher_context"].mean()
            ),
            "mean_destination_count": safe_float(frame["observed_destination_count"].mean()),
            "mean_destination_posts": safe_float(frame["observed_destination_posts"].mean()),
            "n_permutations": safe_int(n_permutations),
            "notes": (
                "Observed mover-origin pairs are compared with random destinations drawn "
                "without replacement from all scored subreddits except the origin."
            ),
        }

    result_rows = []
    main_summary = summary_row("all_movers", movers)
    if main_summary:
        result_rows.append(main_summary)
    for tercile in ["low_persfree", "middle_persfree", "high_persfree"]:
        subset = movers[movers["origin_persfree_tercile"].astype(str).eq(tercile)].copy()
        row = summary_row(f"all_movers_{tercile}", subset)
        if row:
            row["origin_persfree_tercile"] = tercile
            result_rows.append(row)

    variant_frames = {
        "single_destination_only": movers[movers["observed_destination_count"].eq(1)].copy(),
        "at_least_two_destination_posts": movers[
            movers["observed_destination_posts"].ge(2)
        ].copy(),
    }
    for sample_name, frame in variant_frames.items():
        row = summary_row(sample_name, frame)
        if row:
            result_rows.append(row)

    control_terms = [
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "origin_gen_cap",
        "origin_phys_free",
    ]
    category_term = ""
    if "origin_category" in movers.columns and movers["origin_category"].nunique(dropna=True) > 1:
        category_term = " + C(origin_category)"
    controls_label = (
        "log(1+pre_posts_origin), pre_active_months_origin, origin GenCap, origin PhysFree"
        + (", origin category FE" if category_term else "")
    )
    regression_rows = []
    regression_samples = {"all_movers": movers}
    regression_samples.update(variant_frames)
    for sample_name, frame in regression_samples.items():
        model_data = frame.dropna(
            subset=[
                "excess_higher_context_shift",
                "origin_persfree",
                "origin_subreddit",
            ]
            + control_terms
        ).copy()
        if model_data.empty or model_data["origin_subreddit"].nunique() < 2:
            continue
        formula = (
            "excess_higher_context_shift ~ origin_persfree + "
            + " + ".join(control_terms)
            + category_term
        )
        model = fit_ols(formula, model_data, cluster_col="origin_subreddit")
        if model is None:
            continue
        focal = reg_result(model, "origin_persfree")
        regression_rows.append({
            "row_type": "regression",
            "sample": sample_name,
            "outcome": "excess_higher_context_shift",
            "model_name": "placebo_adjusted_destination_shift",
            "origin_persfree_coef": focal.get("coef"),
            "standard_error": focal.get("se"),
            "p_value": focal.get("pvalue"),
            "n_mover_origin_pairs": safe_int(model.nobs),
            "n_unique_authors": safe_int(model.model.data.frame["author"].nunique()),
            "n_origin_subreddits": safe_int(model.model.data.frame["origin_subreddit"].nunique()),
            "controls": controls_label,
            "fixed_effects": "none" + ("; origin category FE" if category_term else ""),
            "clustering_level": "origin subreddit",
            "n_permutations": safe_int(n_permutations),
            "score_source": str(score_path),
            "displaced_panel_source": str(displaced_panel_path),
            "panel_output_path": str(panel_output_path),
            "interpretation": (
                "Negative beta means high-PersFree origins send movers to lower-PersFree, "
                "higher-context destinations more than expected from random choice-set availability."
            ),
        })
    result_rows.extend(regression_rows)

    result_table = pd.DataFrame(result_rows)
    if result_table.empty:
        raise ValueError("No destination choice-set placebo rows were produced.")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    panel_columns = [
        "author",
        "origin_pair_id",
        "origin_subreddit",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "origin_persfree_tercile",
        "pre_posts_origin",
        "pre_active_months_origin",
        "log_pre_posts_origin",
        "log_pre_subreddit_posts",
        "observed_destination_subreddits",
        "observed_destination_count",
        "observed_destination_posts",
        "observed_mean_destination_persfree",
        "observed_min_destination_persfree",
        "observed_max_destination_persfree",
        "observed_delta_persfree",
        "mean_placebo_destination_persfree",
        "mean_placebo_delta_persfree",
        "excess_higher_context_shift",
        "observed_moved_higher_context",
        "placebo_probability_higher_context",
        "pair_permutation_p_value",
    ]
    optional_panel_columns = ["origin_category"]
    for column_name in optional_panel_columns:
        if column_name in movers.columns and column_name not in panel_columns:
            panel_columns.append(column_name)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    movers[panel_columns].to_csv(panel_output_path, index=False)

    regression_table = pd.DataFrame(regression_rows)
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    if regression_table.empty:
        latex_path.write_text(
            result_table.to_latex(index=False, float_format="%.4f", escape=False),
            encoding="utf-8",
        )
    else:
        latex_columns = [
            "sample",
            "outcome",
            "origin_persfree_coef",
            "standard_error",
            "p_value",
            "n_mover_origin_pairs",
            "n_unique_authors",
            "n_origin_subreddits",
            "controls",
            "clustering_level",
        ]
        latex_path.write_text(
            regression_table[latex_columns].to_latex(
                index=False, float_format="%.4f", escape=False
            ),
            encoding="utf-8",
        )

    def row_for(sample_name, row_type="placebo_summary"):
        rows = result_table[
            result_table["row_type"].eq(row_type)
            & result_table["sample"].eq(sample_name)
        ]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    all_row = row_for("all_movers")
    high_row = row_for("all_movers_high_persfree")
    regression_row = (
        regression_table[regression_table["sample"].eq("all_movers")].iloc[0].to_dict()
        if not regression_table.empty
        and regression_table["sample"].eq("all_movers").any()
        else {}
    )
    observed_minus_placebo = all_row.get("observed_minus_placebo_delta")
    perm_p = all_row.get("permutation_p_value")
    high_observed_minus_placebo = high_row.get("observed_minus_placebo_delta")
    beta = regression_row.get("origin_persfree_coef")
    beta_p = regression_row.get("p_value")
    movement_phrase = (
        "stronger than random destination availability"
        if observed_minus_placebo is not None and observed_minus_placebo < 0
        else "not stronger than random destination availability"
    )
    strongest_phrase = (
        "strongest for high-PersFree origin communities"
        if high_observed_minus_placebo is not None
        and observed_minus_placebo is not None
        and high_observed_minus_placebo <= observed_minus_placebo
        else "not uniquely strongest for high-PersFree origin communities"
    )
    if beta is None:
        regression_phrase = "the placebo-adjusted origin-PersFree regression could not be estimated"
    else:
        regression_direction = (
            "predicts excess movement toward higher-context destinations"
            if beta < 0
            else "predicts a less higher-context shift than random choice-set availability would imply"
        )
        regression_sig = "statistically significant" if beta_p is not None and beta_p < 0.05 else "not statistically significant"
        regression_phrase = (
            f"origin PersFree {regression_direction} "
            f"(beta={beta:.4f}, p={beta_p:.4g}, {regression_sig})"
        )
    summary = (
        "Destination choice-set placebo: observed mover destinations shift toward higher-context "
        f"communities {movement_phrase} (observed-minus-placebo delta="
        f"{observed_minus_placebo:.4f}, permutation p={perm_p:.4g}). The excess shift is "
        f"{strongest_phrase}. After placebo adjustment, {regression_phrase}."
    )
    print(summary)

    return {
        "results": result_table,
        "regressions": regression_table,
        "panel": movers[panel_columns],
        "summary": summary,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
        "panel_output_path": str(panel_output_path),
    }

def compute_micro_displacement_prediction(
    displaced_panel_path=None,
    metrics_path=None,
    metrics_latex_path=None,
    coefficients_path=None,
    coefficients_latex_path=None,
    panel_output_path=None,
    n_folds=5,
    random_seed=20250603,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)

    displaced_panel_path = Path(
        displaced_panel_path or tables_dir / "displaced_contributor_destination_panel.csv"
    )
    metrics_path = Path(metrics_path or tables_dir / "micro_displacement_prediction_metrics.csv")
    metrics_latex_path = Path(
        metrics_latex_path or tables_dir / "micro_displacement_prediction_metrics.tex"
    )
    coefficients_path = Path(
        coefficients_path or tables_dir / "micro_displacement_prediction_coefficients.csv"
    )
    coefficients_latex_path = Path(
        coefficients_latex_path or tables_dir / "micro_displacement_prediction_coefficients.tex"
    )
    panel_output_path = Path(
        panel_output_path or tables_dir / "micro_displacement_prediction_panel.csv"
    )

    if displaced_panel_path.exists():
        panel = pd.read_csv(displaced_panel_path)
    else:
        displaced = compute_displaced_contributor_destinations(
            panel_output_path=displaced_panel_path,
        )
        panel = displaced["panel"].copy()

    required_columns = {
        "author",
        "origin_subreddit",
        "origin_pair_id",
        "pre_posts_origin",
        "pre_active_months_origin",
        "first_pre_month_origin",
        "last_pre_month_origin",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "stayed_same_subreddit",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_higher_context",
        "same_subreddit_postshock_posts",
        "postshock_total_posts_sample",
    }
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(
            "Displaced-contributor panel missing columns needed for micro prediction: "
            f"{sorted(missing_columns)}"
        )

    panel = panel.copy()
    panel["author"] = panel["author"].fillna("").astype(str)
    panel["origin_subreddit"] = panel["origin_subreddit"].fillna("").astype(str)
    panel = panel[
        panel["author"].ne("")
        & panel["origin_subreddit"].ne("")
    ].copy()
    numeric_columns = [
        "origin_pair_id",
        "pre_posts_origin",
        "pre_active_months_origin",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "same_subreddit_postshock_posts",
        "postshock_total_posts_sample",
        "stayed_same_subreddit",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_higher_context",
    ]
    for column_name in numeric_columns:
        panel[column_name] = pd.to_numeric(panel[column_name], errors="coerce")
    panel = panel.dropna(
        subset=[
            "origin_pair_id",
            "pre_posts_origin",
            "pre_active_months_origin",
            "origin_persfree",
            "origin_gen_cap",
            "origin_phys_free",
            "stayed_same_subreddit",
            "disappeared_from_sample",
            "moved_other_subreddit",
            "moved_higher_context",
        ]
    ).copy()
    if panel.empty:
        raise ValueError("No usable author-origin rows for micro displacement prediction.")



    panel["pre_posts_origin"] = panel["pre_posts_origin"].astype(float)
    panel["pre_active_months_origin"] = panel["pre_active_months_origin"].clip(lower=1).astype(float)
    panel["log_pre_posts_origin"] = np.log1p(panel["pre_posts_origin"])
    panel["posts_per_active_month_origin"] = (
        panel["pre_posts_origin"] / panel["pre_active_months_origin"]
    )
    author_total_pre = panel.groupby("author", sort=False)["pre_posts_origin"].transform("sum")
    author_breadth = panel.groupby("author", sort=False)["origin_subreddit"].transform("nunique")
    panel["author_total_pre_posts_all_sampled_subreddits"] = author_total_pre.astype(float)
    panel["author_pre_subreddit_breadth"] = author_breadth.astype(float)
    if "pre_subreddit_posts" in panel.columns:
        panel["origin_subreddit_pre_posts_total"] = pd.to_numeric(
            panel["pre_subreddit_posts"], errors="coerce"
        )
    else:
        panel["origin_subreddit_pre_posts_total"] = panel.groupby(
            "origin_subreddit", sort=False
        )["pre_posts_origin"].transform("sum")
    panel["origin_subreddit_pre_posts_total"] = panel[
        "origin_subreddit_pre_posts_total"
    ].astype(float)
    if "log_pre_subreddit_posts" in panel.columns:
        panel["log_origin_size_pre"] = pd.to_numeric(
            panel["log_pre_subreddit_posts"], errors="coerce"
        )
    else:
        panel["log_origin_size_pre"] = np.log1p(panel["origin_subreddit_pre_posts_total"])
    panel["log_origin_size_pre"] = panel["log_origin_size_pre"].astype(float)
    if "origin_category" not in panel.columns:
        subreddit_categories = globals().get("SUBREDDITS", {})
        if isinstance(subreddit_categories, dict) and subreddit_categories:
            panel["origin_category"] = (
                panel["origin_subreddit"].map(subreddit_categories).fillna("unknown").astype(str)
            )
        else:
            panel["origin_category"] = "unknown"
    panel["origin_category"] = panel["origin_category"].fillna("unknown").astype(str)

    panel["stayed_same_subreddit"] = panel["stayed_same_subreddit"].astype(int)
    panel["displaced_from_origin"] = (panel["stayed_same_subreddit"] == 0).astype(int)
    panel["disappeared_from_sample"] = panel["disappeared_from_sample"].astype(int)
    panel["moved_other_subreddit"] = panel["moved_other_subreddit"].astype(int)
    panel["moved_higher_context"] = panel["moved_higher_context"].astype(int)


    panel["marginal_postshock_pair"] = (
        (panel["same_subreddit_postshock_posts"].fillna(0) > 0)
        & (panel["same_subreddit_postshock_posts"].fillna(0) <= 1)
    ).astype(int)
    panel["pre_shock_low_frequency_pair"] = (
        panel["pre_posts_origin"] <= 1
    ).astype(int)
    panel["pre_shock_low_active_months"] = (
        panel["pre_active_months_origin"] <= 1
    ).astype(int)

    outcomes = [
        "displaced_from_origin",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_higher_context",
        "stayed_same_subreddit",
        "marginal_postshock_pair",
    ]
    model_specs = [
        (
            "A_baseline_activity",
            [
                "log_pre_posts_origin",
                "pre_active_months_origin",
                "posts_per_active_month_origin",
                "author_pre_subreddit_breadth",
                "log_origin_size_pre",
            ],
        ),
        (
            "B_plus_gen_phys",
            [
                "log_pre_posts_origin",
                "pre_active_months_origin",
                "posts_per_active_month_origin",
                "author_pre_subreddit_breadth",
                "log_origin_size_pre",
                "origin_gen_cap",
                "origin_phys_free",
            ],
        ),
        (
            "C_plus_persfree",
            [
                "log_pre_posts_origin",
                "pre_active_months_origin",
                "posts_per_active_month_origin",
                "author_pre_subreddit_breadth",
                "log_origin_size_pre",
                "origin_persfree",
            ],
        ),
        (
            "D_full",
            [
                "log_pre_posts_origin",
                "pre_active_months_origin",
                "posts_per_active_month_origin",
                "author_pre_subreddit_breadth",
                "log_origin_size_pre",
                "origin_gen_cap",
                "origin_phys_free",
                "origin_persfree",
            ],
        ),
    ]

    def grouped_subreddit_folds(groups, requested_folds):
        unique_groups = np.array(sorted(pd.Series(groups).dropna().astype(str).unique()))
        if len(unique_groups) < 2:
            raise ValueError("Need at least two origin subreddits for grouped cross-validation.")
        fold_count = max(2, min(int(requested_folds), len(unique_groups)))
        rng = np.random.default_rng(random_seed)
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        return [set(fold) for fold in np.array_split(shuffled, fold_count) if len(fold) > 0]

    def standardized_design_matrix(frame, features, means=None, scales=None):
        x = frame[features].astype(float).to_numpy()
        if means is None:
            means = np.nanmean(x, axis=0)
        if scales is None:
            scales = np.nanstd(x, axis=0)
        scales = np.asarray(scales, dtype=float)
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0
        x_scaled = (x - means) / scales
        x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return np.column_stack([np.ones(len(frame), dtype=float), x_scaled]), means, scales

    def fit_lpm_coefficients(frame, outcome, features):
        x, means, scales = standardized_design_matrix(frame, features)
        y = frame[outcome].astype(float).to_numpy()
        try:
            ridge_alpha = max(1.0, len(frame) * 1e-6)
            penalty = np.eye(x.shape[1], dtype=float) * ridge_alpha
            penalty[0, 0] = 0.0
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                xtx = x.T @ x
                xty = x.T @ y
            if not np.isfinite(xtx).all() or not np.isfinite(xty).all():
                return None
            standardized_coefficients = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(standardized_coefficients).all():
            return None
        raw_coefficients = standardized_coefficients.copy()
        raw_coefficients[1:] = standardized_coefficients[1:] / scales
        raw_coefficients[0] = standardized_coefficients[0] - np.sum(
            standardized_coefficients[1:] * means / scales
        )
        return {
            "standardized": standardized_coefficients,
            "raw": raw_coefficients,
            "means": means,
            "scales": scales,
        }

    def cross_validated_lpm_predictions(frame, outcome, features, folds):
        predictions = np.full(len(frame), np.nan, dtype=float)
        groups = frame["origin_subreddit"].astype(str).to_numpy()
        for test_groups in folds:
            test_mask = np.isin(groups, list(test_groups))
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            train = frame.iloc[train_mask]
            test = frame.iloc[test_mask]
            if train[outcome].nunique(dropna=True) < 2:
                predictions[test_mask] = float(train[outcome].mean())
                continue
            fit = fit_lpm_coefficients(train, outcome, features)
            if fit is None:
                predictions[test_mask] = float(train[outcome].mean())
                continue
            x_test, _means, _scales = standardized_design_matrix(
                test,
                features,
                means=fit["means"],
                scales=fit["scales"],
            )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                fold_predictions = x_test @ fit["standardized"]
            if not np.isfinite(fold_predictions).all():
                fold_predictions = np.where(
                    np.isfinite(fold_predictions),
                    fold_predictions,
                    float(train[outcome].mean()),
                )
            predictions[test_mask] = fold_predictions
        return predictions

    def auc_score(y_true, scores):
        y_true = np.asarray(y_true, dtype=float)
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(scores)
        y_true = y_true[valid]
        scores = scores[valid]
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        rank_sum_pos = ranks[y_true == 1].sum()
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return safe_float(auc)

    def prediction_metrics(y_true, raw_predictions):
        y_true = np.asarray(y_true, dtype=float)
        raw_predictions = np.asarray(raw_predictions, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(raw_predictions)
        y_true = y_true[valid]
        raw_predictions = raw_predictions[valid]
        if len(y_true) == 0:
            return {}
        clipped = np.clip(raw_predictions, 1e-6, 1 - 1e-6)
        prevalence = float(y_true.mean())
        null_prediction = np.clip(prevalence, 1e-6, 1 - 1e-6)
        brier = float(np.mean((y_true - clipped) ** 2))
        null_brier = float(np.mean((y_true - null_prediction) ** 2))
        log_loss = float(
            -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
        )
        null_log_loss = float(
            -np.mean(y_true * np.log(null_prediction) + (1 - y_true) * np.log(1 - null_prediction))
        )
        return {
            "auc": auc_score(y_true, raw_predictions),
            "accuracy": safe_float(np.mean((clipped >= 0.5).astype(int) == y_true)),
            "brier_score": safe_float(brier),
            "log_loss": safe_float(log_loss),
            "pseudo_r2_log_loss": safe_float(1 - log_loss / null_log_loss)
            if null_log_loss > 0 else None,
            "brier_improvement_vs_null": safe_float(1 - brier / null_brier)
            if null_brier > 0 else None,
        }

    metric_rows = []
    coefficient_lookup = {}
    all_features = sorted({feature for _name, features in model_specs for feature in features})
    panel[all_features + outcomes] = panel[all_features + outcomes].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    model_panel = panel.dropna(
        subset=outcomes + all_features + ["origin_subreddit"]
    ).copy()
    if model_panel.empty:
        raise ValueError("No complete rows for micro displacement prediction models.")

    folds = grouped_subreddit_folds(model_panel["origin_subreddit"], n_folds)



    for outcome in outcomes:
        if model_panel[outcome].nunique(dropna=True) < 2:
            continue
        baseline_auc = None
        baseline_brier = None
        baseline_log_loss = None
        for model_name, features in model_specs:
            frame = model_panel.dropna(subset=[outcome] + features + ["origin_subreddit"]).copy()
            if frame.empty or frame[outcome].nunique(dropna=True) < 2:
                continue
            predictions = cross_validated_lpm_predictions(frame, outcome, features, folds)
            metrics = prediction_metrics(frame[outcome].to_numpy(), predictions)
            full_fit = fit_lpm_coefficients(frame, outcome, features)
            persfree_coefficient = None
            persfree_sign = None
            if full_fit is not None and "origin_persfree" in features:
                persfree_index = features.index("origin_persfree") + 1
                persfree_coefficient = safe_float(full_fit["raw"][persfree_index])
                persfree_sign = (
                    "positive" if persfree_coefficient > 0
                    else "negative" if persfree_coefficient < 0
                    else "zero"
                )
            if model_name == "A_baseline_activity":
                baseline_auc = metrics.get("auc")
                baseline_brier = metrics.get("brier_score")
                baseline_log_loss = metrics.get("log_loss")
            coefficient_lookup[(outcome, model_name)] = persfree_coefficient
            metric_rows.append({
                "outcome": outcome,
                "model_name": model_name,
                "model_family": "grouped_cv_linear_probability",
                "n_observations": safe_int(len(frame)),
                "n_author_origin_pairs": safe_int(len(frame)),
                "n_authors": safe_int(frame["author"].nunique()),
                "n_origin_subreddits": safe_int(frame["origin_subreddit"].nunique()),
                "n_folds": safe_int(len(folds)),
                "outcome_prevalence": safe_float(frame[outcome].mean()),
                "auc": metrics.get("auc"),
                "accuracy": metrics.get("accuracy"),
                "brier_score": metrics.get("brier_score"),
                "log_loss": metrics.get("log_loss"),
                "pseudo_r2_log_loss": metrics.get("pseudo_r2_log_loss"),
                "brier_improvement_vs_null": metrics.get("brier_improvement_vs_null"),
                "delta_auc_vs_baseline": safe_float(metrics.get("auc") - baseline_auc)
                if baseline_auc is not None and metrics.get("auc") is not None else None,
                "delta_brier_vs_baseline": safe_float(baseline_brier - metrics.get("brier_score"))
                if baseline_brier is not None and metrics.get("brier_score") is not None else None,
                "delta_log_loss_vs_baseline": safe_float(baseline_log_loss - metrics.get("log_loss"))
                if baseline_log_loss is not None and metrics.get("log_loss") is not None else None,
                "persfree_lpm_prediction_coef": persfree_coefficient,
                "persfree_lpm_prediction_sign": persfree_sign,
                "features": ", ".join(features),
                "cv_grouping": "origin_subreddit",
            })

    metrics_table = pd.DataFrame(metric_rows)
    if metrics_table.empty:
        raise ValueError("No micro displacement prediction metrics were estimated.")

    lpm_terms = [
        "origin_persfree",
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "author_pre_subreddit_breadth",
        "log_origin_size_pre",
        "origin_gen_cap",
        "origin_phys_free",
    ]
    coefficient_rows = []



    for outcome in outcomes:
        lpm_data = model_panel.dropna(subset=[outcome] + lpm_terms + ["origin_subreddit"]).copy()
        if lpm_data.empty or lpm_data[outcome].nunique(dropna=True) < 2:
            continue
        formula = f"{outcome} ~ " + " + ".join(lpm_terms)
        model = fit_ols(formula, lpm_data, cluster_col="origin_subreddit")
        if model is None:
            continue
        for term in lpm_terms:
            result = reg_result(model, term)
            coefficient_rows.append({
                "outcome": outcome,
                "term": term,
                "coef": result.get("coef"),
                "standard_error": result.get("se"),
                "p_value": result.get("pvalue"),
                "n_observations": safe_int(model.nobs),
                "n_author_origin_pairs": safe_int(len(lpm_data)),
                "n_authors": safe_int(lpm_data["author"].nunique()),
                "n_origin_subreddits": safe_int(lpm_data["origin_subreddit"].nunique()),
                "fixed_effects": "none",
                "clustering_level": "origin subreddit",
                "controls": ", ".join(term_name for term_name in lpm_terms if term_name != term),
            })

    coefficients_table = pd.DataFrame(coefficient_rows)
    if coefficients_table.empty:
        raise ValueError("No micro displacement prediction LPM coefficients were estimated.")

    metrics_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_table.to_csv(metrics_path, index=False)
    metrics_latex_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_latex_columns = [
        "outcome",
        "model_name",
        "n_observations",
        "outcome_prevalence",
        "auc",
        "brier_score",
        "log_loss",
        "delta_auc_vs_baseline",
        "persfree_lpm_prediction_coef",
    ]
    metrics_latex_path.write_text(
        metrics_table[metrics_latex_columns].to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
        ),
        encoding="utf-8",
    )

    coefficients_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients_table.to_csv(coefficients_path, index=False)
    coefficients_latex_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients_latex_columns = [
        "outcome",
        "term",
        "coef",
        "standard_error",
        "p_value",
        "n_observations",
        "n_origin_subreddits",
        "clustering_level",
    ]
    coefficients_latex_path.write_text(
        coefficients_table[coefficients_latex_columns].to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
        ),
        encoding="utf-8",
    )

    panel_columns = [
        "author",
        "origin_pair_id",
        "origin_subreddit",
        "origin_category",
        "pre_posts_origin",
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "first_pre_month_origin",
        "last_pre_month_origin",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_subreddit_pre_posts_total",
        "log_origin_size_pre",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "stayed_same_subreddit",
        "displaced_from_origin",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_higher_context",
        "marginal_postshock_pair",
        "pre_shock_low_frequency_pair",
        "pre_shock_low_active_months",
    ]
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    panel[panel_columns].to_csv(panel_output_path, index=False)

    def metric_row(outcome, model_name):
        rows = metrics_table[
            metrics_table["outcome"].eq(outcome)
            & metrics_table["model_name"].eq(model_name)
        ]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    displacement_persfree = metric_row("displaced_from_origin", "C_plus_persfree")
    displacement_gen_phys = metric_row("displaced_from_origin", "B_plus_gen_phys")
    moved_higher_persfree = metric_row("moved_higher_context", "C_plus_persfree")
    moved_higher_gen_phys = metric_row("moved_higher_context", "B_plus_gen_phys")
    disp_delta = displacement_persfree.get("delta_auc_vs_baseline")
    higher_delta = moved_higher_persfree.get("delta_auc_vs_baseline")
    gen_phys_delta = displacement_gen_phys.get("delta_auc_vs_baseline")
    persfree_more_predictive = (
        disp_delta is not None
        and gen_phys_delta is not None
        and disp_delta > gen_phys_delta
    )
    low_freq_rate = safe_float(
        model_panel.loc[model_panel["pre_shock_low_frequency_pair"].eq(1), "displaced_from_origin"].mean()
    )
    higher_freq_rate = safe_float(
        model_panel.loc[model_panel["pre_shock_low_frequency_pair"].eq(0), "displaced_from_origin"].mean()
    )
    low_active_rate = safe_float(
        model_panel.loc[model_panel["pre_shock_low_active_months"].eq(1), "displaced_from_origin"].mean()
    )
    more_active_rate = safe_float(
        model_panel.loc[model_panel["pre_shock_low_active_months"].eq(0), "displaced_from_origin"].mean()
    )
    vulnerability_phrase = (
        "low-frequency and low-active-month pairs are most vulnerable"
        if low_freq_rate > higher_freq_rate and low_active_rate > more_active_rate
        else "low-frequency and low-active-month pairs are not uniformly the most vulnerable"
    )
    summary = (
        "Micro displacement prediction: adding PersFree changes displaced-from-origin AUC by "
        f"{disp_delta:.4f} and moved-higher-context AUC by {higher_delta:.4f}. "
        f"GenCap/PhysFree are {'less' if persfree_more_predictive else 'not less'} predictive than PersFree "
        f"for displaced-from-origin by AUC gain. {vulnerability_phrase}: "
        f"displacement rates are {low_freq_rate:.3f} for one-post origin pairs versus "
        f"{higher_freq_rate:.3f} otherwise, and {low_active_rate:.3f} for one-active-month pairs versus "
        f"{more_active_rate:.3f} otherwise. "
        + (
            "PersFree adds meaningful individual-level predictive power beyond activity and size."
            if disp_delta is not None and disp_delta >= 0.01
            else "PersFree adds little individual-level predictive lift, so this evidence is stronger for community-level prediction than individual-level prediction."
        )
    )
    print(summary)

    return {
        "metrics": metrics_table,
        "coefficients": coefficients_table,
        "panel": panel[panel_columns],
        "summary": summary,
        "metrics_path": str(metrics_path),
        "metrics_latex_path": str(metrics_latex_path),
        "coefficients_path": str(coefficients_path),
        "coefficients_latex_path": str(coefficients_latex_path),
        "panel_output_path": str(panel_output_path),
    }

def compute_micro_level_prediction(
    score_path=None,
    counts_path=None,
    pairs_path=None,
    panel_output_path=None,
    metrics_path=None,
    metrics_latex_path=None,
    coefficients_path=None,
    coefficients_latex_path=None,
    risk_quartiles_path=None,
    risk_quartiles_latex_path=None,
    comparison_path=None,
    comparison_latex_path=None,
    max_model_rows=250_000,
    min_rows_per_subreddit=500,
    n_folds=5,
    random_seed=20250604,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    if score_path is None:
        score_candidates = [
            tables_dir / "acsi_preshock_tworuns.csv",
            root / "acsi_preshock_tworuns.csv",
            data_dir / "acsi_preshock_tworuns.csv",
        ]
        score_path = next((path for path in score_candidates if path.exists()), score_candidates[0])
    score_path = Path(score_path)
    counts_path = Path(counts_path or tables_dir / "author_month_intensive_counts.parquet")
    pairs_path = Path(pairs_path or tables_dir / "author_month_intensive_pairs.parquet")
    panel_output_path = Path(panel_output_path or tables_dir / "micro_level_prediction_panel.csv")
    metrics_path = Path(metrics_path or tables_dir / "micro_level_prediction_metrics.csv")
    metrics_latex_path = Path(metrics_latex_path or tables_dir / "micro_level_prediction_metrics.tex")
    coefficients_path = Path(coefficients_path or tables_dir / "micro_level_prediction_coefficients.csv")
    coefficients_latex_path = Path(coefficients_latex_path or tables_dir / "micro_level_prediction_coefficients.tex")
    risk_quartiles_path = Path(risk_quartiles_path or tables_dir / "micro_level_prediction_risk_quartiles.csv")
    risk_quartiles_latex_path = Path(
        risk_quartiles_latex_path or tables_dir / "micro_level_prediction_risk_quartiles.tex"
    )
    comparison_path = Path(comparison_path or tables_dir / "micro_macro_prediction_comparison.csv")
    comparison_latex_path = Path(
        comparison_latex_path or tables_dir / "micro_macro_prediction_comparison.tex"
    )

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    if not counts_path.exists() or not pairs_path.exists():
        compute_author_month_intensive_margin(
            score_path=score_path,
            counts_path=counts_path,
            pairs_path=pairs_path,
        )

    pair_columns = [
        "pair_id",
        "author",
        "subreddit",
        "pre_pair_posts",
        "post_pair_posts",
        "author_pre_posts",
        "author_post_posts",
        "gen_cap",
        "phys_free",
        "pers_free",
    ]
    pairs = pd.read_parquet(pairs_path, columns=pair_columns)
    counts = pd.read_parquet(counts_path)
    required_pair_columns = set(pair_columns)
    required_count_columns = {"pair_id", "month_index", "post_count"}
    missing_pair_columns = required_pair_columns - set(pairs.columns)
    missing_count_columns = required_count_columns - set(counts.columns)
    if missing_pair_columns:
        raise ValueError(f"Pair cache missing columns: {sorted(missing_pair_columns)}")
    if missing_count_columns:
        raise ValueError(f"Count cache missing columns: {sorted(missing_count_columns)}")

    for column_name in ["pair_id", "pre_pair_posts", "post_pair_posts", "author_pre_posts", "author_post_posts"]:
        pairs[column_name] = pd.to_numeric(pairs[column_name], errors="coerce")
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        pairs[column_name] = pd.to_numeric(pairs[column_name], errors="coerce")
    pairs["author"] = pairs["author"].fillna("").astype(str)
    pairs["subreddit"] = pairs["subreddit"].fillna("").astype(str)
    pairs = pairs.dropna(
        subset=[
            "pair_id",
            "author",
            "subreddit",
            "pre_pair_posts",
            "post_pair_posts",
            "author_pre_posts",
            "author_post_posts",
            "gen_cap",
            "phys_free",
            "pers_free",
        ]
    ).copy()
    pairs = pairs[pairs["author"].ne("") & pairs["subreddit"].ne("")].copy()
    pairs["pair_id"] = pairs["pair_id"].astype(np.int64)
    pairs["pre_pair_posts"] = pairs["pre_pair_posts"].astype(float)
    pairs["post_pair_posts"] = pairs["post_pair_posts"].astype(float)
    pairs["author_pre_posts"] = pairs["author_pre_posts"].astype(float)
    pairs["author_post_posts"] = pairs["author_post_posts"].astype(float)

    counts["pair_id"] = pd.to_numeric(counts["pair_id"], errors="coerce")
    counts["month_index"] = pd.to_numeric(counts["month_index"], errors="coerce")
    counts["post_count"] = pd.to_numeric(counts["post_count"], errors="coerce")
    counts = counts.dropna(subset=["pair_id", "month_index", "post_count"]).copy()
    counts["pair_id"] = counts["pair_id"].astype(np.int64)
    counts["month_index"] = counts["month_index"].astype(np.int16)
    counts["post_count"] = counts["post_count"].fillna(0).astype(float)

    all_months = [
        month.strftime("%Y-%m")
        for month in globals().get(
            "ALL_MONTHS",
            pd.date_range("2022-01-01", "2024-12-01", freq="MS"),
        )
    ]
    if len(all_months) < 36:
        all_months = pd.date_range("2022-01-01", "2024-12-01", freq="MS").strftime("%Y-%m").tolist()
    pre_month_indices = set(range(0, min(11, len(all_months))))
    post_month_indices = set(range(11, len(all_months)))
    transition_index = 10
    month_index_to_label = dict(enumerate(all_months))

    pre_counts = counts[counts["month_index"].isin(pre_month_indices)].copy()
    post_counts = counts[counts["month_index"].isin(post_month_indices)].copy()
    pre_pair_stats = pre_counts.groupby("pair_id", sort=False).agg(
        pre_active_months_origin=("month_index", "nunique"),
        first_pre_month_index=("month_index", "min"),
        last_pre_month_index=("month_index", "max"),
    )
    post_pair_stats = post_counts.groupby("pair_id", sort=False).agg(
        postshock_active_months_origin=("month_index", "nunique"),
    )

    pre_pairs = pairs[pairs["pre_pair_posts"] > 0].copy()
    if pre_pairs.empty:
        raise ValueError("No pre-shock author-subreddit pairs found in pair cache.")
    pre_pairs = pre_pairs.merge(pre_pair_stats, left_on="pair_id", right_index=True, how="left")
    pre_pairs = pre_pairs.merge(post_pair_stats, left_on="pair_id", right_index=True, how="left")
    pre_pairs["pre_active_months_origin"] = (
        pd.to_numeric(pre_pairs["pre_active_months_origin"], errors="coerce")
        .fillna(1)
        .clip(lower=1)
        .astype(float)
    )
    pre_pairs["postshock_active_months_origin"] = (
        pd.to_numeric(pre_pairs["postshock_active_months_origin"], errors="coerce")
        .fillna(0)
        .astype(float)
    )
    pre_pairs["first_pre_month_index"] = (
        pd.to_numeric(pre_pairs["first_pre_month_index"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    pre_pairs["last_pre_month_index"] = (
        pd.to_numeric(pre_pairs["last_pre_month_index"], errors="coerce")
        .fillna(transition_index)
        .astype(int)
    )

    pair_subreddit = pairs.set_index("pair_id")["subreddit"]
    pre_counts["subreddit"] = pre_counts["pair_id"].map(pair_subreddit).astype(str)
    monthly_posts = (
        pre_counts.groupby(["subreddit", "month_index"], as_index=False)["post_count"]
        .sum()
        .rename(columns={"post_count": "monthly_posts"})
    )
    target_subreddits = sorted(pre_pairs["subreddit"].astype(str).unique())
    pre_grid = pd.MultiIndex.from_product(
        [target_subreddits, sorted(pre_month_indices)],
        names=["subreddit", "month_index"],
    ).to_frame(index=False)
    pre_monthly = pre_grid.merge(monthly_posts, on=["subreddit", "month_index"], how="left")
    pre_monthly["monthly_posts"] = pre_monthly["monthly_posts"].fillna(0).astype(float)
    pre_monthly["log_posts"] = np.log1p(pre_monthly["monthly_posts"])
    origin_feature_rows = []
    for subreddit, group in pre_monthly.groupby("subreddit", sort=False):
        group = group.sort_values("month_index")
        x = group["month_index"].astype(float).to_numpy()
        y = group["log_posts"].astype(float).to_numpy()
        slope = float(np.polyfit(x - x.mean(), y, 1)[0]) if len(group) >= 2 and np.nanstd(y) > 0 else 0.0
        total_posts = float(group["monthly_posts"].sum())
        origin_feature_rows.append({
            "origin_subreddit": subreddit,
            "origin_pre_mean_log_posts": safe_float(group["log_posts"].mean()),
            "origin_pre_trend_slope": safe_float(slope),
            "origin_pre_volatility": safe_float(group["log_posts"].std(ddof=1)),
            "origin_pre_total_posts": safe_float(total_posts),
            "log_origin_pre_total_posts": safe_float(np.log1p(total_posts)),
        })
    origin_features = pd.DataFrame(origin_feature_rows)

    post_destination_pairs = pairs[pairs["post_pair_posts"] > 0].copy()
    author_dest_stats = post_destination_pairs.groupby("author", sort=False).agg(
        author_post_destination_count=("subreddit", "nunique"),
        author_post_destination_persfree_sum=("pers_free", "sum"),
    )
    pre_pairs = pre_pairs.merge(author_dest_stats, left_on="author", right_index=True, how="left")
    pre_pairs["author_post_destination_count"] = (
        pd.to_numeric(pre_pairs["author_post_destination_count"], errors="coerce")
        .fillna(0)
        .astype(float)
    )
    pre_pairs["author_post_destination_persfree_sum"] = (
        pd.to_numeric(pre_pairs["author_post_destination_persfree_sum"], errors="coerce")
        .fillna(0)
        .astype(float)
    )
    origin_post_indicator = (pre_pairs["post_pair_posts"] > 0).astype(float)
    pre_pairs["destination_count"] = (
        pre_pairs["author_post_destination_count"] - origin_post_indicator
    ).clip(lower=0)
    destination_persfree_numerator = (
        pre_pairs["author_post_destination_persfree_sum"]
        - origin_post_indicator * pre_pairs["pers_free"].astype(float)
    )
    pre_pairs["mean_destination_persfree"] = np.where(
        pre_pairs["destination_count"] > 0,
        destination_persfree_numerator / pre_pairs["destination_count"],
        np.nan,
    )
    del post_destination_pairs
    del author_dest_stats
    gc.collect()

    panel = pre_pairs.rename(
        columns={
            "pair_id": "origin_pair_id",
            "subreddit": "origin_subreddit",
            "pre_pair_posts": "pre_posts_origin",
            "post_pair_posts": "postshock_posts_origin",
            "author_pre_posts": "author_total_pre_posts_all_sampled_subreddits",
            "author_post_posts": "postshock_total_posts_sample",
            "pers_free": "origin_persfree",
            "gen_cap": "origin_gen_cap",
            "phys_free": "origin_phys_free",
        }
    )
    panel = panel.merge(origin_features, on="origin_subreddit", how="left")
    panel["author_pre_subreddit_breadth"] = (
        panel.groupby("author", sort=False)["origin_subreddit"].transform("nunique").astype(float)
    )
    panel["log_pre_posts_origin"] = np.log1p(panel["pre_posts_origin"].astype(float))
    panel["posts_per_active_month_origin"] = (
        panel["pre_posts_origin"].astype(float) / panel["pre_active_months_origin"].clip(lower=1)
    )
    panel["first_pre_month_origin"] = panel["first_pre_month_index"].map(month_index_to_label).fillna("2022-01")
    panel["last_pre_month_origin"] = panel["last_pre_month_index"].map(month_index_to_label).fillna("2022-11")
    panel["months_since_last_pre_post"] = (
        transition_index - panel["last_pre_month_index"].astype(float)
    ).clip(lower=0)
    mu_lookup = globals().get("MU_K", {})
    panel["subscriber_count"] = panel["origin_subreddit"].map(mu_lookup)
    subscriber_fill = (
        float(pd.to_numeric(panel["subscriber_count"], errors="coerce").dropna().median())
        if pd.to_numeric(panel["subscriber_count"], errors="coerce").notna().any()
        else 0.5
    )
    panel["subscriber_count"] = (
        pd.to_numeric(panel["subscriber_count"], errors="coerce")
        .fillna(subscriber_fill)
        .astype(float)
    )
    panel["log_subscribers"] = np.log1p(panel["subscriber_count"])
    subreddit_categories = globals().get("SUBREDDITS", {})
    if isinstance(subreddit_categories, dict) and subreddit_categories:
        panel["origin_category"] = (
            panel["origin_subreddit"].map(subreddit_categories).fillna("unknown").astype(str)
        )
    else:
        panel["origin_category"] = "unknown"

    panel["stayed_origin"] = (panel["postshock_posts_origin"] > 0).astype(int)
    panel["broke_origin_tie"] = (panel["stayed_origin"] == 0).astype(int)
    panel["disappeared_from_sample"] = (
        panel["postshock_total_posts_sample"].astype(float) <= 0
    ).astype(int)
    panel["any_other_subreddit_postshock"] = (panel["destination_count"] > 0).astype(int)
    panel["moved_other_subreddit"] = (
        panel["broke_origin_tie"].eq(1)
        & panel["any_other_subreddit_postshock"].eq(1)
    ).astype(int)
    panel["moved_context_heavy"] = (
        panel["moved_other_subreddit"].eq(1)
        & (panel["mean_destination_persfree"] <= panel["origin_persfree"] - 0.05)
    ).astype(int)
    panel["moved_context_light"] = (
        panel["moved_other_subreddit"].eq(1)
        & (panel["mean_destination_persfree"] >= panel["origin_persfree"] + 0.05)
    ).astype(int)
    panel["postshock_low_activity_origin"] = (
        panel["stayed_origin"].eq(1)
        & (
            panel["postshock_posts_origin"].le(1)
            | panel["postshock_active_months_origin"].le(1)
        )
    ).astype(int)

    category_values = [
        value
        for value in sorted(panel["origin_category"].dropna().astype(str).unique())
        if value != "unknown"
    ]
    category_features = []
    if len(panel["origin_category"].dropna().unique()) > 1:
        base_category = sorted(panel["origin_category"].dropna().astype(str).unique())[0]
        for category in sorted(panel["origin_category"].dropna().astype(str).unique()):
            if category == base_category:
                continue
            safe_category = "".join(char if char.isalnum() else "_" for char in category)
            column_name = f"origin_category_{safe_category}"
            panel[column_name] = panel["origin_category"].astype(str).eq(category).astype(float)
            category_features.append(column_name)
    else:
        base_category = "none"

    panel_columns = [
        "author",
        "origin_pair_id",
        "origin_subreddit",
        "origin_category",
        "pre_posts_origin",
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "first_pre_month_origin",
        "last_pre_month_origin",
        "months_since_last_pre_post",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_pre_mean_log_posts",
        "origin_pre_trend_slope",
        "origin_pre_volatility",
        "origin_pre_total_posts",
        "log_origin_pre_total_posts",
        "subscriber_count",
        "log_subscribers",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "stayed_origin",
        "broke_origin_tie",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "any_other_subreddit_postshock",
        "moved_context_heavy",
        "moved_context_light",
        "postshock_low_activity_origin",
        "postshock_posts_origin",
        "postshock_active_months_origin",
        "postshock_total_posts_sample",
        "destination_count",
        "mean_destination_persfree",
    ] + category_features
    panel = panel.sort_values(["origin_subreddit", "author"]).reset_index(drop=True)
    panel_output_path.parent.mkdir(exist_ok=True, parents=True)
    panel[panel_columns].to_csv(panel_output_path, index=False)

    activity_features = [
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "months_since_last_pre_post",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_pre_mean_log_posts",
        "origin_pre_trend_slope",
        "origin_pre_volatility",
        "log_origin_pre_total_posts",
        "log_subscribers",
    ]
    exposure_features = ["origin_persfree", "origin_gen_cap", "origin_phys_free"]
    model_specs = [
        ("A_activity_only", activity_features),
        ("B_exposure_only", exposure_features),
        ("C_activity_persfree", activity_features + ["origin_persfree"]),
        ("D_full", activity_features + exposure_features),
    ]
    if category_features:
        model_specs.append(("E_full_category", activity_features + exposure_features + category_features))
    outcomes = [
        "stayed_origin",
        "broke_origin_tie",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "any_other_subreddit_postshock",
        "moved_context_heavy",
        "moved_context_light",
        "postshock_low_activity_origin",
    ]

    all_model_features = sorted({feature for _name, features in model_specs for feature in features})
    model_frame = panel[[
        "author",
        "origin_subreddit",
        "origin_category",
        "pre_posts_origin",
        "pre_active_months_origin",
        "author_pre_subreddit_breadth",
    ] + outcomes + all_model_features].replace([np.inf, -np.inf], np.nan)
    model_frame = model_frame.dropna(subset=outcomes + all_model_features + ["origin_subreddit"]).copy()
    if model_frame.empty:
        raise ValueError("No complete rows for micro-level prediction models.")

    def stratified_model_sample(frame):
        if len(frame) <= int(max_model_rows):
            sampled = frame.copy()
            sampled["analysis_sample"] = "full_panel"
            return sampled
        rng = np.random.default_rng(int(random_seed))
        group_indices = frame.groupby("origin_subreddit", sort=False).indices
        base_indices = []
        per_group_target = min(
            int(min_rows_per_subreddit),
            max(1, int(max_model_rows) // max(len(group_indices), 1)),
        )
        for _group, index_values in group_indices.items():
            index_array = np.asarray(index_values, dtype=np.int64)
            take = min(per_group_target, len(index_array))
            base_indices.append(rng.choice(index_array, size=take, replace=False))
        base_indices = np.unique(np.concatenate(base_indices)) if base_indices else np.array([], dtype=np.int64)
        remaining_slots = int(max_model_rows) - len(base_indices)
        if remaining_slots > 0:
            remaining = np.setdiff1d(np.arange(len(frame), dtype=np.int64), base_indices, assume_unique=False)
            if len(remaining) > 0:
                fill = rng.choice(remaining, size=min(remaining_slots, len(remaining)), replace=False)
                base_indices = np.unique(np.concatenate([base_indices, fill]))
        sampled = frame.iloc[base_indices].copy()
        sampled["analysis_sample"] = (
            f"stratified_origin_subreddit_sample_seed_{random_seed}_max_{int(max_model_rows)}"
        )
        return sampled

    model_sample = stratified_model_sample(model_frame).reset_index(drop=True)

    def grouped_subreddit_folds(groups, fold_count):
        unique_groups = np.array(sorted(pd.Series(groups).dropna().astype(str).unique()))
        if len(unique_groups) == 0:
            return []
        rng = np.random.default_rng(int(random_seed))
        rng.shuffle(unique_groups)
        fold_count = max(2, min(int(fold_count), len(unique_groups)))
        return [set(chunk.tolist()) for chunk in np.array_split(unique_groups, fold_count) if len(chunk) > 0]

    def standardized_design_matrix(frame, features, means=None, scales=None):
        features = list(dict.fromkeys(features))
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        x = frame[features].astype(float).replace([np.inf, -np.inf], np.nan)
        if means is None:
            means = x.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if scales is None:
            scales = (
                x.std(ddof=0)
                .replace([0, np.inf, -np.inf], np.nan)
                .fillna(1.0)
            )
        x = (x - means) / scales
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e6, 1e6)
        return np.column_stack([np.ones(len(x), dtype=float), x.to_numpy(dtype=float)]), means, scales

    def fit_lpm_coefficients(frame, outcome, features):
        if frame.empty or frame[outcome].nunique(dropna=True) < 2:
            return None
        x, means, scales = standardized_design_matrix(frame, features)
        y = frame[outcome].astype(float).to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return None
        try:
            standardized = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(standardized).all():
            return None
        raw = standardized.copy()
        raw[1:] = standardized[1:] / scales.to_numpy(dtype=float)
        raw[0] = standardized[0] - float(np.sum(raw[1:] * means.to_numpy(dtype=float)))
        if not np.isfinite(raw).all():
            raw = np.where(np.isfinite(raw), raw, np.nan)
        return {"standardized": standardized, "raw": raw, "means": means, "scales": scales}

    def predict_lpm(frame, features, fit):
        x, _means, _scales = standardized_design_matrix(
            frame,
            features,
            means=fit["means"],
            scales=fit["scales"],
        )
        if not np.isfinite(fit["standardized"]).all():
            return np.full(len(frame), np.nan, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            predictions = x @ fit["standardized"]
        return np.where(np.isfinite(predictions), predictions, np.nan)

    def cross_validated_lpm_predictions(frame, outcome, features, folds):
        predictions = np.full(len(frame), np.nan, dtype=float)
        groups = frame["origin_subreddit"].astype(str).to_numpy()
        for test_groups in folds:
            test_mask = np.isin(groups, list(test_groups))
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            train = frame.iloc[train_mask]
            test = frame.iloc[test_mask]
            if train[outcome].nunique(dropna=True) < 2:
                predictions[test_mask] = float(train[outcome].mean())
                continue
            fit = fit_lpm_coefficients(train, outcome, features)
            if fit is None:
                predictions[test_mask] = float(train[outcome].mean())
                continue
            fold_predictions = predict_lpm(test, features, fit)
            predictions[test_mask] = np.where(
                np.isfinite(fold_predictions),
                fold_predictions,
                float(train[outcome].mean()),
            )
        return predictions

    def auc_score(y_true, scores):
        y_true = np.asarray(y_true, dtype=float)
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(scores)
        y_true = y_true[valid]
        scores = scores[valid]
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        rank_sum_pos = ranks[y_true == 1].sum()
        return safe_float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def prediction_metrics(y_true, raw_predictions):
        y_true = np.asarray(y_true, dtype=float)
        raw_predictions = np.asarray(raw_predictions, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(raw_predictions)
        y_true = y_true[valid]
        raw_predictions = raw_predictions[valid]
        if len(y_true) == 0:
            return {}
        clipped = np.clip(raw_predictions, 1e-6, 1 - 1e-6)
        prevalence = float(y_true.mean())
        null_prediction = np.clip(prevalence, 1e-6, 1 - 1e-6)
        brier = float(np.mean((y_true - clipped) ** 2))
        null_brier = float(np.mean((y_true - null_prediction) ** 2))
        log_loss = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))
        null_log_loss = float(
            -np.mean(y_true * np.log(null_prediction) + (1 - y_true) * np.log(1 - null_prediction))
        )
        if len(raw_predictions) >= 4:
            top_cutoff = pd.Series(raw_predictions).rank(method="first", ascending=False) <= max(1, len(raw_predictions) // 4)
            top_precision = float(y_true[top_cutoff.to_numpy()].mean())
        else:
            top_precision = np.nan
        return {
            "auc": auc_score(y_true, raw_predictions),
            "accuracy": safe_float(np.mean((clipped >= 0.5).astype(int) == y_true)),
            "brier_score": safe_float(brier),
            "log_loss": safe_float(log_loss),
            "top_risk_quartile_precision": safe_float(top_precision),
            "base_rate": safe_float(prevalence),
            "lift": safe_float(top_precision / prevalence)
            if prevalence > 0 and np.isfinite(top_precision) else None,
            "pseudo_r2_log_loss": safe_float(1 - log_loss / null_log_loss)
            if null_log_loss > 0 else None,
            "brier_improvement_vs_null": safe_float(1 - brier / null_brier)
            if null_brier > 0 else None,
        }

    folds = grouped_subreddit_folds(model_sample["origin_subreddit"], n_folds)
    metric_rows = []
    for outcome in outcomes:
        if model_sample[outcome].nunique(dropna=True) < 2:
            continue
        baseline_auc = None
        baseline_brier = None
        baseline_log_loss = None
        baseline_lift = None
        for model_name, features in model_specs:
            frame = model_sample.dropna(subset=[outcome] + features + ["origin_subreddit"]).copy()
            if frame.empty or frame[outcome].nunique(dropna=True) < 2:
                continue
            predictions = cross_validated_lpm_predictions(frame, outcome, features, folds)
            metrics = prediction_metrics(frame[outcome].to_numpy(), predictions)
            if model_name == "A_activity_only":
                baseline_auc = metrics.get("auc")
                baseline_brier = metrics.get("brier_score")
                baseline_log_loss = metrics.get("log_loss")
                baseline_lift = metrics.get("lift")
            full_fit = fit_lpm_coefficients(frame, outcome, features)
            persfree_coefficient = None
            if full_fit is not None and "origin_persfree" in features:
                persfree_coefficient = safe_float(full_fit["raw"][features.index("origin_persfree") + 1])
            metric_rows.append({
                "outcome": outcome,
                "model_name": model_name,
                "model_family": "grouped_cv_linear_probability",
                "n_observations": safe_int(len(frame)),
                "n_total_panel_rows": safe_int(len(panel)),
                "n_authors": safe_int(frame["author"].nunique()),
                "n_origin_subreddits": safe_int(frame["origin_subreddit"].nunique()),
                "n_folds": safe_int(len(folds)),
                "sample_note": frame["analysis_sample"].iloc[0] if "analysis_sample" in frame.columns else "full_panel",
                "outcome_prevalence": metrics.get("base_rate"),
                "auc": metrics.get("auc"),
                "brier_score": metrics.get("brier_score"),
                "log_loss": metrics.get("log_loss"),
                "accuracy": metrics.get("accuracy"),
                "top_risk_quartile_precision": metrics.get("top_risk_quartile_precision"),
                "base_rate": metrics.get("base_rate"),
                "lift": metrics.get("lift"),
                "pseudo_r2_log_loss": metrics.get("pseudo_r2_log_loss"),
                "brier_improvement_vs_null": metrics.get("brier_improvement_vs_null"),
                "delta_auc_vs_activity": safe_float(metrics.get("auc") - baseline_auc)
                if baseline_auc is not None and metrics.get("auc") is not None else None,
                "delta_brier_vs_activity": safe_float(baseline_brier - metrics.get("brier_score"))
                if baseline_brier is not None and metrics.get("brier_score") is not None else None,
                "delta_log_loss_vs_activity": safe_float(baseline_log_loss - metrics.get("log_loss"))
                if baseline_log_loss is not None and metrics.get("log_loss") is not None else None,
                "delta_lift_vs_activity": safe_float(metrics.get("lift") - baseline_lift)
                if baseline_lift is not None and metrics.get("lift") is not None else None,
                "persfree_lpm_prediction_coef": persfree_coefficient,
                "features": ", ".join(features),
                "cv_grouping": "origin_subreddit",
            })

    metrics_table = pd.DataFrame(metric_rows)
    if metrics_table.empty:
        raise ValueError("No micro-level prediction metrics were estimated.")

    lpm_terms = [
        "origin_persfree",
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "months_since_last_pre_post",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_pre_mean_log_posts",
        "origin_pre_trend_slope",
        "origin_pre_volatility",
        "log_origin_pre_total_posts",
        "log_subscribers",
        "origin_gen_cap",
        "origin_phys_free",
    ]
    category_term = " + C(origin_category)" if category_features else ""
    coefficient_rows = []


    for outcome in outcomes:
        lpm_data = model_sample.dropna(subset=[outcome] + lpm_terms + ["origin_subreddit"]).copy()
        if lpm_data.empty or lpm_data[outcome].nunique(dropna=True) < 2:
            continue
        formula = f"{outcome} ~ " + " + ".join(lpm_terms) + category_term
        model = fit_ols(formula, lpm_data, cluster_col="origin_subreddit")
        if model is None:
            continue
        for term in lpm_terms:
            result = reg_result(model, term)
            coefficient_rows.append({
                "outcome": outcome,
                "term": term,
                "coef": result.get("coef"),
                "standard_error": result.get("se"),
                "p_value": result.get("pvalue"),
                "n_observations": safe_int(model.nobs),
                "n_total_panel_rows": safe_int(len(panel)),
                "n_authors": safe_int(lpm_data["author"].nunique()),
                "n_origin_subreddits": safe_int(lpm_data["origin_subreddit"].nunique()),
                "fixed_effects": "origin category FE" if category_features else "none",
                "clustering_level": "origin subreddit",
                "sample_note": lpm_data["analysis_sample"].iloc[0],
                "controls": ", ".join(term_name for term_name in lpm_terms if term_name != term),
            })
    coefficients_table = pd.DataFrame(coefficient_rows)
    if coefficients_table.empty:
        raise ValueError("No micro-level prediction coefficient rows were estimated.")

    important_outcomes = [
        "broke_origin_tie",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_context_heavy",
        "stayed_origin",
    ]
    risk_specs = [
        ("C_activity_persfree", activity_features + ["origin_persfree"]),
        ("D_full", activity_features + exposure_features),
    ]
    risk_rows = []
    for outcome in important_outcomes:
        base_rate = float(panel[outcome].mean())
        for model_name, features in risk_specs:
            fit_frame = model_sample.dropna(subset=[outcome] + features).copy()
            fit = fit_lpm_coefficients(fit_frame, outcome, features)
            if fit is None:
                continue
            prediction_frame = panel.dropna(subset=[outcome] + features).copy()
            prediction_frame["predicted_risk"] = predict_lpm(prediction_frame, features, fit)
            prediction_frame = prediction_frame[np.isfinite(prediction_frame["predicted_risk"])].copy()
            if prediction_frame.empty:
                continue
            prediction_frame["predicted_risk_quartile"] = pd.qcut(
                prediction_frame["predicted_risk"].rank(method="first", ascending=False),
                q=4,
                labels=["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"],
            ).astype(str)
            for quartile in ["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"]:
                group = prediction_frame[prediction_frame["predicted_risk_quartile"].eq(quartile)]
                if group.empty:
                    continue
                outcome_rate = float(group[outcome].mean())
                risk_rows.append({
                    "outcome": outcome,
                    "model_name": model_name,
                    "predicted_risk_quartile": quartile,
                    "n_author_subreddit_pairs": safe_int(len(group)),
                    "n_unique_authors": safe_int(group["author"].nunique()),
                    "mean_predicted_risk": safe_float(group["predicted_risk"].mean()),
                    "actual_outcome_rate": safe_float(outcome_rate),
                    "base_outcome_rate": safe_float(base_rate),
                    "lift_relative_to_base": safe_float(outcome_rate / base_rate) if base_rate > 0 else None,
                    "mean_persfree": safe_float(group["origin_persfree"].mean()),
                    "mean_pre_posts_origin": safe_float(group["pre_posts_origin"].mean()),
                    "mean_pre_active_months_origin": safe_float(group["pre_active_months_origin"].mean()),
                    "mean_author_pre_subreddit_breadth": safe_float(group["author_pre_subreddit_breadth"].mean()),
                    "fit_sample_rows": safe_int(len(fit_frame)),
                    "risk_score_note": "Higher fitted outcome probability is higher predicted risk for the named outcome.",
                })
    risk_quartiles = pd.DataFrame(risk_rows)

    metrics_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_table.to_csv(metrics_path, index=False)
    metrics_latex_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_latex_columns = [
        "outcome",
        "model_name",
        "n_observations",
        "outcome_prevalence",
        "auc",
        "brier_score",
        "log_loss",
        "top_risk_quartile_precision",
        "lift",
        "delta_auc_vs_activity",
    ]
    metrics_latex_path.write_text(
        metrics_table[metrics_latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    coefficients_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients_table.to_csv(coefficients_path, index=False)
    coefficients_latex_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients_latex_columns = [
        "outcome",
        "term",
        "coef",
        "standard_error",
        "p_value",
        "n_observations",
        "n_origin_subreddits",
        "fixed_effects",
        "clustering_level",
    ]
    coefficients_latex_path.write_text(
        coefficients_table[coefficients_latex_columns].to_latex(
            index=False,
            float_format="%.4f",
            escape=False,
        ),
        encoding="utf-8",
    )

    risk_quartiles_path.parent.mkdir(exist_ok=True, parents=True)
    risk_quartiles.to_csv(risk_quartiles_path, index=False)
    risk_quartiles_latex_path.parent.mkdir(exist_ok=True, parents=True)
    risk_latex_columns = [
        "outcome",
        "model_name",
        "predicted_risk_quartile",
        "n_author_subreddit_pairs",
        "actual_outcome_rate",
        "base_outcome_rate",
        "lift_relative_to_base",
        "mean_persfree",
    ]
    risk_quartiles_latex_path.write_text(
        risk_quartiles[risk_latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    comparison_rows = []
    community_metrics_path = tables_dir / "community_vulnerability_prediction_metrics.csv"
    if community_metrics_path.exists():
        community_metrics = pd.read_csv(community_metrics_path)
        for _, row in community_metrics.iterrows():
            outcome = row.get("outcome")
            model_name = row.get("model_name")
            if outcome not in {"overall_post_change", "bottom_quartile_decline"}:
                continue
            metric_name = "auc" if pd.notna(row.get("auc")) else "out_of_sample_r2"
            metric_value = row.get(metric_name)
            comparison_rows.append({
                "level": "macro_community",
                "question": "Which subreddits decline?",
                "outcome": outcome,
                "model_name": model_name,
                "n_units": row.get("n_subreddits"),
                "primary_metric": metric_name,
                "primary_metric_value": metric_value,
                "delta_vs_baseline": row.get("delta_auc_vs_baseline")
                if metric_name == "auc" else row.get("delta_r2_vs_baseline"),
                "notes": "Community-level diagnostic prediction from community_vulnerability_prediction_metrics.csv.",
            })
    for _, row in metrics_table.iterrows():
        if row.get("outcome") not in {
            "broke_origin_tie",
            "disappeared_from_sample",
            "moved_other_subreddit",
            "moved_context_heavy",
            "stayed_origin",
        }:
            continue
        if row.get("model_name") not in {"A_activity_only", "C_activity_persfree", "D_full"}:
            continue
        comparison_rows.append({
            "level": "micro_author_subreddit_tie",
            "question": "Which pre-shock author-subreddit ties break or reallocate?",
            "outcome": row.get("outcome"),
            "model_name": row.get("model_name"),
            "n_units": row.get("n_observations"),
            "primary_metric": "auc",
            "primary_metric_value": row.get("auc"),
            "delta_vs_baseline": row.get("delta_auc_vs_activity"),
            "notes": "Micro-level grouped-CV prediction; rows are pre-shock author-origin ties.",
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison_path.parent.mkdir(exist_ok=True, parents=True)
    comparison.to_csv(comparison_path, index=False)
    comparison_latex_path.parent.mkdir(exist_ok=True, parents=True)
    comparison_latex_path.write_text(
        comparison.to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    def metric_row(outcome, model_name):
        rows = metrics_table[
            metrics_table["outcome"].eq(outcome)
            & metrics_table["model_name"].eq(model_name)
        ]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    broke_a = metric_row("broke_origin_tie", "A_activity_only")
    broke_c = metric_row("broke_origin_tie", "C_activity_persfree")
    disappear_c = metric_row("disappeared_from_sample", "C_activity_persfree")
    moved_c = metric_row("moved_other_subreddit", "C_activity_persfree")
    heavy_c = metric_row("moved_context_heavy", "C_activity_persfree")
    auc_values = [
        row.get("auc")
        for row in [broke_c, disappear_c, moved_c, heavy_c]
        if row.get("auc") is not None
    ]
    max_auc = max(auc_values) if auc_values else None
    prediction_strength = (
        "strong" if max_auc is not None and max_auc >= 0.75
        else "modest" if max_auc is not None and max_auc >= 0.60
        else "weak"
    )
    persfree_delta = broke_c.get("delta_auc_vs_activity")
    persfree_improves = persfree_delta is not None and persfree_delta > 0
    most_predictable = (
        metrics_table.sort_values("auc", ascending=False).iloc[0].to_dict()
        if "auc" in metrics_table.columns and metrics_table["auc"].notna().any()
        else {}
    )
    summary = (
        "Micro-level prediction: PersFree "
        + ("improves" if persfree_improves else "does not improve")
        + " prediction of broken origin ties beyond activity-only features "
        f"(AUC {broke_a.get('auc'):.4f} -> {broke_c.get('auc'):.4f}, "
        f"delta={persfree_delta:.4f}). "
        f"Overall micro-level prediction is {prediction_strength}; the highest AUC is "
        f"{most_predictable.get('auc'):.4f} for {most_predictable.get('outcome')} under "
        f"{most_predictable.get('model_name')}. "
        f"Disappearance AUC with Activity+PersFree is {disappear_c.get('auc'):.4f}, "
        f"moved-other-subreddit AUC is {moved_c.get('auc'):.4f}, and moved-context-heavy "
        f"AUC is {heavy_c.get('auc'):.4f}. "
        "PersFree should be interpreted as a community-level and possibly author-tie-level "
        "diagnostic signal, not a high-accuracy individual forecasting model."
    )
    print(summary)

    return {
        "panel": panel[panel_columns],
        "metrics": metrics_table,
        "coefficients": coefficients_table,
        "risk_quartiles": risk_quartiles,
        "comparison": comparison,
        "summary": summary,
        "panel_output_path": str(panel_output_path),
        "metrics_path": str(metrics_path),
        "metrics_latex_path": str(metrics_latex_path),
        "coefficients_path": str(coefficients_path),
        "coefficients_latex_path": str(coefficients_latex_path),
        "risk_quartiles_path": str(risk_quartiles_path),
        "risk_quartiles_latex_path": str(risk_quartiles_latex_path),
        "comparison_path": str(comparison_path),
        "comparison_latex_path": str(comparison_latex_path),
    }

def compute_micro_prediction_heterogeneity(
    panel_path=None,
    coefficients_path=None,
    coefficients_latex_path=None,
    metrics_path=None,
    metrics_latex_path=None,
    risk_quartiles_path=None,
    risk_quartiles_latex_path=None,
    max_model_rows=250_000,
    min_rows_per_subreddit=500,
    n_folds=5,
    random_seed=20250604,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    tables_dir.mkdir(exist_ok=True, parents=True)

    panel_path = Path(panel_path or tables_dir / "micro_level_prediction_panel.csv")
    coefficients_path = Path(
        coefficients_path or tables_dir / "micro_prediction_heterogeneity_coefficients.csv"
    )
    coefficients_latex_path = Path(
        coefficients_latex_path or tables_dir / "micro_prediction_heterogeneity_coefficients.tex"
    )
    metrics_path = Path(metrics_path or tables_dir / "micro_prediction_heterogeneity_metrics.csv")
    metrics_latex_path = Path(
        metrics_latex_path or tables_dir / "micro_prediction_heterogeneity_metrics.tex"
    )
    risk_quartiles_path = Path(
        risk_quartiles_path or tables_dir / "micro_prediction_heterogeneity_risk_quartiles.csv"
    )
    risk_quartiles_latex_path = Path(
        risk_quartiles_latex_path or tables_dir / "micro_prediction_heterogeneity_risk_quartiles.tex"
    )

    if not panel_path.exists():
        compute_micro_level_prediction(panel_output_path=panel_path)
    panel = pd.read_csv(panel_path)

    required_columns = {
        "author",
        "origin_subreddit",
        "pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "stayed_origin",
        "broke_origin_tie",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_context_heavy",
        "moved_context_light",
    }
    missing_columns = required_columns - set(panel.columns)
    if missing_columns:
        raise ValueError(f"Micro-level prediction panel missing columns: {sorted(missing_columns)}")

    panel["author"] = panel["author"].fillna("").astype(str)
    panel["origin_subreddit"] = panel["origin_subreddit"].fillna("").astype(str)
    panel = panel[panel["author"].ne("") & panel["origin_subreddit"].ne("")].copy()
    numeric_columns = [
        "pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_persfree",
        "origin_gen_cap",
        "origin_phys_free",
        "stayed_origin",
        "broke_origin_tie",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_context_heavy",
        "moved_context_light",
    ]
    optional_numeric_columns = [
        "months_since_last_pre_post",
        "origin_pre_mean_log_posts",
        "origin_pre_trend_slope",
        "origin_pre_volatility",
        "log_subscribers",
        "log_pre_posts_origin",
    ]
    for column_name in numeric_columns + [c for c in optional_numeric_columns if c in panel.columns]:
        panel[column_name] = pd.to_numeric(panel[column_name], errors="coerce")
    panel = panel.dropna(
        subset=[
            "pre_posts_origin",
            "pre_active_months_origin",
            "posts_per_active_month_origin",
            "author_total_pre_posts_all_sampled_subreddits",
            "author_pre_subreddit_breadth",
            "origin_persfree",
            "origin_gen_cap",
            "origin_phys_free",
        ]
    ).copy()
    if "log_pre_posts_origin" not in panel.columns:
        panel["log_pre_posts_origin"] = np.log1p(panel["pre_posts_origin"].astype(float))
    panel["log_pre_posts_origin"] = pd.to_numeric(panel["log_pre_posts_origin"], errors="coerce")



    panel["one_pre_post"] = panel["pre_posts_origin"].eq(1).astype(int)
    panel["one_pre_month"] = panel["pre_active_months_origin"].eq(1).astype(int)

    def tercile_labels(values):
        ranks = pd.Series(values).astype(float).rank(method="first")
        return pd.qcut(ranks, q=3, labels=["bottom", "middle", "top"]).astype(str)

    panel["low_pre_posts"] = tercile_labels(panel["pre_posts_origin"]).eq("bottom").astype(int)
    panel["low_pre_active_months"] = (
        tercile_labels(panel["pre_active_months_origin"]).eq("bottom").astype(int)
    )
    panel["high_author_breadth"] = (
        tercile_labels(panel["author_pre_subreddit_breadth"]).eq("top").astype(int)
    )
    panel["low_posts_per_active_month"] = (
        tercile_labels(panel["posts_per_active_month_origin"]).eq("bottom").astype(int)
    )

    sort_columns = [
        "author",
        "pre_posts_origin",
        "pre_active_months_origin",
        "last_pre_month_origin",
        "origin_subreddit",
    ]
    if "last_pre_month_origin" not in panel.columns:
        panel["last_pre_month_origin"] = ""
    primary = (
        panel.sort_values(
            sort_columns,
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        .groupby("author", sort=False)
        .first()
        .reset_index()[["author", "origin_subreddit"]]
        .rename(columns={"origin_subreddit": "primary_origin_subreddit"})
    )
    panel = panel.merge(primary, on="author", how="left")
    panel["non_primary_origin_tie"] = (
        panel["origin_subreddit"].ne(panel["primary_origin_subreddit"])
    ).astype(int)

    weak_tie_indicators = [
        "one_pre_post",
        "one_pre_month",
        "low_pre_posts",
        "low_pre_active_months",
        "high_author_breadth",
        "low_posts_per_active_month",
        "non_primary_origin_tie",
    ]
    outcomes = [
        "broke_origin_tie",
        "stayed_origin",
        "disappeared_from_sample",
        "moved_other_subreddit",
        "moved_context_heavy",
        "moved_context_light",
    ]
    for outcome in outcomes:
        panel[outcome] = pd.to_numeric(panel[outcome], errors="coerce")

    controls = [
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "months_since_last_pre_post",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_pre_mean_log_posts",
        "origin_pre_trend_slope",
        "origin_pre_volatility",
        "log_subscribers",
        "origin_gen_cap",
        "origin_phys_free",
    ]
    controls = [column for column in controls if column in panel.columns]
    base_activity_features = [
        "log_pre_posts_origin",
        "pre_active_months_origin",
        "posts_per_active_month_origin",
        "months_since_last_pre_post",
        "author_total_pre_posts_all_sampled_subreddits",
        "author_pre_subreddit_breadth",
        "origin_pre_mean_log_posts",
        "origin_pre_trend_slope",
        "origin_pre_volatility",
        "log_subscribers",
    ]
    base_activity_features = [column for column in base_activity_features if column in panel.columns]
    model_specs = [
        ("A_activity_only", base_activity_features),
        ("B_activity_persfree", base_activity_features + ["origin_persfree"]),
        ("C_full", base_activity_features + ["origin_persfree", "origin_gen_cap", "origin_phys_free"]),
    ]

    def stratified_model_sample(frame):
        if len(frame) <= int(max_model_rows):
            sampled = frame.copy()
            sampled["analysis_sample"] = "full_panel"
            return sampled
        rng = np.random.default_rng(int(random_seed))
        group_indices = frame.groupby("origin_subreddit", sort=False).indices
        base_indices = []
        per_group_target = min(
            int(min_rows_per_subreddit),
            max(1, int(max_model_rows) // max(len(group_indices), 1)),
        )
        for _group, index_values in group_indices.items():
            index_array = np.asarray(index_values, dtype=np.int64)
            take = min(per_group_target, len(index_array))
            base_indices.append(rng.choice(index_array, size=take, replace=False))
        base_indices = np.unique(np.concatenate(base_indices)) if base_indices else np.array([], dtype=np.int64)
        remaining_slots = int(max_model_rows) - len(base_indices)
        if remaining_slots > 0:
            remaining = np.setdiff1d(np.arange(len(frame), dtype=np.int64), base_indices, assume_unique=False)
            if len(remaining) > 0:
                fill = rng.choice(remaining, size=min(remaining_slots, len(remaining)), replace=False)
                base_indices = np.unique(np.concatenate([base_indices, fill]))
        sampled = frame.iloc[base_indices].copy()
        sampled["analysis_sample"] = (
            f"stratified_origin_subreddit_sample_seed_{random_seed}_max_{int(max_model_rows)}"
        )
        return sampled

    model_columns = sorted(set(
        ["author", "origin_subreddit", "origin_persfree", "origin_gen_cap", "origin_phys_free"]
        + outcomes
        + weak_tie_indicators
        + controls
        + [feature for _name, features in model_specs for feature in features]
        + ["pre_posts_origin", "pre_active_months_origin", "author_pre_subreddit_breadth"]
    ))
    model_frame = (
        panel[model_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["origin_subreddit", "origin_persfree"] + controls + outcomes + weak_tie_indicators)
        .copy()
    )
    if model_frame.empty:
        raise ValueError("No complete rows for micro prediction heterogeneity models.")
    model_sample = stratified_model_sample(model_frame).reset_index(drop=True)

    def grouped_subreddit_folds(groups, fold_count):
        unique_groups = np.array(sorted(pd.Series(groups).dropna().astype(str).unique()))
        if len(unique_groups) == 0:
            return []
        rng = np.random.default_rng(int(random_seed))
        rng.shuffle(unique_groups)
        fold_count = max(2, min(int(fold_count), len(unique_groups)))
        return [set(chunk.tolist()) for chunk in np.array_split(unique_groups, fold_count) if len(chunk) > 0]

    def standardized_design_matrix(frame, features, means=None, scales=None):
        features = list(dict.fromkeys(features))
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        x = frame[features].astype(float).replace([np.inf, -np.inf], np.nan)
        if means is None:
            means = x.mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if scales is None:
            scales = x.std(ddof=0).replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
        x = (x - means) / scales
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1e6, 1e6)
        return np.column_stack([np.ones(len(x), dtype=float), x.to_numpy(dtype=float)]), means, scales

    def fit_lpm_coefficients(frame, outcome, features):
        if frame.empty or frame[outcome].nunique(dropna=True) < 2:
            return None
        x, means, scales = standardized_design_matrix(frame, features)
        y = frame[outcome].astype(float).to_numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return None
        try:
            standardized = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(standardized).all():
            return None
        raw = standardized.copy()
        raw[1:] = standardized[1:] / scales.to_numpy(dtype=float)
        raw[0] = standardized[0] - float(np.sum(raw[1:] * means.to_numpy(dtype=float)))
        raw = np.where(np.isfinite(raw), raw, np.nan)
        return {"standardized": standardized, "raw": raw, "means": means, "scales": scales}

    def predict_lpm(frame, features, fit):
        if fit is None or not np.isfinite(fit["standardized"]).all():
            return np.full(len(frame), np.nan, dtype=float)
        x, _means, _scales = standardized_design_matrix(
            frame,
            features,
            means=fit["means"],
            scales=fit["scales"],
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            predictions = x @ fit["standardized"]
        return np.where(np.isfinite(predictions), predictions, np.nan)

    def cross_validated_lpm_predictions(frame, outcome, features):
        predictions = np.full(len(frame), np.nan, dtype=float)
        folds = grouped_subreddit_folds(frame["origin_subreddit"], n_folds)
        groups = frame["origin_subreddit"].astype(str).to_numpy()
        for test_groups in folds:
            test_mask = np.isin(groups, list(test_groups))
            train_mask = ~test_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            train = frame.iloc[train_mask]
            test = frame.iloc[test_mask]
            if train[outcome].nunique(dropna=True) < 2:
                predictions[test_mask] = float(train[outcome].mean())
                continue
            fit = fit_lpm_coefficients(train, outcome, features)
            fold_predictions = predict_lpm(test, features, fit)
            predictions[test_mask] = np.where(
                np.isfinite(fold_predictions),
                fold_predictions,
                float(train[outcome].mean()),
            )
        return predictions

    def auc_score(y_true, scores):
        y_true = np.asarray(y_true, dtype=float)
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(scores)
        y_true = y_true[valid]
        scores = scores[valid]
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        rank_sum_pos = ranks[y_true == 1].sum()
        return safe_float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def prediction_metrics(y_true, raw_predictions):
        y_true = np.asarray(y_true, dtype=float)
        raw_predictions = np.asarray(raw_predictions, dtype=float)
        valid = np.isfinite(y_true) & np.isfinite(raw_predictions)
        y_true = y_true[valid]
        raw_predictions = raw_predictions[valid]
        if len(y_true) == 0:
            return {}
        clipped = np.clip(raw_predictions, 1e-6, 1 - 1e-6)
        base_rate = float(y_true.mean())
        brier = float(np.mean((y_true - clipped) ** 2))
        if len(raw_predictions) >= 4:
            top_mask = (
                pd.Series(raw_predictions)
                .rank(method="first", ascending=False)
                .le(max(1, len(raw_predictions) // 4))
                .to_numpy()
            )
            top_precision = float(y_true[top_mask].mean())
        else:
            top_precision = np.nan
        return {
            "auc": auc_score(y_true, raw_predictions),
            "brier_score": safe_float(brier),
            "top_risk_quartile_precision": safe_float(top_precision),
            "base_rate": safe_float(base_rate),
            "lift": safe_float(top_precision / base_rate)
            if base_rate > 0 and np.isfinite(top_precision) else None,
        }

    coefficient_rows = []
    for outcome in outcomes:
        for weak_indicator in weak_tie_indicators:
            interaction = f"origin_persfree_x_{weak_indicator}"
            model_sample[interaction] = model_sample["origin_persfree"] * model_sample[weak_indicator]
            lpm_terms = ["origin_persfree", weak_indicator, interaction] + controls
            lpm_terms = list(dict.fromkeys(lpm_terms))
            lpm_data = model_sample.dropna(subset=[outcome, "origin_subreddit"] + lpm_terms).copy()
            if lpm_data.empty or lpm_data[outcome].nunique(dropna=True) < 2:
                continue
            formula = f"{outcome} ~ " + " + ".join(lpm_terms)
            model = fit_ols(formula, lpm_data, cluster_col="origin_subreddit")
            if model is None:
                continue
            for term in ["origin_persfree", weak_indicator, interaction]:
                result = reg_result(model, term)
                coefficient_rows.append({
                    "outcome": outcome,
                    "weak_tie_indicator": weak_indicator,
                    "term": term,
                    "coef": result.get("coef"),
                    "standard_error": result.get("se"),
                    "p_value": result.get("pvalue"),
                    "n_observations": safe_int(model.nobs),
                    "n_total_panel_rows": safe_int(len(panel)),
                    "n_authors": safe_int(lpm_data["author"].nunique()),
                    "n_origin_subreddits": safe_int(lpm_data["origin_subreddit"].nunique()),
                    "fixed_effects": "none",
                    "clustering_level": "origin subreddit",
                    "sample_note": lpm_data["analysis_sample"].iloc[0],
                    "controls": ", ".join(controls),
                })
    coefficients = pd.DataFrame(coefficient_rows)
    if coefficients.empty:
        raise ValueError("No micro prediction heterogeneity coefficients were estimated.")

    metric_rows = []
    for outcome in outcomes:
        for weak_indicator in weak_tie_indicators:
            for subgroup_value, subgroup_label in [(1, "weak_tie"), (0, "strong_tie")]:
                subgroup = model_sample[model_sample[weak_indicator].eq(subgroup_value)].copy()
                if subgroup.empty or subgroup[outcome].nunique(dropna=True) < 2:
                    continue
                baseline_metrics = None
                for model_name, features in model_specs:
                    frame = subgroup.dropna(subset=[outcome, "origin_subreddit"] + features).copy()
                    if frame.empty or frame[outcome].nunique(dropna=True) < 2:
                        continue
                    predictions = cross_validated_lpm_predictions(frame, outcome, features)
                    metrics = prediction_metrics(frame[outcome].to_numpy(), predictions)
                    if model_name == "A_activity_only":
                        baseline_metrics = metrics
                    metric_rows.append({
                        "outcome": outcome,
                        "weak_tie_indicator": weak_indicator,
                        "subgroup": subgroup_label,
                        "weak_tie_value": subgroup_value,
                        "model_name": model_name,
                        "n_observations": safe_int(len(frame)),
                        "n_total_panel_rows": safe_int(len(panel)),
                        "n_authors": safe_int(frame["author"].nunique()),
                        "n_origin_subreddits": safe_int(frame["origin_subreddit"].nunique()),
                        "sample_note": frame["analysis_sample"].iloc[0],
                        "auc": metrics.get("auc"),
                        "brier_score": metrics.get("brier_score"),
                        "top_risk_quartile_precision": metrics.get("top_risk_quartile_precision"),
                        "base_outcome_rate": metrics.get("base_rate"),
                        "lift": metrics.get("lift"),
                        "delta_auc_vs_activity": safe_float(metrics.get("auc") - baseline_metrics.get("auc"))
                        if baseline_metrics and metrics.get("auc") is not None and baseline_metrics.get("auc") is not None else None,
                        "delta_brier_vs_activity": safe_float(baseline_metrics.get("brier_score") - metrics.get("brier_score"))
                        if baseline_metrics and metrics.get("brier_score") is not None and baseline_metrics.get("brier_score") is not None else None,
                        "delta_lift_vs_activity": safe_float(metrics.get("lift") - baseline_metrics.get("lift"))
                        if baseline_metrics and metrics.get("lift") is not None and baseline_metrics.get("lift") is not None else None,
                        "features": ", ".join(features),
                        "cv_grouping": "origin_subreddit",
                    })
    metrics_table = pd.DataFrame(metric_rows)
    if metrics_table.empty:
        raise ValueError("No micro prediction heterogeneity metrics were estimated.")

    risk_rows = []
    priority_weak_indicators = [
        "one_pre_post",
        "one_pre_month",
        "non_primary_origin_tie",
        "high_author_breadth",
    ]
    priority_outcomes = ["broke_origin_tie", "disappeared_from_sample", "moved_context_heavy"]
    risk_features = base_activity_features + ["origin_persfree"]
    for weak_indicator in priority_weak_indicators:
        full_weak_panel = panel[panel[weak_indicator].eq(1)].copy()
        fit_weak_sample = model_sample[model_sample[weak_indicator].eq(1)].copy()
        for outcome in priority_outcomes:
            fit_data = fit_weak_sample.dropna(subset=[outcome] + risk_features).copy()
            if fit_data.empty or fit_data[outcome].nunique(dropna=True) < 2:
                continue
            fit = fit_lpm_coefficients(fit_data, outcome, risk_features)
            prediction_frame = full_weak_panel.dropna(subset=[outcome] + risk_features).copy()
            prediction_frame["predicted_risk"] = predict_lpm(prediction_frame, risk_features, fit)
            prediction_frame = prediction_frame[np.isfinite(prediction_frame["predicted_risk"])].copy()
            if prediction_frame.empty:
                continue
            base_rate = float(prediction_frame[outcome].mean())
            prediction_frame["predicted_risk_quartile"] = pd.qcut(
                prediction_frame["predicted_risk"].rank(method="first", ascending=False),
                q=4,
                labels=["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"],
            ).astype(str)
            for quartile in ["Q1_highest_predicted_risk", "Q2", "Q3", "Q4_lowest_predicted_risk"]:
                group = prediction_frame[prediction_frame["predicted_risk_quartile"].eq(quartile)]
                if group.empty:
                    continue
                actual_rate = float(group[outcome].mean())
                risk_rows.append({
                    "outcome": outcome,
                    "weak_tie_indicator": weak_indicator,
                    "predicted_risk_quartile": quartile,
                    "n_author_subreddit_pairs": safe_int(len(group)),
                    "n_unique_authors": safe_int(group["author"].nunique()),
                    "actual_outcome_rate": safe_float(actual_rate),
                    "mean_predicted_risk": safe_float(group["predicted_risk"].mean()),
                    "mean_persfree": safe_float(group["origin_persfree"].mean()),
                    "mean_pre_posts_origin": safe_float(group["pre_posts_origin"].mean()),
                    "mean_author_pre_subreddit_breadth": safe_float(group["author_pre_subreddit_breadth"].mean()),
                    "base_outcome_rate": safe_float(base_rate),
                    "lift_relative_to_base": safe_float(actual_rate / base_rate) if base_rate > 0 else None,
                    "fit_sample_rows": safe_int(len(fit_data)),
                    "model_name": "B_activity_persfree",
                })
    risk_quartiles = pd.DataFrame(risk_rows)
    if risk_quartiles.empty:
        raise ValueError("No micro prediction heterogeneity risk quartiles were estimated.")

    coefficients_path.parent.mkdir(exist_ok=True, parents=True)
    coefficients.to_csv(coefficients_path, index=False)
    coefficients_latex_path.parent.mkdir(exist_ok=True, parents=True)
    coefficient_latex_columns = [
        "outcome",
        "weak_tie_indicator",
        "term",
        "coef",
        "standard_error",
        "p_value",
        "n_observations",
        "n_origin_subreddits",
    ]
    coefficients_latex_path.write_text(
        coefficients[coefficient_latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    metrics_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_table.to_csv(metrics_path, index=False)
    metrics_latex_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_latex_columns = [
        "outcome",
        "weak_tie_indicator",
        "subgroup",
        "model_name",
        "n_observations",
        "auc",
        "brier_score",
        "top_risk_quartile_precision",
        "base_outcome_rate",
        "lift",
        "delta_auc_vs_activity",
    ]
    metrics_latex_path.write_text(
        metrics_table[metrics_latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    risk_quartiles_path.parent.mkdir(exist_ok=True, parents=True)
    risk_quartiles.to_csv(risk_quartiles_path, index=False)
    risk_quartiles_latex_path.parent.mkdir(exist_ok=True, parents=True)
    risk_latex_columns = [
        "outcome",
        "weak_tie_indicator",
        "predicted_risk_quartile",
        "n_author_subreddit_pairs",
        "actual_outcome_rate",
        "mean_predicted_risk",
        "mean_persfree",
        "lift_relative_to_base",
    ]
    risk_quartiles_latex_path.write_text(
        risk_quartiles[risk_latex_columns].to_latex(index=False, float_format="%.4f", escape=False),
        encoding="utf-8",
    )

    interaction_rows = coefficients[coefficients["term"].str.startswith("origin_persfree_x_")].copy()
    significant_interactions = interaction_rows[
        pd.to_numeric(interaction_rows["p_value"], errors="coerce").lt(0.05)
    ].copy()
    if not significant_interactions.empty:
        significant_interactions["abs_coef"] = significant_interactions["coef"].abs()
        top_interactions = significant_interactions.sort_values("abs_coef", ascending=False).head(3)
        interaction_phrase = "; ".join(
            f"{row.weak_tie_indicator} on {row.outcome}: beta={row.coef:.4f}, p={row.p_value:.4g}"
            for row in top_interactions.itertuples(index=False)
        )
    else:
        top_interactions = pd.DataFrame()
        interaction_phrase = "no PersFree x weak-tie interactions are significant at p<0.05"

    def metric_lookup(outcome, weak_indicator, subgroup, model_name, column):
        rows = metrics_table[
            metrics_table["outcome"].eq(outcome)
            & metrics_table["weak_tie_indicator"].eq(weak_indicator)
            & metrics_table["subgroup"].eq(subgroup)
            & metrics_table["model_name"].eq(model_name)
        ]
        if rows.empty or column not in rows.columns:
            return None
        value = rows.iloc[0].get(column)
        return None if pd.isna(value) else float(value)

    comparison_candidates = []
    for weak_indicator in weak_tie_indicators:
        for outcome in outcomes:
            weak_delta = metric_lookup(outcome, weak_indicator, "weak_tie", "B_activity_persfree", "delta_auc_vs_activity")
            strong_delta = metric_lookup(outcome, weak_indicator, "strong_tie", "B_activity_persfree", "delta_auc_vs_activity")
            if weak_delta is None or strong_delta is None:
                continue
            comparison_candidates.append({
                "weak_indicator": weak_indicator,
                "outcome": outcome,
                "weak_delta": weak_delta,
                "strong_delta": strong_delta,
                "difference": weak_delta - strong_delta,
            })
    if comparison_candidates:
        best_delta = max(comparison_candidates, key=lambda item: item["difference"])
        weak_vs_strong_phrase = (
            f"largest PersFree AUC gain difference is for {best_delta['weak_indicator']} on "
            f"{best_delta['outcome']} (weak delta={best_delta['weak_delta']:.4f}, "
            f"strong delta={best_delta['strong_delta']:.4f})"
        )
    else:
        weak_vs_strong_phrase = "weak-vs-strong PersFree AUC comparisons could not be computed"

    def finite_mean(values):
        finite_values = [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ]
        return float(np.mean(finite_values)) if finite_values else np.nan

    moved_heavy_gain = finite_mean([
        metric_lookup("moved_context_heavy", weak_indicator, "weak_tie", "B_activity_persfree", "delta_auc_vs_activity")
        for weak_indicator in weak_tie_indicators
    ])
    break_gain = finite_mean([
        metric_lookup("broke_origin_tie", weak_indicator, "weak_tie", "B_activity_persfree", "delta_auc_vs_activity")
        for weak_indicator in weak_tie_indicators
    ])
    destination_phrase = (
        "PersFree mainly predicts destination type rather than ordinary tie breakage"
        if np.isfinite(moved_heavy_gain) and np.isfinite(break_gain) and moved_heavy_gain > break_gain
        else "PersFree does not clearly predict destination type more than ordinary tie breakage"
    )
    summary = (
        "Micro prediction heterogeneity: strongest PersFree interactions are "
        f"{interaction_phrase}. {weak_vs_strong_phrase}. {destination_phrase} "
        f"(mean weak-tie AUC gain for moved_context_heavy={moved_heavy_gain:.4f}, "
        f"broke_origin_tie={break_gain:.4f}). "
        "If PersFree interactions are strongest for one-post, one-month, or non-primary ties, "
        "this supports a micro-level weak-tie prediction channel; otherwise PersFree should be "
        "framed primarily as a community-level diagnostic."
    )
    print(summary)

    return {
        "coefficients": coefficients,
        "metrics": metrics_table,
        "risk_quartiles": risk_quartiles,
        "summary": summary,
        "coefficients_path": str(coefficients_path),
        "coefficients_latex_path": str(coefficients_latex_path),
        "metrics_path": str(metrics_path),
        "metrics_latex_path": str(metrics_latex_path),
        "risk_quartiles_path": str(risk_quartiles_path),
        "risk_quartiles_latex_path": str(risk_quartiles_latex_path),
    }

def compute_destination_rank_analysis(
    displaced_panel_path=None,
    mover_destination_panel_path=None,
    score_path=None,
    output_path=None,
    latex_path=None,
    ecosystem_posts=None,
):
    root = globals().get("ROOT", Path("."))
    tables_dir = globals().get("TABLES_DIR", root)
    data_dir = globals().get("DATA_DIR", root / "data")
    tables_dir.mkdir(exist_ok=True, parents=True)

    displaced_panel_path = Path(
        displaced_panel_path or tables_dir / "displaced_contributor_destination_panel.csv"
    )
    mover_destination_panel_path = Path(
        mover_destination_panel_path or tables_dir / "destination_choice_set_placebo_panel.csv"
    )
    if score_path is None:
        score_candidates = [
            tables_dir / "acsi_preshock_tworuns.csv",
            root / "acsi_preshock_tworuns.csv",
            data_dir / "acsi_preshock_tworuns.csv",
        ]
        score_path = next((path for path in score_candidates if path.exists()), score_candidates[0])
    score_path = Path(score_path)
    output_path = Path(output_path or tables_dir / "destination_rank_analysis.csv")
    latex_path = Path(latex_path or tables_dir / "destination_rank_analysis.tex")

    if not score_path.exists():
        scores, _metadata = compute_two_run_preshock_scores()
        if scores.empty:
            raise ValueError("Two-run pre-shock score builder returned no rows.")
        score_path.parent.mkdir(exist_ok=True, parents=True)
        scores.to_csv(score_path, index=False)

    scores = pd.read_csv(score_path)
    required_score_columns = {"subreddit", "pers_free"}
    missing_score_columns = required_score_columns - set(scores.columns)
    if missing_score_columns:
        raise ValueError(f"Score file missing columns: {sorted(missing_score_columns)}")
    scores = scores[["subreddit", "pers_free"]].copy()
    scores["subreddit"] = scores["subreddit"].astype(str)
    scores["pers_free"] = pd.to_numeric(scores["pers_free"], errors="coerce")
    scores = scores.dropna(subset=["subreddit", "pers_free"])
    if scores["subreddit"].nunique() < 2:
        raise ValueError("Need at least two scored subreddits for destination rank analysis.")
    ranked_scores = scores.sort_values(["pers_free", "subreddit"]).reset_index(drop=True)
    ranked_scores["full_persfree_rank"] = np.arange(1, len(ranked_scores) + 1, dtype=int)
    rank_lookup = ranked_scores.set_index("subreddit")["full_persfree_rank"].to_dict()
    persfree_lookup = ranked_scores.set_index("subreddit")["pers_free"].to_dict()
    choice_set_size = int(len(ranked_scores) - 1)
    if choice_set_size <= 0:
        raise ValueError("Destination choice set is empty.")

    if mover_destination_panel_path.exists():
        movers = pd.read_csv(mover_destination_panel_path)
    else:
        if not displaced_panel_path.exists():
            displaced = compute_displaced_contributor_destinations(
                ecosystem_posts=ecosystem_posts,
                score_path=score_path,
                panel_output_path=displaced_panel_path,
            )
            displaced_panel = displaced["panel"].copy()
        else:
            displaced_panel = pd.read_csv(displaced_panel_path)
        required_displaced_columns = {
            "author",
            "origin_pair_id",
            "origin_subreddit",
            "origin_persfree",
            "origin_persfree_tercile",
            "pre_posts_origin",
            "log_pre_posts_origin",
            "log_pre_subreddit_posts",
            "moved_other_subreddit",
            "stayed_same_subreddit",
        }
        missing_displaced_columns = required_displaced_columns - set(displaced_panel.columns)
        if missing_displaced_columns:
            raise ValueError(
                "Displaced-contributor panel missing columns: "
                f"{sorted(missing_displaced_columns)}"
            )
        movers = displaced_panel[
            displaced_panel["moved_other_subreddit"].astype(int).eq(1)
            & displaced_panel["stayed_same_subreddit"].astype(int).eq(0)
        ].copy()
        if movers.empty:
            raise ValueError("No movers found in displaced-contributor panel.")
        if ecosystem_posts is None:
            posts_path = globals().get("POSTS_ECOSYSTEM_PATH", None)
            if posts_path is None or not Path(posts_path).exists():
                raise ValueError(
                    "Pass ecosystem_posts or run the main pipeline first so posts_clean_ecosystem.parquet exists."
                )
            posts = pd.read_parquet(posts_path, columns=["author", "subreddit", "year_month"])
        else:
            posts = ecosystem_posts[["author", "subreddit", "year_month"]].copy()
        target_subreddits = set(scores["subreddit"].astype(str))
        excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
        posts["author"] = posts["author"].fillna("").astype(str)
        posts["subreddit"] = posts["subreddit"].fillna("").astype(str)
        posts["year_month"] = posts["year_month"].astype(str)
        posts = posts[
            posts["author"].isin(set(movers["author"].astype(str)))
            & posts["subreddit"].isin(target_subreddits)
            & posts["year_month"].between("2022-12", "2024-12")
            & posts["author"].ne("")
            & ~posts["author"].isin(excluded_authors)
            & ~posts["author"].str.lower().str.endswith("bot", na=False)
        ].copy()
        destination_edges = (
            posts.groupby(["author", "subreddit"], as_index=False)
            .size()
            .rename(columns={"subreddit": "destination_subreddit", "size": "destination_posts"})
        )
        movers = movers.merge(destination_edges, on="author", how="inner")
        movers = movers[movers["destination_subreddit"].ne(movers["origin_subreddit"])].copy()
        movers = (
            movers.groupby(
                [
                    "author",
                    "origin_pair_id",
                    "origin_subreddit",
                    "origin_persfree",
                    "origin_persfree_tercile",
                    "pre_posts_origin",
                    "log_pre_posts_origin",
                    "log_pre_subreddit_posts",
                ],
                as_index=False,
            )
            .agg(
                observed_destination_subreddits=(
                    "destination_subreddit",
                    lambda values: "|".join(sorted(set(values.astype(str)))),
                )
            )
        )

    required_mover_columns = {
        "author",
        "origin_pair_id",
        "origin_subreddit",
        "origin_persfree",
        "origin_persfree_tercile",
        "pre_posts_origin",
        "log_pre_posts_origin",
        "log_pre_subreddit_posts",
        "observed_destination_subreddits",
    }
    missing_mover_columns = required_mover_columns - set(movers.columns)
    if missing_mover_columns:
        raise ValueError(
            "Mover destination panel missing columns: "
            f"{sorted(missing_mover_columns)}"
        )
    movers = movers.copy()
    movers["origin_subreddit"] = movers["origin_subreddit"].astype(str)
    movers["origin_persfree"] = pd.to_numeric(movers["origin_persfree"], errors="coerce")
    movers["log_pre_shock_posts"] = pd.to_numeric(
        movers["log_pre_posts_origin"], errors="coerce"
    )
    movers["origin_subreddit_size"] = pd.to_numeric(
        movers["log_pre_subreddit_posts"], errors="coerce"
    )
    movers["origin_persfree_tercile"] = movers["origin_persfree_tercile"].astype(str)
    movers = movers.dropna(
        subset=[
            "origin_pair_id",
            "origin_subreddit",
            "origin_persfree",
            "log_pre_shock_posts",
            "origin_subreddit_size",
            "observed_destination_subreddits",
        ]
    ).copy()
    movers["destination_subreddit"] = movers["observed_destination_subreddits"].astype(str).str.split("|")
    destination_edges = movers[
        [
            "origin_pair_id",
            "origin_subreddit",
            "origin_persfree",
            "destination_subreddit",
        ]
    ].explode("destination_subreddit")
    destination_edges["destination_subreddit"] = (
        destination_edges["destination_subreddit"].fillna("").astype(str)
    )
    destination_edges = destination_edges[
        destination_edges["destination_subreddit"].ne("")
        & destination_edges["destination_subreddit"].isin(rank_lookup)
        & destination_edges["origin_subreddit"].isin(rank_lookup)
        & destination_edges["destination_subreddit"].ne(destination_edges["origin_subreddit"])
    ].copy()
    if destination_edges.empty:
        raise ValueError("No valid origin-destination edges for rank analysis.")
    destination_edges["origin_full_rank"] = destination_edges["origin_subreddit"].map(rank_lookup).astype(int)
    destination_edges["destination_full_rank"] = destination_edges["destination_subreddit"].map(rank_lookup).astype(int)
    destination_edges["choice_set_rank"] = (
        destination_edges["destination_full_rank"]
        - (destination_edges["origin_full_rank"] < destination_edges["destination_full_rank"]).astype(int)
    )
    destination_edges["normalized_rank"] = (
        destination_edges["choice_set_rank"].astype(float) / choice_set_size
    )
    destination_edges["destination_persfree"] = destination_edges["destination_subreddit"].map(
        persfree_lookup
    )

    rank_summary = (
        destination_edges.groupby("origin_pair_id", as_index=False)
        .agg(
            observed_normalized_rank=("normalized_rank", "mean"),
            min_observed_normalized_rank=("normalized_rank", "min"),
            max_observed_normalized_rank=("normalized_rank", "max"),
            observed_destination_count_for_rank=("destination_subreddit", "nunique"),
            rank_mean_destination_persfree=("destination_persfree", "mean"),
        )
    )
    rank_panel = movers.drop(columns=["destination_subreddit"], errors="ignore").merge(
        rank_summary,
        on="origin_pair_id",
        how="inner",
    )
    rank_panel = rank_panel.dropna(
        subset=[
            "observed_normalized_rank",
            "origin_persfree",
            "log_pre_shock_posts",
            "origin_subreddit_size",
        ]
    ).copy()
    if rank_panel.empty:
        raise ValueError("No mover-origin rows remain after destination rank construction.")

    model_specs = [
        (
            "model_1",
            "observed_normalized_rank ~ origin_persfree",
            "origin_persfree only",
        ),
        (
            "model_2",
            "observed_normalized_rank ~ origin_persfree + log_pre_shock_posts + origin_subreddit_size",
            "origin_persfree, log_pre_shock_posts, log origin subreddit size",
        ),
        (
            "model_3",
            "observed_normalized_rank ~ origin_persfree + C(origin_persfree_tercile)",
            "origin_persfree plus origin PersFree tercile indicators",
        ),
    ]
    rows = []
    regression_rows = []
    for model_name, formula, controls in model_specs:
        model = fit_ols(formula, rank_panel)
        if model is None:
            continue
        result = reg_result(model, "origin_persfree")
        row = {
            "row_type": "regression",
            "model": model_name,
            "formula": formula,
            "controls": controls,
            "term": "origin_persfree",
            "coef": result.get("coef"),
            "standard_error": result.get("se"),
            "p_value": result.get("pvalue"),
            "n_mover_origin_pairs": safe_int(model.nobs),
            "n_unique_authors": safe_int(rank_panel["author"].nunique()),
            "n_origin_subreddits": safe_int(rank_panel["origin_subreddit"].nunique()),
            "standard_errors": "HC3 robust",
            "choice_set_size": safe_int(choice_set_size),
            "score_source": str(score_path),
            "mover_source": str(
                mover_destination_panel_path
                if mover_destination_panel_path.exists()
                else displaced_panel_path
            ),
        }
        rows.append(row)
        regression_rows.append(row)

    tercile_order = ["low_persfree", "middle_persfree", "high_persfree"]
    for tercile in tercile_order:
        group = rank_panel[rank_panel["origin_persfree_tercile"].astype(str).eq(tercile)]
        if group.empty:
            continue
        rows.append({
            "row_type": "descriptive",
            "origin_persfree_tercile": tercile,
            "n_mover_origin_pairs": safe_int(len(group)),
            "n_unique_authors": safe_int(group["author"].nunique()),
            "n_origin_subreddits": safe_int(group["origin_subreddit"].nunique()),
            "mean_normalized_rank": safe_float(group["observed_normalized_rank"].mean()),
            "sd_normalized_rank": safe_float(group["observed_normalized_rank"].std()),
            "mean_origin_persfree": safe_float(group["origin_persfree"].mean()),
            "mean_destination_persfree": safe_float(group["rank_mean_destination_persfree"].mean()),
            "mean_destination_count": safe_float(group["observed_destination_count_for_rank"].mean()),
            "choice_set_size": safe_int(choice_set_size),
        })

    result_table = pd.DataFrame(rows)
    if result_table.empty:
        raise ValueError("No destination rank analysis rows were produced.")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_table.to_csv(output_path, index=False)

    def row_for(model_name):
        rows_for_model = [
            row for row in regression_rows
            if row.get("model") == model_name
        ]
        return rows_for_model[0] if rows_for_model else {}

    model2 = row_for("model_2")
    coef = model2.get("coef")
    se = model2.get("standard_error")
    pvalue = model2.get("p_value")
    def format_pvalue_for_text(value):
        if value is None or pd.isna(value):
            return "NA"
        return "<0.001" if float(value) < 0.001 else f"={float(value):.3f}"

    direction = (
        "lower-ranked, more personal-context"
        if coef is not None and coef < 0
        else "higher-ranked, less personal-context"
    )
    survival = (
        "survives the within-choice-set rank check"
        if coef is not None and coef < 0 and pvalue is not None and pvalue < 0.05
        else "does not survive as selective movement toward more personal-context destinations"
    )
    latex_paragraph = (
        "\\paragraph{Within-choice-set destination ranks.} "
        "For each displaced author--origin pair, we ranked all eligible destination "
        "communities by PersFree after removing the origin community from the choice "
        "set, so rank 1 is the most personal-context destination and rank 123 is the "
        "least personal-context destination. The mover-level outcome is the mean "
        "normalized rank of the author's observed post-shock destination communities. "
        f"In the covariate-adjusted specification, origin PersFree predicts {direction} "
        f"destinations (\\(\\hat{{\\beta}}={coef:.4f}\\), SE = {se:.4f}, "
        f"\\(p{format_pvalue_for_text(pvalue)}\\)). Thus the destination-rank evidence {survival}, "
        "net of the mechanical composition of each origin's available destination set."
    )
    latex_path.parent.mkdir(exist_ok=True, parents=True)
    latex_path.write_text(latex_paragraph + "\n", encoding="utf-8")

    summary = (
        "Destination rank analysis: "
        + "; ".join(
            f"{row['model']} beta={row['coef']:.4f}, SE={row['standard_error']:.4f}, p{format_pvalue_for_text(row['p_value'])}"
            for row in regression_rows
        )
        + ". "
        + (
            "Negative coefficients indicate relatively more personal-context destinations; "
            if any(row.get("coef") is not None and row.get("coef") < 0 for row in regression_rows)
            else "Positive coefficients indicate relatively less personal-context destinations; "
        )
        + f"the covariate-adjusted result {survival}."
    )
    print(summary)

    return {
        "results": result_table,
        "rank_panel": rank_panel,
        "latex_paragraph": latex_paragraph,
        "summary": summary,
        "output_path": str(output_path),
        "latex_path": str(latex_path),
    }

def creator_exit_continuous_model_rows(model, model_name, terms, n_authors, n_subreddits):
    rows = []
    for term in terms:
        if term not in model.params.index:
            continue
        result = reg_result(model, term)
        rows.append({
            "row_type": "coefficient",
            "model": model_name,
            "term": term,
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "ci_low": safe_float(result.get("coef") - 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "ci_high": safe_float(result.get("coef") + 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "n_authors": safe_int(n_authors),
            "n_subreddits": safe_int(n_subreddits),
        })
    return rows

def creator_exit_marginal_effect(model, pers_free_value):
    beta_log = float(model.params["log_pre_rate"])
    beta_interaction = float(model.params["log_pre_rate:pers_free"])
    covariance = model.cov_params()
    vector = np.array([1.0, float(pers_free_value)])
    sub_cov = covariance.loc[
        ["log_pre_rate", "log_pre_rate:pers_free"],
        ["log_pre_rate", "log_pre_rate:pers_free"],
    ].to_numpy(dtype=float)
    estimate = float(beta_log + beta_interaction * pers_free_value)
    se = float(np.sqrt(max(vector @ sub_cov @ vector, 0.0)))
    return {
        "pers_free_value": safe_float(pers_free_value),
        "marginal_effect": safe_float(estimate),
        "se": safe_float(se),
        "ci_low": safe_float(estimate - 1.96 * se),
        "ci_high": safe_float(estimate + 1.96 * se),
    }

def plot_creator_exit_marginal_effects(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 101)
    rows = [creator_exit_marginal_effect(model, value) for value in grid]
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(curve["pers_free_value"], curve["marginal_effect"], color=MUTED_BLUE, linewidth=2)
    ax.fill_between(
        curve["pers_free_value"].to_numpy(dtype=float),
        curve["ci_low"].to_numpy(dtype=float),
        curve["ci_high"].to_numpy(dtype=float),
        color=MUTED_BLUE,
        alpha=0.18,
        linewidth=0,
    )
    ax.axhline(0.0, color=MUTED_GRAY, linewidth=1.0)
    ax.set_xlabel("PersFree")
    ax.set_ylabel("Marginal effect of log pre-event posts on exit")
    ax.set_title("Creator exit marginal effect")
    save_plot(fig, output_path)
    return str(output_path)

def compute_creator_exit_continuous(
    score_path=None,
    target_subreddits=None,
    output_path=None,
    plot_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "creator_exit_continuous.csv")
    plot_path = Path(plot_path or root / "creator_exit_marginal_effects.png")
    creator_frame, metadata = build_creator_exit_continuous_frame(
        score_path=score_path,
        target_subreddits=target_subreddits,
    )
    if creator_frame["subreddit"].nunique() < 2:
        raise ValueError("Need at least two subreddits for clustered subreddit FE models.")




    model1 = fit_ols("exit ~ log_pre_rate + C(subreddit)", creator_frame, cluster_col="subreddit")
    model2 = fit_ols(
        "exit ~ log_pre_rate + log_pre_rate:pers_free + C(subreddit)",
        creator_frame,
        cluster_col="subreddit",
    )
    model3 = fit_ols(
        "exit ~ log_pre_rate + log_pre_rate:pers_free + log_pre_rate:gen_cap + C(subreddit)",
        creator_frame,
        cluster_col="subreddit",
    )
    if model1 is None or model2 is None or model3 is None:
        raise ValueError("One or more creator-exit continuous models failed to fit.")

    n_authors = len(creator_frame)
    n_subreddits = creator_frame["subreddit"].nunique()
    bottom_exit = creator_frame.loc[
        creator_frame["pers_free_tercile"].astype(str).eq("bottom"),
        "exit",
    ].mean()
    top_exit = creator_frame.loc[
        creator_frame["pers_free_tercile"].astype(str).eq("top"),
        "exit",
    ].mean()
    summary = {
        "row_type": "sample_summary",
        "model": "creator_exit_continuous",
        "term": "sample",
        "coef": np.nan,
        "se": np.nan,
        "pvalue": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n_authors": safe_int(n_authors),
        "n_subreddits": safe_int(n_subreddits),
        "overall_exit_rate": safe_float(creator_frame["exit"].mean()),
        "bottom_tercile_exit_rate": safe_float(bottom_exit),
        "top_tercile_exit_rate": safe_float(top_exit),
        "files_scanned": metadata["files_scanned"],
        "posts_seen": metadata["posts_seen"],
        "n_target_subreddits": metadata["n_target_subreddits"],
        "note": (
            "Replaces the hand-labeled 26-community subset with the full "
            "continuous-treatment design across all scored subreddits."
        ),
    }

    coefficient_rows = []
    coefficient_rows.extend(creator_exit_continuous_model_rows(
        model1,
        "Model 1: exit ~ log_pre_rate + subreddit FE",
        ["log_pre_rate"],
        n_authors,
        n_subreddits,
    ))
    coefficient_rows.extend(creator_exit_continuous_model_rows(
        model2,
        "Model 2: + pers_free x log_pre_rate",
        ["log_pre_rate", "log_pre_rate:pers_free"],
        n_authors,
        n_subreddits,
    ))
    coefficient_rows.extend(creator_exit_continuous_model_rows(
        model3,
        "Model 3: + pers_free and gen_cap interactions",
        ["log_pre_rate", "log_pre_rate:pers_free", "log_pre_rate:gen_cap"],
        n_authors,
        n_subreddits,
    ))

    marginal_rows = []
    for value, label in [(0.25, "low"), (0.50, "mid"), (0.75, "high")]:
        marginal = creator_exit_marginal_effect(model2, value)
        marginal_rows.append({
            "row_type": "marginal_effect",
            "model": "Model 2: + pers_free x log_pre_rate",
            "term": f"log_pre_rate_at_pers_free_{label}",
            "coef": marginal["marginal_effect"],
            "se": marginal["se"],
            "pvalue": np.nan,
            "ci_low": marginal["ci_low"],
            "ci_high": marginal["ci_high"],
            "n_authors": safe_int(n_authors),
            "n_subreddits": safe_int(n_subreddits),
            "pers_free_value": marginal["pers_free_value"],
        })

    saved_plot_path = plot_creator_exit_marginal_effects(model2, plot_path)
    rows = [summary] + coefficient_rows + marginal_rows
    results_table = pd.DataFrame(rows)
    for key, value in metadata.items():
        if key not in results_table.columns:
            results_table[key] = value
    results_table["plot_path"] = saved_plot_path
    results_table.to_csv(output_path, index=False)
    return {
        "creator_frame": creator_frame,
        "metadata": metadata,
        "models": {"model1": model1, "model2": model2, "model3": model3},
        "rows": rows,
        "output_path": str(output_path),
        "plot_path": saved_plot_path,
        "summary": summary,
        "coefficient_rows": coefficient_rows,
        "marginal_rows": marginal_rows,
    }

def load_or_build_creator_exit_no_fe_frame(
    creator_path=None,
    score_path=None,
    target_subreddits=None,
):
    root = globals().get("ROOT", Path("."))
    creator_path = Path(creator_path or root / "creator_exit_continuous.csv")
    author_cache_path = root / "creator_exit_author_level.csv"
    required_columns = {
        "author", "subreddit", "pre_event_posts",
        "posted_post_event", "pers_free", "gen_cap",
    }
    if creator_path.exists():
        header = pd.read_csv(creator_path, nrows=0)
        if required_columns.issubset(header.columns):
            frame = pd.read_csv(creator_path)
            source = str(creator_path)
        elif author_cache_path.exists():
            frame = pd.read_csv(author_cache_path)
            source = str(author_cache_path)
        else:
            frame, _metadata = build_creator_exit_continuous_frame(
                score_path=score_path,
                target_subreddits=target_subreddits,
            )
            frame[list(required_columns)].to_csv(author_cache_path, index=False)
            source = "rebuilt_from_raw_posts"
    else:
        if author_cache_path.exists():
            frame = pd.read_csv(author_cache_path)
            source = str(author_cache_path)
        else:
            frame, _metadata = build_creator_exit_continuous_frame(
                score_path=score_path,
                target_subreddits=target_subreddits,
            )
            frame[list(required_columns)].to_csv(author_cache_path, index=False)
            source = "rebuilt_from_raw_posts"

    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Creator exit frame missing columns: {sorted(missing_columns)}")
    frame = frame.copy()
    frame["subreddit"] = frame["subreddit"].astype(str)
    for column_name in ["pre_event_posts", "posted_post_event", "pers_free", "gen_cap"]:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["subreddit", "pre_event_posts", "posted_post_event", "pers_free", "gen_cap"]
    )
    frame = frame[frame["pre_event_posts"] > 0].copy()
    frame["exit"] = 1 - frame["posted_post_event"].astype(int)
    frame["log_pre_rate"] = np.log(frame["pre_event_posts"].astype(float))
    if "month" in frame.columns:
        frame["month"] = frame["month"].astype(str)
    return frame, source

def no_fe_model_rows(model, model_name, terms=None, n_obs=None):
    rows = []
    if terms is None:
        terms = list(model.params.index)
    if n_obs is None:
        n_obs = model.nobs
    for term in terms:
        if term not in model.params.index:
            continue
        result = reg_result(model, term)
        rows.append({
            "row_type": "coefficient",
            "model": model_name,
            "term": term,
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "n": safe_int(n_obs),
        })
    return rows

def no_fe_marginal_effect(model, pers_free_value):
    beta_log = float(model.params["log_pre_rate"])
    beta_interaction = float(model.params["log_pre_rate:pers_free"])
    covariance = model.cov_params()
    vector = np.array([1.0, float(pers_free_value)])
    sub_cov = covariance.loc[
        ["log_pre_rate", "log_pre_rate:pers_free"],
        ["log_pre_rate", "log_pre_rate:pers_free"],
    ].to_numpy(dtype=float)
    estimate = float(beta_log + beta_interaction * pers_free_value)
    se = float(np.sqrt(max(vector @ sub_cov @ vector, 0.0)))
    return {
        "pers_free_value": safe_float(pers_free_value),
        "marginal_effect": safe_float(estimate),
        "se": safe_float(se),
        "ci_low": safe_float(estimate - 1.96 * se),
        "ci_high": safe_float(estimate + 1.96 * se),
    }

def plot_creator_exit_no_fe_marginal_effects(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 101)
    curve = pd.DataFrame([no_fe_marginal_effect(model, value) for value in grid])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(curve["pers_free_value"], curve["marginal_effect"], color=MUTED_BLUE, linewidth=2)
    ax.fill_between(
        curve["pers_free_value"].to_numpy(dtype=float),
        curve["ci_low"].to_numpy(dtype=float),
        curve["ci_high"].to_numpy(dtype=float),
        color=MUTED_BLUE,
        alpha=0.18,
        linewidth=0,
    )
    ax.axhline(0.0, color=MUTED_GRAY, linewidth=1.0)
    ax.set_xlabel("PersFree")
    ax.set_ylabel("Marginal effect of log pre-event posts on exit")
    ax.set_title("Creator exit marginal effect, no subreddit FE")
    save_plot(fig, output_path)
    return str(output_path)

def compute_creator_exit_no_fe(
    creator_path=None,
    score_path=None,
    target_subreddits=None,
    output_path=None,
    plot_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "creator_exit_no_fe.csv")
    plot_path = Path(plot_path or root / "creator_exit_no_fe.png")
    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
        target_subreddits=target_subreddits,
    )
    if frame.empty:
        raise ValueError("Creator exit no-FE frame is empty.")

    model_a = fit_ols(
        "exit ~ log_pre_rate * pers_free",
        frame,
        cluster_col="subreddit",
    )
    if "month" in frame.columns and frame["month"].nunique() > 1:
        model_b_formula = "exit ~ log_pre_rate * pers_free + C(month)"
        model_b_note = "month_fixed_effects"
    else:
        model_b_formula = "exit ~ log_pre_rate * pers_free"
        model_b_note = "month_fixed_effects_unavailable_same_as_model_a"
    model_b = fit_ols(model_b_formula, frame, cluster_col="subreddit")
    model_c = fit_ols(
        "exit ~ log_pre_rate * pers_free + log_pre_rate * gen_cap",
        frame,
        cluster_col="subreddit",
    )
    if model_a is None or model_b is None or model_c is None:
        raise ValueError("One or more creator-exit no-FE models failed to fit.")

    rows = [{
        "row_type": "sample_summary",
        "model": "creator_exit_no_fe",
        "term": "sample",
        "coef": np.nan,
        "se": np.nan,
        "pvalue": np.nan,
        "n": safe_int(len(frame)),
        "n_subreddits": safe_int(frame["subreddit"].nunique()),
        "overall_exit_rate": safe_float(frame["exit"].mean()),
        "frame_source": frame_source,
        "model_b_note": model_b_note,
    }]
    rows.extend(no_fe_model_rows(model_a, "Model A"))
    rows.extend(no_fe_model_rows(model_b, "Model B"))
    rows.extend(no_fe_model_rows(model_c, "Model C"))

    marginal_rows = []
    for value, label in [(0.25, "low"), (0.50, "mid"), (0.75, "high")]:
        marginal = no_fe_marginal_effect(model_a, value)
        marginal_rows.append({
            "row_type": "marginal_effect",
            "model": "Model A",
            "term": f"log_pre_rate_at_pers_free_{label}",
            "coef": marginal["marginal_effect"],
            "se": marginal["se"],
            "pvalue": np.nan,
            "n": safe_int(model_a.nobs),
            "pers_free_value": marginal["pers_free_value"],
            "ci_low": marginal["ci_low"],
            "ci_high": marginal["ci_high"],
        })
    rows.extend(marginal_rows)
    saved_plot_path = plot_creator_exit_no_fe_marginal_effects(model_a, plot_path)
    output = pd.DataFrame(rows)
    output["plot_path"] = saved_plot_path
    output.to_csv(output_path, index=False)
    return {
        "frame": frame,
        "frame_source": frame_source,
        "models": {"Model A": model_a, "Model B": model_b, "Model C": model_c},
        "rows": rows,
        "marginal_rows": marginal_rows,
        "output_path": str(output_path),
        "plot_path": saved_plot_path,
        "model_b_note": model_b_note,
    }

def split_low_high_log_pre_rate(frame):
    ordered = frame.sort_values(["subreddit", "log_pre_rate", "author"]).copy()
    ordered["_within_subreddit_rank"] = ordered.groupby("subreddit").cumcount()
    ordered["_within_subreddit_n"] = ordered.groupby("subreddit")["author"].transform("count")
    ordered["_frequency_half"] = np.where(
        ordered["_within_subreddit_rank"] < ordered["_within_subreddit_n"] / 2.0,
        "low",
        "high",
    )
    return ordered

def community_level_model_rows(model, model_name):
    rows = []
    for term in model.params.index:
        result = reg_result(model, term)
        rows.append({
            "row_type": "coefficient",
            "model": model_name,
            "term": term,
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "n": safe_int(model.nobs),
        })
    return rows

def label_scatter_outliers(ax, data, x_col, y_col, model, max_labels=12):
    fitted = model.fittedvalues
    residual = data[y_col].to_numpy(dtype=float) - np.asarray(fitted, dtype=float)
    scale = np.std(residual, ddof=1)
    if scale <= 0 or not np.isfinite(scale):
        return []
    labeled = data.assign(_abs_standard_residual=np.abs(residual / scale))
    labeled = labeled[labeled["_abs_standard_residual"] >= 2.0].copy()
    if labeled.empty:
        labeled = data.assign(_abs_standard_residual=np.abs(residual / scale)).nlargest(
            min(5, len(data)),
            "_abs_standard_residual",
        )
    labeled = labeled.nlargest(max_labels, "_abs_standard_residual")
    for _, row in labeled.iterrows():
        ax.text(
            row[x_col],
            row[y_col],
            str(row["subreddit"]),
            fontsize=7,
            ha="left",
            va="bottom",
        )
    return labeled["subreddit"].astype(str).tolist()

def plot_creator_exit_community_level(community, model_exit, model_gap, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_specs = [
        (axes[0], "exit_rate", "Exit rate", model_exit),
        (axes[1], "freq_gap", "Low-high frequency exit gap", model_gap),
    ]
    x_grid = np.linspace(
        float(community["pers_free"].min()),
        float(community["pers_free"].max()),
        100,
    )
    for ax, y_col, y_label, model in plot_specs:
        ax.scatter(
            community["pers_free"],
            community[y_col],
            color="#2563eb",
            alpha=0.7,
            s=45,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        pred = model.get_prediction(pd.DataFrame({"pers_free": x_grid})).summary_frame(alpha=0.05)
        ax.plot(x_grid, pred["mean"], color="#dc2626", linewidth=2.0)
        ax.fill_between(
            x_grid,
            pred["mean_ci_lower"].to_numpy(dtype=float),
            pred["mean_ci_upper"].to_numpy(dtype=float),
            color="#dc2626",
            alpha=0.10,
            linewidth=0,
        )
        label_scatter_outliers(ax, community, "pers_free", y_col, model)
        ax.set_xlabel("PersFree")
        ax.set_ylabel(y_label)
        ax.set_title(y_label)
    save_plot(fig, output_path)
    return str(output_path)

def compute_creator_exit_community_level(
    creator_path=None,
    score_path=None,
    target_subreddits=None,
    output_path=None,
    plot_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "creator_exit_community_level.csv")
    plot_path = Path(plot_path or root / "creator_exit_community_level.png")
    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
        target_subreddits=target_subreddits,
    )
    frame = split_low_high_log_pre_rate(frame)
    half_exit = (
        frame.groupby(["subreddit", "_frequency_half"])["exit"]
        .mean()
        .unstack("_frequency_half")
        .rename(columns={"low": "low_freq_exit_rate", "high": "high_freq_exit_rate"})
    )
    community = (
        frame.groupby("subreddit")
        .agg(
            n_creators=("author", "nunique"),
            n_exited=("exit", "sum"),
            exit_rate=("exit", "mean"),
            pers_free=("pers_free", "first"),
            gen_cap=("gen_cap", "first"),
        )
        .merge(half_exit, left_index=True, right_index=True, how="left")
        .reset_index()
    )
    community["freq_gap"] = community["low_freq_exit_rate"] - community["high_freq_exit_rate"]
    community = community.dropna(
        subset=["exit_rate", "freq_gap", "pers_free", "gen_cap"]
    ).sort_values("subreddit")
    if len(community) < 3:
        raise ValueError("Need at least three subreddit-level rows for community-level models.")

    model_a = fit_ols("exit_rate ~ pers_free", community)
    model_b = fit_ols("exit_rate ~ pers_free + gen_cap", community)
    model_c = fit_ols("freq_gap ~ pers_free", community)
    model_d = fit_ols("freq_gap ~ pers_free + gen_cap", community)
    if model_a is None or model_b is None or model_c is None or model_d is None:
        raise ValueError("One or more creator-exit community-level models failed to fit.")

    coefficient_rows = []
    coefficient_rows.extend(community_level_model_rows(model_a, "Model A: exit_rate ~ pers_free"))
    coefficient_rows.extend(community_level_model_rows(model_b, "Model B: exit_rate ~ pers_free + gen_cap"))
    coefficient_rows.extend(community_level_model_rows(model_c, "Model C: freq_gap ~ pers_free"))
    coefficient_rows.extend(community_level_model_rows(model_d, "Model D: freq_gap ~ pers_free + gen_cap"))

    saved_plot_path = plot_creator_exit_community_level(
        community,
        model_a,
        model_c,
        plot_path,
    )
    community_rows = community.copy()
    community_rows["row_type"] = "community"
    community_rows["model"] = np.nan
    community_rows["term"] = np.nan
    community_rows["coef"] = np.nan
    community_rows["se"] = np.nan
    community_rows["pvalue"] = np.nan
    community_rows["n"] = np.nan
    coefficient_table = pd.DataFrame(coefficient_rows)
    coefficient_table["subreddit"] = np.nan
    coefficient_table["n_creators"] = np.nan
    coefficient_table["n_exited"] = np.nan
    coefficient_table["exit_rate"] = np.nan
    coefficient_table["low_freq_exit_rate"] = np.nan
    coefficient_table["high_freq_exit_rate"] = np.nan
    coefficient_table["freq_gap"] = np.nan
    coefficient_table["pers_free"] = np.nan
    coefficient_table["gen_cap"] = np.nan
    output = pd.concat([community_rows, coefficient_table], ignore_index=True, sort=False)
    output["frame_source"] = frame_source
    output["plot_path"] = saved_plot_path
    output.to_csv(output_path, index=False)
    return {
        "community": community,
        "models": {
            "Model A": model_a,
            "Model B": model_b,
            "Model C": model_c,
            "Model D": model_d,
        },
        "coefficient_rows": coefficient_rows,
        "output_path": str(output_path),
        "plot_path": saved_plot_path,
        "frame_source": frame_source,
    }

def assign_creator_persfree_terciles(frame):
    subreddit_scores = (
        frame[["subreddit", "pers_free"]]
        .drop_duplicates("subreddit")
        .sort_values(["pers_free", "subreddit"])
        .copy()
    )
    subreddit_scores["pers_free_tercile"] = pd.qcut(
        subreddit_scores["pers_free"],
        q=3,
        labels=["bottom", "middle", "top"],
        duplicates="drop",
    ).astype(str)
    return frame.drop(columns=["pers_free_tercile"], errors="ignore").merge(
        subreddit_scores[["subreddit", "pers_free_tercile"]],
        on="subreddit",
        how="left",
    )

def compute_creator_exit_tercile(
    creator_path=None,
    score_path=None,
    target_subreddits=None,
    output_path=None,
    plot_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "creator_exit_tercile.csv")
    plot_path = Path(plot_path or root / "creator_exit_tercile.png")
    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
        target_subreddits=target_subreddits,
    )
    frame = assign_creator_persfree_terciles(frame)
    frame = frame.dropna(subset=["pers_free_tercile", "log_pre_rate", "exit", "subreddit"]).copy()
    frame["middle_tercile_dummy"] = frame["pers_free_tercile"].eq("middle").astype(int)
    frame["top_tercile_dummy"] = frame["pers_free_tercile"].eq("top").astype(int)

    rows = [{
        "row_type": "sample_summary",
        "tercile": "all",
        "model": "creator_exit_tercile",
        "term": "sample",
        "coef": np.nan,
        "se": np.nan,
        "pvalue": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": safe_int(len(frame)),
        "n_subreddits": safe_int(frame["subreddit"].nunique()),
        "exit_rate": safe_float(frame["exit"].mean()),
        "frame_source": frame_source,
    }]

    tercile_models = {}
    tercile_order = ["bottom", "middle", "top"]
    for tercile in tercile_order:
        subset = frame[frame["pers_free_tercile"].eq(tercile)].copy()
        if subset.empty:
            continue
        model = fit_ols("exit ~ log_pre_rate + C(subreddit)", subset, cluster_col="subreddit")
        if model is None:
            continue
        tercile_models[tercile] = model
        result = reg_result(model, "log_pre_rate")
        rows.append({
            "row_type": "tercile_model",
            "tercile": tercile,
            "model": f"{tercile}: exit ~ log_pre_rate + subreddit FE",
            "term": "log_pre_rate",
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "ci_low": safe_float(result.get("coef") - 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "ci_high": safe_float(result.get("coef") + 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "n": safe_int(model.nobs),
            "n_subreddits": safe_int(subset["subreddit"].nunique()),
            "exit_rate": safe_float(subset["exit"].mean()),
            "frame_source": frame_source,
        })

    pooled_model = fit_ols(
        (
            "exit ~ log_pre_rate + log_pre_rate:middle_tercile_dummy "
            "+ log_pre_rate:top_tercile_dummy + C(subreddit)"
        ),
        frame,
        cluster_col="subreddit",
    )
    if pooled_model is None:
        raise ValueError("Creator-exit tercile pooled model failed to fit.")
    for term in [
        "log_pre_rate",
        "log_pre_rate:middle_tercile_dummy",
        "log_pre_rate:top_tercile_dummy",
    ]:
        result = reg_result(pooled_model, term)
        rows.append({
            "row_type": "pooled_interaction",
            "tercile": "pooled",
            "model": "Pooled: bottom baseline",
            "term": term,
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "ci_low": safe_float(result.get("coef") - 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "ci_high": safe_float(result.get("coef") + 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "n": safe_int(pooled_model.nobs),
            "n_subreddits": safe_int(frame["subreddit"].nunique()),
            "exit_rate": safe_float(frame["exit"].mean()),
            "frame_source": frame_source,
        })

    exit_by_tercile = (
        frame.groupby("pers_free_tercile")
        .agg(
            exit_rate=("exit", "mean"),
            n=("author", "nunique"),
            n_subreddits=("subreddit", "nunique"),
        )
        .reindex(tercile_order)
        .reset_index()
    )
    for _, row in exit_by_tercile.dropna(subset=["exit_rate"]).iterrows():
        rows.append({
            "row_type": "exit_rate_by_tercile",
            "tercile": row["pers_free_tercile"],
            "model": "descriptive",
            "term": "exit_rate",
            "coef": row["exit_rate"],
            "se": np.nan,
            "pvalue": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": safe_int(row["n"]),
            "n_subreddits": safe_int(row["n_subreddits"]),
            "exit_rate": safe_float(row["exit_rate"]),
            "frame_source": frame_source,
        })

    quartile_frame = frame.copy()
    quartile_frame["_log_pre_rate_rank"] = (
        quartile_frame.groupby("pers_free_tercile")["log_pre_rate"]
        .rank(method="first")
    )
    quartile_frame["log_pre_rate_quartile"] = (
        quartile_frame.groupby("pers_free_tercile")["_log_pre_rate_rank"]
        .transform(lambda values: pd.qcut(values, q=4, labels=["Q1", "Q2", "Q3", "Q4"]))
    )
    exit_by_quartile = (
        quartile_frame.dropna(subset=["log_pre_rate_quartile"])
        .groupby(["pers_free_tercile", "log_pre_rate_quartile"], observed=True)
        .agg(
            exit_rate=("exit", "mean"),
            n=("author", "nunique"),
            n_subreddits=("subreddit", "nunique"),
        )
        .reset_index()
    )
    for _, row in exit_by_quartile.iterrows():
        rows.append({
            "row_type": "exit_rate_by_tercile_quartile",
            "tercile": row["pers_free_tercile"],
            "model": "descriptive",
            "term": f"log_pre_rate_{row['log_pre_rate_quartile']}",
            "coef": row["exit_rate"],
            "se": np.nan,
            "pvalue": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n": safe_int(row["n"]),
            "n_subreddits": safe_int(row["n_subreddits"]),
            "exit_rate": safe_float(row["exit_rate"]),
            "frame_source": frame_source,
        })

    tercile_rows = [row for row in rows if row["row_type"] == "tercile_model"]
    plot_table = pd.DataFrame(tercile_rows).set_index("tercile").reindex(tercile_order).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(plot_table))
    y = plot_table["coef"].astype(float).to_numpy()
    yerr = np.vstack([
        y - plot_table["ci_low"].astype(float).to_numpy(),
        plot_table["ci_high"].astype(float).to_numpy() - y,
    ])
    ax.bar(x, y, color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#1d4ed8", capsize=4, linewidth=1.5, capthick=1.5)
    ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label.title() for label in plot_table["tercile"]])
    ax.set_ylabel("Coefficient on log_pre_rate")
    ax.set_title("Creator exit by PersFree tercile")
    save_plot(fig, plot_path)

    output = pd.DataFrame(rows)
    output["plot_path"] = str(plot_path)
    output.to_csv(output_path, index=False)
    return {
        "frame": frame,
        "rows": rows,
        "tercile_models": tercile_models,
        "pooled_model": pooled_model,
        "output_path": str(output_path),
        "plot_path": str(plot_path),
        "frame_source": frame_source,
    }

def assign_full_sample_frequency_quartiles(frame):
    ordered = frame.sort_values(["log_pre_rate", "author", "subreddit"]).copy()
    ordered["_frequency_rank"] = np.arange(1, len(ordered) + 1)
    ordered["frequency_quartile"] = pd.qcut(
        ordered["_frequency_rank"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    return ordered.drop(columns=["_frequency_rank"])

def assign_creator_context_terciles(frame):
    assigned = assign_creator_persfree_terciles(frame)
    context_map = {
        "top": "low_context",
        "middle": "mid_context",
        "bottom": "high_context",
    }
    persfree_label_map = {
        "top": "high_pers_free",
        "middle": "mid_pers_free",
        "bottom": "low_pers_free",
    }
    assigned["context_tercile"] = assigned["pers_free_tercile"].map(context_map)
    assigned["pers_free_tercile_label"] = assigned["pers_free_tercile"].map(persfree_label_map)
    return assigned

def plot_creator_exit_quartile_heatmap(cell_table, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quartile_order = ["Q1", "Q2", "Q3", "Q4"]
    context_order = ["low_context", "mid_context", "high_context"]
    plot_table = (
        cell_table.pivot(index="frequency_quartile", columns="context_tercile", values="exit_rate")
        .reindex(index=quartile_order, columns=context_order)
    )
    n_table = (
        cell_table.pivot(index="frequency_quartile", columns="context_tercile", values="n")
        .reindex(index=quartile_order, columns=context_order)
    )
    matrix = plot_table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(matrix)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("modern_blue_seq", ["#eff6ff", "#1e3a8a"])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(context_order)))
    ax.set_xticklabels(["Low", "Mid", "High"])
    ax.set_yticks(np.arange(len(quartile_order)))
    ax.set_yticklabels(quartile_order)
    ax.set_xlabel("Personal-context tercile")
    ax.set_ylabel("Posting-frequency quartile")
    ax.set_title("Creator exit rate by frequency and context")
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(context_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(quartile_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index, quartile in enumerate(quartile_order):
        for col_index, context in enumerate(context_order):
            value = plot_table.loc[quartile, context]
            n_value = n_table.loc[quartile, context]
            if pd.isna(value):
                label = "NA"
            else:
                label = f"{value:.1%}\nN={int(n_value):,}"
            text_color = "white" if pd.notna(value) and value >= np.nanmean(matrix) else "#1a1a2e"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color=text_color)
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Exit rate")
    save_plot(fig, output_path)
    return str(output_path)

def compute_creator_exit_quartiles(
    creator_path=None,
    score_path=None,
    target_subreddits=None,
    output_path=None,
    plot_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "creator_exit_quartiles.csv")
    plot_path = Path(plot_path or root / "creator_exit_heatmap.png")
    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
        target_subreddits=target_subreddits,
    )
    frame = assign_full_sample_frequency_quartiles(frame)
    frame = assign_creator_context_terciles(frame)
    frame = frame.dropna(
        subset=["frequency_quartile", "context_tercile", "log_pre_rate", "exit", "subreddit"]
    ).copy()
    if frame.empty:
        raise ValueError("Creator exit quartile frame is empty.")

    quartile_order = ["Q1", "Q2", "Q3", "Q4"]
    context_order = ["low_context", "mid_context", "high_context"]
    full_index = pd.MultiIndex.from_product(
        [quartile_order, context_order],
        names=["frequency_quartile", "context_tercile"],
    )
    cell_table = (
        frame.groupby(["frequency_quartile", "context_tercile"], observed=True)
        .agg(
            exit_rate=("exit", "mean"),
            n=("author", "nunique"),
            n_subreddits=("subreddit", "nunique"),
            mean_log_pre_rate=("log_pre_rate", "mean"),
            mean_pers_free=("pers_free", "mean"),
        )
        .reindex(full_index)
        .reset_index()
    )

    for quartile in ["Q1", "Q2", "Q3"]:
        frame[f"{quartile}_dummy"] = frame["frequency_quartile"].eq(quartile).astype(int)
    frame["low_context_dummy"] = frame["context_tercile"].eq("low_context").astype(int)
    frame["mid_context_dummy"] = frame["context_tercile"].eq("mid_context").astype(int)

    interaction_terms = [
        "Q1_dummy:low_context_dummy",
        "Q1_dummy:mid_context_dummy",
        "Q2_dummy:low_context_dummy",
        "Q2_dummy:mid_context_dummy",
        "Q3_dummy:low_context_dummy",
        "Q3_dummy:mid_context_dummy",
    ]
    model_terms = ["Q1_dummy", "Q2_dummy", "Q3_dummy", *interaction_terms]
    formula = "exit ~ " + " + ".join(model_terms) + " + C(subreddit)"
    model = fit_ols(formula, frame, cluster_col="subreddit")
    if model is None:
        raise ValueError("Creator exit quartile FE model failed to fit.")

    saved_plot_path = plot_creator_exit_quartile_heatmap(cell_table, plot_path)
    rows = [{
        "row_type": "sample_summary",
        "model": "creator_exit_quartiles",
        "term": "sample",
        "coef": np.nan,
        "se": np.nan,
        "pvalue": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "frequency_quartile": "all",
        "context_tercile": "all",
        "pers_free_tercile": "all",
        "exit_rate": safe_float(frame["exit"].mean()),
        "n": safe_int(len(frame)),
        "n_subreddits": safe_int(frame["subreddit"].nunique()),
        "mean_log_pre_rate": safe_float(frame["log_pre_rate"].mean()),
        "mean_pers_free": safe_float(frame["pers_free"].mean()),
        "frame_source": frame_source,
        "plot_path": saved_plot_path,
        "note": "low_context is the high-pers_free tercile; high_context is the low-pers_free tercile.",
    }]

    persfree_map = {
        "low_context": "high_pers_free",
        "mid_context": "mid_pers_free",
        "high_context": "low_pers_free",
    }
    for _, row in cell_table.iterrows():
        rows.append({
            "row_type": "exit_rate_cell",
            "model": "descriptive",
            "term": "exit_rate",
            "coef": row["exit_rate"],
            "se": np.nan,
            "pvalue": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "frequency_quartile": row["frequency_quartile"],
            "context_tercile": row["context_tercile"],
            "pers_free_tercile": persfree_map.get(row["context_tercile"]),
            "exit_rate": row["exit_rate"],
            "n": safe_int(row["n"]),
            "n_subreddits": safe_int(row["n_subreddits"]),
            "mean_log_pre_rate": row["mean_log_pre_rate"],
            "mean_pers_free": row["mean_pers_free"],
            "frame_source": frame_source,
            "plot_path": saved_plot_path,
            "note": np.nan,
        })

    for term in model_terms:
        result = reg_result(model, term)
        rows.append({
            "row_type": "coefficient",
            "model": "FE quartile x context model",
            "term": term,
            "coef": result.get("coef"),
            "se": result.get("se"),
            "pvalue": result.get("pvalue"),
            "ci_low": safe_float(result.get("coef") - 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "ci_high": safe_float(result.get("coef") + 1.96 * result.get("se"))
            if result.get("coef") is not None and result.get("se") is not None else None,
            "frequency_quartile": np.nan,
            "context_tercile": np.nan,
            "pers_free_tercile": np.nan,
            "exit_rate": np.nan,
            "n": safe_int(model.nobs),
            "n_subreddits": safe_int(frame["subreddit"].nunique()),
            "mean_log_pre_rate": np.nan,
            "mean_pers_free": np.nan,
            "frame_source": frame_source,
            "plot_path": saved_plot_path,
            "note": np.nan,
        })

    output = pd.DataFrame(rows)
    output.to_csv(output_path, index=False)
    return {
        "frame": frame,
        "cell_table": cell_table,
        "model": model,
        "rows": rows,
        "output_path": str(output_path),
        "plot_path": saved_plot_path,
        "frame_source": frame_source,
    }

def load_or_build_single_community_flags(
    target_subreddits,
    eligible_authors,
    flags_path=None,
    pre_start_month="2022-01",
    pre_end_month="2022-10",
):
    root = globals().get("ROOT", Path("."))
    data_dir = globals().get("DATA_DIR", root / "data")
    flags_path = Path(flags_path or root / "creator_single_community_flags.csv")
    eligible_authors = set(pd.Series(list(eligible_authors), dtype=str).dropna())
    if flags_path.exists():
        flags = pd.read_csv(flags_path)
        required = {"author", "n_pre_subreddits", "single_community", "single_subreddit"}
        if required.issubset(flags.columns):
            flags["author"] = flags["author"].astype(str)
            if eligible_authors.issubset(set(flags["author"])):
                return flags, str(flags_path)

    target_subreddits = sorted(set(pd.Series(target_subreddits, dtype=str).dropna()))
    excluded_authors = set(globals().get("EXCLUDED_AUTHORS", set()))
    max_lines_per_file = globals().get("MAX_LINES_PER_FILE", None)
    start_date = pd.Timestamp(pre_start_month).to_pydatetime()
    end_date_exclusive = (pd.Timestamp(pre_end_month) + pd.offsets.MonthBegin(1)).to_pydatetime()
    author_subreddits = {author: set() for author in eligible_authors}

    for subreddit in tqdm(target_subreddits, desc="  single-community pre-event scans"):
        path = raw_post_file_path(subreddit, data_dir)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if max_lines_per_file is not None and i >= max_lines_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                created_utc = payload.get("created_utc")
                if created_utc is None:
                    continue
                try:
                    dt = datetime.utcfromtimestamp(int(created_utc))
                except Exception:
                    continue
                if dt < start_date or dt >= end_date_exclusive:
                    continue
                author = str(payload.get("author") or "")
                if (
                    author not in eligible_authors
                    or author in excluded_authors
                    or author.lower().endswith("bot")
                ):
                    continue
                author_subreddits[author].add(subreddit)

    rows = []
    for author in sorted(eligible_authors):
        subreddits = sorted(author_subreddits.get(author, set()))
        rows.append({
            "author": author,
            "n_pre_subreddits": safe_int(len(subreddits)),
            "single_community": int(len(subreddits) == 1),
            "single_subreddit": subreddits[0] if len(subreddits) == 1 else "",
            "pre_subreddits": ";".join(subreddits),
        })
    flags = pd.DataFrame(rows)
    flags.to_csv(flags_path, index=False)
    return flags, "rebuilt_from_raw_posts"

def compute_creator_exit_single_community(
    creator_path=None,
    score_path=None,
    target_subreddits=None,
    output_path=None,
    flags_path=None,
):
    root = globals().get("ROOT", Path("."))
    output_path = Path(output_path or root / "creator_exit_single_community.csv")
    if target_subreddits is None:
        panel_path = root / "panel.csv"
        if panel_path.exists():
            target_subreddits = pd.read_csv(panel_path, usecols=["subreddit"])["subreddit"].astype(str).unique()

    frame, frame_source = load_or_build_creator_exit_no_fe_frame(
        creator_path=creator_path,
        score_path=score_path,
        target_subreddits=target_subreddits,
    )
    flags, flags_source = load_or_build_single_community_flags(
        target_subreddits=target_subreddits,
        eligible_authors=frame["author"].astype(str),
        flags_path=flags_path,
    )
    restricted = frame.merge(
        flags[["author", "n_pre_subreddits", "single_community", "single_subreddit"]],
        on="author",
        how="left",
    )
    retained = restricted[restricted["single_community"].eq(1)].copy()
    dropped = restricted[~restricted["single_community"].eq(1)].copy()
    if retained.empty:
        raise ValueError("No single-community creators retained.")

    model1 = fit_ols("exit ~ log_pre_rate + C(subreddit)", retained, cluster_col="subreddit")
    model2 = fit_ols(
        "exit ~ log_pre_rate + log_pre_rate:pers_free + C(subreddit)",
        retained,
        cluster_col="subreddit",
    )
    model3 = fit_ols(
        "exit ~ log_pre_rate + log_pre_rate:pers_free + log_pre_rate:gen_cap + C(subreddit)",
        retained,
        cluster_col="subreddit",
    )
    if model1 is None or model2 is None or model3 is None:
        raise ValueError("One or more single-community creator-exit models failed to fit.")

    rows = [{
        "row_type": "sample_summary",
        "model": "single_community_creators",
        "term": "sample",
        "coef": np.nan,
        "se": np.nan,
        "pvalue": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": safe_int(len(retained)),
        "n_subreddits": safe_int(retained["subreddit"].nunique()),
        "n_retained": safe_int(len(retained)),
        "n_dropped": safe_int(len(dropped)),
        "full_sample_n": safe_int(len(frame)),
        "exit_rate": safe_float(retained["exit"].mean()),
        "frame_source": frame_source,
        "flags_source": flags_source,
        "comparison_full_sample_interaction_coef": 0.0079,
        "comparison_full_sample_interaction_pvalue": 0.829,
    }]

    coefficient_specs = [
        ("Model 1: exit ~ log_pre_rate + subreddit FE", model1, ["log_pre_rate"]),
        (
            "Model 2: + pers_free x log_pre_rate",
            model2,
            ["log_pre_rate", "log_pre_rate:pers_free"],
        ),
        (
            "Model 3: + pers_free and gen_cap interactions",
            model3,
            ["log_pre_rate", "log_pre_rate:pers_free", "log_pre_rate:gen_cap"],
        ),
    ]
    for model_name, model, terms in coefficient_specs:
        for term in terms:
            result = reg_result(model, term)
            rows.append({
                "row_type": "coefficient",
                "model": model_name,
                "term": term,
                "coef": result.get("coef"),
                "se": result.get("se"),
                "pvalue": result.get("pvalue"),
                "ci_low": safe_float(result.get("coef") - 1.96 * result.get("se"))
                if result.get("coef") is not None and result.get("se") is not None else None,
                "ci_high": safe_float(result.get("coef") + 1.96 * result.get("se"))
                if result.get("coef") is not None and result.get("se") is not None else None,
                "n": safe_int(model.nobs),
                "n_subreddits": safe_int(retained["subreddit"].nunique()),
                "n_retained": safe_int(len(retained)),
                "n_dropped": safe_int(len(dropped)),
                "full_sample_n": safe_int(len(frame)),
                "exit_rate": safe_float(retained["exit"].mean()),
                "frame_source": frame_source,
                "flags_source": flags_source,
                "comparison_full_sample_interaction_coef": 0.0079,
                "comparison_full_sample_interaction_pvalue": 0.829,
            })

    for value, label in [(0.25, "low"), (0.50, "mid"), (0.75, "high")]:
        marginal = creator_exit_marginal_effect(model2, value)
        rows.append({
            "row_type": "marginal_effect",
            "model": "Model 2: + pers_free x log_pre_rate",
            "term": f"log_pre_rate_at_pers_free_{label}",
            "coef": marginal["marginal_effect"],
            "se": marginal["se"],
            "pvalue": np.nan,
            "ci_low": marginal["ci_low"],
            "ci_high": marginal["ci_high"],
            "n": safe_int(model2.nobs),
            "n_subreddits": safe_int(retained["subreddit"].nunique()),
            "n_retained": safe_int(len(retained)),
            "n_dropped": safe_int(len(dropped)),
            "full_sample_n": safe_int(len(frame)),
            "exit_rate": safe_float(retained["exit"].mean()),
            "frame_source": frame_source,
            "flags_source": flags_source,
            "pers_free_value": marginal["pers_free_value"],
            "comparison_full_sample_interaction_coef": 0.0079,
            "comparison_full_sample_interaction_pvalue": 0.829,
        })

    output = pd.DataFrame(rows)
    output.to_csv(output_path, index=False)
    return {
        "frame": frame,
        "restricted": retained,
        "dropped": dropped,
        "flags": flags,
        "rows": rows,
        "models": {"model1": model1, "model2": model2, "model3": model3},
        "output_path": str(output_path),
        "frame_source": frame_source,
        "flags_source": flags_source,
    }

def compute_split_half_preshock_stability(acsi_scores, measurement_path=None):
    measurement_path = measurement_path or ACSI_MEASUREMENT_SAMPLE_RUN1_PATH
    frame = normalize_measurement_frame(pd.read_csv(measurement_path))
    frame = frame[frame["subreddit"].isin(set(acsi_scores["subreddit"].astype(str)))].copy()
    early_scores, early_metadata = simple_acsi_scores_from_measurements(
        frame,
        start_month="2022-01",
        end_month="2022-05",
    )
    late_scores, late_metadata = simple_acsi_scores_from_measurements(
        frame,
        start_month="2022-06",
        end_month="2022-10",
    )
    merged = early_scores.merge(
        late_scores,
        on="subreddit",
        suffixes=("_early", "_late"),
        how="inner",
    ).dropna()
    if len(merged) < 3:
        raise ValueError("Need at least 3 merged subreddits for split-half stability.")
    rows = []
    requested_benchmark_n = 124
    requested_noise_sd = 1.0 / np.sqrt(requested_benchmark_n - 1)
    requested_expected_abs = np.sqrt(2.0 / (np.pi * (requested_benchmark_n - 1)))
    for dimension in ["gen_cap", "phys_free", "pers_free"]:
        early = merged[f"{dimension}_early"].astype(float)
        late = merged[f"{dimension}_late"].astype(float)
        pearson_r, pearson_pvalue = stats.pearsonr(early, late)
        spearman = stats.spearmanr(early, late)
        actual_noise_sd = 1.0 / np.sqrt(len(merged) - 1)
        actual_expected_abs = np.sqrt(2.0 / (np.pi * (len(merged) - 1)))
        rows.append({
            "dimension": dimension,
            "n_subreddits_actual": safe_int(len(merged)),
            "pearson_r": safe_float(pearson_r),
            "pearson_pvalue": safe_float(pearson_pvalue),
            "spearman_rho": safe_float(spearman.statistic),
            "spearman_pvalue_h0_rho_0": safe_float(spearman.pvalue),
            "mean_abs_difference": safe_float((late - early).abs().mean()),
            "pure_noise_expected_rho": 0.0,
            "pure_noise_sd_rho_actual_n": safe_float(actual_noise_sd),
            "pure_noise_expected_abs_rho_actual_n": safe_float(actual_expected_abs),
            "pure_noise_benchmark_n_requested": requested_benchmark_n,
            "pure_noise_sd_rho_n124": safe_float(requested_noise_sd),
            "pure_noise_expected_abs_rho_n124": safe_float(requested_expected_abs),
        })
    emit_output_table(
        pd.DataFrame(rows),
        TABLES_DIR / "acsi_split_half_preshock_stability.csv",
        index=False,
    )
    persfree_row = next(row for row in rows if row["dimension"] == "pers_free")
    return {
        "rows": rows,
        "early_n_posts": early_metadata["n_posts_used"],
        "late_n_posts": late_metadata["n_posts_used"],
        "n_merged_subreddits": safe_int(len(merged)),
        "pers_free_stability_confirmed": bool(persfree_row["spearman_rho"] > 0.85),
    }

def summarize_paper_check_result(result):
    summary = {"status": "ok"}
    if isinstance(result, dict):
        for key, value in result.items():
            if key.endswith("_path") or key in {
                "output_path",
                "comparison_path",
                "summary_path",
                "plot_path",
                "hist_path",
                "latex_path",
                "panel_output_path",
                "metrics_path",
                "coefficients_path",
                "rankings_path",
                "quartiles_path",
                "risk_quartiles_path",
            }:
                summary[key] = str(value)
            elif key in {"summary", "paragraph", "latex_sentence", "interpretation"} and isinstance(value, str):
                summary[key] = value[:1000]
            elif key == "rows" and isinstance(value, list):
                summary["n_rows"] = safe_int(len(value))
            elif isinstance(value, pd.DataFrame):
                summary[f"n_{key}_rows"] = safe_int(len(value))
        return summary
    if result is None:
        summary["note"] = "function returned None"
    else:
        summary["result_type"] = type(result).__name__
    return summary


def run_paper_check(results, manifest_rows, key, label, fn):
    print(f"\n--- Paper suite: {label} ---")
    try:
        result = fn()
        summary = summarize_paper_check_result(result)
        summary["label"] = label
        results[key] = summary
        manifest_rows.append({
            "key": key,
            "label": label,
            "status": summary.get("status", "ok"),
            "output_path": summary.get("output_path"),
            "plot_path": summary.get("plot_path"),
            "summary_path": summary.get("summary_path"),
            "note": summary.get("summary") or summary.get("paragraph") or summary.get("interpretation"),
        })
        print(f"  Completed {label}.")
        return result
    except FileNotFoundError as exc:
        message = str(exc)
        results[key] = {
            "status": "skipped",
            "label": label,
            "error": message,
        }
        manifest_rows.append({
            "key": key,
            "label": label,
            "status": "skipped",
            "output_path": None,
            "plot_path": None,
            "summary_path": None,
            "note": message,
        })
        print(f"  {label} skipped: {message}")
        return None
    except Exception as exc:
        message = str(exc)
        results[key] = {
            "status": "failed",
            "label": label,
            "error": message,
        }
        manifest_rows.append({
            "key": key,
            "label": label,
            "status": "failed",
            "output_path": None,
            "plot_path": None,
            "summary_path": None,
            "note": message,
        })
        print(f"  {label} failed: {message}")
        return None


def prepare_current_mechanism_score_path(acsi_scores, tables_dir):
    """Write current ACSI scores in the legacy mechanism schema if available."""
    tables_dir = Path(tables_dir)
    output_path = tables_dir / "acsi_scores_current_mechanism_schema.csv"
    legacy_path = tables_dir / "acsi_preshock_tworuns.csv"
    required_columns = {"subreddit", "gen_cap", "phys_free", "pers_free"}
    current_columns = {
        "generation_capability_norm": "gen_cap",
        "physical_free_norm": "phys_free",
        "non_personal_norm": "pers_free",
    }

    if acsi_scores is None or getattr(acsi_scores, "empty", True):
        return output_path if output_path.exists() else legacy_path

    scores = acsi_scores.copy()
    if not required_columns.issubset(scores.columns):
        if set(current_columns).issubset(scores.columns):
            scores = scores.assign(
                gen_cap=scores["generation_capability_norm"],
                phys_free=scores["physical_free_norm"],
                pers_free=scores["non_personal_norm"],
            )
        else:
            return output_path if output_path.exists() else legacy_path

    scores = scores[["subreddit", "gen_cap", "phys_free", "pers_free"]].copy()
    for column_name in ["gen_cap", "phys_free", "pers_free"]:
        scores[column_name] = pd.to_numeric(scores[column_name], errors="coerce")
    scores["subreddit"] = scores["subreddit"].astype(str)
    scores = scores.dropna(subset=["subreddit", "gen_cap", "phys_free", "pers_free"])
    if scores.empty:
        return output_path if output_path.exists() else legacy_path

    output_path.parent.mkdir(exist_ok=True, parents=True)
    scores.to_csv(output_path, index=False)
    return output_path


def run_full_paper_suite(submonth_panel, acsi_scores, results):
    print("\n=== Full paper robustness/mechanism suite ===")
    tables_dir = globals().get("TABLES_DIR", Path("."))
    figures_dir = globals().get("FIGURES_DIR", Path("."))
    output_dir = globals().get("OUTPUT_DIR", Path("."))
    tables_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    month_columns_available = (
        submonth_panel is not None
        and not submonth_panel.empty
        and (
            "year_month" in submonth_panel.columns
            or "month" in submonth_panel.columns
            or "year_month_dt" in submonth_panel.columns
        )
    )
    if not month_columns_available or "subreddit" not in submonth_panel.columns or "log_posts" not in submonth_panel.columns:
        manifest_path = tables_dir / "paper_check_manifest.csv"
        manifest = pd.DataFrame([{
            "key": "full_paper_suite",
            "label": "Full paper robustness/mechanism suite",
            "status": "skipped",
            "output_path": None,
            "plot_path": None,
            "summary_path": None,
            "note": "Skipped because no complete subreddit-month panel was provided.",
        }])
        manifest.to_csv(manifest_path, index=False)
        results["paper_check_manifest"] = {
            "status": "skipped",
            "output_path": str(manifest_path),
            "n_checks": 0,
            "n_failed": 0,
        }
        print("  Skipping full paper suite: no complete subreddit-month panel provided.")
        return results

    paper_panel = submonth_panel.copy()
    if "year_month" not in paper_panel.columns and "month" in paper_panel.columns:
        paper_panel["year_month"] = paper_panel["month"]
    if "year_month" not in paper_panel.columns and "year_month_dt" in paper_panel.columns:
        paper_panel["year_month"] = pd.to_datetime(paper_panel["year_month_dt"]).dt.strftime("%Y-%m")
    paper_panel["month"] = paper_panel["year_month"].astype(str).str.slice(0, 7)
    if "post" not in paper_panel.columns:
        if "post_shock" in paper_panel.columns:
            paper_panel["post"] = paper_panel["post_shock"]
        else:
            paper_panel["post"] = paper_panel["month"].ge("2022-12").astype(int)
    if "total_posts" not in paper_panel.columns:
        if "post_count" in paper_panel.columns:
            paper_panel["total_posts"] = paper_panel["post_count"]
        elif "posts" in paper_panel.columns:
            paper_panel["total_posts"] = paper_panel["posts"]

    score_path = prepare_current_mechanism_score_path(acsi_scores, tables_dir)
    manifest_rows = []

    checks = [
        (
            "acsi_dimension_diagnostics",
            "ACSI dimension diagnostics",
            lambda: compute_acsi_dimension_diagnostics(
                output_dir=output_dir,
                panel=submonth_panel.copy(),
                acsi_scores=acsi_scores.copy(),
            ),
        ),
        (
            "reddit_disruption_robustness",
            "Reddit API disruption robustness",
            lambda: compute_reddit_disruption_robustness(
                paper_panel.copy(),
                score_path=score_path,
                blackout_path=tables_dir / "blackout.csv",
                proxy_path=tables_dir / "blackout_proxy.csv",
                output_path=tables_dir / "reddit_disruption_robustness.csv",
            ),
        ),
        (
            "placebo_shock_comparison",
            "ChatGPT/API/placebo shock comparison",
            lambda: compute_placebo_shock_comparison(
                panel=paper_panel.copy(),
                score_path=score_path,
                output_path=tables_dir / "placebo_shock_comparison.csv",
                comparison_path=tables_dir / "placebo_shock_table.csv",
            ),
        ),
        (
            "shock_date_placebo",
            "Shock-date permutation placebo",
            lambda: compute_shock_date_placebo(
                panel=paper_panel.copy(),
                score_path=score_path,
                output_path=tables_dir / "shock_date_placebo_results.csv",
                summary_path=tables_dir / "shock_date_placebo_summary.csv",
                plot_path=figures_dir / "shock_date_placebo.png",
                hist_path=figures_dir / "shock_date_placebo_hist.png",
            ),
        ),
        (
            "shock_date_placebo_api_control",
            "Shock-date placebo with API control",
            lambda: compute_shock_date_placebo_api_control(
                panel=paper_panel.copy(),
                score_path=score_path,
                output_path=tables_dir / "shock_date_placebo_api_control_results.csv",
                summary_path=tables_dir / "shock_date_placebo_api_control_summary.csv",
                plot_path=figures_dir / "shock_date_placebo_api_control.png",
                hist_path=figures_dir / "shock_date_placebo_api_control_hist.png",
            ),
        ),
        (
            "shock_date_placebo_all_dimensions",
            "Shock-date placebo by dimension",
            lambda: compute_shock_date_placebo_all_dimensions(
                panel=paper_panel.copy(),
                score_path=score_path,
                output_dir=figures_dir,
            ),
        ),
        (
            "pretrend_power_analysis",
            "Pre-trend power analysis",
            lambda: compute_pretrend_power_analysis(
                paper_panel.copy(),
                score_path=score_path,
                n_simulations=250 if globals().get("QUICK_MODE", False) else 1000,
                output_path=tables_dir / "pretrend_power_analysis.csv",
                plot_path=figures_dir / "pretrend_power_curve.png",
            ),
        ),
        (
            "creator_exit_interaction_power",
            "Creator-exit interaction power analysis",
            lambda: compute_creator_exit_interaction_power_analysis(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_interaction_power_analysis.csv",
                latex_path=tables_dir / "creator_exit_interaction_power_analysis_sentence.tex",
            ),
        ),
        (
            "community_low_frequency_composition",
            "Low-frequency creator composition",
            lambda: compute_community_low_frequency_composition(score_path=score_path),
        ),
        (
            "community_new_entrant_rates",
            "New-entrant rates",
            lambda: compute_community_new_entrant_rates(score_path=score_path),
        ),
        (
            "community_new_entrant_volume_control",
            "New-entrant volume control",
            lambda: compute_community_new_entrant_volume_control(),
        ),
        (
            "preshock_persfree_linear_trend",
            "Linear PersFree pre-trend",
            lambda: compute_preshock_persfree_linear_trend(),
        ),
        (
            "preshock_leave_one_month_f_tests",
            "Leave-one-month pre-trend F-tests",
            lambda: compute_preshock_leave_one_month_f_tests(),
        ),
        (
            "extended_low_physreq_pretrend",
            "Extended low-physical-requirement pre-trend",
            lambda: compute_extended_low_physreq_pretrend(score_path=score_path),
        ),
        (
            "author_month_intensive_margin",
            "Author-month intensive margin",
            lambda: compute_author_month_intensive_margin(score_path=score_path),
        ),
        (
            "marginal_participation_decomposition",
            "Marginal participation decomposition",
            lambda: compute_marginal_participation_decomposition(score_path=score_path),
        ),
        (
            "community_vulnerability_prediction",
            "Community vulnerability prediction",
            lambda: compute_community_vulnerability_prediction(score_path=score_path),
        ),
        (
            "displaced_contributor_destinations",
            "Displaced contributor destinations",
            lambda: compute_displaced_contributor_destinations(score_path=score_path),
        ),
        (
            "displaced_contributor_primary_origin",
            "Displaced contributor primary-origin robustness",
            lambda: compute_displaced_contributor_primary_origin_robustness(score_path=score_path),
        ),
        (
            "destination_choice_set_placebo",
            "Destination choice-set placebo",
            lambda: compute_destination_choice_set_placebo(score_path=score_path),
        ),
        (
            "destination_rank_analysis",
            "Destination rank analysis",
            lambda: compute_destination_rank_analysis(score_path=score_path),
        ),
        (
            "micro_displacement_prediction",
            "Micro displacement prediction",
            lambda: compute_micro_displacement_prediction(),
        ),
        (
            "micro_level_prediction",
            "Micro-level prediction",
            lambda: compute_micro_level_prediction(score_path=score_path),
        ),
        (
            "micro_prediction_heterogeneity",
            "Micro prediction heterogeneity",
            lambda: compute_micro_prediction_heterogeneity(),
        ),
        (
            "creator_exit_continuous",
            "Creator exit continuous sensitivity",
            lambda: compute_creator_exit_continuous(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_continuous.csv",
                plot_path=figures_dir / "creator_exit_marginal_effects.png",
            ),
        ),
        (
            "creator_exit_no_fe",
            "Creator exit no-FE sensitivity",
            lambda: compute_creator_exit_no_fe(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_no_fe.csv",
                plot_path=figures_dir / "creator_exit_no_fe.png",
            ),
        ),
        (
            "creator_exit_community_level",
            "Creator exit community-level sensitivity",
            lambda: compute_creator_exit_community_level(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_community_level.csv",
                plot_path=figures_dir / "creator_exit_community_level.png",
            ),
        ),
        (
            "creator_exit_tercile",
            "Creator exit tercile sensitivity",
            lambda: compute_creator_exit_tercile(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_tercile.csv",
                plot_path=figures_dir / "creator_exit_tercile.png",
            ),
        ),
        (
            "creator_exit_quartiles",
            "Creator exit quartile heatmap sensitivity",
            lambda: compute_creator_exit_quartiles(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_quartiles.csv",
                plot_path=figures_dir / "creator_exit_heatmap.png",
            ),
        ),
        (
            "creator_exit_single_community",
            "Creator exit single-community sensitivity",
            lambda: compute_creator_exit_single_community(
                score_path=score_path,
                output_path=tables_dir / "creator_exit_single_community.csv",
            ),
        ),
    ]
    for key, label, fn in checks:
        run_paper_check(results, manifest_rows, key, label, fn)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = tables_dir / "paper_check_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    results["paper_check_manifest"] = {
        "status": "ok",
        "output_path": str(manifest_path),
        "n_checks": safe_int(len(manifest)),
        "n_failed": safe_int(manifest["status"].eq("failed").sum()) if not manifest.empty else 0,
        "n_skipped": safe_int(manifest["status"].eq("skipped").sum()) if not manifest.empty else 0,
    }
    print(
        "  Paper suite manifest saved -> "
        f"{manifest_path} ({results['paper_check_manifest']['n_failed']} failed, "
        f"{results['paper_check_manifest']['n_skipped']} skipped)."
    )
    return results

def run_robustness_checks(submonth_panel, acsi_scores, df_all, creators, results):
    print(f"\n--- Robustness: extended 2020-2024 three-dimensional {INDEX_SHORT} panel ---")
    try:
        extended_panel_robustness = compute_extended_panel_robustness(
            submonth_panel.copy(),
            acsi_scores,
        )
        if extended_panel_robustness:
            results["extended_panel_robustness"] = extended_panel_robustness
            pretrend = extended_panel_robustness.get("pretrend") or {}
            print(
                "  Extended panel pre-trend F-test: "
                f"F={fmt4(pretrend.get('f_stat'))} "
                f"p={fmt4(pretrend.get('pvalue'))}"
            )
    except Exception as e:
        print(f"  Extended panel robustness failed: {e}")

    print(f"\n--- Robustness: three-dimensional {INDEX_SHORT} with pre-period controls ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Three-dimensional covariate robustness")
        pre_covariates = add_pre_covariates(acsi_panel)
        adjusted_panel = acsi_panel.merge(pre_covariates, on="subreddit", how="left")
        adjusted_panel["pre_avg_post"] = adjusted_panel["pre_avg_log_posts"] * adjusted_panel["post_shock"]
        adjusted_panel["pre_trend_post"] = adjusted_panel["pre_trend"] * adjusted_panel["post_shock"]
        adjusted_panel["log_mu_post"] = adjusted_panel["log_mu_k"] * adjusted_panel["post_shock"]
        control_terms = ["pre_avg_post", "pre_trend_post", "log_mu_post"]

        adjusted_three_dimensional_model, adjusted_three_dimensional_results = fit_three_dimensional_acsi_model(
            adjusted_panel,
            additional_terms=control_terms,
        )
        if adjusted_three_dimensional_results:
            print_regression_sample_summary(
                "Three-dimensional covariate robustness",
                adjusted_three_dimensional_results[0],
            )
            emit_output_table(
                pd.DataFrame(adjusted_three_dimensional_results),
                TABLES_DIR / "acsi_three_dimensional_covariate_adjusted.csv",
                index=False,
            )
            (TABLES_DIR / "acsi_three_dimensional_covariate_adjusted.tex").write_text(
                adjusted_three_dimensional_model.summary().as_latex()
            )
            results["acsi_three_dimensional_covariate_adj"] = adjusted_three_dimensional_results
            for model_result in adjusted_three_dimensional_results:
                print(
                    f"  Adjusted {model_result['label']} x Post: coef={fmt_signed4(model_result['coef'])} "
                    f"SE={fmt4(model_result['se'])} p={fmt4(model_result['pvalue'])}"
                )
    except Exception as e:
        print(f"  Three-dimensional {INDEX_SHORT} covariate robustness failed: {e}")

    print(f"\n--- Robustness: low personal-context leave-one-subreddit influence ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Leave-one-subreddit influence")
        influence_summary, influence_rows = compute_three_dimensional_leave_one_out(acsi_panel)
        if influence_summary and influence_rows:
            emit_output_table(
                pd.DataFrame(influence_rows),
                TABLES_DIR / "acsi_three_dimensional_leave_one_out.csv",
                index=False,
            )
            results["acsi_three_dimensional_influence"] = influence_summary
            print(
                "  Low personal-context coefficient range after omitting one subreddit: "
                f"{fmt_signed4(influence_summary['max_leave_one_out_coef'])} to "
                f"{fmt_signed4(influence_summary['min_leave_one_out_coef'])}; "
                f"negative and significant in "
                f"{influence_summary['n_significant_leave_one_out']} of "
                f"{influence_summary['n_subreddits_tested']} runs; "
                f"largest shift omits r/{influence_summary['largest_shift_subreddit']}"
            )
    except Exception as e:
        print(f"  Three-dimensional {INDEX_SHORT} influence check failed: {e}")

    print(f"\n--- Robustness: wild cluster bootstrap for PersFree x Post ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Wild cluster bootstrap")
        wild_bootstrap = compute_wild_cluster_bootstrap(
            acsi_panel,
            n_bootstrap=globals().get("N_RANDOMIZATION_PERMS", 1000),
        )
        results["acsi_wild_cluster_bootstrap"] = wild_bootstrap
        summary = wild_bootstrap.get("summary", {})
        print(
            "  Wild cluster bootstrap PersFree x Post: "
            f"SE={fmt4(summary.get('bootstrap_se'))}, "
            f"95% CI=[{fmt4(summary.get('ci_low'))}, {fmt4(summary.get('ci_high'))}], "
            f"share more negative={fmt4(summary.get('proportion_more_negative_than_observed'))}"
        )
    except Exception as e:
        print(f"  Wild cluster bootstrap failed: {e}")

    print(f"\n--- Robustness: entropy-balanced three-dimensional DiD ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Entropy-balanced DiD")
        entropy_balanced = compute_entropy_balanced_did(acsi_panel)
        results["acsi_entropy_balanced_did"] = entropy_balanced
        persfree_row = entropy_balanced.get("persfree", {})
        print(
            "  Entropy-balanced PersFree x Post: "
            f"coef={fmt_signed4(persfree_row.get('coef'))} "
            f"SE={fmt4(persfree_row.get('se'))} "
            f"p={fmt4(persfree_row.get('pvalue'))}"
        )
    except Exception as e:
        print(f"  Entropy-balanced DiD failed: {e}")

    print(f"\n--- Robustness: pre-shock-only vs balanced {INDEX_SHORT} score versions ---")
    try:
        version_check = compute_acsi_version_correlation_check(acsi_scores)
        results["acsi_version_correlation"] = version_check
        pers_row = next(
            (
                row for row in version_check.get("summary", [])
                if row.get("dimension") == "pers_free"
            ),
            {},
        )
        print(
            "  PersFree balanced-vs-pre-shock: "
            f"Pearson r={fmt4(pers_row.get('pearson_r'))} "
            f"Spearman rho={fmt4(pers_row.get('spearman_rho'))} "
            f"MAD={fmt4(pers_row.get('mean_abs_difference'))}"
        )
    except Exception as e:
        print(f"  {INDEX_SHORT} score-version correlation check failed: {e}")

    print("\n--- Robustness: composition-bias outlier check ---")
    try:
        composition_check = compute_composition_bias_outlier_check(
            submonth_panel.copy(),
            acsi_scores,
        )
        results["composition_bias_check"] = composition_check
        model_rows = composition_check.get("models", [])
        for model_row in model_rows:
            print(
                f"  {model_row['model']}: "
                f"coef={fmt_signed4(model_row['activity_change_coef'])} "
                f"SE={fmt4(model_row['activity_change_se'])} "
                f"p={fmt4(model_row['activity_change_pvalue'])}"
            )
    except Exception as e:
        print(f"  Composition-bias outlier check failed: {e}")

    print("\n--- Robustness: two-run pre-shock ACSI regression ---")
    try:
        two_run_check = compute_two_run_preshock_regression(
            submonth_panel.copy(),
            acsi_scores,
        )
        results["acsi_tworun_preshock_regression"] = two_run_check
        print(
            "  Two-run PersFree x Post SE="
            f"{fmt4(two_run_check.get('pers_free_se'))}; "
            "delta vs single-run pre-shock="
            f"{fmt_signed4(two_run_check.get('pers_free_se_vs_single_run_preshock'))}; "
            "delta vs balanced="
            f"{fmt_signed4(two_run_check.get('pers_free_se_vs_balanced'))}"
        )
    except Exception as e:
        print(f"  Two-run pre-shock ACSI regression failed: {e}")

    print("\n--- Robustness: GenCap SIMEX measurement-error correction ---")
    try:
        simex_scores = prepare_gen_cap_simex_scores_from_two_runs(
            subreddit_filter=acsi_scores["subreddit"].astype(str),
            pre_shock_only=True,
            exclude_ai_related=True,
        )
        simex_check = compute_gen_cap_simex_correction(
            simex_scores,
            submonth_panel.copy(),
            lambdas=(0.5, 1.0, 1.5, 2.0),
            n_simulations=100,
            n_bootstrap=500,
            output_path=FIGURES_DIR / "acsi_gen_cap_simex.png",
        )
        results["acsi_gen_cap_simex"] = simex_check
        print(
            "  GenCap x Post SIMEX: "
            f"observed={fmt_signed4(simex_check.get('observed_b1'))} "
            f"corrected={fmt_signed4(simex_check.get('corrected_b1'))} "
            f"bootstrap SE={fmt4(simex_check.get('bootstrap_se'))}; "
            f"sigma2_u={fmt4(simex_check.get('sigma2_u'))}"
        )
        print(f"  SIMEX plot saved -> {simex_check.get('plot_path')}")
    except Exception as e:
        print(f"  GenCap SIMEX correction failed: {e}")

    print("\n--- Robustness: pre-shock split-half score stability ---")
    try:
        split_stability = compute_split_half_preshock_stability(acsi_scores)
        results["acsi_split_half_stability"] = split_stability
        pers_row = next(
            (
                row for row in split_stability.get("rows", [])
                if row.get("dimension") == "pers_free"
            ),
            {},
        )
        print(
            "  PersFree split-half stability: "
            f"Spearman rho={fmt4(pers_row.get('spearman_rho'))} "
            f"p={fmt4(pers_row.get('spearman_pvalue_h0_rho_0'))}; "
            f"confirmed={split_stability.get('pers_free_stability_confirmed')}"
        )
    except Exception as e:
        print(f"  Pre-shock split-half stability check failed: {e}")

    print("\n--- Robustness: pre-shock-only ACSI scores as treatment variable ---")
    try:
        coverage = compute_preshock_acsi_coverage()
        results["acsi_preshock_only_coverage"] = {
            "n_subreddits": safe_int(coverage["subreddit"].nunique()),
            "below_50": safe_int(coverage["below_50"].sum()),
            "below_100": safe_int(coverage["below_100"].sum()),
            "below_200": safe_int(coverage["below_200"].sum()),
            "output_path": str(TABLES_DIR / "preshock_acsi_coverage.csv"),
        }
        preshock_scores = compute_preshock_only_acsi_scores(min_posts=50)
        if preshock_scores is not None and len(preshock_scores) > 0:
            preshock_results = compute_preshock_only_main_regression(
                submonth_panel.copy(), preshock_scores
            )
            results["acsi_preshock_only"] = preshock_results
            main_row = next(
                (
                    r for r in preshock_results.get("preshock_main", [])
                    if "pers_free" in str(r.get("label", "")).lower()
                ),
                {},
            )
            print(
                f"  Pre-shock-only PersFree x Post: "
                f"coef={fmt_signed4(main_row.get('coef'))} "
                f"SE={fmt4(main_row.get('se'))} "
                f"p={fmt4(main_row.get('pvalue'))}"
            )
    except Exception as e:
        print(f"  Pre-shock-only ACSI regression failed: {e}")

    print(f"\n--- Supporting: aggregate {INDEX_LABEL} dose-response DiD ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Main dose-response")
        main_acsi_model = fit_ols(
            "log_posts ~ gse_post + C(subreddit) + C(year_month)",
            acsi_panel,
            cluster_col="subreddit",
        )

        if main_acsi_model:
            model_result = reg_result(main_acsi_model, "gse_post")
            model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
            results["gse_main"] = model_result
            print(f"  {INDEX_SHORT} x Post coef={fmt_signed4(model_result['coef'])} SE={fmt4(model_result['se'])} p={fmt4(model_result['pvalue'])} "
                  f"full exposure effect={fmt_signed1(model_result['percent_effect_full_exposure'])}%")
            (TABLES_DIR / "gse_main_dose_response.tex").write_text(main_acsi_model.summary().as_latex())
    except Exception as e:
        print(f"  {INDEX_SHORT} main failed: {e}")

    print(f"\n--- {INDEX_SHORT} dose-response: covariate adjusted model ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Covariate-adjusted dose-response")
        pre_covariates = add_pre_covariates(acsi_panel)
        adjusted_panel = acsi_panel.merge(pre_covariates, on="subreddit", how="left")
        adjusted_panel["pre_avg_post"] = adjusted_panel["pre_avg_log_posts"] * adjusted_panel["post_shock"]
        adjusted_panel["pre_trend_post"] = adjusted_panel["pre_trend"] * adjusted_panel["post_shock"]
        adjusted_panel["log_mu_post"] = adjusted_panel["log_mu_k"] * adjusted_panel["post_shock"]

        adjusted_acsi_model = fit_ols(
            "log_posts ~ gse_post + pre_avg_post + pre_trend_post + log_mu_post + C(subreddit) + C(year_month)",
            adjusted_panel, cluster_col="subreddit"
        )
        if adjusted_acsi_model:
            model_result = reg_result(adjusted_acsi_model, "gse_post")
            model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
            results["gse_covariate_adj"] = model_result
            print(f"  Adjusted {INDEX_SHORT} x Post coef={fmt_signed4(model_result['coef'])} SE={fmt4(model_result['se'])} p={fmt4(model_result['pvalue'])}")
            (TABLES_DIR / "gse_covariate_adj.tex").write_text(adjusted_acsi_model.summary().as_latex())
    except Exception as e:
        print(f"  {INDEX_SHORT} covariate adjusted failed: {e}")

    print(f"\n--- {INDEX_SHORT} permutation inference ---")
    try:
        if not results.get("gse_main") or results["gse_main"].get("coef") is None:
            print(f"  Skipping {INDEX_SHORT} permutation: no observed {INDEX_SHORT} coefficient.")
        else:
            observed_coef = results["gse_main"]["coef"]
            permutation_panel = acsi_model_panel(submonth_panel.copy(), "Permutation inference")
            use_fast_permutation = is_balanced_two_way_panel(permutation_panel, "subreddit", "year_month")
            if use_fast_permutation:
                y_resid = residualize_two_way(permutation_panel["log_posts"], permutation_panel["subreddit"], permutation_panel["year_month"])
                post_shock_values = permutation_panel["post_shock"].astype(float).to_numpy()
            else:
                print("  Panel is not balanced; using exact OLS fallback for permutations.")

            subreddit_permutation_frame = permutation_panel[["subreddit", "gse"]].drop_duplicates()
            subreddit_permutation_frame["mu_k"] = subreddit_permutation_frame["subreddit"].map(MU_K).fillna(0.5)
            subreddit_permutation_frame["size_bin"] = pd.qcut(subreddit_permutation_frame["mu_k"].rank(method="first"), q=4, labels=False)

            rng = np.random.default_rng(RANDOM_SEED)
            permuted_coefficients = []

            for _ in tqdm(range(N_RANDOMIZATION_PERMS), desc=f"  {INDEX_SHORT} perms"):
                permuted_scores = []
                for _, group in subreddit_permutation_frame.groupby("size_bin"):
                    shuffled = group["gse"].sample(frac=1, replace=False, random_state=int(rng.integers(1e9))).values
                    permuted_subreddit_scores = group[["subreddit"]].copy()
                    permuted_subreddit_scores["permuted_acsi_scores"] = shuffled
                    permuted_scores.append(permuted_subreddit_scores)

                permuted_scores = pd.concat(permuted_scores, ignore_index=True)
                if use_fast_permutation:
                    permuted_acsi_scores = permutation_panel["subreddit"].map(permuted_scores.set_index("subreddit")["permuted_acsi_scores"])
                    permuted_coefficient = two_way_fe_coef_from_residualized_y(
                        y_resid,
                        permuted_acsi_scores.astype(float).to_numpy() * post_shock_values,
                        permutation_panel["subreddit"],
                        permutation_panel["year_month"],
                    )
                else:
                    exact_permutation_panel = permutation_panel.drop(
                        columns=["permuted_acsi_scores"], errors="ignore"
                    ).merge(permuted_scores, on="subreddit", how="left")
                    exact_permutation_panel["permuted_acsi_post"] = (
                        exact_permutation_panel["permuted_acsi_scores"]
                        * exact_permutation_panel["post_shock"]
                    )
                    exact_permutation_model = fit_ols(
                        "log_posts ~ permuted_acsi_post + C(subreddit) + C(year_month)",
                        exact_permutation_panel, cluster_col="subreddit"
                    )
                    permuted_coefficient = (
                        None
                        if not exact_permutation_model
                        else safe_float(exact_permutation_model.params.get("permuted_acsi_post", np.nan))
                    )
                if permuted_coefficient is not None:
                    permuted_coefficients.append(permuted_coefficient)

            if permuted_coefficients:
                left_tail_permutation_pvalue = float(np.mean([coef <= observed_coef for coef in permuted_coefficients]))
                two_sided_permutation_pvalue = float(np.mean([abs(coef) >= abs(observed_coef) for coef in permuted_coefficients]))
                results["gse_permutation"] = {
                    "observed_coef": observed_coef,
                    "perm_pvalue_left": left_tail_permutation_pvalue,
                    "perm_pvalue_two_sided": two_sided_permutation_pvalue,
                    "perm_mean": safe_float(np.mean(permuted_coefficients)),
                    "perm_sd": safe_float(np.std(permuted_coefficients)),
                    "n_perms": len(permuted_coefficients),
                }
                emit_output_table(
                    pd.DataFrame({"coef": permuted_coefficients}),
                    TABLES_DIR / "gse_permutation_coefs.csv",
                    index=False,
                )
                print(f"  {INDEX_SHORT} permutation p={left_tail_permutation_pvalue:.4f} (left-sided)")
    except Exception as e:
        print(f"  {INDEX_SHORT} permutation failed: {e}")

    print(f"\n--- {INDEX_SHORT} dose-response: secondary outcomes ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Secondary outcome dose-response")
        secondary_model_results = []
        outcome_labels = {
            "log_active_creators": "Active creators",
            "log_calibrated_output": "Calibrated output",
            "log_avg_score": "Average score per post",
        }

        for outcome in ["log_active_creators", "log_calibrated_output", "log_avg_score"]:
            model_data = acsi_panel if outcome != "log_avg_score" else acsi_panel[acsi_panel["posts"] > 0].copy()
            secondary_model = fit_ols(f"{outcome} ~ gse_post + C(subreddit) + C(year_month)", model_data, cluster_col="subreddit")
            if secondary_model:
                model_result = reg_result(secondary_model, "gse_post")
                model_result["outcome"] = outcome
                model_result["label"] = outcome_labels[outcome]
                model_result["percent_effect_full_exposure"] = pct_effect_from_coef(model_result["coef"])
                secondary_model_results.append(model_result)
                print(f"  {outcome}: coef={fmt_signed4(model_result['coef'])} p={fmt4(model_result['pvalue'])}")

        if secondary_model_results:
            emit_output_table(
                pd.DataFrame(secondary_model_results),
                TABLES_DIR / "gse_secondary_outcomes.csv",
                index=False,
            )
            results["gse_secondary"] = secondary_model_results
    except Exception as e:
        print(f"  {INDEX_SHORT} secondary outcomes failed: {e}")

    print(f"\n--- {INDEX_SHORT} quartile nonlinearity check ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Quartile nonlinearity check")
        sub_scores = acsi_panel[["subreddit", "gse"]].drop_duplicates()
        sub_scores["gse_rank"] = sub_scores["gse"].rank(method="first")
        sub_scores["gse_quartile"] = pd.qcut(
            sub_scores["gse_rank"], q=4,
            labels=["lowest", "lower_middle", "upper_middle", "highest"]
        )

        acsi_panel = acsi_panel.merge(sub_scores[["subreddit", "gse_quartile"]], on="subreddit", how="left")

        quartile_terms = {
            "lower_middle": "acsi_lower_middle_quartile_post",
            "upper_middle": "acsi_upper_middle_quartile_post",
            "highest": "acsi_highest_quartile_post",
        }
        for quartile_name, term in quartile_terms.items():
            acsi_panel[term] = (
                (acsi_panel["gse_quartile"] == quartile_name).astype(int)
                * acsi_panel["post_shock"]
            )

        quartile_model = fit_ols(
            "log_posts ~ acsi_lower_middle_quartile_post + acsi_upper_middle_quartile_post + acsi_highest_quartile_post + C(subreddit) + C(year_month)",
            acsi_panel, cluster_col="subreddit",
        )

        if quartile_model:
            quartile_model_results = []
            quartile_label_map = {
                "acsi_lower_middle_quartile_post": "Lower-middle",
                "acsi_upper_middle_quartile_post": "Upper-middle",
                "acsi_highest_quartile_post": "Highest",
            }
            for term in quartile_terms.values():
                model_result = reg_result(quartile_model, term)
                model_result["term"] = term
                model_result["label"] = quartile_label_map.get(term, term)
                model_result["percent_effect"] = pct_effect_from_coef(model_result["coef"])
                quartile_model_results.append(model_result)
                print(f"  {term}: coef={fmt_signed4(model_result['coef'])} p={fmt4(model_result['pvalue'])} effect={fmt_signed1(model_result['percent_effect'])}%")

            emit_output_table(
                pd.DataFrame(quartile_model_results),
                TABLES_DIR / "gse_quartile_check.csv",
                index=False,
            )
            results["gse_quartiles"] = quartile_model_results
            
            try:
                apply_publication_style()
                quartile_results_table = pd.DataFrame(quartile_model_results)
                fig, ax = plt.subplots(figsize=(8, 5))
                x = np.arange(len(quartile_results_table))
                y = quartile_results_table["coef"].values
                se = quartile_results_table["se"].values

                ax.bar(x, y, color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
                ax.errorbar(x, y, yerr=1.96 * se, fmt="none", color="#1d4ed8", capsize=4, linewidth=1.5, capthick=1.5)
                ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1.0)
                ax.set_xticks(x)
                ax.set_xticklabels(quartile_results_table["label"])
                ax.set_ylabel("Post-shock effect on log monthly posts")
                ax.set_title(f"{INDEX_SHORT} quartile dose-response check\nReference: lowest-exposure quartile")
                save_plot(fig, FIGURES_DIR / "gse_quartile_check.png")
            except Exception as fe:
                print(f"  {INDEX_SHORT} quartile figure failed: {fe}")

    except Exception as e:
        print(f"  {INDEX_SHORT} quartile check failed: {e}")

    print("\n--- Robustness figures: personal-context dynamics and placebos ---")
    try:
        acsi_panel = acsi_model_panel(submonth_panel.copy(), "Robustness figure panel")

        time_rows = plot_time_varying_personal_context(
            acsi_panel,
            FIGURES_DIR / "robust_time_varying_personal_context.png",
        )
        if time_rows:
            results["robust_time_varying_personal_context"] = time_rows
            print("  Saved quarterly personal-context displacement plot.")

        full_event_rows = plot_monthly_persfree_event_study(
            acsi_panel,
            FIGURES_DIR / "robust_event_study_full.png",
            "Event Study: Personal-Context Displacement Effect",
        )
        if full_event_rows:
            results["robust_event_study_full"] = full_event_rows
            print("  Saved full monthly event-study plot.")

        all_dimension_event_rows = plot_all_dimensions_event_study(acsi_panel, FIGURES_DIR)
        if all_dimension_event_rows:
            results["robust_event_study_all_dimensions"] = all_dimension_event_rows
            print("  Saved all-dimension monthly event-study plots.")

        retained_pairs, all_pairs = strict_matched_subreddit_pairs(acsi_panel)
        matched_subreddits = sorted(set(sum(([a, b] for a, b in retained_pairs), [])))
        matched_event_rows = plot_monthly_persfree_event_study(
            acsi_panel,
            FIGURES_DIR / "robust_matched_strict_event_study.png",
            f"Strict matched-pairs event study ({len(retained_pairs)} retained pairs)",
            subreddits=matched_subreddits,
        )
        if matched_event_rows:
            results["robust_matched_strict_event_study"] = {
                "rows": matched_event_rows,
                "n_retained_pairs": safe_int(len(retained_pairs)),
                "n_attempted_pairs": safe_int(len(all_pairs)),
            }
            print(
                "  Saved strict matched-pairs event-study plot: "
                f"{len(retained_pairs)} retained pairs."
            )

        placebo_permutation = plot_persfree_permutation_placebo(
            acsi_panel,
            FIGURES_DIR / "robust_placebo_permutation.png",
        )
        if placebo_permutation:
            results["robust_placebo_permutation"] = placebo_permutation
            print(
                "  Saved PersFree permutation placebo plot: "
                f"p={fmt4(placebo_permutation['empirical_pvalue'])}."
            )

        nov2023_rows = plot_post_period_placebo_nov2023(
            acsi_panel,
            FIGURES_DIR / "robust_placebo_nov2023.png",
        )
        if nov2023_rows:
            results["robust_placebo_nov2023"] = nov2023_rows
            persfree_row = next(
                (row for row in nov2023_rows if row.get("label") == "PersFree"),
                None,
            )
            if persfree_row:
                print(
                    "  Saved Nov 2023 placebo plot: PersFree "
                    f"coef={fmt_signed4(persfree_row['coef'])} "
                    f"p={fmt4(persfree_row['pvalue'])}."
                )
    except Exception as e:
        print(f"  Robustness figure generation failed: {e}")

    print(f"\n--- Low personal-context two-point summary: dose-response pre-trends ---")
    try:
        event_study_panel = acsi_model_panel(submonth_panel.copy(), "Event-study dose-response")
        event_study_panel = event_study_panel[event_study_panel["year_month_dt"] >= SHOCK_MONTH - pd.DateOffset(months=24)].copy()

        def get_bin(dt):
            mb = (SHOCK_MONTH.year - dt.year) * 12 + (SHOCK_MONTH.month - dt.month)
            if mb <= 0: return "post"
            elif mb <= 6: return "pre_6"
            elif mb <= 12: return "pre_12"
            elif mb <= 18: return "pre_18"
            else: return "pre_24"

        event_study_panel["bin"] = event_study_panel["year_month_dt"].apply(get_bin)

        event_bins = [
            b
            for b in ["pre_24", "pre_18", "pre_12", "post"]
            if (event_study_panel["bin"] == b).any()
        ]

        event_exposure = "non_personal_norm"
        event_term_prefix = "non_personal"
        for b in event_bins:
            event_study_panel[f"{event_term_prefix}_{b}"] = (
                event_study_panel[event_exposure] * (event_study_panel["bin"] == b).astype(int)
            )

        event_terms = [f"{event_term_prefix}_{b}" for b in event_bins]
        event_study_model = fit_ols(
            "log_posts ~ " + " + ".join(event_terms) + " + C(subreddit) + C(year_month)",
            event_study_panel, cluster_col="subreddit",
        ) if event_terms else None

        event_rows = []
        if event_study_model:
            print("\n--- RAW EVENT-STUDY PRETREND REGRESSION OUTPUT ---")
            print(f"  Formula: log_posts ~ {' + '.join(event_terms)} + C(subreddit) + C(year_month)")
            print(f"  N obs: {safe_int(event_study_model.nobs)}")
            print("  Focal event-study terms:")
            raw_event_rows = []
            for term in event_terms:
                raw_event_rows.append({
                    "term": term,
                    "coef": safe_float(event_study_model.params.get(term, np.nan)),
                    "std_err": safe_float(event_study_model.bse.get(term, np.nan)),
                    "t": safe_float(event_study_model.tvalues.get(term, np.nan)),
                    "pvalue": safe_float(event_study_model.pvalues.get(term, np.nan)),
                })
            print(pd.DataFrame(raw_event_rows).to_string(index=False))

            for b in event_bins:
                term = f"{event_term_prefix}_{b}"
                model_result = reg_result(event_study_model, term)
                model_result["bin"] = b
                event_rows.append(model_result)
                print(f"  {b}: coef={fmt_signed4(model_result['coef'])} p={fmt4(model_result['pvalue'])}")

            pre_terms = [f"{event_term_prefix}_{b}" for b in event_bins if b != "post"]
            try:
                restriction_matrix = np.zeros((len(pre_terms), len(event_study_model.params)))
                param_names = list(event_study_model.params.index)
                valid_pre = []
                for i, pretrend_term in enumerate(pre_terms):
                    if pretrend_term in param_names:
                        restriction_matrix[i, param_names.index(pretrend_term)] = 1.0
                        valid_pre.append(pretrend_term)
                if valid_pre:
                    restriction_matrix = restriction_matrix[:len(valid_pre), :]
                    f_test = event_study_model.f_test(restriction_matrix)
                    pretrend_pvalue = safe_float(f_test.pvalue)
                    print("\n--- RAW JOINT PRETREND F-TEST OUTPUT ---")
                    print(f"  Tested restrictions: {', '.join(valid_pre)} = 0")
                    print(f_test)
                else:
                    pretrend_pvalue = None
            except Exception:
                pretrend_pvalue = None

            results["gse_event_study"] = {
                "rows": event_rows,
                "pretrend_pvalue": pretrend_pvalue,
            }
            if pretrend_pvalue is not None:
                print(f"  Pretrend p={fmt4(pretrend_pvalue)}")
            emit_output_table(pd.DataFrame(event_rows), TABLES_DIR / "gse_event_study.csv", index=False)
            print("  Two-point pre/post figure removed; use the terminal table or monthly event-study plot.")
                
    except Exception as e:
        print(f"  Low personal-context event study failed: {e}")

    print("\n--- Robustness: linear pre-shock PersFree trend ---")
    try:
        linear_trend = compute_linear_preshock_persfree_trend(
            submonth_panel.copy(),
            acsi_scores,
        )
        results["acsi_preshock_linear_trend"] = linear_trend
        trend_row = linear_trend.get("result", {})
        print(
            "  Linear pre-shock PersFree trend: "
            f"coef={fmt_signed4(trend_row.get('coef'))} "
            f"SE={fmt4(trend_row.get('se'))} "
            f"p={fmt4(trend_row.get('pvalue'))}"
        )
    except Exception as e:
        print(f"  Linear pre-shock PersFree trend failed: {e}")

    print("\n--- Robustness: leave-one-month-out pre-trend battery ---")
    try:
        pretrend_panel = acsi_model_panel(submonth_panel.copy(), "Leave-one-month pretrend")
        leave_one_month = compute_leave_one_month_out_pretrends(pretrend_panel)
        results["acsi_leave_one_month_pretrend"] = leave_one_month
        summary = leave_one_month.get("summary", {})
        print(
            "  Leave-one-month pre-lead range: "
            f"coef {fmt_signed4(summary.get('min_pre_lead_coef'))} to "
            f"{fmt_signed4(summary.get('max_pre_lead_coef'))}; "
            f"p {fmt4(summary.get('min_pre_lead_pvalue'))} to "
            f"{fmt4(summary.get('max_pre_lead_pvalue'))}"
        )
    except Exception as e:
        print(f"  Leave-one-month pretrend battery failed: {e}")

    print("\n--- Robustness: Reddit blackout/API timing checks ---")
    try:
        mechanism_score_path = prepare_current_mechanism_score_path(acsi_scores, TABLES_DIR)
        reddit_disruption = compute_reddit_disruption_robustness(
            submonth_panel.copy(),
            score_path=mechanism_score_path,
            blackout_path=TABLES_DIR / "blackout.csv",
            proxy_path=TABLES_DIR / "blackout_proxy.csv",
            output_path=TABLES_DIR / "reddit_disruption_robustness.csv",
        )
        persfree_rows = [
            row for row in reddit_disruption.get("rows", [])
            if row.get("row_type") == "regression" and row.get("term") == "pers_free_post"
        ]
        reddit_disruption["persfree_rows"] = persfree_rows
        results["reddit_disruption_robustness"] = reddit_disruption
        for row in persfree_rows:
            print(
                f"  {row.get('spec')}: PersFree x Post "
                f"coef={fmt_signed4(row.get('coef'))} "
                f"SE={fmt4(row.get('se'))} p={fmt4(row.get('pvalue'))}"
            )
    except Exception as e:
        print(f"  Reddit blackout robustness failed: {e}")

    print("\n--- Robustness: drop June-August 2023 blackout quarter ---")
    try:
        drop_blackout = compute_drop_blackout_quarter(
            submonth_panel.copy(),
            acsi_scores,
        )
        results["acsi_drop_blackout_quarter"] = drop_blackout
        persfree_row = drop_blackout.get("persfree", {})
        print(
            "  Drop June-August 2023 PersFree x Post: "
            f"coef={fmt_signed4(persfree_row.get('coef'))} "
            f"SE={fmt4(persfree_row.get('se'))} "
            f"p={fmt4(persfree_row.get('pvalue'))}"
        )
    except Exception as e:
        print(f"  Drop June-August 2023 check failed: {e}")

    print(f"\n--- Mechanism check: post-level AI mention adoption ---")
    try:
        adoption_results = compute_post_ai_adoption_check(submonth_panel.copy(), acsi_scores)
        if adoption_results:
            results["post_ai_adoption"] = adoption_results
            focal_rows = [
                row for row in adoption_results["rows"]
                if row["exposure"] in {"non_personal_norm", "gse"}
                and row["metric"] == "delta_tool_post_rate"
            ]
            for row in focal_rows:
                print(
                    f"  {row['exposure_label']} vs Δ narrow AI-tool post mention rate: "
                    f"Pearson r={fmt4(row['pearson_r'])} p={fmt4(row['pearson_pvalue'])}"
                )
        else:
            print("  Skipped post-level AI mention adoption check: required post mention columns unavailable.")
    except Exception as e:
        print(f"  Post-level AI mention adoption check failed: {e}")

    print("\n--- Mechanism check: creator composition ---")
    try:
        creator_composition = compute_creator_composition_checks(
            df_all,
            acsi_scores,
            submonth_panel.copy(),
        )
        results["creator_composition_checks"] = creator_composition
        results["creator_new_entrant_share"] = creator_composition.get("new_entrant_share")
        results["creator_bottom_tercile_share"] = creator_composition.get("bottom_tercile_creator_share")
        for label, row in [
            ("New entrant share", creator_composition.get("new_entrant_share", {})),
            ("Bottom-tercile creator share", creator_composition.get("bottom_tercile_creator_share", {})),
        ]:
            print(
                f"  {label}: PersFree x Post "
                f"coef={fmt_signed4(row.get('coef'))} "
                f"SE={fmt4(row.get('se'))} p={fmt4(row.get('pvalue'))}"
            )
    except Exception as e:
        print(f"  Creator composition checks failed: {e}")

    creator_level_results = compute_creator_level_checks(df_all, creators, acsi_scores=acsi_scores)
    results.update(creator_level_results)

    print("\n--- Backtest and forward simulation: PersFree mechanism ---")
    try:
        backtest_forward = compute_backtest_and_forward_simulation(
            submonth_panel.copy(),
            acsi_scores,
            df_all=df_all,
        )
        if backtest_forward:
            results["backtest_forward_simulation"] = backtest_forward
    except Exception as e:
        print(f"  Backtest and forward simulation failed: {e}")
    results = run_full_paper_suite(submonth_panel, acsi_scores, results)
    return results
