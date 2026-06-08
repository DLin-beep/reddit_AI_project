"""Supplementary subreddit-month measurement outcomes.

This module is intentionally separate from robustness_checks.py. It computes
additional measurement outcomes from the ecosystem-clean post sample and runs
two-way fixed-effect DiD regressions against PersFree x Post.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_root():
    script_dir = Path(__file__).parent.resolve()
    if script_dir.name == "scripts":
        return Path(__file__).parent.parent.resolve()
    return script_dir


ROOT = resolve_root()
DEFAULT_OUTPUT_DIR = ROOT / "output" / "latest"
DEFAULT_TABLES_DIR = DEFAULT_OUTPUT_DIR / "tables"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"
DEFAULT_POSTS_PATH = DEFAULT_OUTPUT_DIR / "posts_clean_ecosystem.parquet"
DEFAULT_MAIN_PANEL_PATH = DEFAULT_OUTPUT_DIR / "subreddit_month_gse_panel.parquet"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_RAW_DATA_DIR = DEFAULT_DATA_DIR / "raw_files"
DEFAULT_SCORE_CANDIDATES = [
    DEFAULT_TABLES_DIR / "acsi_preshock_tworuns.csv",
    DEFAULT_TABLES_DIR / "acsi_scores_computed.csv",
    ROOT / "data" / "acsi_scores.csv",
    ROOT / "data" / "acsi_preshock_tworuns.csv",
]

START_MONTH = pd.Timestamp("2022-01-01")
END_MONTH = pd.Timestamp("2024-12-01")
POST_MONTH = "2022-12"
EVENT_REFERENCE_MONTH = "2022-10"
EVENT_SHOCK_MONTH = "2022-11"
EXCLUDED_AUTHORS = {"[deleted]", "[removed]", "AutoModerator", ""}

OUTCOME_SPECS = [
    ("log_unique_posters", "log(1 + unique active posters)"),
    ("posts_per_author", "Posts per active author"),
    ("new_poster_share", "New poster share"),
    ("avg_post_length", "Average post length"),
    ("log_total_words", "log(1 + total words)"),
    ("author_survival_rate", "Author survival rate"),
    ("hhi", "HHI of posting concentration"),
]
ROBUSTNESS_OUTCOMES = [
    ("log_unique_posters", "log(1 + unique active posters)"),
    ("log_total_words", "log(1 + total words)"),
]
ROBUSTNESS_ITERATIONS = 1000
ROBUSTNESS_RANDOM_SEED = 20260606

MODERN_GRID_COLOR = "#e5e7eb"
MODERN_SPINE_COLOR = "#d1d5db"
MODERN_TEXT_DARK = "#1a1a2e"
MODERN_AXIS_LABEL = "#2d2d2d"
MODERN_TICK_LABEL = "#4a4a4a"
MUTED_BLUE = "#2563eb"
MUTED_GRAY = "#9ca3af"


def apply_publication_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.titlecolor": MODERN_TEXT_DARK,
        "axes.labelsize": 12,
        "axes.labelweight": "medium",
        "axes.labelcolor": MODERN_AXIS_LABEL,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": MODERN_TICK_LABEL,
        "ytick.color": MODERN_TICK_LABEL,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": MODERN_GRID_COLOR,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def apply_modern_style(ax):
    ax.set_facecolor("#ffffff")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=MODERN_GRID_COLOR, linewidth=0.5, linestyle="--", zorder=0)
    ax.grid(False, axis="x")
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["bottom", "left"]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(MODERN_SPINE_COLOR)
        ax.spines[spine_name].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=MODERN_TICK_LABEL, labelsize=10, width=0.8)
    ax.title.set_color(MODERN_TEXT_DARK)
    ax.xaxis.label.set_color(MODERN_AXIS_LABEL)
    ax.yaxis.label.set_color(MODERN_AXIS_LABEL)
    return ax


apply_publication_style()

def safe_float(value):
    try:
        if value is None:
            return None
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    except Exception:
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def fmt4(value):
    return "NA" if value is None else f"{value:.4f}"


def fmt_signed4(value):
    return "NA" if value is None else f"{value:+.4f}"


def fit_ols(formula, data, cluster_col=None):
    if data is None or data.empty:
        return None
    try:
        model = smf.ols(formula, data=data)
        if cluster_col is not None:
            return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        return model.fit(cov_type="HC3")
    except Exception as exc:
        print(f"  OLS failed for {formula}: {exc}")
        return None


def reg_result(model, term):
    if model is None:
        return {"coef": None, "se": None, "pvalue": None, "n_obs": 0}
    return {
        "coef": safe_float(model.params.get(term, np.nan)),
        "se": safe_float(model.bse.get(term, np.nan)),
        "pvalue": safe_float(model.pvalues.get(term, np.nan)),
        "n_obs": safe_int(model.nobs),
    }


def term_result(model, term, prefix):
    result = reg_result(model, term)
    return {
        f"{prefix}_coef": result["coef"],
        f"{prefix}_se": result["se"],
        f"{prefix}_pvalue": result["pvalue"],
    }


def scalar_stat(value):
    try:
        array = np.asarray(value)
        if array.size == 1:
            return safe_float(array.reshape(-1)[0])
        return safe_float(value)
    except Exception:
        return safe_float(value)


def month_term(month):
    return f"event_{str(month).replace('-', '_')}"


def sorted_panel_months(panel):
    return sorted(pd.Series(panel["year_month"].dropna().astype(str).unique()).tolist())


def normalized_month_strings(frame):
    if "year_month" in frame.columns:
        return pd.to_datetime(frame["year_month"], errors="coerce").dt.to_period("M").astype(str)
    if "year_month_dt" in frame.columns:
        return pd.to_datetime(frame["year_month_dt"], errors="coerce").dt.to_period("M").astype(str)
    if "month" in frame.columns:
        return pd.to_datetime(frame["month"], errors="coerce").dt.to_period("M").astype(str)
    raise ValueError("Panel diagnostics require year_month, year_month_dt, or month.")


def print_exposure_diagnostics(frame, label):
    print(f"\n{label} diagnostics:")
    if "subreddit" not in frame.columns:
        print("  subreddit column missing")
        return
    if "non_personal_norm" not in frame.columns:
        print("  non_personal_norm column missing")
        return
    months = normalized_month_strings(frame)
    exposure = pd.to_numeric(frame["non_personal_norm"], errors="coerce").dropna()
    print(f"  unique subreddits: {frame['subreddit'].nunique()}")
    print(f"  unique months: {months.dropna().nunique()}")
    print(
        "  non_personal_norm: "
        f"mean={fmt4(safe_float(exposure.mean()))} "
        f"SD={fmt4(safe_float(exposure.std()))} "
        f"min={fmt4(safe_float(exposure.min()))} "
        f"max={fmt4(safe_float(exposure.max()))}"
    )


def print_panel_diagnostics(panel, output_dir=None):
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    print_exposure_diagnostics(panel, "Additional measurement panel")
    main_panel_path = output_dir / "subreddit_month_gse_panel.parquet"
    if not main_panel_path.exists() and DEFAULT_MAIN_PANEL_PATH.exists():
        main_panel_path = DEFAULT_MAIN_PANEL_PATH
    if not main_panel_path.exists():
        print(f"\nMain paper panel diagnostics skipped: file not found -> {main_panel_path}")
        return
    main_panel = pd.read_parquet(main_panel_path)
    print_exposure_diagnostics(main_panel, f"Main paper panel ({main_panel_path})")


def add_event_terms(frame, months, exposure_col="non_personal_norm"):
    frame = frame.copy()
    for month in months:
        frame[month_term(month)] = frame[exposure_col] * frame["year_month"].eq(month).astype(float)
    return frame


def fit_event_study_model(panel, months, reference_month=EVENT_REFERENCE_MONTH):
    event_months = [month for month in months if month != reference_month]
    model_data = panel.dropna(
        subset=["log_unique_posters", "non_personal_norm", "subreddit", "year_month"]
    ).copy()
    if model_data.empty:
        return None, event_months, model_data
    model_data["year_month"] = model_data["year_month"].astype(str)
    model_data = add_event_terms(model_data, event_months)
    formula = (
        "log_unique_posters ~ "
        + " + ".join(month_term(month) for month in event_months)
        + " + C(subreddit) + C(year_month)"
    )
    return fit_ols(formula, model_data, cluster_col="subreddit"), event_months, model_data


def event_study_table_from_model(model, months, reference_month=EVENT_REFERENCE_MONTH):
    rows = []
    conf = model.conf_int() if model is not None else pd.DataFrame()
    for month in months:
        term = month_term(month)
        if month == reference_month:
            coef = se = pvalue = 0.0
            ci_low = ci_high = 0.0
        elif model is None or term not in model.params:
            coef = se = pvalue = ci_low = ci_high = None
        else:
            coef = safe_float(model.params.get(term))
            se = safe_float(model.bse.get(term))
            pvalue = safe_float(model.pvalues.get(term))
            ci_low = safe_float(conf.loc[term, 0]) if term in conf.index else None
            ci_high = safe_float(conf.loc[term, 1]) if term in conf.index else None
        rows.append({
            "year_month": month,
            "term": term,
            "coef": coef,
            "se": se,
            "pvalue": pvalue,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "reference_month": month == reference_month,
        })
    return pd.DataFrame(rows)


def run_event_study_unique_posters(panel, output_dir=None):
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    months = sorted_panel_months(panel)
    model, _, _ = fit_event_study_model(panel, months, reference_month=EVENT_REFERENCE_MONTH)
    event_table = event_study_table_from_model(model, months, reference_month=EVENT_REFERENCE_MONTH)

    plot_data = event_table.dropna(subset=["coef", "ci_low", "ci_high"]).copy()
    plot_data["month_dt"] = pd.to_datetime(plot_data["year_month"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axhline(0, color=MUTED_GRAY, linewidth=1.0)
    ax.axvline(pd.Timestamp(f"{EVENT_SHOCK_MONTH}-01"), color=MUTED_GRAY, linestyle="--", linewidth=1.0)
    ax.plot(plot_data["month_dt"], plot_data["coef"], color=MUTED_BLUE, linewidth=1.8, marker="o", markersize=3.5)
    ax.fill_between(
        plot_data["month_dt"],
        plot_data["ci_low"],
        plot_data["ci_high"],
        color=MUTED_BLUE,
        alpha=0.16,
        linewidth=0,
    )
    ax.set_title("Event Study: Unique Active Posters")
    ax.set_xlabel("Month")
    ax.set_ylabel("PersFree interaction coefficient")
    ax.tick_params(axis="x", rotation=45)
    apply_modern_style(ax)
    fig.tight_layout()
    output_path = figures_dir / "event_study_unique_posters.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Event-study figure saved -> {output_path}")
    return event_table


def run_pretrend_test_unique_posters(panel):
    pre_months = [
        month for month in sorted_panel_months(panel)
        if month < EVENT_SHOCK_MONTH
    ]
    print("\nPre-trend F-test coverage: log(1 + unique active posters)")
    print(f"  pre-period months: {', '.join(pre_months) if pre_months else 'none'}")
    coverage = panel[panel["year_month"].astype(str).isin(pre_months)].copy()
    if coverage.empty:
        print("  subreddit counts by month: none")
    else:
        month_counts = (
            coverage.groupby("year_month", sort=True)["subreddit"]
            .nunique()
            .reindex(pre_months)
            .fillna(0)
            .astype(int)
        )
        print("  subreddit counts by month:")
        for month, count in month_counts.items():
            print(f"    {month}: {count}")
    if EVENT_REFERENCE_MONTH not in pre_months:
        print("  Pre-trend test skipped: October 2022 reference month is absent.")
        return {
            "model": "event_study_pretrend",
            "outcome": "log_unique_posters",
            "label": "Unique active posters pre-trend test",
            "pretrend_f_stat": None,
            "pretrend_pvalue": None,
            "pretrend_linear_slope": None,
            "pretrend_linear_se": None,
            "pretrend_linear_pvalue": None,
            "n_obs": 0,
            "n_subreddits": 0,
        }

    model, event_months, model_data = fit_event_study_model(
        panel[panel["year_month"].astype(str).isin(pre_months)].copy(),
        pre_months,
        reference_month=EVENT_REFERENCE_MONTH,
    )
    f_stat = pvalue = None
    if model is not None and event_months:
        restrictions = ", ".join(f"{month_term(month)} = 0" for month in event_months)
        try:
            f_test = model.f_test(restrictions)
            f_stat = scalar_stat(f_test.fvalue)
            pvalue = scalar_stat(f_test.pvalue)
        except Exception as exc:
            print(f"  Pre-trend joint F-test failed: {exc}")

    event_table = event_study_table_from_model(model, pre_months, reference_month=EVENT_REFERENCE_MONTH)
    trend_data = event_table.dropna(subset=["coef"]).copy()
    trend_slope = trend_se = trend_pvalue = None
    if len(trend_data) >= 3 and trend_data["coef"].nunique() > 1:
        trend_data = trend_data.sort_values("year_month").reset_index(drop=True)
        trend_data["time_index"] = np.arange(len(trend_data), dtype=float)
        trend_model = smf.ols("coef ~ time_index", data=trend_data).fit()
        trend_slope = safe_float(trend_model.params.get("time_index"))
        trend_se = safe_float(trend_model.bse.get("time_index"))
        trend_pvalue = safe_float(trend_model.pvalues.get("time_index"))

    print("\nPre-trend diagnostics: log(1 + unique active posters)")
    print(f"  Joint pre-shock F-test: F={fmt4(f_stat)} p={fmt4(pvalue)}")
    print(
        "  Linear pre-shock event-coefficient trend: "
        f"slope={fmt_signed4(trend_slope)} SE={fmt4(trend_se)} p={fmt4(trend_pvalue)}"
    )
    return {
        "model": "event_study_pretrend",
        "outcome": "log_unique_posters",
        "label": "Unique active posters pre-trend test",
        "pretrend_f_stat": f_stat,
        "pretrend_pvalue": pvalue,
        "pretrend_linear_slope": trend_slope,
        "pretrend_linear_se": trend_se,
        "pretrend_linear_pvalue": trend_pvalue,
        "n_obs": safe_int(len(model_data)),
        "n_subreddits": safe_int(model_data["subreddit"].nunique()) if not model_data.empty else 0,
    }


def add_panel_pre_covariates(panel):
    cov_panel = panel.copy()
    cov_panel["year_month_dt"] = pd.to_datetime(cov_panel["year_month"], errors="coerce")
    cov_panel["log_posts"] = np.log1p(pd.to_numeric(cov_panel["n_posts"], errors="coerce"))
    pre = cov_panel[cov_panel["post"].eq(0)].dropna(subset=["subreddit", "year_month_dt", "log_posts"]).copy()
    if pre.empty:
        return cov_panel

    pre_avg = pre.groupby("subreddit")["log_posts"].mean().rename("pre_avg_log_posts")
    pre_mu = pre.groupby("subreddit")["n_posts"].mean().rename("mu_k")
    min_month = pre["year_month_dt"].min()
    pre["t"] = (
        (pre["year_month_dt"].dt.year - min_month.year) * 12
        + (pre["year_month_dt"].dt.month - min_month.month)
    )
    trend_rows = []
    for subreddit, group in pre.groupby("subreddit"):
        if len(group) >= 6 and group["log_posts"].nunique() > 1:
            slope = float(np.polyfit(group["t"], group["log_posts"], 1)[0])
        else:
            slope = 0.0
        trend_rows.append({"subreddit": subreddit, "pre_trend": slope})

    covariates = (
        pre_avg.reset_index()
        .merge(pre_mu.reset_index(), on="subreddit", how="left")
        .merge(pd.DataFrame(trend_rows), on="subreddit", how="left")
    )
    covariates["mu_k"] = pd.to_numeric(covariates["mu_k"], errors="coerce").fillna(0.0)
    covariates["log_mu_k"] = np.log1p(covariates["mu_k"])
    cov_panel = cov_panel.merge(
        covariates[["subreddit", "pre_avg_log_posts", "pre_trend", "log_mu_k"]],
        on="subreddit",
        how="left",
    )
    cov_panel["pre_avg_post"] = cov_panel["pre_avg_log_posts"] * cov_panel["post"]
    cov_panel["pre_trend_post"] = cov_panel["pre_trend"] * cov_panel["post"]
    cov_panel["log_mu_post"] = cov_panel["log_mu_k"] * cov_panel["post"]
    return cov_panel


def run_three_dimensional_did(frame, outcome, model_name, label, group=None):
    required = {
        outcome,
        "generation_capability_norm",
        "physical_free_norm",
        "non_personal_norm",
        "post",
        "subreddit",
        "year_month",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Three-dimensional DiD frame missing columns: {sorted(missing)}")
    model_data = frame.dropna(subset=sorted(required)).copy()
    model_data["generation_capability_post"] = model_data["generation_capability_norm"] * model_data["post"]
    model_data["physical_free_post"] = model_data["physical_free_norm"] * model_data["post"]
    model_data["non_personal_post"] = model_data["non_personal_norm"] * model_data["post"]
    terms = ["generation_capability_post", "physical_free_post", "non_personal_post"]
    if model_data.empty or model_data["subreddit"].nunique() < 2:
        result = {"coef": None, "se": None, "pvalue": None, "n_obs": safe_int(len(model_data))}
        gen = {"generation_capability_coef": None, "generation_capability_se": None, "generation_capability_pvalue": None}
        phys = {"physical_free_coef": None, "physical_free_se": None, "physical_free_pvalue": None}
    else:
        model = fit_ols(
            f"{outcome} ~ {' + '.join(terms)} + C(subreddit) + C(year_month)",
            model_data,
            cluster_col="subreddit",
        )
        result = reg_result(model, "non_personal_post")
        gen = term_result(model, "generation_capability_post", "generation_capability")
        phys = term_result(model, "physical_free_post", "physical_free")
    row = {
        "model": model_name,
        "outcome": outcome,
        "label": label,
        "group": group,
        "coef": result["coef"],
        "se": result["se"],
        "pvalue": result["pvalue"],
        "n_obs": result["n_obs"],
        "n_subreddits": safe_int(model_data["subreddit"].nunique()) if not model_data.empty else 0,
    }
    row.update(gen)
    row.update(phys)
    return row


def prepare_three_dimensional_model_data(frame, outcome, nonpersonal_col="non_personal_norm"):
    required = {
        outcome,
        nonpersonal_col,
        "generation_capability_norm",
        "physical_free_norm",
        "post",
        "subreddit",
        "year_month",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Three-dimensional model data missing columns: {sorted(missing)}")
    model_data = frame.dropna(subset=sorted(required)).copy()
    model_data["generation_capability_post"] = (
        pd.to_numeric(model_data["generation_capability_norm"], errors="coerce") * model_data["post"]
    )
    model_data["physical_free_post"] = (
        pd.to_numeric(model_data["physical_free_norm"], errors="coerce") * model_data["post"]
    )
    model_data["non_personal_post"] = (
        pd.to_numeric(model_data[nonpersonal_col], errors="coerce") * model_data["post"]
    )
    model_data = model_data.dropna(
        subset=[outcome, "generation_capability_post", "physical_free_post", "non_personal_post"]
    )
    return model_data


def fit_three_dimensional_model(frame, outcome, nonpersonal_col="non_personal_norm"):
    model_data = prepare_three_dimensional_model_data(
        frame, outcome, nonpersonal_col=nonpersonal_col
    )
    terms = ["generation_capability_post", "physical_free_post", "non_personal_post"]
    formula = f"{outcome} ~ {' + '.join(terms)} + C(subreddit) + C(year_month)"
    return fit_ols(formula, model_data, cluster_col="subreddit"), model_data


def residualize_two_way(values, group_a, group_b, max_iter=100, tol=1e-10):
    array = np.asarray(values, dtype=float)
    is_vector = array.ndim == 1
    if is_vector:
        array = array.reshape(-1, 1)
    residual = array.copy()
    group_a = pd.Series(group_a).reset_index(drop=True)
    group_b = pd.Series(group_b).reset_index(drop=True)
    for _ in range(max_iter):
        previous = residual.copy()
        residual -= pd.DataFrame(residual).groupby(group_a, sort=False).transform("mean").to_numpy()
        residual -= pd.DataFrame(residual).groupby(group_b, sort=False).transform("mean").to_numpy()
        if np.nanmax(np.abs(residual - previous)) < tol:
            break
    return residual[:, 0] if is_vector else residual


def fit_three_dimensional_fast(frame, outcome, nonpersonal_col="non_personal_norm"):
    model_data = prepare_three_dimensional_model_data(
        frame, outcome, nonpersonal_col=nonpersonal_col
    )
    term_columns = ["generation_capability_post", "physical_free_post", "non_personal_post"]
    y = pd.to_numeric(model_data[outcome], errors="coerce").to_numpy(dtype=float)
    x = model_data[term_columns].to_numpy(dtype=float)
    y_resid = residualize_two_way(y, model_data["subreddit"], model_data["year_month"])
    x_resid = residualize_two_way(x, model_data["subreddit"], model_data["year_month"])
    beta = np.linalg.lstsq(x_resid, y_resid, rcond=None)[0]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        resid = y_resid - x_resid @ beta
    return {
        "model_data": model_data,
        "terms": term_columns,
        "beta": beta,
        "persfree_coef": safe_float(beta[term_columns.index("non_personal_post")]),
        "resid": resid,
        "y_resid": y_resid,
        "x_resid": x_resid,
        "clusters": model_data["subreddit"].astype(str).to_numpy(),
    }


def wild_cluster_bootstrap_persfree(panel, outcome, n_iter=ROBUSTNESS_ITERATIONS, seed=ROBUSTNESS_RANDOM_SEED):
    fit = fit_three_dimensional_fast(panel, outcome)
    x_resid = fit["x_resid"]
    beta_hat = fit["beta"]
    resid = fit["resid"]
    clusters = fit["clusters"]
    unique_clusters = pd.Index(pd.unique(clusters))
    cluster_codes = unique_clusters.get_indexer(clusters)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        xtx_inv_xt = np.linalg.pinv(x_resid.T @ x_resid) @ x_resid.T
    rng = np.random.default_rng(seed)
    boot_coefs = np.empty(n_iter, dtype=float)
    null_coefs = np.empty(n_iter, dtype=float)
    persfree_index = fit["terms"].index("non_personal_post")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        fitted = x_resid @ beta_hat
    restricted_columns = [index for index in range(x_resid.shape[1]) if index != persfree_index]
    x_restricted = x_resid[:, restricted_columns]
    beta_restricted = np.linalg.lstsq(x_restricted, fit["y_resid"], rcond=None)[0]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        restricted_fitted = x_restricted @ beta_restricted
    restricted_resid = fit["y_resid"] - restricted_fitted
    for iteration in range(n_iter):
        weights = rng.choice([-1.0, 1.0], size=len(unique_clusters))
        y_boot = fitted + resid * weights[cluster_codes]
        beta_boot = xtx_inv_xt @ y_boot
        boot_coefs[iteration] = beta_boot[persfree_index]
        y_null = restricted_fitted + restricted_resid * weights[cluster_codes]
        beta_null = xtx_inv_xt @ y_null
        null_coefs[iteration] = beta_null[persfree_index]
    observed = fit["persfree_coef"]
    pvalue = safe_float(np.mean(np.abs(null_coefs) >= abs(observed)))
    ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])
    return {
        "wild_bootstrap_pvalue": pvalue,
        "wild_bootstrap_ci_low": safe_float(ci_low),
        "wild_bootstrap_ci_high": safe_float(ci_high),
        "wild_bootstrap_iterations": safe_int(n_iter),
    }


def leave_one_subreddit_out(panel, outcome):
    coefficients = []
    pvalues = []
    omitted_subreddits = sorted(panel["subreddit"].astype(str).dropna().unique(), key=str.lower)
    for omitted in omitted_subreddits:
        model, model_data = fit_three_dimensional_model(
            panel[panel["subreddit"].astype(str).ne(omitted)].copy(),
            outcome,
        )
        result = reg_result(model, "non_personal_post")
        coefficients.append(result["coef"])
        pvalues.append(result["pvalue"])
    coefficient_series = pd.Series(coefficients, dtype="float64")
    pvalue_series = pd.Series(pvalues, dtype="float64")
    return {
        "loo_min_coef": safe_float(coefficient_series.min()),
        "loo_max_coef": safe_float(coefficient_series.max()),
        "loo_significant_count": safe_int((pvalue_series < 0.05).sum()),
        "loo_runs": safe_int(len(omitted_subreddits)),
    }


def permutation_placebo_persfree(panel, outcome, true_coef, n_iter=ROBUSTNESS_ITERATIONS, seed=ROBUSTNESS_RANDOM_SEED):
    base = prepare_three_dimensional_model_data(panel, outcome)
    y = pd.to_numeric(base[outcome], errors="coerce").to_numpy(dtype=float)
    y_resid = residualize_two_way(y, base["subreddit"], base["year_month"])
    post = pd.to_numeric(base["post"], errors="coerce").to_numpy(dtype=float)
    gen_post = pd.to_numeric(base["generation_capability_norm"], errors="coerce").to_numpy(dtype=float) * post
    phys_post = pd.to_numeric(base["physical_free_norm"], errors="coerce").to_numpy(dtype=float) * post
    subreddit_scores = (
        base[["subreddit", "non_personal_norm"]]
        .drop_duplicates("subreddit")
        .sort_values("subreddit")
        .reset_index(drop=True)
    )
    subreddits = subreddit_scores["subreddit"].astype(str).to_numpy()
    scores = pd.to_numeric(subreddit_scores["non_personal_norm"], errors="coerce").to_numpy(dtype=float)
    subreddit_codes = pd.Index(subreddits).get_indexer(base["subreddit"].astype(str))
    rng = np.random.default_rng(seed)
    permuted_coefs = np.empty(n_iter, dtype=float)
    for iteration in range(n_iter):
        permuted_scores = rng.permutation(scores)
        nonpersonal_post = permuted_scores[subreddit_codes] * post
        x = np.column_stack([gen_post, phys_post, nonpersonal_post])
        x_resid = residualize_two_way(x, base["subreddit"], base["year_month"])
        beta = np.linalg.lstsq(x_resid, y_resid, rcond=None)[0]
        permuted_coefs[iteration] = beta[2]
    more_negative_share = safe_float(np.mean(permuted_coefs <= true_coef))
    return {
        "permutation_more_negative_share": more_negative_share,
        "permutation_iterations": safe_int(n_iter),
    }


def run_log_outcome_robustness(panel, main_results, n_iter=ROBUSTNESS_ITERATIONS):
    rows = []
    print("\nAdditional measurement robustness battery")
    for offset, (outcome, label) in enumerate(ROBUSTNESS_OUTCOMES):
        print(f"\n  Robustness outcome: {label}")
        model, _ = fit_three_dimensional_model(panel, outcome)
        result = reg_result(model, "non_personal_post")
        main_row = main_results[main_results["outcome"].eq(outcome)]
        main_row = main_row.iloc[0] if not main_row.empty else {}
        print(
            "    Covariate adjusted: "
            f"coef={fmt_signed4(main_row.get('covariate_adj_coef'))} "
            f"SE={fmt4(main_row.get('covariate_adj_se'))} "
            f"p={fmt4(main_row.get('covariate_adj_pvalue'))}"
        )
        bootstrap = wild_cluster_bootstrap_persfree(
            panel,
            outcome,
            n_iter=n_iter,
            seed=ROBUSTNESS_RANDOM_SEED + 101 * offset,
        )
        print(
            "    Wild cluster bootstrap: "
            f"p={fmt4(bootstrap['wild_bootstrap_pvalue'])} "
            f"95% CI=[{fmt_signed4(bootstrap['wild_bootstrap_ci_low'])}, "
            f"{fmt_signed4(bootstrap['wild_bootstrap_ci_high'])}]"
        )
        leave_one = leave_one_subreddit_out(panel, outcome)
        print(
            "    Leave-one-subreddit-out: "
            f"coef range=[{fmt_signed4(leave_one['loo_min_coef'])}, "
            f"{fmt_signed4(leave_one['loo_max_coef'])}], "
            f"significant={leave_one['loo_significant_count']}/{leave_one['loo_runs']}"
        )
        permutation = permutation_placebo_persfree(
            panel,
            outcome,
            true_coef=result["coef"],
            n_iter=n_iter,
            seed=ROBUSTNESS_RANDOM_SEED + 1009 * (offset + 1),
        )
        print(
            "    Permutation placebo: "
            f"share more negative than true={fmt4(permutation['permutation_more_negative_share'])}"
        )
        row = {
            "outcome": outcome,
            "label": label,
            "coef": result["coef"],
            "se": result["se"],
            "pvalue": result["pvalue"],
            "covariate_adj_coef": main_row.get("covariate_adj_coef"),
            "covariate_adj_se": main_row.get("covariate_adj_se"),
            "covariate_adj_pvalue": main_row.get("covariate_adj_pvalue"),
        }
        row.update(bootstrap)
        row.update(leave_one)
        row.update(permutation)
        rows.append(row)
    return pd.DataFrame(rows)


def choose_score_path(score_path=None, output_dir=None):
    if score_path is not None:
        score_path = Path(score_path)
        if not score_path.exists():
            raise FileNotFoundError(score_path)
        return score_path

    if output_dir is not None:
        output_dir = Path(output_dir)
        candidates = [
            output_dir / "tables" / "acsi_preshock_tworuns.csv",
            output_dir / "tables" / "acsi_scores_computed.csv",
            ROOT / "data" / "acsi_scores.csv",
            ROOT / "data" / "acsi_preshock_tworuns.csv",
        ]
    else:
        candidates = DEFAULT_SCORE_CANDIDATES

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No ACSI score file found in expected output/data locations.")


def load_posts(posts_path=None, output_dir=None):
    posts_path = Path(posts_path or (Path(output_dir) / "posts_clean_ecosystem.parquet" if output_dir else DEFAULT_POSTS_PATH))
    if not posts_path.exists():
        raise FileNotFoundError(posts_path)
    print(f"  Loading posts -> {posts_path}")
    return pd.read_parquet(posts_path)


def load_score_lookup(score_path=None, output_dir=None):
    score_path = choose_score_path(score_path=score_path, output_dir=output_dir)
    print(f"  Loading ACSI scores -> {score_path}")
    scores = pd.read_csv(score_path)
    if "subreddit" not in scores.columns:
        raise ValueError(f"{score_path} must include subreddit.")

    lookup = scores[["subreddit"]].copy()
    lookup["subreddit"] = lookup["subreddit"].astype(str)

    if "pers_free" in scores.columns:
        lookup["non_personal_norm"] = pd.to_numeric(scores["pers_free"], errors="coerce")
    elif "non_personal_norm" in scores.columns:
        lookup["non_personal_norm"] = pd.to_numeric(scores["non_personal_norm"], errors="coerce")
    elif "avg_personal_req_0_to_3" in scores.columns:
        lookup["non_personal_norm"] = 1.0 - pd.to_numeric(scores["avg_personal_req_0_to_3"], errors="coerce") / 3.0
    elif "personal_req" in scores.columns:
        lookup["non_personal_norm"] = (5.0 - pd.to_numeric(scores["personal_req"], errors="coerce")) / 4.0
    else:
        raise ValueError(
            f"{score_path} needs pers_free, non_personal_norm, avg_personal_req_0_to_3, or personal_req."
        )

    if "gen_cap" in scores.columns:
        lookup["generation_capability_norm"] = pd.to_numeric(scores["gen_cap"], errors="coerce")
    elif "generation_capability_norm" in scores.columns:
        lookup["generation_capability_norm"] = pd.to_numeric(scores["generation_capability_norm"], errors="coerce")

    if "phys_free" in scores.columns:
        lookup["physical_free_norm"] = pd.to_numeric(scores["phys_free"], errors="coerce")
    elif "physical_free_norm" in scores.columns:
        lookup["physical_free_norm"] = pd.to_numeric(scores["physical_free_norm"], errors="coerce")

    acsi_score_path = ROOT / "data" / "acsi_scores.csv"
    missing_dimension_controls = {
        "generation_capability_norm",
        "physical_free_norm",
    } - set(lookup.columns)
    if missing_dimension_controls and acsi_score_path.exists() and acsi_score_path.resolve() != score_path.resolve():
        acsi_scores = pd.read_csv(acsi_score_path)
        available = ["subreddit", *sorted(missing_dimension_controls & set(acsi_scores.columns))]
        if len(available) > 1:
            supplement = acsi_scores[available].copy()
            supplement["subreddit"] = supplement["subreddit"].astype(str)
            lookup = lookup.merge(supplement.drop_duplicates("subreddit"), on="subreddit", how="left")

    lookup["non_personal_norm"] = lookup["non_personal_norm"].clip(0, 1)
    for column_name in ["generation_capability_norm", "physical_free_norm"]:
        if column_name in lookup.columns:
            lookup[column_name] = pd.to_numeric(lookup[column_name], errors="coerce").clip(0, 1)
    return lookup.dropna(subset=["subreddit", "non_personal_norm"]).drop_duplicates("subreddit")


def normalize_month_column(frame):
    if "year_month_dt" in frame.columns:
        return pd.to_datetime(frame["year_month_dt"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "year_month" in frame.columns:
        return pd.to_datetime(frame["year_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "month" in frame.columns:
        return pd.to_datetime(frame["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if "created_utc" in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame["created_utc"]):
            created = pd.to_datetime(frame["created_utc"], errors="coerce")
        else:
            created = pd.to_datetime(frame["created_utc"], unit="s", utc=True, errors="coerce")
        if getattr(created.dt, "tz", None) is not None:
            created = created.dt.tz_localize(None)
        return created.dt.to_period("M").dt.to_timestamp()
    raise ValueError("Post table needs year_month_dt, year_month, month, or created_utc.")


def compute_vectorized_author_survival_rate(author_months):
    """Share of subreddit-month authors who also appear in the same subreddit next month."""
    required_columns = {"subreddit", "author", "year_month"}
    missing_columns = required_columns - set(author_months.columns)
    if missing_columns:
        raise ValueError(f"author_months missing columns: {sorted(missing_columns)}")

    current = (
        author_months[["subreddit", "author", "year_month"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    if current.empty:
        return pd.DataFrame(columns=["subreddit", "year_month", "author_survival_rate"])

    current["subreddit"] = current["subreddit"].astype(str)
    current["author"] = current["author"].astype(str)
    current["year_month"] = pd.to_datetime(current["year_month"]).dt.to_period("M").dt.to_timestamp()
    current["next_month"] = current["year_month"] + pd.offsets.MonthBegin(1)

    future = (
        current[["subreddit", "author", "year_month"]]
        .rename(columns={"year_month": "next_month"})
        .drop_duplicates()
        .copy()
    )
    future["survived_next_month"] = 1.0

    survived = current.merge(
        future,
        on=["subreddit", "author", "next_month"],
        how="left",
        validate="one_to_one",
    )
    survived["survived_next_month"] = survived["survived_next_month"].fillna(0.0)
    return (
        survived.groupby(["subreddit", "year_month"], as_index=False)["survived_next_month"]
        .mean()
        .rename(columns={"survived_next_month": "author_survival_rate"})
    )


def raw_post_id(payload):
    post_id = payload.get("post_id") or payload.get("id")
    if post_id:
        return str(post_id)
    name = payload.get("name")
    if isinstance(name, str) and name.startswith("t3_"):
        return name[3:]
    return None


def resolve_raw_data_dir(data_dir=None):
    base = Path(data_dir or DEFAULT_RAW_DATA_DIR)
    nested = base / "raw_files"
    return nested if nested.exists() else base


def compute_word_count_panel(posts_df, data_dir=None):
    required_columns = {"subreddit", "post_id", "year_month"}
    missing_columns = required_columns - set(posts_df.columns)
    if missing_columns:
        print(f"  Warning: cannot compute raw word counts; missing columns: {sorted(missing_columns)}")
        return pd.DataFrame(columns=["subreddit", "year_month", "avg_post_length", "log_total_words"])

    data_dir = resolve_raw_data_dir(data_dir)
    posts_for_words = posts_df[["subreddit", "post_id", "year_month"]].dropna().copy()
    posts_for_words["subreddit"] = posts_for_words["subreddit"].astype(str)
    posts_for_words["post_id"] = posts_for_words["post_id"].astype(str)
    posts_for_words["year_month"] = (
        pd.to_datetime(posts_for_words["year_month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    posts_for_words = posts_for_words.dropna(subset=["subreddit", "post_id", "year_month"])
    if posts_for_words.empty:
        return pd.DataFrame(columns=["subreddit", "year_month", "avg_post_length", "log_total_words"])

    wanted_ids = set(posts_for_words["post_id"])
    subreddits = sorted(posts_for_words["subreddit"].unique(), key=str.lower)
    progress = globals().get("tqdm")
    iterator = progress(subreddits, desc="raw word counts") if progress is not None else subreddits
    word_rows = []

    for subreddit in iterator:
        if progress is None:
            print(f"  raw word counts: r/{subreddit}")
        path = data_dir / f"r_{subreddit}_posts.jsonl"
        if not path.exists():
            print(f"  Warning: raw post file not found, skipping r/{subreddit}: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                post_id = raw_post_id(payload)
                if post_id is None or post_id not in wanted_ids:
                    continue
                title = str(payload.get("title") or "")
                selftext = str(payload.get("selftext") or "")
                word_rows.append({
                    "post_id": post_id,
                    "word_count": len((title + " " + selftext).split()),
                })

    if not word_rows:
        print("  Warning: no raw word counts matched cleaned post IDs.")
        return pd.DataFrame(columns=["subreddit", "year_month", "avg_post_length", "log_total_words"])

    word_counts = pd.DataFrame(word_rows).drop_duplicates("post_id")
    merged = posts_for_words.merge(word_counts, on="post_id", how="left", validate="many_to_one")
    matched_posts = int(merged["word_count"].notna().sum())
    total_posts = int(len(merged))
    print(f"  Raw word counts matched {matched_posts:,} of {total_posts:,} cleaned posts.")
    merged = merged.dropna(subset=["word_count"])
    if merged.empty:
        return pd.DataFrame(columns=["subreddit", "year_month", "avg_post_length", "log_total_words"])

    word_panel = (
        merged.groupby(["subreddit", "year_month"], as_index=False)
        .agg(
            avg_post_length=("word_count", "mean"),
            total_words=("word_count", "sum"),
        )
    )
    word_panel["log_total_words"] = np.log1p(word_panel["total_words"])
    return word_panel[["subreddit", "year_month", "avg_post_length", "log_total_words"]]


def build_outcome_panel(posts, data_dir=None):
    required_columns = {"subreddit", "author"}
    missing_columns = required_columns - set(posts.columns)
    if missing_columns:
        raise ValueError(f"Post table missing columns: {sorted(missing_columns)}")

    df = posts.copy()
    print("  Loaded post columns:")
    print("    " + ", ".join(map(str, df.columns)))
    df["subreddit"] = df["subreddit"].astype(str)
    df["author"] = df["author"].astype(str)
    df["year_month"] = normalize_month_column(df)
    df = df[df["year_month"].between(START_MONTH, END_MONTH)].copy()
    df = df[~df["author"].isin(EXCLUDED_AUTHORS)]
    df = df[~df["author"].str.endswith("bot", na=False)]
    if df.empty:
        raise ValueError("No usable posts remain in the Jan 2022-Dec 2024 measurement window.")

    first_month = (
        df.groupby(["subreddit", "author"])["year_month"]
        .min()
        .rename("first_month")
        .reset_index()
    )
    df = df.merge(first_month, on=["subreddit", "author"], how="left")
    df["_is_new"] = df["year_month"].eq(df["first_month"])

    panel = (
        df.groupby(["subreddit", "year_month"], as_index=False)
        .agg(
            n_posts=("author", "size"),
            unique_authors=("author", "nunique"),
            new_posts=("_is_new", "sum"),
        )
    )
    panel["log_unique_posters"] = np.log1p(panel["unique_authors"])
    panel["posts_per_author"] = panel["n_posts"] / panel["unique_authors"].clip(lower=1)
    panel["new_poster_share"] = panel["new_posts"] / panel["n_posts"].clip(lower=1)

    word_panel = compute_word_count_panel(df, data_dir=data_dir)
    panel = panel.merge(word_panel, on=["subreddit", "year_month"], how="left")

    author_months = df[["subreddit", "author", "year_month"]].drop_duplicates()
    survival = compute_vectorized_author_survival_rate(author_months)
    panel = panel.merge(survival, on=["subreddit", "year_month"], how="left")

    author_posts = (
        df.groupby(["subreddit", "year_month", "author"])
        .size()
        .rename("author_posts")
        .reset_index()
    )
    author_posts["month_total_posts"] = author_posts.groupby(
        ["subreddit", "year_month"]
    )["author_posts"].transform("sum")
    author_posts["share_sq"] = (
        author_posts["author_posts"] / author_posts["month_total_posts"].clip(lower=1)
    ) ** 2
    hhi = (
        author_posts.groupby(["subreddit", "year_month"], as_index=False)["share_sq"]
        .sum()
        .rename(columns={"share_sq": "hhi"})
    )
    panel = panel.merge(hhi, on=["subreddit", "year_month"], how="left")
    panel["year_month"] = pd.to_datetime(panel["year_month"]).dt.strftime("%Y-%m")
    panel["post"] = panel["year_month"].ge(POST_MONTH).astype(float)
    return panel


def run_measurement_regressions(panel, three_dimensional=False):
    panel = add_panel_pre_covariates(panel)
    panel["non_personal_post"] = panel["non_personal_norm"] * panel["post"]
    terms = ["non_personal_post"]
    if three_dimensional:
        control_columns = {"generation_capability_norm", "physical_free_norm"}
        missing_controls = control_columns - set(panel.columns)
        if missing_controls:
            raise ValueError(
                "--three-dimensional requires generation_capability_norm and physical_free_norm "
                f"in the merged ACSI score table. Missing: {sorted(missing_controls)}"
            )
        panel["generation_capability_post"] = panel["generation_capability_norm"] * panel["post"]
        panel["physical_free_post"] = panel["physical_free_norm"] * panel["post"]
        terms = ["generation_capability_post", "physical_free_post", "non_personal_post"]

    covariate_terms = ["pre_avg_post", "pre_trend_post", "log_mu_post"]
    rows = []
    for outcome, label in OUTCOME_SPECS:
        model_data = panel.dropna(subset=[outcome, *terms, "subreddit", "year_month"]).copy()
        if model_data.empty or model_data["subreddit"].nunique() < 2:
            rows.append({
                "model": "three_dimensional" if three_dimensional else "persfree_only",
                "outcome": outcome,
                "label": label,
                "coef": None,
                "se": None,
                "pvalue": None,
                "generation_capability_coef": None,
                "generation_capability_se": None,
                "generation_capability_pvalue": None,
                "physical_free_coef": None,
                "physical_free_se": None,
                "physical_free_pvalue": None,
                "covariate_adj_coef": None,
                "covariate_adj_se": None,
                "covariate_adj_pvalue": None,
                "n_obs": safe_int(len(model_data)),
                "n_subreddits": safe_int(model_data["subreddit"].nunique()) if not model_data.empty else 0,
            })
            continue
        model = fit_ols(
            f"{outcome} ~ {' + '.join(terms)} + C(subreddit) + C(year_month)",
            model_data,
            cluster_col="subreddit",
        )
        result = reg_result(model, "non_personal_post")
        row = {
            "model": "three_dimensional" if three_dimensional else "persfree_only",
            "outcome": outcome,
            "label": label,
            "coef": result["coef"],
            "se": result["se"],
            "pvalue": result["pvalue"],
            "generation_capability_coef": None,
            "generation_capability_se": None,
            "generation_capability_pvalue": None,
            "physical_free_coef": None,
            "physical_free_se": None,
            "physical_free_pvalue": None,
            "n_obs": result["n_obs"],
            "n_subreddits": safe_int(model_data["subreddit"].nunique()),
        }
        if three_dimensional:
            row.update(term_result(model, "generation_capability_post", "generation_capability"))
            row.update(term_result(model, "physical_free_post", "physical_free"))
        covariate_model_data = panel.dropna(
            subset=[outcome, *terms, *covariate_terms, "subreddit", "year_month"]
        ).copy()
        if covariate_model_data.empty or covariate_model_data["subreddit"].nunique() < 2:
            covariate_result = {"coef": None, "se": None, "pvalue": None}
        else:
            covariate_model = fit_ols(
                f"{outcome} ~ {' + '.join([*terms, *covariate_terms])} + C(subreddit) + C(year_month)",
                covariate_model_data,
                cluster_col="subreddit",
            )
            covariate_result = reg_result(covariate_model, "non_personal_post")
        row.update({
            "covariate_adj_coef": covariate_result["coef"],
            "covariate_adj_se": covariate_result["se"],
            "covariate_adj_pvalue": covariate_result["pvalue"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_unique_posters_decomposition(posts_df, panel):
    required_columns = {"subreddit", "author"}
    missing_columns = required_columns - set(posts_df.columns)
    if missing_columns:
        raise ValueError(f"Post table missing columns for decomposition: {sorted(missing_columns)}")

    df = posts_df[["subreddit", "author"]].copy()
    df["year_month"] = normalize_month_column(posts_df)
    df["subreddit"] = df["subreddit"].astype(str)
    df["author"] = df["author"].astype(str)
    df = df[df["year_month"].between(START_MONTH, END_MONTH)].copy()
    df = df[~df["author"].isin(EXCLUDED_AUTHORS)]
    df = df[~df["author"].str.endswith("bot", na=False)]
    if df.empty:
        return pd.DataFrame()

    print("\nUnique-poster decomposition: classifying stable pairs")
    df["post"] = df["year_month"].ge(pd.Timestamp(f"{POST_MONTH}-01"))
    group_rows = {
        "stable_pairs": [],
        "nonstable_pairs": [],
        "one_month_pairs": [],
        "low_commitment_repeat_pairs": [],
        "preshock_low_frequency_pairs": [],
    }
    stable_pair_total = 0
    n_pre_months = int(df.loc[~df["post"], "year_month"].nunique())
    low_commitment_threshold = 0.5 * n_pre_months

    def append_group_counts(rows, subreddit, group, eligible_authors):
        if not eligible_authors:
            return
        group_counts = (
            group.loc[group["author"].isin(eligible_authors)]
            .groupby("year_month", sort=False)["author"]
            .nunique()
        )
        for month, count in group_counts.items():
            rows.append({
                "subreddit": subreddit,
                "year_month": month,
                "unique_authors": count,
            })

    for subreddit, group in df.groupby("subreddit", sort=False):
        print(f"  decomposition counts: r/{subreddit}")
        pre_authors = set(group.loc[~group["post"], "author"].unique())
        post_authors = set(group.loc[group["post"], "author"].unique())
        stable_authors = pre_authors & post_authors
        stable_pair_total += len(stable_authors)

        all_authors = set(group["author"].unique())
        nonstable_authors = all_authors - stable_authors
        active_months = group.groupby("author", sort=False)["year_month"].nunique()
        one_month_authors = set(active_months[active_months.eq(1)].index)
        low_commitment_repeat_authors = set(
            active_months[
                active_months.gt(1) & active_months.lt(low_commitment_threshold)
            ].index
        )
        pre_counts = group.loc[~group["post"]].groupby("author", sort=False).size()
        if pre_counts.empty:
            preshock_low_frequency_authors = set()
        else:
            bottom_tercile_cutoff = pre_counts.quantile(1.0 / 3.0)
            preshock_low_frequency_authors = set(pre_counts[pre_counts.le(bottom_tercile_cutoff)].index)

        append_group_counts(group_rows["stable_pairs"], subreddit, group, stable_authors)
        append_group_counts(group_rows["nonstable_pairs"], subreddit, group, nonstable_authors)
        append_group_counts(group_rows["one_month_pairs"], subreddit, group, one_month_authors)
        append_group_counts(
            group_rows["low_commitment_repeat_pairs"],
            subreddit,
            group,
            low_commitment_repeat_authors,
        )
        append_group_counts(
            group_rows["preshock_low_frequency_pairs"],
            subreddit,
            group,
            preshock_low_frequency_authors,
        )
    print(f"  Stable author-subreddit pairs: {stable_pair_total:,}")

    group_counts = {
        group_name: pd.DataFrame(rows)
        for group_name, rows in group_rows.items()
    }
    for counts_frame in group_counts.values():
        if not counts_frame.empty:
            counts_frame["year_month"] = pd.to_datetime(counts_frame["year_month"]).dt.strftime("%Y-%m")

    base_columns = [
        "subreddit",
        "year_month",
        "post",
        "non_personal_norm",
        "generation_capability_norm",
        "physical_free_norm",
    ]
    missing_panel_columns = set(base_columns) - set(panel.columns)
    if missing_panel_columns:
        raise ValueError(f"Panel missing columns for decomposition: {sorted(missing_panel_columns)}")
    base_panel = panel[base_columns].drop_duplicates(["subreddit", "year_month"]).copy()

    rows = []
    print("\nUnique-poster decomposition: three-dimensional DiD")
    for group_name, label in [
        ("stable_pairs", "Stable author-subreddit pairs"),
        ("nonstable_pairs", "Nonstable author-subreddit pairs"),
        ("one_month_pairs", "One-month author-subreddit pairs"),
        ("low_commitment_repeat_pairs", "Low-commitment repeat author-subreddit pairs"),
        ("preshock_low_frequency_pairs", "Pre-shock low-frequency author-subreddit pairs"),
    ]:
        counts = group_counts[group_name]
        if counts.empty:
            counts = pd.DataFrame(columns=["subreddit", "year_month", "unique_authors"])
        else:
            counts = counts[["subreddit", "year_month", "unique_authors"]]
        group_panel = base_panel.merge(counts, on=["subreddit", "year_month"], how="left")
        group_panel["unique_authors"] = group_panel["unique_authors"].fillna(0.0)
        group_panel["log_unique_posters"] = np.log1p(group_panel["unique_authors"])
        row = run_three_dimensional_did(
            group_panel,
            outcome="log_unique_posters",
            model_name="unique_posters_decomposition_three_dimensional",
            label=label,
            group=group_name,
        )
        rows.append(row)
        print(
            f"  {label}: coef={fmt_signed4(row['coef'])} "
            f"SE={fmt4(row['se'])} p={fmt4(row['pvalue'])}"
        )
    return pd.DataFrame(rows)


def compute_additional_measurements(posts_path=None, score_path=None, output_dir=None, three_dimensional=False):
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    posts = load_posts(posts_path=posts_path, output_dir=output_dir)
    scores = load_score_lookup(score_path=score_path, output_dir=output_dir)
    panel = build_outcome_panel(posts).merge(scores, on="subreddit", how="inner")
    if panel.empty:
        raise ValueError("No subreddit-month rows merged to ACSI scores.")

    print_panel_diagnostics(panel, output_dir=output_dir)

    main_results = run_measurement_regressions(panel, three_dimensional=three_dimensional)
    event_table = run_event_study_unique_posters(panel, output_dir=output_dir)
    pretrend_row = run_pretrend_test_unique_posters(panel)
    decomposition_results = run_unique_posters_decomposition(posts, panel)
    robustness_results = run_log_outcome_robustness(panel, main_results)
    results = pd.concat(
        [
            main_results,
            pd.DataFrame([pretrend_row]),
            decomposition_results,
        ],
        ignore_index=True,
        sort=False,
    )
    panel_path = tables_dir / "additional_measurement_panel.csv"
    results_path = tables_dir / "additional_measurement_results.csv"
    event_path = tables_dir / "additional_measurement_event_study_unique_posters.csv"
    robustness_path = tables_dir / "additional_measurement_robustness.csv"
    decomposition_path = tables_dir / "additional_measurement_unique_posters_decomposition.csv"
    latex_path = tables_dir / "additional_measurement_results.tex"
    panel.to_csv(panel_path, index=False)
    results.to_csv(results_path, index=False)
    event_table.to_csv(event_path, index=False)
    robustness_results.to_csv(robustness_path, index=False)
    decomposition_results.to_csv(decomposition_path, index=False)
    latex_path.write_text(results.to_latex(index=False, float_format="%.4f"), encoding="utf-8")

    model_label = "three-dimensional" if three_dimensional else "PersFree-only"
    print(f"\nAdditional measurement {model_label} DiD coefficients:")
    for _, row in main_results.iterrows():
        print(
            f"  {row['label']}: coef={fmt_signed4(row['coef'])} "
            f"SE={fmt4(row['se'])} p={fmt4(row['pvalue'])}; "
            f"cov-adj={fmt_signed4(row['covariate_adj_coef'])} "
            f"SE={fmt4(row['covariate_adj_se'])} p={fmt4(row['covariate_adj_pvalue'])}"
        )
    print(f"  Panel saved -> {panel_path}")
    print(f"  Results saved -> {results_path}")
    print(f"  Event-study coefficients saved -> {event_path}")
    print(f"  Robustness summary saved -> {robustness_path}")
    print(f"  Unique-poster decomposition saved -> {decomposition_path}")
    print(f"  Event-study figure saved -> {figures_dir / 'event_study_unique_posters.png'}")
    return panel, results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute additional measurement outcomes.")
    parser.add_argument("--posts-path", default=None, help="Path to posts_clean_ecosystem.parquet.")
    parser.add_argument("--score-path", default=None, help="Path to ACSI score CSV.")
    parser.add_argument("--output-dir", default=None, help="Output directory, default output/latest.")
    parser.add_argument(
        "--three-dimensional",
        action="store_true",
        help="Control for GenCap x Post and PhysFree x Post in each DiD regression.",
    )
    parser.add_argument(
        "--write-csvs",
        action="store_true",
        help="Accepted for compatibility; CSV and TeX outputs are always written.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    compute_additional_measurements(
        posts_path=args.posts_path,
        score_path=args.score_path,
        output_dir=args.output_dir,
        three_dimensional=args.three_dimensional,
    )


if __name__ == "__main__":
    main()
