#!/usr/bin/env python3
"""
Run ACSI dimension-level diagnostics from an existing subreddit-month panel.

This is intentionally downstream-only: it does not rescan raw Reddit files.
It reads the cleaned panel and ACSI score table produced by scripts/run.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run import MU_K  # noqa: E402


DIMENSIONS = [
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


DOMAIN_MAP = {
    # Writing/text communities
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
    # Art/design/game/media communities
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
    # School, admissions, technical, and career-help communities
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
    # Physical/craft/food/home communities
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
    # Remaining activity, travel, and miscellaneous communities
    "chess": "travel_fitness_misc",
    "programminghumor": "travel_fitness_misc",
    "rowing": "travel_fitness_misc",
    "running": "travel_fitness_misc",
    "swimming": "travel_fitness_misc",
    "solotravel": "travel_fitness_misc",
    "travel": "travel_fitness_misc",
}


def pct_effect(coef: float | None, scale: float = 1.0) -> float | None:
    if coef is None or pd.isna(coef):
        return None
    return float(100 * (np.exp(coef * scale) - 1))


def fit_ols(formula: str, data: pd.DataFrame):
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data["subreddit"]})


def reg_row(model, term: str) -> dict:
    return {
        "term": term,
        "coef": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "pvalue": float(model.pvalues.get(term, np.nan)),
        "n_obs": int(model.nobs),
    }


def add_pre_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    pre = panel[panel["post_shock"].eq(0)].copy()
    pre_avg = pre.groupby("subreddit")["log_posts"].mean().rename("pre_avg_log_posts")

    min_month = pre["year_month_dt"].min()
    pre["t"] = (
        (pre["year_month_dt"].dt.year - min_month.year) * 12
        + (pre["year_month_dt"].dt.month - min_month.month)
    )

    trend_rows = []
    for subreddit, group in pre.groupby("subreddit"):
        if len(group) >= 6 and group["log_posts"].nunique() > 1:
            slope = np.polyfit(group["t"], group["log_posts"], 1)[0]
        else:
            slope = 0.0
        trend_rows.append({"subreddit": subreddit, "pre_trend": slope})

    cov = pre_avg.reset_index().merge(pd.DataFrame(trend_rows), on="subreddit", how="left")
    cov["mu_k"] = cov["subreddit"].map(MU_K).fillna(0.5)
    cov["log_mu_k"] = np.log1p(cov["mu_k"])
    return cov


def prepare_panel(output_dir: Path) -> pd.DataFrame:
    panel_path = output_dir / "subreddit_month_gse_panel.parquet"
    score_path = output_dir / "tables" / "acsi_scores_computed.csv"
    legacy_score_path = output_dir / "tables" / "gse_scores_computed.csv"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)
    if not score_path.exists():
        if legacy_score_path.exists():
            score_path = legacy_score_path
        else:
            raise FileNotFoundError(score_path)

    panel = pd.read_parquet(panel_path)
    scores = pd.read_csv(score_path)
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
    scores = scores[[c for c in keep_cols if c in scores.columns]].copy()

    # Normalize the 1-5 component scores to 0-1. For physical/personal
    # constraints, reverse the direction so every exposure variable means
    # "more AI-substitutable" when it is higher.
    scores["direct_gen_exp"] = (scores["direct_gen"] - 1) / 4
    scores["usefulness_exp"] = (scores["usefulness"] - 1) / 4
    scores["quality_comp_exp"] = (scores["quality_comp"] - 1) / 4
    scores["physical_free_exp"] = (5 - scores["physical_req"]) / 4
    scores["non_personal_exp"] = (5 - scores["personal_req"]) / 4
    scores["domain"] = scores["subreddit"].map(DOMAIN_MAP).fillna("other")

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


def run_dimension_models(panel: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = []
    cov = add_pre_covariates(panel)
    adjusted = panel.merge(cov, on="subreddit", how="left")
    adjusted["pre_avg_post"] = adjusted["pre_avg_log_posts"] * adjusted["post_shock"]
    adjusted["pre_trend_post"] = adjusted["pre_trend"] * adjusted["post_shock"]
    adjusted["log_mu_post"] = adjusted["log_mu_k"] * adjusted["post_shock"]

    for dim in DIMENSIONS:
        var = dim["var"]
        term = f"{var}_post"
        panel[term] = panel[var] * panel["post_shock"]
        adjusted[term] = adjusted[var] * adjusted["post_shock"]

        for model_name, data, extra in [
            ("fixed_effects", panel, ""),
            ("covariate_adjusted", adjusted, " + pre_avg_post + pre_trend_post + log_mu_post"),
        ]:
            model = fit_ols(f"log_posts ~ {term}{extra} + C(subreddit) + C(year_month)", data)
            row = reg_row(model, term)
            observed_range = float(data[var].max() - data[var].min())
            row.update(
                {
                    "model": model_name,
                    "dimension": var,
                    "source_score": dim["source"],
                    "label": dim["label"],
                    "interpretation": dim["interpretation"],
                    "observed_min": float(data[var].min()),
                    "observed_max": float(data[var].max()),
                    "percent_effect_full_0_to_1": pct_effect(row["coef"]),
                    "percent_effect_observed_range": pct_effect(row["coef"], observed_range),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "acsi_dimension_regressions.csv", index=False)
    return out


def run_joint_models(panel: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    dims = [d["var"] for d in DIMENSIONS]
    terms = [f"{v}_post" for v in dims]
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
        model = fit_ols(formula, data)
        for dim in DIMENSIONS:
            term = f"{dim['var']}_post"
            row = reg_row(model, term)
            row.update(
                {
                    "model": model_name,
                    "dimension": dim["var"],
                    "source_score": dim["source"],
                    "label": dim["label"],
                    "percent_effect_full_0_to_1": pct_effect(row["coef"]),
                }
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "acsi_dimension_joint_model.csv", index=False)
    return out


def run_domain_tables(panel: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    summary["mean_percent_change"] = summary["mean_pre_post_log_change"].apply(pct_effect)
    summary.to_csv(tables_dir / "acsi_content_area_summary.csv", index=False)

    reg_panel = panel.copy()
    reference = "craft_physical"
    domains = sorted(d for d in reg_panel["domain"].unique() if d != reference)
    terms = []
    for domain in domains:
        clean = domain.replace("-", "_")
        term = f"domain_{clean}_post"
        reg_panel[term] = reg_panel["post_shock"] * reg_panel["domain"].eq(domain).astype(int)
        terms.append((domain, term))

    formula = f"log_posts ~ {' + '.join(term for _, term in terms)} + C(subreddit) + C(year_month)"
    model = fit_ols(formula, reg_panel)
    rows = []
    for domain, term in terms:
        row = reg_row(model, term)
        row.update(
            {
                "domain": domain,
                "reference_domain": reference,
                "label": f"{domain} vs {reference}",
                "percent_effect_vs_reference": pct_effect(row["coef"]),
            }
        )
        rows.append(row)

    reg = pd.DataFrame(rows)
    reg.to_csv(tables_dir / "acsi_content_area_regression.csv", index=False)
    sub_scores.to_csv(tables_dir / "acsi_subreddit_dimension_summary.csv", index=False)
    return summary, reg


def run_correlation_table(panel: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    sub_scores = panel[["subreddit"] + [d["var"] for d in DIMENSIONS] + ["gse"]].drop_duplicates("subreddit")
    corr = sub_scores.drop(columns=["subreddit"]).corr()
    corr.to_csv(tables_dir / "acsi_dimension_correlations.csv")
    return corr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "latest",
        help="Existing run output directory containing subreddit_month_gse_panel.parquet.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    panel = prepare_panel(output_dir)
    dim = run_dimension_models(panel.copy(), tables_dir)
    joint = run_joint_models(panel.copy(), tables_dir)
    domain_summary, domain_reg = run_domain_tables(panel.copy(), tables_dir)
    run_correlation_table(panel.copy(), tables_dir)

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
    print(f"\nWrote diagnostics to {tables_dir}")


if __name__ == "__main__":
    main()
