import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.run as run


TEST_SUBREDDITS = ["alpha", "beta"]
TEST_MONTHS = pd.date_range("2022-11-01", periods=2, freq="MS")


def make_score_table():
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


def make_required_results():
    return {
        "gse_main": {"coef": 0.0},
        "gse_covariate_adj": {"coef": 0.0},
        "gse_dimensions": [{"coef": 0.0} for _ in run.ACSI_DIMENSION_SPECS],
        "acsi_component_correlations": [{"component": "test"}],
        "acsi_three_dimensional": [{"coef": 0.0} for _ in run.ACSI_MECHANISM_SPECS],
        "acsi_mechanisms": [{"coef": 0.0} for _ in run.ACSI_MECHANISM_SPECS],
        "acsi_mechanisms_joint": [{"coef": 0.0} for _ in run.ACSI_MECHANISM_SPECS],
        "binary_did_consistency": {"coef": 0.0},
        "substitutability_hypothesis": {"conclusion": "test"},
        "gse_permutation": {"observed_coef": 0.0},
        "gse_secondary": [{"coef": 0.0} for _ in range(3)],
        "gse_quartiles": [{"coef": 0.0} for _ in range(3)],
        "gse_event_study": {"rows": []},
    }


class RunValidationTests(unittest.TestCase):
    def setUp(self):
        self.previous_output_dir = run.OUTPUT_DIR
        self.test_output_dir = run.OUTPUT_ROOT / "test_validation"
        shutil.rmtree(self.test_output_dir, ignore_errors=True)
        run.configure_output_dir(self.test_output_dir)

    def tearDown(self):
        run.configure_output_dir(self.previous_output_dir)
        shutil.rmtree(self.test_output_dir, ignore_errors=True)

    def panel_context(self):
        return patch.multiple(
            run,
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
            run.validate_subreddit_month_panel(make_valid_panel(), make_score_table(), validation_errors)

        self.assertEqual(validation_errors, [])

    def test_subreddit_month_panel_validation_detects_duplicate_rows(self):
        panel = make_valid_panel()
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

        with self.panel_context():
            validation_errors = []
            run.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

        self.assertTrue(any("duplicate subreddit-month rows" in error for error in validation_errors))

    def test_subreddit_month_panel_validation_detects_out_of_range_scores(self):
        panel = make_valid_panel()
        panel.loc[0, "gse"] = 1.5

        with self.panel_context():
            validation_errors = []
            run.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

        self.assertTrue(any("column gse has values outside" in error for error in validation_errors))

    def test_subreddit_month_panel_validation_accepts_negative_raw_average_score(self):
        panel = make_valid_panel()
        panel.loc[0, "avg_score"] = -0.25
        panel.loc[0, "log_avg_score"] = 0.0

        with self.panel_context():
            validation_errors = []
            run.validate_subreddit_month_panel(panel, make_score_table(), validation_errors)

        self.assertEqual(validation_errors, [])

    def test_load_acsi_scores_rejects_duplicate_subreddit_rows(self):
        duplicate_scores_path = self.test_output_dir / "duplicate_scores.csv"
        pd.DataFrame(
            [
                {"subreddit": "alpha", "direct_gen": 3, "usefulness": 3, "quality_comp": 3, "physical_req": 2, "personal_req": 2, "n_used": 50},
                {"subreddit": "alpha", "direct_gen": 4, "usefulness": 3, "quality_comp": 3, "physical_req": 2, "personal_req": 2, "n_used": 50},
                {"subreddit": "beta", "direct_gen": 3, "usefulness": 3, "quality_comp": 3, "physical_req": 2, "personal_req": 2, "n_used": 50},
            ]
        ).to_csv(duplicate_scores_path, index=False)

        with patch.object(run, "ACSI_SCORE_PATH", duplicate_scores_path), patch.object(
            run, "SUBREDDITS", {"alpha": "treatment", "beta": "control"}
        ):
            with self.assertRaisesRegex(ValueError, "duplicate subreddit rows"):
                run.load_acsi_scores()

    def test_final_validation_gate_reads_saved_panel_and_passes_clean_outputs(self):
        panel = make_valid_panel()
        score_table = make_score_table()
        panel.to_parquet(run.SUBMONTH_PANEL_PATH, index=False)

        with self.panel_context(), patch.object(run, "validate_required_artifacts"), patch.object(
            run, "validate_cache_metadata"
        ):
            run.validate_run_outputs(
                submonth_panel=panel,
                score_table=score_table,
                analysis_results=make_required_results(),
                run_legacy_models=False,
                author_cap_enabled=True,
            )


if __name__ == "__main__":
    unittest.main()
