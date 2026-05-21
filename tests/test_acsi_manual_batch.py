import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import scripts.acsi_manual_batch as acsi_manual_batch


class AcsiManualBatchTests(unittest.TestCase):
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
            pd.DataFrame([coded_row], columns=acsi_manual_batch.OUTPUT_COLUMNS).to_csv(coded_path, index=False)

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
                acsi_manual_batch.aggregate_scores(args)

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

            with patch.object(acsi_manual_batch, "DEFAULT_RUN1_OUTPUT", run1_path), patch.object(
                acsi_manual_batch, "DEFAULT_RUN2_OUTPUT", run2_path
            ), patch.object(acsi_manual_batch, "DEFAULT_FINAL_OUTPUT", final_path), contextlib.redirect_stdout(
                io.StringIO()
            ):
                acsi_manual_batch.print_summary(args)

            self.assertFalse(run1_path.exists())
            self.assertFalse(run2_path.exists())
            self.assertFalse(final_path.exists())


if __name__ == "__main__":
    unittest.main()
