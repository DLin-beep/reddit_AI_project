# Reddit GenAI Pipeline

## Setup

```bash
pip install -r requirements.txt
```

## Data

Drop Reddit `.jsonl` files into the `data/` folder.  
Files should be named like `r_writing_posts.jsonl` and `r_writing_comments.jsonl`.

The ACSI measurement sample lives in `data/acsi_measurement_sample_run1.csv`;
subreddit-level component scores live in `data/acsi_scores.csv`.

Raw Reddit files are intentionally kept in `data/`. Derived analysis outputs
live under `output/latest/`. Full raw rescans may create `data/cache/` with
reusable monthly aggregates; use `--reuse-gse-panel` when you want to rerun the
models from `output/latest/subreddit_month_gse_panel.parquet` without recreating
that cache folder.

## Run

```bash
python scripts/run.py --gse-only --reuse-gse-panel
```

Other Useful commands:

```bash
python scripts/run.py --dashboard-only --quick
python scripts/run.py --gse-only --force-rebuild-gse-panel
python scripts/run.py --reuse-clean-posts
```

## Tests

## Timelime

- Pre-shock: Jan 2020 – Nov 2022
- Shock month in models: Dec 2022
- Post-shock: Dec 2022 – Dec 2024

## Subreddits
