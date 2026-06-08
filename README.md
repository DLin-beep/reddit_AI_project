# Reddit GenAI Pipeline

## Setup

```bash
pip install -r requirements.txt
```

## Data

Drop Reddit `.jsonl` files into the `data/raw_files/` folder.
Files should be named like `r_writing_posts.jsonl`.

The ACSI run-1 annotation sample lives in `data/acsi_annotated.csv`;
subreddit-level component scores live in `data/acsi_scores.csv`.

Build and score annotation samples through the single annotation entry point:

```bash
.venv/bin/python scripts/annotate.py build
.venv/bin/python scripts/annotate.py next --run 1 -n 10
.venv/bin/python scripts/annotate.py aggregate
```

Raw Reddit files are intentionally kept in `data/raw_files/`. Derived analysis outputs
live under `output/latest/`. The pipeline rebuilds cleaned post data from the
raw JSONL files on every full run.

## Run

```bash
.venv/bin/python scripts/run.py
```

## Tests

```bash
.venv/bin/python -m pytest
```

## Timeline

- Full analysis window: Jan 2020 – Dec 2024
- Pre-shock split-window diagnostic: Jan 2020 – Nov 2022
- Shock month in models: Dec 2022
- Post-shock split-window diagnostic: Dec 2022 – Dec 2024

## Subreddits
