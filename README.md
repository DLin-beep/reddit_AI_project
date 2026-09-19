# Reddit and AI research

Code, methods, and aggregate results for studying Reddit posting and AI availability. The newer semantic measurement pilot and the existing ACSI annotation/analysis workflow have separate entry points and data requirements.

## Start here

| Task | Entry point |
| --- | --- |
| Review the semantic pilot or reproduce its methods | [Semantic pilot](semantic_pilot/README.md) |
| Confirm that collaborators use the same 60,000 posts | [Semantic data access and identity checks](semantic_pilot/data-access.md) |
| Read completed numerical findings | [Pilot summary](semantic_pilot/reports/SUMMARY.md) and [full report](semantic_pilot/reports/RESULTS.md) |
| Use the existing annotation/analysis workflow | Instructions below and [data guide](data/README.md) |

## Semantic pilot status

The September 14–15, 2026 records establish verified embeddings for **60,000 reference posts** and **72 completed calculations**: 24 spherical k-means fits and 24 vMF fits converged; 24 density calculations finished. Comparisons assess reproducibility and coverage, not classification accuracy or AI effects.

The prepared content-review packet covers **60 regions across four alternative maps**, with 25 sampled posts per region: **1,500 appearances of 549 unique posts**. No final map has been selected or frozen, and no region-description API pass has run. Human interpretation and region-level event-window support remain pending. See the [pilot plan](semantic_pilot/docs/SEMANTIC_PILOT_PLAN.md).

This public repository provides code, documentation, and selected aggregate outputs. Restricted texts, embeddings, post-level identifiers/assignments, and individual ratings are supplied separately through the [data handoff](semantic_pilot/data-access.md).

## Existing ACSI annotation and analysis workflow

The original [scripts](scripts/) remain available. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Supply the required local inputs described in the [data guide](data/README.md). Raw files use names such as `data/raw_files/r_writing_posts.jsonl`; derived outputs go to `output/latest/`.

The annotation entry point builds samples, presents the next batch, accepts completed scores, and aggregates them. `build` defaults to the prepared `output/latest/posts_clean_all.parquet` input; `aggregate` requires completed annotations.

```bash
.venv/bin/python scripts/annotate.py build
.venv/bin/python scripts/annotate.py next --run 1 -n 10
.venv/bin/python scripts/annotate.py --help
```

After the required annotations and local data are ready:

```bash
.venv/bin/python scripts/annotate.py aggregate
.venv/bin/python scripts/run.py
```

The existing workflow uses January 2020–December 2024, with December 2022 as the shock month, a January 2020–November 2022 pre-shock diagnostic window, and a December 2022–December 2024 post-shock diagnostic window. These dates describe that workflow; the semantic pilot has its own reference-data and later event-data requirements.

For semantic synthetic tests and their separate pinned environment, follow the [pilot setup](semantic_pilot/README.md#run-the-synthetic-checks).

The older `annotation_code/` directory is retained for reference. Its original `pipeline_utils.py` and sampling configuration are not included in this checkout, so it is not a standalone runnable pipeline. Use the separate entry points above.
