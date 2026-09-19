# Semantic mapping pilot

This pilot compares spherical k-means, von Mises–Fisher mixtures, and density-based regions using embeddings of 60,000 Reddit posts. The numerical comparison is complete. Content review and selection of a final map remain in progress.

- [Methods](docs/METHODS.md): sample design, fitting procedures, and evaluation.
- [Results](reports/RESULTS.md): findings and numerical tables.
- [Region review](docs/REVIEW.md): sampling, interpretation, and the proposed API descriptions.
- [Data](data-access.md): required files and their Longleaf locations.

## Setup

Use Python 3.12 on Linux, macOS, or WSL. From this directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-semantic-pilot.txt
```

## Running the comparison

Reuse the saved embeddings and the original A/B/C folds. With access to the project data, this command checks the inputs without fitting models:

```bash
.venv/bin/python code/run_semantic_threefold_pilot.py \
  --config semantic_threefold_pilot_config.json \
  --data-root /path/to/project/data --preflight
```

To fit the comparison, replace `--preflight` with `--output /path/to/new/results`. The runner also accepts `--task-id` for individual fits and `--aggregate` to assemble completed results. Use a compute node for the full experiment.

The configuration records the initial 100-iteration pass. The final vMF results also require `code/run_vmf_convergence_extension.py`: supply `--source-results`, `--data-root`, and a new `--output`; run `--prepare`, then `--task-id` for each reported pending task, followed by `--aggregate`. Exit code 75 indicates a saved pause and requires resuming that task. All options are listed by `--help`.

Keep the original inputs and results unchanged. `run_semantic_region_pilot.py` provides shared functions for the comparison; its older standalone configuration is not included.
