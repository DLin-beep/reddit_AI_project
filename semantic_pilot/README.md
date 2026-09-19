# Semantic measurement pilot

This package records the comparison of semantic-region methods on **60,000 reference posts**, with 300 posts from each of 200 communities. It includes implementation code, settings, synthetic tests, methods, and selected aggregate results. Restricted data and saved models are supplied separately; start with [data access and identity checks](data-access.md) to ensure collaborators use the same corpus.

## What is complete

The September 14–15, 2026 records verify all 60,000 normalized UNC Azure `text-embedding-3-large` embeddings at 3,072 dimensions. The three-fold comparison finished all 72 calculations: 24 spherical k-means fits and 24 vMF fits converged; 24 density calculations completed. The final report incorporates continuation of the 15 vMF fits that initially reached 100 iterations.

Read the [summary](reports/SUMMARY.md), [full numerical report](reports/RESULTS.md), and [continuation record](docs/VMF_CONVERGENCE_EXTENSION.md). Agreement statistics measure repeatability of assignments; they are not accuracy percentages. Convergence does not establish useful topics or a global optimum, and this comparison estimates no AI-availability effect.

Manual content interpretation, final map selection/checking/freezing, and region-level event-window support remain pending. No region-description API pass has run. The prepared review packet samples 25 posts from each of 60 regions across four alternative maps: 1,500 appearances of 549 unique posts. These are candidate maps, not one final 60-region map.

## Find the materials

| Material | Location |
| --- | --- |
| Pilot goal, status, and remaining deliverables | [Pilot plan](docs/SEMANTIC_PILOT_PLAN.md) |
| Balanced A/B/C design and method grid | [Three-fold design](docs/THREEFOLD_SEMANTIC_PILOT.md) |
| Initial fitting configuration | [semantic_threefold_pilot_config.json](semantic_threefold_pilot_config.json) |
| Implementation and synthetic tests | [code](code/) and [code/tests](code/tests/) |
| Numerical results | [reports](reports/) |
| Content-review procedure | [Review guide](docs/SEMANTIC_REGION_REVIEW.md) |
| Proposed model-generated descriptions | [Description rubric](docs/SEMANTIC_REGION_DESCRIPTION_RUBRIC.md) and [API prompt/schema](code/prompts/) |
| Restricted inputs and version checks | [Data-access guide](data-access.md) |

## Run the synthetic checks

From this `semantic_pilot/` directory on Linux/macOS (or Windows through WSL), use a separate Python 3.12 environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-azure-semantic-pilot.txt pytest
.venv/bin/python -m pytest code/tests -q
```

GitHub runs this suite through the [test workflow](../.github/workflows/semantic-pilot-tests.yml) when the package changes. The included tests use synthetic fixtures and require neither the private corpus nor API credentials. Passing them checks implementation behavior; it does not reproduce the real-data results or validate semantic interpretation.

## Reproduce the recorded comparison

First obtain the authenticated private inputs and saved run metadata described in [data access](data-access.md). Reuse the completed embedding cache; rerunning embedding requests is unnecessary for the comparison. Preserve post IDs, text hashes, shard-row mappings, and saved A/B/C assignments. Earlier STM roles are provenance, not these new folds.

The root configuration records the **initial 100-iteration pass**. The converged result also requires the saved continuation configuration and source-run artifacts described in the [continuation record](docs/VMF_CONVERGENCE_EXTENSION.md). Use the converged report when comparing outputs. Keep reruns in a new results directory and retain the pinned numerical environment and source/version records.

Review reproducibility, coverage, content, and computational feasibility before selecting settings. Then fit, check, and freeze the chosen map on all 60,000 reference posts before applying it to event data. The reference sample alone cannot provide later event-window support.

The scripts under `code/jobs/` preserve the original Longleaf deployment commands, paths, and task array. Treat them as recorded examples: adapt scheduling and paths for a new run rather than submitting them unchanged. `run_semantic_region_pilot.py` is included as a dependency of the current workflow; its older standalone default configuration is not part of this package. The Azure preparation tool also requires explicit source/configuration inputs; the existing verified cache is the input for this comparison.

## Review blinding

Save initial content descriptions before consulting method identities, numerical diagnostics, or historical ratings, as specified in the [review guide](docs/SEMANTIC_REGION_REVIEW.md). The public code and recorded alias seeds can reconstruct map/region aliases: this is a procedural separation for the first reading, not a secret key or a guarantee that a reviewer cannot identify methods. Keep the initial reading packet separate from those materials.
