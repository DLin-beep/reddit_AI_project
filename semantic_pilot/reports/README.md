# Completed numerical results

Start with [SUMMARY.md](SUMMARY.md), then [RESULTS.md](RESULTS.md). These are the completed September 14–15, 2026 measurement-pilot results, including the continuation of unfinished vMF fits. They do not estimate an AI-availability effect.

| File | Contents |
| --- | --- |
| [fit_summary.csv](fit_summary.csv) | Fit/evaluation coverage, convergence, runtime, and model diagnostics |
| [partition_stability.csv](partition_stability.csv) | Whole-partition comparisons, including independent-fit and seed stability |
| [region_stability.csv](region_stability.csv) | Region correspondences, Jaccard matches, and aggregate overlap counts; download this larger file to inspect it |
| [region_sizes.csv](region_sizes.csv) | Training/evaluation region sizes and available mixture diagnostics |
| [sample_size_comparison.csv](sample_size_comparison.csv) | Nested 20k-versus-40k training comparisons |
| [vmf_convergence_summary.csv](vmf_convergence_summary.csv) | Stopping evidence for all 24 converged vMF fits |
| [source_deployment_manifest.json](source_deployment_manifest.json) | Recorded code hashes and source/completed experiment locations |
| [continuation_validation.json](continuation_validation.json) | Aggregate checks that the original experiment was preserved and continuation finished |

The CSV files and verification receipt are unchanged from the recorded outputs. The Markdown reports have only portability, artifact-location, and terminology clarifications. The source and report checksums are in [MANIFEST.json](../MANIFEST.json).

Post-level assignments, examples, original human ratings, and saved models are provided through the [data handoff](../data-access.md). The 9,006 representative examples described in the original report are distinct from the later review packet's 1,500 appearances of 549 unique posts. Neither packet is a completed interpretation result.
