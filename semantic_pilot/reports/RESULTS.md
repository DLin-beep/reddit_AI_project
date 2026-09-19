## Optimization continuation

All 24 vMF fits now satisfy the original mean log-likelihood improvement tolerance of 1e-05. The original 100-iteration cutoff applied to the first experiment only. Continuation resumed the saved parameters with no total EM iteration limit; checkpoint chunks and scheduler deadlines were pause boundaries, not completion criteria. Training posts, evaluation posts, seeds, tolerance, and concentration cap were unchanged. Previously converged vMF fits and all other methods retain their original numerical outputs. Continuation elapsed times include data loading and checkpoint I/O; they are not directly comparable to the original pure fitting timers. See `vmf_convergence_summary.csv` for the exact iterations and stopping evidence for every vMF fit.

# Semantic-region measurement pilot

Numerical status: **complete** — 72 of 72 planned calculations completed. Human interpretation, final map selection, and event-data readiness remain pending.

The three balanced reference folds separate fitting from evaluation. Single-fold fits are compared on their common unseen third fold. Two-fold fits are evaluated on the remaining fold; comparison with either constituent fit measures consistency under a larger, overlapping training sample.

| Method configuration | Training posts | Evaluation runs | Median coverage | Median regions | Completed fits/calculations | Median fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| density_nn15_q0.5 | 20,000 | 6 | 69.9% | 1 | 3/3 | 12.6 |
| density_nn15_q0.5 | 40,000 | 3 | 66.5% | 1 | 3/3 | 28.3 |
| density_nn15_q0.8 | 20,000 | 6 | 36.1% | 2 | 3/3 | 6.7 |
| density_nn15_q0.8 | 40,000 | 3 | 35.1% | 4 | 3/3 | 1.3 |
| density_nn30_q0.5 | 20,000 | 6 | 82.5% | 1 | 3/3 | 0.9 |
| density_nn30_q0.5 | 40,000 | 3 | 78.2% | 1 | 3/3 | 1.4 |
| density_nn30_q0.8 | 20,000 | 6 | 50.4% | 1 | 3/3 | 0.7 |
| density_nn30_q0.8 | 40,000 | 3 | 46.2% | 3 | 3/3 | 1.3 |
| spherical_kmeans_k10 | 20,000 | 12 | 100.0% | 10 | 6/6 | 6.2 |
| spherical_kmeans_k10 | 40,000 | 6 | 100.0% | 10 | 6/6 | 14.0 |
| spherical_kmeans_k20 | 20,000 | 12 | 100.0% | 20 | 6/6 | 8.4 |
| spherical_kmeans_k20 | 40,000 | 6 | 100.0% | 20 | 6/6 | 12.2 |
| vmf_k10 | 20,000 | 12 | 100.0% | 10 | 6/6 | 58.8 |
| vmf_k10 | 40,000 | 6 | 100.0% | 10 | 6/6 | 56.1 |
| vmf_k20 | 20,000 | 12 | 100.0% | 20 | 6/6 | 89.1 |
| vmf_k20 | 40,000 | 6 | 100.0% | 20 | 6/6 | 79.4 |

| Comparison | Configuration | Comparisons | Median joint coverage | Median nondegenerate ARI | Comparisons with warnings |
|---|---|---:|---:|---:|---:|
| independent fit reproducibility | density_nn15_q0.5 | 3 | 59.0% | undefined | 3 |
| independent fit reproducibility | density_nn15_q0.8 | 3 | 27.2% | undefined | 3 |
| independent fit reproducibility | density_nn30_q0.5 | 3 | 74.9% | undefined | 3 |
| independent fit reproducibility | density_nn30_q0.8 | 3 | 41.0% | undefined | 3 |
| independent fit reproducibility | spherical_kmeans_k10 | 6 | 100.0% | 0.573 | 0 |
| independent fit reproducibility | spherical_kmeans_k20 | 6 | 100.0% | 0.573 | 0 |
| independent fit reproducibility | vmf_k10 | 6 | 100.0% | 0.615 | 0 |
| independent fit reproducibility | vmf_k20 | 6 | 100.0% | 0.615 | 0 |
| nested training size consistency | density_nn15_q0.5 | 6 | 59.5% | undefined | 6 |
| nested training size consistency | density_nn15_q0.8 | 6 | 28.6% | 0.165 | 2 |
| nested training size consistency | density_nn30_q0.5 | 6 | 73.8% | undefined | 6 |
| nested training size consistency | density_nn30_q0.8 | 6 | 40.7% | undefined | 6 |
| nested training size consistency | spherical_kmeans_k10 | 12 | 100.0% | 0.563 | 0 |
| nested training size consistency | spherical_kmeans_k20 | 12 | 100.0% | 0.586 | 0 |
| nested training size consistency | vmf_k10 | 12 | 100.0% | 0.646 | 0 |
| nested training size consistency | vmf_k20 | 12 | 100.0% | 0.578 | 0 |
| seed stability | spherical_kmeans_k10 | 9 | 100.0% | 0.531 | 0 |
| seed stability | spherical_kmeans_k20 | 9 | 100.0% | 0.563 | 0 |
| seed stability | vmf_k10 | 9 | 100.0% | 0.516 | 0 |
| seed stability | vmf_k20 | 9 | 100.0% | 0.581 | 0 |

## How to assess the results

`fit_summary.csv` gives coverage, convergence, runtime and available model diagnostics for each fit/evaluation fold. Rejected posts are reported as rejected, rather than treated as one extra semantic region. ARI is calculated only on jointly assigned posts and is left undefined for all-rejected or single-region joint partitions. Read it together with coverage and warnings.

`region_stability.csv` gives each region’s best and one-to-one Jaccard match in both directions, including members rejected by the other map. The `overlaps_json` field retains aggregate overlap counts and directional fractions so potential splits and merges remain visible. Numeric region IDs are local to each fit. A match is an empirical correspondence on this evaluation sample, not a semantic identity claim.

`region_sizes.csv` lists training/evaluation counts and flags regions below a descriptive count threshold or absent from evaluation. That threshold is not a guarantee of statistical reliability. vMF rows include effective training mass from mixture weights, hard assignment counts, and concentration where available. Mixture components are not automatically density peaks or meaningful topics.

There are 0 evaluation summaries with one or more vMF concentration-cap hits (a fit may occur in more than one summary). Check these and convergence before interpreting components. A cap may constrain cohesive content; it is not evidence of an AI effect.

`partition_stability.csv` separates independent-fit reproducibility and seed stability; `sample_size_comparison.csv` contains only nested training-size comparisons. The 20k-versus-40k comparisons share training posts and cannot establish independent reproducibility or prove improvement. For density, retaining a fixed neighbor count while increasing the training size also changes the neighbor-count-to-sample-size ratio and therefore the smoothing scale. This sensitivity belongs in the interpretation of any recovered smaller regions.

The private Longleaf report’s `representative_posts_restricted.csv` contains 9006 selected post examples, of which 9006 include hash-verified original text. Each available region packet draws up to three deterministic random examples, three cosine-central examples, and two lower-confidence examples where that method supplies a score; duplicates are removed. The random selection is drawn before central and confidence examples. A lower-confidence tail is not a calibrated boundary or comparable uncertainty across methods. These examples require manual reading; no interpretability winner has been selected.

The private report’s `review_assignments_restricted.csv` and `region_review_summary.csv` preserve Derek’s and Arya’s original personal-context scores separately and identify reviews used in fitting. These historically selected reviews are not a random sample of each new region. Their means and agreement describe the reviewed subset, not population personal-context prevalence. Personal context is one interpretation tool; use the blank description/other-explanation fields to consider other content patterns. No new ratings were generated.

Post-level files and review-score summaries are excluded from this public package. See [data access](../data-access.md) for the recorded private locations.

## Limits and next decision

Event-window sample sizes, posting rates, regional shares during availability events, and AI effects are **unavailable**, not zero. This reference-data pilot cannot estimate those outcomes. The balanced community design describes this reference sample rather than Reddit-wide prevalence. Split rotations reuse data, so their summaries are not independent replications or confidence intervals.

Use per-region reproducibility, coverage, convergence and manual interpretation together to shortlist a method and settings. Inspect failures and smaller regions before deciding whether additional fitting data help. Only after that decision should the chosen map be fitted on all 60,000 reference posts, checked, frozen, and applied to separate availability-event data. Discovery-selected responses must later be tested on held-out events.
