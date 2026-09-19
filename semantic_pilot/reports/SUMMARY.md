# Completed semantic measurement pilot — September 14, 2026

**All 72 pilot fits are finished: 24 spherical k-means fits converged, 24 vMF fits converged, and 24 density calculations completed.** The 15 previously unfinished vMF fits resumed from their saved parameters and reached the original stopping criterion at **119–274 total iterations**. None stopped at an iteration cutoff.

The continuation kept the same posts, initializations, evaluation groups, convergence tolerance (`1e-5` improvement in mean log likelihood), and concentration cap (4,000). It used checkpoints without a total EM iteration limit. The other 57 fits retained their numerical results. All original experiment files remain unchanged, and no new embeddings were requested.

## Updated comparison

All 60,000 posts participate in the three-group design: maps fitted on separate 20,000-post groups are compared on a common evaluation group, alongside the corresponding 40,000-post fits. This is a comparison of candidate maps; a final map has not been selected or frozen.

| Method | Completed result | Interpretation |
|---|---|---|
| Spherical k-means | All 24 converged. Median independent-fit agreement is 0.573 for both 10 and 20 groups. | A useful benchmark whose regions still need content review. |
| vMF mixtures | All 24 converged. Median independent-fit agreement is 0.615 for both 10 and 20 components. | The unfinished-optimization concern is resolved. Agreement is somewhat higher here, but seed stability is mixed, so this does not establish a winner. |
| Density regions | All 24 calculations finished; 15 returned one region. Every fit's largest region contains at least 94.9% of assigned training posts. | These particular settings mainly recover one broad concentration, rather than several useful hot spots. |

Agreement is measured by adjusted Rand index (ARI), where 1 means identical grouping. It is not an accuracy percentage. All 12 independent density comparisons have undefined ARI because their jointly assigned posts form a degenerate grouping. Read agreement alongside coverage and individual-region overlap; the sample rotations reuse data and do not provide independent replications.

The full Longleaf report includes the updated assignments and all **144 planned comparisons**; this public package contains aggregate results. All 24 vMF likelihood histories passed the numerical checks, with no concentration-cap hits. Numerical convergence does not establish a global optimum or prove that a component is a meaningful topic. Continuation timings include loading and checkpoint work and should not be compared directly with the original fitting timers.

## Next decision

Review the content and boundaries of the converged vMF and spherical k-means maps, using individual-region stability and the existing annotations. The density construction needs revision if we want it to separate several hot spots. After selecting a method and settings, fit and check the final map on all 60,000 reference posts, then freeze it before availability-event analysis.

Both original scores for all **300 reviewed posts** remain preserved. They support interpretation, including explanations beyond personal context; they are not representative samples of each new region. No new human labels or AI-availability effects were produced. Suitable event-window data, manual interpretation, and held-out-event validation remain pending.

See the [full numerical report](RESULTS.md), [vMF stopping evidence](vmf_convergence_summary.csv), and [continuation verification](continuation_validation.json).

Models, checkpoints, and post-level review materials remain on Longleaf; the completed results and later review packet have separate locations listed in the [data-access guide](../data-access.md). This public package contains aggregate reports only. The original `semantic_threefold_20260914` results remain an unchanged record of the initial 100-iteration pass.
