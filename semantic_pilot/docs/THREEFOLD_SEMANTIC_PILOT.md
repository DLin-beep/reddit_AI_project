# Semantic-region measurement pilot

**Numerical fitting finished — September 14, 2026:** all 24 spherical k-means and all 24 vMF fits converged, and all 24 density calculations finished on Longleaf. The 15 vMF fits left unfinished by the initial 100-iteration pass have now converged after continuation. Read the [latest findings and next decision](../reports/SUMMARY.md). No final map has been selected or frozen; human interpretation and event analysis remain pending.

## Purpose

Compare density regions, von Mises–Fisher (vMF) mixtures, and spherical k-means on the verified 60,000 Azure embeddings. Determine whether candidate regions reproduce across samples, cover enough posts, and have interpretable content. Personal-context ratings support interpretation; they do not determine the fitting sample or the winning method.

## Data and comparisons

- Divide the 300 posts from each of 200 communities into groups A, B, and C, with 100 posts per community in each group: **20,000 posts per group**. Use deterministic, seeded allocation independent of the existing human scores. Preserve earlier STM training/evaluation roles as provenance, rather than constraints on these new groups. These historical posts have been examined in earlier work; this is exploratory measurement validation.
- For each method and setting, fit A and B separately and compare their assignments on C. Rotate the evaluation group. Compare the same evaluation posts using label-invariant agreement, regional overlap, and coverage, including rejected posts.
- Fit the corresponding union of two groups (**40,000 posts**) and evaluate on the third group. Compare this with the two 20,000-post fits. This is a sample-size sensitivity check: the fitting samples overlap, so agreement with the larger fit is not an independent replication or proof that it is better.
- Assess seed variation separately from variation between fitting samples. The three rotations share data and are not independent experiments or confidence intervals.

## Bounded method grid

| Method | Settings |
|---|---|
| Spherical k-means | 10 or 20 groups; seeds 11 and 29 |
| vMF mixtures | 10 or 20 components; seeds 11 and 29; concentration cap 4,000 |
| Density regions | 15 or 30 neighbors; retain the densest 50% or 20%; minimum region size 30 |

The six fitting sets (A, B, C, AB, AC, BC) produce **72 distinct fits**. The initial pass allowed 100 iterations per stochastic fit: all 24 spherical k-means fits and 9 vMF fits converged, while 15 vMF fits reached the limit. Those 15 were resumed from their saved parameters, without reinitialization, and converged at 119–274 total iterations. The continuation retained the original `1e-5` improvement-in-mean-log-likelihood tolerance and concentration cap of 4,000, with no total-iteration cutoff. Checkpoints and cluster wall-time boundaries were resumable pauses, not completion criteria. Report convergence, objective behavior, and vMF cap hits; convergence is not proof of a global optimum or useful semantic regions. See the [continuation record](VMF_CONVERGENCE_EXTENSION.md).

The density model is a neighbor-based concentration graph, not a normalized probability density. Its neighbor geometry is cached and reused across density thresholds. The absolute 30-post minimum is a pilot setting whose effective share changes with fitting size; it is not a scientifically established minimum. Fixed neighbor counts also examine a smaller share of the data in a 40k fit. Density differences therefore reflect both increased fitting data and this change in resolution, rather than a pure sample-size effect.

Jobs ran on Longleaf with bounded concurrency and per-task checkpoints. Completed results are saved as `experiments/semantic_threefold_converged_20260914/results/` in the [private handoff](../data-access.md). The [public configuration](../semantic_threefold_pilot_config.json) records the initial `max_iter=100` pass; the saved continuation configuration governs the completed vMF extension. The original `semantic_threefold_20260914` experiment remains an unchanged archive of the first pass. Original embeddings and their integrity manifests remain unchanged. No additional embedding requests were needed, and private post examples remain on Longleaf.

## Output and interpretation

The completed report includes fitting diagnostics and runtime, assignment coverage, partition and individual-region reproducibility, split/merge overlap diagnostics, and the 20k-versus-40k comparison. All 300 existing reviews were matched by authenticated post identity, preserving Derek's and Arya's separate scores and whether a reviewed post was in the fitting sample.

The saved run includes a restricted example packet with central, randomly selected, and low-confidence examples where available; obtain it through the [private handoff](../data-access.md). The separate [focused first-reading packet](SEMANTIC_REGION_REVIEW.md) uses 25 randomly sampled posts per region across four existing maps. These examples support human interpretation; selecting examples or summarizing old scores does not create new human labels. Existing reviews were not sampled within the new regions and are not representative estimates of each region's personal-context dependence.

The current corpus ends before the later availability events. Region-level event-window sample counts therefore remain unavailable until suitable event data are supplied. The later analysis must report total posting volume, each region's posting rate, and each region's share separately, with consistent population and window definitions. None of these event outcomes has been estimated by this reference-data pilot. Numerical completion leaves manual descriptive interpretation and event-window support pending; it is not completion of every requested measurement-pilot deliverable.

After reviewing reproducibility, coverage, content, and sample-size sensitivity, select a method and settings. A final fit on all 60,000 reference posts must be checked and frozen before availability-response discovery and held-out-event validation. This pilot does not automatically select or freeze that final map.

## Rationale

The separation of two fitting samples from a shared evaluation sample adapts the evaluation logic of [Rinaldo et al., *Stability of Density-Based Clustering*](https://jmlr.org/papers/volume13/rinaldo12a/rinaldo12a.pdf). Their theoretical results concern kernel density estimators and do not automatically apply to this neighbor-graph implementation. Equal thirds provide a concrete reproducibility design, not an optimality guarantee. The current diagnostics are not the prediction-strength procedure, which has a different definition.
