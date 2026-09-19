# Methods

The pilot compares three ways of grouping Reddit posts by semantic content. Its purpose is to establish a useful measurement before examining responses to AI availability. Personal context may help interpret a region, but it is not a criterion for defining or selecting regions.

## Data and comparison

The reference corpus contains 60,000 posts: 300 from each of 200 communities. Posts were embedded with Azure `text-embedding-3-large` and normalized to unit length. Each vector has 3,072 dimensions.

Posts were divided into three groups, A, B, and C, each containing 100 posts per community. Two 20,000-post groups were fitted separately, and their assignments were compared on the same third group. Rotating the groups tests whether similar regions are recovered from different training samples.

Corresponding 40,000-post fits were also evaluated on the remaining group. These comparisons assess sensitivity to training size; they share training posts and are not independent replications. Seed variation was assessed separately. The rotations reuse data, so their summaries should not be treated as independent experiments.

## Models

| Method | Settings | Fitting procedure |
| --- | --- | --- |
| Spherical k-means | 10 or 20 regions; seeds 11 and 29 | Alternate assignment by cosine similarity and updates to normalized region centers. |
| von Mises–Fisher (vMF) mixtures | 10 or 20 components; seeds 11 and 29; concentration capped at 4,000 | Initialize from spherical k-means, then use expectation–maximization to update membership probabilities, component directions, weights, and concentrations. |
| Density regions | 15 or 30 neighbors; retain the densest 50% or 20%; minimum region size 30 | Use neighbor distances as a density proxy and connected components of a neighborhood graph to define regions. Posts outside accepted regions can remain unassigned. |

The six training sets—A, B, C, AB, AC, and BC—give 72 runs. The density construction is a graph-based concentration measure, not a fitted probability density. Its fixed neighbor counts and minimum region size have different relative scales in 20k and 40k training samples.

The initial configuration allowed 100 iterations. Fifteen vMF fits reached that limit and were resumed from their saved parameters until the original mean log-likelihood improvement criterion of `1e-5` was met. They required 119–274 total iterations. The data, initialization, tolerance, and concentration cap were unchanged. All 24 vMF and 24 spherical k-means fits converged; all 24 density calculations completed.

## Evaluation

Adjusted Rand index (ARI) compares complete group assignments on common evaluation posts. Region-level Jaccard overlap measures correspondence between individual regions. Coverage records the fraction of posts assigned by each map. ARI is calculated on jointly assigned posts and is left undefined when that comparison is degenerate. Region size, convergence, and concentration diagnostics accompany the stability results.

These measures assess repeatability rather than semantic accuracy. Content review is needed to determine what the regions capture. The review uses four maps fitted on AB and evaluated on C: both directional methods at 10 and 20 regions, using seed 11. The [review procedure](REVIEW.md) describes the sampling and interpretation.

After choosing a method and settings, the selected map will be fitted and checked on all reference posts, then fixed before event analysis. That analysis will report total posting volume, regional posting rates, and regional shares separately. Regions selected for large responses will require validation on held-out events. The present pilot does not estimate an AI effect.
