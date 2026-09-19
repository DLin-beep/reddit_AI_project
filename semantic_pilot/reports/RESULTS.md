# Pilot results

All 72 runs are complete: 24 spherical k-means fits and 24 vMF fits converged, and 24 density calculations finished. Fifteen vMF fits required continuation beyond the initial 100 iterations, reaching convergence at 119–274 total iterations. None reached the concentration cap.

## Stability

The main comparison fits maps on separate 20,000-post samples and compares their assignments on a common third sample. Each independent-fit median summarizes six comparisons across sample rotations and seeds. ARI measures agreement between entire partitions; it is not an accuracy percentage or a median over individual posts.

| Method | Regions | Median independent-fit ARI | Median seed-stability ARI |
| --- | ---: | ---: | ---: |
| Spherical k-means | 10 | 0.573 | 0.531 |
| Spherical k-means | 20 | 0.573 | 0.563 |
| vMF | 10 | 0.615 | 0.516 |
| vMF | 20 | 0.615 | 0.581 |

Seed-stability medians summarize nine comparisons on common evaluation posts. Both methods assign all evaluation posts. vMF has somewhat higher agreement across independent training samples, while seed stability is mixed. These results support further content review rather than selection of a clear winner.

The density settings produced one accepted region in 15 of 24 runs. In every run, the largest region contained at least 94.9% of assigned training posts. All 12 independent-fit density comparisons had undefined ARI because the jointly assigned posts formed a degenerate grouping. Coverage varied, so this does not mean the entire corpus belonged to one region. The tested construction did not provide a useful separation into several substantial regions.

## Numerical files

| File | Contents |
| --- | --- |
| [fit_summary.csv](fit_summary.csv) | Coverage, convergence, runtime, and model diagnostics for each fit/evaluation group |
| [partition_stability.csv](partition_stability.csv) | Independent-fit and seed comparisons |
| [region_stability.csv](region_stability.csv) | Region-level Jaccard matches and overlap counts; download to inspect the full table |
| [region_sizes.csv](region_sizes.csv) | Training/evaluation counts, mixture weights, and concentrations where applicable |
| [sample_size_comparison.csv](sample_size_comparison.csv) | Comparisons between 20k and 40k training samples |
| [vmf_convergence_summary.csv](vmf_convergence_summary.csv) | Iterations and stopping evidence for each vMF fit |

Region identifiers are specific to each fit; correspondence is established through overlap on common evaluation posts. ARI is calculated only for jointly assigned posts and should be read alongside coverage. The 20k/40k comparisons share training data, and fixed neighbor counts also change the relative smoothing scale for density regions. Continuation timings include loading and checkpoint operations, so they are not directly comparable with the original fitting timers.

The sample rotations reuse posts and do not constitute independent replications or confidence intervals. The balanced community sample describes the reference corpus rather than Reddit-wide prevalence.

Content review, final map selection, and regional event-window sample sizes remain pending. Once selected, the map will be fitted and checked on all reference posts and fixed for the availability analysis. That analysis will examine total volume, regional posting rates, and regional shares, with validation on held-out events. No AI-availability effect has been estimated in this pilot.
