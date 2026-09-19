# Semantic distribution study: implementation plan

**Goal:** discover which regions of semantic space respond to changes in AI availability, then interpret those responses. Personal context is one possible explanation, assessed with the existing rubric; recovering it is not a requirement for a useful region map. No new STM fits are planned.

**Current stage — September 14:** all 60,000 Azure post embeddings are complete and verified on Longleaf. Numerical fitting for the 72-fit three-group comparison is now finished: all 24 spherical k-means and all 24 vMF fits converged, and all 24 density calculations finished. The initial 100-iteration pass left 15 vMF fits unfinished; continuation from their saved states brought them to convergence at 119–274 total iterations, using the unchanged tolerance and no total-iteration cutoff. See the [latest report](../reports/SUMMARY.md) and [continuation record](VMF_CONVERGENCE_EXTENSION.md). Manual interpretation and final map selection/freezing remain pending. No availability responses have been estimated in this pilot.

The embeddings remain usable under the clarified goal: they encode post text without personal-context labels. Use them for the small method comparison and later availability analysis described below.

**Scope clarification — September 15:** the numerical comparison is complete, but the full measurement pilot requested by Fan still needs manual interpretation and suitable event-window sample support. Interpretability means being able to describe a region's content from its posts. It does not require a personal-context label, and descriptive patterns do not establish an AI effect.

**Preparation completed — September 15:** a blinded four-map packet is ready on Longleaf: 25 random posts per region, 60 regions, and 549 unique posts across 1,500 appearances. A terminal reader saves manual notes on Longleaf; no descriptions have been entered. The event-data inventory confirms 25 readable source archives and seven dependencies, while event-text extraction, regional support, and genuinely untouched validation events remain pending. [Start the review](SEMANTIC_REGION_REVIEW.md).

**Proposed next step:** generate open-ended descriptions for the prepared packet using the [description rubric](SEMANTIC_REGION_DESCRIPTION_RUBRIC.md). Check initial outputs against the source posts and measure actual API usage before completing the packet. Expand reading where ambiguity remains. These would be model-generated descriptions with human checks; no paid description calls have been made.

**Next deliverable: the pilot report Yao requested to decide whether to proceed.** Use the existing comparison to finish these three items:

| Requested item | Current position | Next work |
|---|---|---|
| Region stability | Numerical comparisons are complete. | Summarize individual-region reproducibility, coverage, and failures alongside the content review. |
| Interpretability | The focused, blinded packet and terminal reader are ready; manual descriptions are pending. | Review the vMF and spherical k-means candidates at 10 and 20 regions. Describe what each region captures, including mixed or unclear content. Use the rubric as optional context. |
| Event-window sample sizes | Source archives are readable and old count panels exist; new regional support remains unavailable. | Use the [data-readiness audit](SEMANTIC_EVENT_DATA_READINESS.md) to establish windows, populations, coverage, and reference/discovery/validation roles. After selecting and freezing the reference map, assign development-event posts and report region/window support. Keep held-out responses uninspected. |

Keep this comparison bounded: use a small, recorded selection of existing maps for manual review, with the other fits supplying stability evidence; do not review all 72 fits separately. Record the density result as a limitation of the tested construction. Additional density, manifold, OT, or UOT work is not required to finish this pilot report. Prepare event-data requirements alongside the content review; assess readiness for availability analysis once the three items above are documented. Define development windows and coverage rules before the support audit, and use that audit to assess feasibility rather than to select a map for large event responses.

The current implementation uses UNC Azure embeddings. Reuse the verified cache for reproduction; see the [package instructions](../README.md#reproduce-the-recorded-comparison) and [data-access guide](../data-access.md).

**Storage:** private inputs, embedding caches, and saved model outputs remain on Longleaf. This package contains code, methods, and selected aggregate reports; use the [data-access guide](../data-access.md) for the separate coauthor handoff.

**1. Complete: embed all 60,000 posts through UNC Azure.**

Vectors were generated for the entire indexed 60,000-post corpus through UNC's `/openai/v1/embeddings` endpoint, including all 300 previously reviewed posts. Post identities, returned dimensions, and normalization were verified, and model settings were recorded. The full embedding cache can support the initial small method comparison and later larger runs without embedding the same posts again.

The UNC deployment `text-embedding-3-large` returned 3,072-dimensional vectors. All 60,000 normalized post vectors are complete and verified. These pre-release posts can supply reference data for map development; they do not supply text for later availability-event windows. The implemented measurement comparison uses all 60,000 reference posts through balanced three-group rotations.

**2. Complete: run the three-group comparison of three representations.**

The implemented design divides all 60,000 posts into three balanced groups of 20,000. Maps were fitted on two groups separately and compared on the third, rotating the evaluation group. Their 40,000-post union was also fitted and evaluated on the same third group to assess sample-size sensitivity. Existing historical roles remain provenance. See the [execution design](THREEFOLD_SEMANTIC_PILOT.md). This grid implements the reference-data comparison; Yao's message does not prescribe or endorse its exact sample split or settings.

| Method | What to implement |
|---|---|
| Density regions | Estimate local density from neighbor distances, build a neighborhood graph, and track connected high-density regions as the threshold changes. |
| vMF mixtures | Fit groups suited to normalized embeddings, retaining each post’s membership probabilities. Cap concentration and check for tiny or collapsed components. |
| Spherical k-means | Fit a simpler grouping baseline using the same embeddings. |

The three-group runner uses a separate configuration bound to the verified Azure inputs. The earlier 4k/4k template remains a historical smoke-test configuration. The grid has 72 fits across three single-group and three paired-group training sets. The original `semantic_threefold_20260914` experiment remains unchanged as an archive of the first pass; completed continuation results are saved as `experiments/semantic_threefold_converged_20260914/results/` in the [private handoff](../data-access.md). The public configuration records the initial `max_iter=100` pass; the saved continuation configuration removes the total EM iteration cutoff. Choose the representation using stability, coverage, interpretability, and computational feasibility on reference data. Numerical convergence alone does not establish a useful semantic map. Before examining availability responses, freeze the reference-post IDs, encoder settings, any dimension reduction, fitted regions, and assignment/poor-fit rules. Keep that map fixed across events.

**3. Reuse the 300 completed human reviews.**

Join the [existing ratings](../data-access.md) to the new region assignments by post ID. Preserve Derek’s and Arya’s separate scores: **0 means generic; 3 means deeply dependent on personal context.**

For each region, inspect reviewed-post coverage, representative content, boundaries, and disagreements. Record open-ended descriptions and plausible explanations alongside personal-context scores. The existing ratings were not sampled by these new regions and do not replace reading posts for descriptive interpretation. Where supplemental review is needed, start with roughly 20–30 randomly sampled posts per region, then add review where uncertainty remains; review very small regions in full where feasible. This is an initial screen, not a fixed precision guarantee. Record which fitted maps and sampling rules are used before reading the packet, and do not choose examples for high personal-context scores or expected AI responses.

**4. Decide which representation is useful.**

Use the completed comparisons across starts and training samples, which compare assignments on the same evaluation posts. Assess whether regions contain understandable content, remain recognizable across runs, and adequately represent posts, including uncertain assignments. Select the method using this reference-data evidence; neither personal-context alignment nor the size of later event responses is a selection criterion.

Existing unstable STM topics may be compared on the same posts as a measurement diagnostic. Instability does not establish an AI-availability effect. Any relationship between personal context and instability is an optional measurement question, not a criterion for selecting AI-responsive regions. A shared manifold representation can be a later extension if it improves the map.

**Pilot output:** a small method comparison and region table reporting stability, coverage, interpretability, annotation coverage/uncertainty, and event-window sample support. The support audit should show region/event/window post counts, represented communities, window durations, and data/sampling coverage for development events. Distinguish missing data from zero posts. Report unsupported windows as unavailable; the pre-release reference sample alone cannot fill this table for later events.

**5. Proceed to the availability analysis only after the measurement pilot.**

Obtain eligible text and counts for defined availability events and matched comparison windows. Specify event windows, target population, denominators, sampling weights, and the statistical design. Report three quantities separately under the frozen map:

- **Total posting volume:** the number of eligible posts across the target population in each window, with the window duration and an overall rate when comparing different durations.
- **Each region's posting rate:** eligible posts assigned to that region per unit time.
- **Each region's share:** that region's posts divided by all eligible posts in the same population and window.

Use consistent population definitions, coverage information, and weights across these quantities. A region's share can rise while its posting rate falls, so report both alongside total activity. The balanced reference sample is insufficient for estimating these population outcomes during later events.

Separate discovery events from genuinely held-out events before screening for large responses. Freeze the selected regions, hypotheses, and testing rules after discovery, then test them on the held-out events with appropriate uncertainty and multiplicity handling. Posts held out within an already inspected event do not make the event untouched. Use prespecified coverage/baseline eligibility rules, and do not inspect held-out response counts while choosing the map or regions.

Interpret responses using sampled content, the personal-context rubric, and other plausible explanations. OT/Sinkhorn, UOT, and MMD remain optional summaries of overall change; they do not replace region-level rates/shares or the availability-event design. Any causal interpretation depends on that design's assumptions.

**Implementation status:** the completed numerical comparison provides partition-level and individual-region stability, coverage, 20k/40k comparisons, existing-score summaries, and an open-ended review packet on Longleaf. Manual interpretation, final map selection/checking/freezing, a deployment-ready assignment pipeline, event-window support, and held-out-event response tests remain pending. The existing event-support audit covers three already-inspected episodes at aggregate count level, so it supplies neither region-level support nor untouched validation events.
