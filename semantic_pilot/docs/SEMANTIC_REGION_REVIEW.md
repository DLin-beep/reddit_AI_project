# Review the candidate semantic regions

The purpose is to decide whether the maps provide understandable, reproducible descriptions of content before studying AI availability. Personal context is optional interpretation, not a required label. The packet is prepared and verified, but manual interpretation has not started. Obtain the restricted packet through the [data-access guide](../data-access.md); it is not included in this public package.

## Prepared packet

Four existing maps: spherical k-means and vMF, each with 10 and 20 regions. All use the same AB reference training set (40,000 posts), seed 11, and evaluation set C (20,000 posts). This convention was chosen before reading the new packet; it is not a selection of the best-looking fit. The other completed fits supply stability evidence. This is not the final 60,000-post map.

For each evaluation region, select 25 posts by a reproducible random ranking, or all posts if fewer than 25 exist. Selection does not use old scores, centrality, confidence, or expected event responses. A shared ranking allows the same posts to recur where maps overlap. The completed packet contains 60 regions, 1,500 post appearances, and 549 unique posts; repeated appearances are not different posts or independent observations.

Method names, stability results, old ratings, community metadata, and original identifiers are hidden in the first reading view. Post wording may still reveal its context. The key and both original scores for the 300 previously reviewed posts are kept separately for later interpretation. The public code and recorded alias seeds can reconstruct map/region aliases: blinding is a procedural separation during the first reading, not a secret-key guarantee. Keep code that reconstructs aliases, method diagnostics, and historical ratings out of that reading session.

## How to review

Use the terminal reader below on Longleaf, or open `reviewer/reader.html` and record notes in the accompanying region-notes table. Start with the two maps containing 10 regions; then use the 20-region maps to judge whether the finer divisions add useful distinctions. This is an ordering for the review, not a decision to prefer 10 regions.

For each region, record:

- **Description:** one or two sentences explaining what the posts have in common.
- **Coherence:** whether that description covers most posts, with examples of exceptions.
- **Overlap or mixed content:** themes that seem combined or difficult to distinguish from another region.
- **Uncertainty:** what remains unclear and whether more randomly sampled posts are needed.
- **Other explanations:** any meaningful characteristic, including personal context if it helps.

Twenty-five posts is an initial screen, not a guarantee of precision. A region can be useful without a single neat topic label. Do not force a personal/generic classification. Existing scores do not replace this descriptive reading, and new solo review does not create evidence of agreement between two raters.

After the initial descriptions are saved, consult the method key, stability evidence, and historical-score context. Compare content quality, coverage, and reproducibility together. Record the choice and its limitations before fitting, checking, and freezing the selected map on all reference posts.

## Read and save notes with the private packet

After obtaining access, work on Longleaf from the public package's `semantic_pilot/` directory. Set `SEMANTIC_REVIEW_PACKET` to the actual private `review_packet/` directory supplied in the [handoff](../data-access.md); the path below is a placeholder.

```bash
SEMANTIC_REVIEW_PACKET=/path/to/private/review_packet
python3 code/semantic_region_review_terminal.py --posts "$SEMANTIC_REVIEW_PACKET/reviewer/sampled_posts_restricted.csv" --list
```

Choose a map/region from the list, for example:

```bash
python3 code/semantic_region_review_terminal.py --posts "$SEMANTIC_REVIEW_PACKET/reviewer/sampled_posts_restricted.csv" --map "Map B" --region "Region 01"
```

Press Enter to advance between posts, then answer the short prompts and type `s` to save. Notes are stored as `manual_region_notes.json` beside the private sampled-post CSV. Saving a region preserves notes for other regions; quitting leaves saved notes unchanged. The terminal notes and blank CSV are alternative ways to record the same review, so use one consistently.

## Completion and storage

The packet is preparation for manual review; its creation does not mean interpretability has been established. No AI effect is measured here. Event-data feasibility is recorded separately in the [event-data audit](SEMANTIC_EVENT_DATA_READINESS.md).

Private text, identity keys, and historical-score mappings stay in the restricted Longleaf packet described in [data access](../data-access.md). This public package contains the reader code and procedure, not the private packet or completed notes. Completed model outputs remain unchanged. No region-description API pass has run; the [description rubric](SEMANTIC_REGION_DESCRIPTION_RUBRIC.md) records the proposed model-assisted step.
