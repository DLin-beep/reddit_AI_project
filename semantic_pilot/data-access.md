# Data

Post text, embeddings, saved models, assignments, and individual ratings are stored on Longleaf. Access is arranged separately from this repository.

The corpus contains 60,000 posts, with 300 from each of 200 communities. Embeddings use `text-embedding-3-large`, with 3,072 dimensions and unit-length normalization. The saved A/B/C folds contain 20,000 posts each.

Data root: `/work/users/d/e/derekl/azure_semantic_corpus_20260914/`.

| Material | Path relative to the data root |
| --- | --- |
| Post text | `experiments/semantic_threefold_20260914/inputs/texts_restricted.csv` |
| Embeddings, index, and manifest | `output/azure_semantic_corpus_20260914_inputs/azure_cache_4141107435ef8996/` |
| Historical roles and human ratings | `output/azure_semantic_corpus_20260914_inputs/roles.csv` and `human_reviews.csv` |
| Completed fits, assignments, and folds | `experiments/semantic_threefold_converged_20260914/results/` |
| Initial fits used for vMF continuation | `experiments/semantic_threefold_20260914/results/` |
| Review sample and notes | `experiments/semantic_readiness_20260915/review_packet/` |

Keep `post_embedding_shards/`, `post_embedding_index.csv`, and `embedding_cache_manifest.json` together. The saved experiment includes `config.json`, `inputs.json`, `tasks.json`, `extension.json`, fold assignments, and each task's model and predictions; these files are required for reproduction.

The [configuration](semantic_threefold_pilot_config.json) records input checksums. Match files by their recorded post/annotation identifiers and vector index rather than row position. Use the saved folds, and preserve the two raters' original scores separately. Historical STM roles are distinct from the A/B/C folds.

For the first content review, use the sample in `reviewer/` before consulting `restricted_key/`, model diagnostics, or historical ratings. See the [review procedure](docs/REVIEW.md).
