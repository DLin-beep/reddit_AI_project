# Data access and corpus identity

The public repository contains methods, code, prompts, and aggregate results. The matching post-level files and saved models are recorded on Longleaf. Sharing this GitHub link does not grant access to those files; Derek and Arya should confirm access under their own accounts before transferring or rerunning anything.

## The five material groups

| Group | What to share | Location |
| --- | --- | --- |
| 1. Code and settings | Fitting, comparison, continuation, review preparation, dependencies, and synthetic tests | This [pilot package](README.md) |
| 2. Methodology and review instructions | Research plan, comparison design, description rubric, prompt/schema, and sampling instructions | [Methods](docs/) and [prompts](code/prompts/) |
| 3. Aggregate results and provenance | Stability, region sizes, convergence evidence, input identities, and source checksums | [Reports](reports/) and [package manifest](MANIFEST.json) |
| 4. Exact corpus and metadata | The 60,000 texts, identifiers, vector index, saved folds, and historical review metadata | Longleaf paths below |
| 5. Embeddings and post-level artifacts | Vector shards, saved models/assignments, sampled review posts, and individual ratings | Longleaf paths below |

## Recorded Longleaf layout

Data root: `/work/users/d/e/derekl/azure_semantic_corpus_20260914/`.

All paths below are relative to that root. These are recorded locations, not confirmation that a collaborator currently has access.

| Material | Relative path |
| --- | --- |
| Exact post texts | `experiments/semantic_threefold_20260914/inputs/texts_restricted.csv` |
| Vector index and manifest | `output/azure_semantic_corpus_20260914_inputs/azure_cache_4141107435ef8996/post_embedding_index.csv` and `embedding_cache_manifest.json` in the same directory |
| Embeddings | `output/azure_semantic_corpus_20260914_inputs/azure_cache_4141107435ef8996/post_embedding_shards/` |
| Historical roles and separate human ratings | `output/azure_semantic_corpus_20260914_inputs/roles.csv` and `human_reviews.csv` |
| Completed comparison and saved A/B/C folds | `experiments/semantic_threefold_converged_20260914/results/` |
| Original run needed to reproduce continuation | `experiments/semantic_threefold_20260914/results/` |
| Prepared review packet | `experiments/semantic_readiness_20260915/review_packet/` |

Share each embedding shard together with the index and manifest; a vector matrix without its row mapping is insufficient. The completed results directory contains `fold_assignments_restricted.csv`, configuration/input/task records, `extension.json`, and per-task models, predictions, diagnostics, and post assignments. Preserve both human raters' original scores separately.

For the initial content review, use the review packet's `reviewer/` materials before consulting `restricted_key/`, method identities, or numerical results. The public alias-generation code makes this procedural blinding, not a secret mapping. No region-description API responses have been produced yet.

## Check that the inputs match

The corpus is 300 posts from each of 200 communities. All 60,000 posts have normalized `text-embedding-3-large` vectors with 3,072 dimensions. Each saved A/B/C fold contains 20,000 posts. Equal counts alone do not establish identical inputs.

The [configuration](semantic_threefold_pilot_config.json) records full SHA-256 hashes for the index, embedding manifest, text file, roles, and human reviews. Use those hashes and the saved run provenance to verify identity. Join files using their recorded post/annotation identifiers and shard-row mapping; do not join independently sorted rows by position. Historical STM roles are separate from the A/B/C folds.

After obtaining the authorized inputs and installing the [pilot environment](README.md#run-the-synthetic-checks), run this from `semantic_pilot/` on a compute node with enough memory for the full embedding matrix:

```bash
PILOT_DATA_ROOT=/work/users/d/e/derekl/azure_semantic_corpus_20260914
.venv/bin/python code/run_semantic_threefold_pilot.py \
  --config semantic_threefold_pilot_config.json \
  --data-root "$PILOT_DATA_ROOT" \
  --preflight
```

Preflight verifies inputs and vectors without fitting or making embedding requests. It needs the private inputs; the public checkout alone cannot run it. Reuse the completed embeddings and saved fits for review. The published configuration describes the initial 100-iteration pass; the final vMF results also require the [continuation procedure](docs/VMF_CONVERGENCE_EXTENSION.md).

The public [manifest](MANIFEST.json) records hashes of the files in this package. It verifies the distributed code/results, while the configuration's input hashes identify the private corpus. No post text, vector rows, individual ratings, or access credentials are included in either manifest.
