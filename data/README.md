# Data guide

The public repository contains code and rubrics. The local data paths below describe inputs and generated artifacts; they are not download links or a claim that the data are included.

## Semantic pilot: the same 60,000 posts

Use the [semantic data-access guide](../semantic_pilot/data-access.md) to obtain the restricted coauthor handoff and verify corpus identity, text hashes, embedding cache, and saved A/B/C folds. Matching a row count is insufficient: collaborators must use the same post IDs and text versions.

The [semantic pilot](../semantic_pilot/README.md) uses the verified September 14, 2026 Azure corpus. Its prepared review packet contains 1,500 post appearances across four candidate maps, drawn from 549 unique posts; that packet is not the full 60,000-post corpus. The older ACSI files below are a separate workflow and do not establish semantic-corpus identity.

Raw texts, tokenized content, embedding shards, post-level indexes/assignments, individual ratings, review excerpts, and interpretation keys remain outside this public package. Follow the access guide for the coauthor data handoff.

## Existing ACSI workflow

Paths are relative to the repository root:

| Path | Role |
| --- | --- |
| `data/raw_files/r_*_posts.jsonl` | Locally supplied Reddit source files, such as `r_writing_posts.jsonl` |
| `output/latest/posts_clean_all.parquet` | Prepared input used by the default annotation `build` command |
| `data/acsi_data.csv` | Annotation sample produced by `scripts/annotate.py build` |
| `data/acsi_annotated.csv` | Run-1 completed annotation records |
| `data/acsi_scores.csv` | Aggregated subreddit component scores |
| `output/latest/` | Generated analysis outputs |

Use [the data rubric](acsi_data_rubric.md) with the [annotation entry point](../scripts/annotate.py). Setup and commands are in the [root README](../README.md#existing-acsi-annotation-and-analysis-workflow). The [annotation_code](../annotation_code/) directory preserves the existing supporting scripts and rubric.
