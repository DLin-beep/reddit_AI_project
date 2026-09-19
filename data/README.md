# Data

The semantic pilot uses a fixed reference sample of 60,000 posts. See [data access](../semantic_pilot/data-access.md) for the corpus, embeddings, and saved sample splits.

The existing ACSI workflow uses the following local paths, relative to the repository root:

| Path | Contents |
| --- | --- |
| `data/raw_files/r_*_posts.jsonl` | Reddit source files |
| `output/latest/posts_clean_all.parquet` | Prepared input for annotation sampling |
| `data/acsi_data.csv` | Annotation sample |
| `data/acsi_annotated.csv` | Run-1 annotation records |
| `data/acsi_scores.csv` | Subreddit component scores |
| `output/latest/` | Generated analysis outputs |

Use the [annotation rubric](acsi_data_rubric.md) with [scripts/annotate.py](../scripts/annotate.py). Its default `build` command reads the prepared Parquet file. Setup and analysis commands are in the [repository README](../README.md).
