# Post embeddings

The saved embeddings cover the 60,000 reference posts, with 300 posts from each of 200 communities. They were generated with UNC Azure `text-embedding-3-large` and normalized to unit length. Each vector contains 3,072 float32 values.

| File | Contents |
| --- | --- |
| `post_embedding_shards/` | 235 NumPy archives containing the vectors and their annotation identifiers |
| `post_embedding_index.csv` | Post identifiers, community/month metadata, and each vector's shard and row |
| `embedding_cache_manifest.json` | Model settings, corpus identity, and checksums for the shards |

Keep these files together. Match vectors to posts using the index and annotation identifiers. The index's `split_half` field is historical metadata; it is not the A/B/C split used in the method comparison.

The files contain post identifiers and associated metadata, but no post text or human ratings. Access to those materials and the saved models is described in the [data guide](../../semantic_pilot/data-access.md).

To read a shard with NumPy:

```python
import numpy as np

with np.load("data/embeddings/post_embedding_shards/posts_000000_000256.npz", allow_pickle=False) as shard:
    vectors = shard["embeddings"]
    annotation_ids = shard["annotation_ids"]
```

The saved vectors can be used directly; no new embedding requests are needed.
