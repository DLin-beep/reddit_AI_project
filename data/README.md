# Data Directory

This directory is intentionally mostly ignored by Git.

Tracked lightweight files:

- `acsi_data_rubric.md`: current ACSI post measurement rubric.
- `acsi_scores.csv`: subreddit-level ACSI component scores used by the model.

Ignored local files:

- `r_*_posts.jsonl` and `r_*_comments.jsonl`: raw Reddit exports.
- `acsi_data.csv`: sampled post text/input rows for manual measurement.
- `acsi_measurement_sample*.csv`: measured post-level rows with post excerpts and rationales.
- `cache/`: regenerated monthly aggregate caches.

Keep raw Reddit files and post-level measurement samples local unless you have
explicit permission to distribute them.
