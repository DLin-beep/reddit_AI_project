# Region review

Review the sampled posts to describe what each candidate region contains and whether its boundaries are useful. Descriptions may concern subject matter, expressed purpose, situations, or other recurring features. Personal context is one possible interpretation; no personal-versus-generic label is required.

## Sample

The prepared sample contains 25 posts from each region across four maps: spherical k-means and vMF, each with 10 and 20 regions. All four maps use the same 40,000 training posts and 20,000 evaluation posts. There are 60 region samples, containing 1,500 appearances of 549 unique posts. Posts recur because the maps overlap; these are not 60 separate divisions of the full corpus.

Posts were selected using a shared, reproducible random ranking, without reference to existing ratings, model confidence, or event responses. Twenty-five posts provides an initial reading; sample more where the content remains unclear.

## Human review

Read the sample before consulting the method identity, stability results, or previous ratings. Record:

- A short description of the common content.
- Whether that description covers the sample, with examples of exceptions.
- Mixed content, unclear boundaries, and overlaps with other regions.
- Any additional posts needed to clarify the interpretation.

Use `reviewer/reader.html` and the accompanying notes table, or the terminal reader:

```bash
python code/semantic_region_review_terminal.py \
  --posts /path/to/review_packet/reviewer/sampled_posts_restricted.csv --list
```

Choose the listed map and region with `--map` and `--region`. The terminal reader saves notes beside the input file. The method aliases support an independent first reading, but can be reconstructed from the code.

After recording the descriptions, compare them with stability and coverage. Existing personal-context ratings can provide additional context, although they were not sampled within these regions and do not represent each region as a whole.

## API-assisted descriptions

The [prompt](../code/prompts/semantic_region_description_v1.md) and [response schema](../code/prompts/semantic_region_description_v1.schema.json) are prepared; no description requests have been run. Each request uses the complete text of one region sample, labelled P01–P25, without method names, group counts, community metadata, historical ratings, or event results. The response should contain:

| Field | Requirement |
| --- | --- |
| Sample description | One or two sentences describing the supported commonalities and differences. |
| Patterns | Up to five observable patterns, each supported by at least two posts and one or two exact excerpts of no more than 20 words each. No pattern is required if none is supported. |
| Other content | Exceptions and individual posts not covered by a recurring pattern. |
| Uninterpretable posts | Posts with insufficient information, with a reason. |
| Uncertainties | Up to three entries covering the remaining ambiguities. |

Every post must be accounted for. Patterns may overlap. Descriptions should distinguish what authors discuss from what they explicitly seek, without inferring hidden motives or forcing a shared theme. Aim for 150–250 words excluding identifiers and excerpts, allowing more where needed to retain meaningful differences.

Before using the descriptions, check the response format, identifiers, complete post coverage, and quotation accuracy. A post marked uninterpretable must not also support an interpretation. Human checks should look for unsupported claims, invented purposes, and omitted exceptions. Include mixed samples and compare several responses after changing post order; read some samples before viewing their generated descriptions.

Revise the prompt if needed, check fresh examples, and record the version used. Save the model, input and prompt hashes, sampling settings, token usage, request status, and human corrections with the outputs. Text, excerpts, and generated descriptions remain with the [project data](../data-access.md). Describe these outputs as model-generated observations about the sample, not validated labels or estimates of whole-region prevalence.
