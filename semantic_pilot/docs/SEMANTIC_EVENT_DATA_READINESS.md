# Semantic event-data readiness — September 15, 2026

**The three inspected outage episodes have authenticated count panels and readable source archives. Event text, semantic scoring, and a frozen final map remain pending. No unused validation events have been established.**

This readiness audit uses existing metadata, extractor code, and prior discovery-count summaries. No text extraction, model fitting, or event-response estimation was performed. The supporting `event_data_readiness.json` contains exact phase geometry, required fields, source hashes, and verification results. It is outside this public package; see the [data-access guide](../data-access.md) for the supporting handoff.

## What is available

- Six local source hashes match the prior support audit. The exported outage files contain aggregate counts, durations, and flags, not text, post IDs, or region assignments.
- The September 15 remote inventory (`remote_event_source_inventory.json`; see [data access](../data-access.md) for supporting records) confirms that all **25 archives**, `RS_2022-12.zst` through `RS_2024-12.zst`, exist, are readable, and match historical byte sizes. Seven sample/config/author-filter dependencies are also readable. This is a metadata check, not fresh content-hash or decompression verification.
- Raw archives, the author table `pair_table.csv.gz`, and its two provenance summaries remain on Longleaf. They are not included in this public package; the [data-access guide](../data-access.md) covers the separate handoff.
- The **60,000 historical reference embeddings and 72 candidate-map fits are complete**. These reference posts predate November 2022; they do not supply event text. Final map selection/freezing and interpretation remain pending.

## Inherited discovery support

| Inspected episode | ACTIVE UTC intervals | Duration | Eligible posts | Communities with posts | Clean control weeks; nearest four offsets |
|---|---|---:|---:|---:|---|
| `2023_12_13` | Dec 14, 2023 01:32–02:10 | 2,280 s | 471 | 138/200 | 11; `+1, -2, +2, +3` |
| `2024_06_04` | Jun 4, 2024 06:49–11:10; 14:14–17:07 | 26,040 s | 4,962 | 196/200 | 15; `-1, +1, -3, +3` |
| `2024_11_07_08` | Nov 8, 2024 05:00–07:00; Nov 9 00:06–00:41 | 9,300 s | 1,652 | 185/200 | 16; `-1, +1, -2, +2` |

These are existing half-open UTC intervals, not newly verified incident times. December PRE and restoration overlap other catalog incidents; June PRE also overlaps one. November's adjacent phases and all three ACTIVE unions are catalog-clean.

The inherited extraction covers **10,053 selected hours**, not a continuous 25-month panel: its earliest/latest hour starts are December 17, 2022 01:00 and December 29, 2024 00:00. Configured corpus coverage ends January 1, 2025 exclusive. Six positive-offset November candidates fall outside it. Complete output cells establish bookkeeping, not complete capture of all Reddit submissions.

For reconciliation, the inherited population is submissions in the fixed 200 communities, excluding blank/placeholder accounts, AutoModerator, bot-suffix accounts, and authors outside the authenticated allowlist or above the corrected 50-post/day active-span rule. Comments are excluded. No reference-sampling cap or self-post-only restriction applies. **The author rule uses 2015–2024 history:** reusing this retrospective population and the three-episode calendar is a development option, not adoption of a final causal design. Population weights, region assignment, and any revised filtering need explicit decisions.

## What the event pilot needs

1. Select, check, and freeze the final map: input hashes, text construction, embedding identity, normalization, parameters, labels, assignment rule, and residual handling. Score event posts with that map without refitting it on event responses.
2. Record a discovery extraction specification and build restricted post-level text/metadata. Preserve exact timestamps, source identity, eligibility reasons, text hashes, and interval membership. Authenticate source bytes and reconcile counts. Record duplicate handling: the original loop counts retained archive records without explicit post-ID deduplication.
3. Measure text availability, embedding success, region coverage, and ordinary discovery-period variation. Missing extraction, no posts, and unscoreable posts must remain distinct. Counts alone do not establish region-level power. The supporting `event_data_readiness.json` lists the concrete extraction, scoring, and aggregate-panel fields.

For duration `T` hours, eligible posts `N`, scored posts `S`, and assigned region mass `M[r]`, report together:

- **Total volume:** `N`, with total posting rate `N/T`.
- **Region rate:** `M[r]/T`.
- **Primary region share:** `M[r]/N`, retaining unscoreable/unassigned mass as an explicit residual.
- **Coverage:** `S/N`; optional `M[r]/S` must be labelled conditional scored-post share.

With incomplete scoring, region rates/shares describe observed assigned mass; latent allocation of missing content remains unknown. If `N=0` in a completely extracted phase, counts/rates are zero and shares are undefined. If `N>0, S=0`, observed assigned mass is zero, with all mass residual and latent composition unknown. Missing extraction is never encoded as zero. Freeze pooled-post versus fixed-community weighting separately.

## Held-out events

New region outcomes do not make the three episodes untouched. Other inherited ChatGPT/API sets were also extracted, so their eligibility cannot be assumed. Genuine confirmation needs an access ledger covering prior text/count/plot/fit inspection, source-backed intervals, contamination, and calendar overlap; unknown history remains unknown. Authenticate unused eligibility and freeze measurement, population, contrasts, inference, multiplicity, and coverage/power rules before confirmation outcome access. Dates after December 2024 require separately authenticated corpus coverage. Semantic regions supply measurement, not causal identification.

Sources: the inherited `event_support_report.md`, `outage_hourly_extraction_manifest.json`, and `extract_chatgpt_outage_outcomes.py` are outside this public package (see [data access](../data-access.md)); the [current semantic report](../reports/SUMMARY.md) is included. Source fingerprints are in the supporting `event_data_readiness.json`.
