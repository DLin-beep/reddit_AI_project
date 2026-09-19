# Finish the vMF optimization

**Finished — September 14, 2026:** all 24 vMF fits now meet the original convergence rule. The 15 resumed fits converged at 119–274 total iterations. All 24 spherical k-means fits had already converged, and all 24 density calculations had finished. No numerical fit remains unfinished.

The first pilot ran all 72 scheduled calculations, but 15 of its 24 vMF fits reached 100 iterations before meeting the convergence rule. That was an iteration cutoff, not completed optimization. The original `semantic_threefold_20260914` experiment and its outputs remain unchanged as an archive of that initial pass.

- Each unfinished mixture resumed from its saved centers, weights, concentrations, and likelihood history. Training posts, evaluation groups, initialization, the concentration cap of 4,000, and the tolerance stayed unchanged.
- Continuation stopped only when the increase in mean log likelihood met the original `1e-5` rule. There was no total-iteration cutoff. Convergence describes numerical optimization, not proof of a global optimum or meaningful semantic regions.
- Checkpoints were saved every 10 iterations. Processing chunks and cluster wall-time boundaries were resumable pauses, never completion criteria.
- The 24 completed spherical k-means fits, 24 density calculations, and 9 already-converged vMF fits were reused. The updated report uses the final vMF assignments and the original evaluation posts, preserving both original scores for all 300 reviewed posts. No new embedding requests or human labels were needed.

The completed results are saved as `experiments/semantic_threefold_converged_20260914/results/` in the [private handoff](../data-access.md). Use the [latest aggregate report](../reports/SUMMARY.md). Models, checkpoints, and private post examples remain on Longleaf; this public package contains selected aggregate reports. The [public configuration](../semantic_threefold_pilot_config.json) retains `max_iter=100` for the original fit/initialization. Reproducing the completed result also requires the saved continuation configuration, source-run artifacts, and pinned numerical environment.

Next, review the revised stability results and region content before choosing and freezing a final semantic map. Manual interpretation, final map selection and freezing, and availability-event analysis remain pending.

## Reproduce the continuation

Reading the saved results needs no refitting. To reproduce the extension, use the authenticated original first-pass results and a new output directory on a Longleaf compute node. Install the [pinned environment](../README.md#run-the-synthetic-checks) and work from `semantic_pilot/`:

```bash
PILOT_DATA_ROOT=/path/to/authorized/data-root
PILOT_SOURCE="$PILOT_DATA_ROOT/experiments/semantic_threefold_20260914/results"
PILOT_EXTENSION=/path/to/new/continuation-results

.venv/bin/python code/run_vmf_convergence_extension.py \
  --source-results "$PILOT_SOURCE" --data-root "$PILOT_DATA_ROOT" \
  --output "$PILOT_EXTENSION" --prepare
```

Preparation reports `pending_task_ids` and records them in the new `extension.json`. For each reported task ID, run the following in a scheduled job, replacing `TASK_ID` with that integer:

```bash
.venv/bin/python code/run_vmf_convergence_extension.py \
  --source-results "$PILOT_SOURCE" --data-root "$PILOT_DATA_ROOT" \
  --output "$PILOT_EXTENSION" --task-id TASK_ID --wall-seconds 6600
```

Exit code 75 indicates a saved pause: rerun that task to resume it. Inspect other failures rather than treating them as completion. Once every pending task has converged:

```bash
.venv/bin/python code/run_vmf_convergence_extension.py \
  --source-results "$PILOT_SOURCE" --data-root "$PILOT_DATA_ROOT" \
  --output "$PILOT_EXTENSION" --aggregate
```

Keep the original source/results unchanged. The historical [job wrappers](../code/jobs/) show the original resources and task array; a new run must use its own pending IDs, paths, and scheduler configuration. Numerical reproducibility also depends on the recorded environment; the public synthetic tests do not rerun these real-data fits.
