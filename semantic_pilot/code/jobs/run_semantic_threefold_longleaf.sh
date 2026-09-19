#!/bin/bash
#SBATCH --job-name=semantic_3fold
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --array=0-71%4
#SBATCH --output=logs/semantic_3fold_%A_%a.log

set -euo pipefail
unset UNC_AI_API_KEY

DATA_ROOT="${SEMANTIC_DATA_ROOT:-/work/users/d/e/derekl/azure_semantic_corpus_20260914}"
EXPERIMENT_ROOT="${SEMANTIC_EXPERIMENT_ROOT:-$DATA_ROOT/experiments/semantic_threefold_20260914}"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'Submit this script as a SLURM array.' >&2; exit 2; }
[[ -f "$EXPERIMENT_ROOT/config.json" ]] || { echo 'Experiment configuration is missing.' >&2; exit 2; }

module load python/3.12.4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
cd "$EXPERIMENT_ROOT"

exec "$DATA_ROOT/.venv-azure/bin/python" code/run_semantic_threefold_pilot.py \
  --config "$EXPERIMENT_ROOT/config.json" \
  --data-root "$DATA_ROOT" \
  --output "$EXPERIMENT_ROOT/results" \
  --task-id "$SLURM_ARRAY_TASK_ID"
