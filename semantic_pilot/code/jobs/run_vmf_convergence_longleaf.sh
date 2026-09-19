#!/bin/bash
#SBATCH --job-name=vmf_finish
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --array=4,6,7,16,17,19,28,29,30,31,41,43,55,64,66%4
#SBATCH --output=logs/vmf_finish_%A_%a.log

set -euo pipefail
unset UNC_AI_API_KEY
DATA_ROOT="${SEMANTIC_DATA_ROOT:-/work/users/d/e/derekl/azure_semantic_corpus_20260914}"
SOURCE_ROOT="$DATA_ROOT/experiments/semantic_threefold_20260914"
EXPERIMENT_ROOT="${SEMANTIC_EXPERIMENT_ROOT:-$DATA_ROOT/experiments/semantic_threefold_converged_20260914}"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo 'Submit as an array with the original unfinished task IDs.' >&2; exit 2; }
module load python/3.12.4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
cd "$EXPERIMENT_ROOT"

# The wall limit is a checkpoint boundary, not a convergence criterion.
# Exit 75 means submit this task again to resume its committed checkpoint.
exec "$DATA_ROOT/.venv-azure/bin/python" code/run_vmf_convergence_extension.py \
  --source-results "$SOURCE_ROOT/results" \
  --data-root "$DATA_ROOT" \
  --output "$EXPERIMENT_ROOT/results" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --wall-seconds 6600
