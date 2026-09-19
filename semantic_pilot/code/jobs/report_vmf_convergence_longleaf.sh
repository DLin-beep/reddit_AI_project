#!/bin/bash
#SBATCH --job-name=vmf_report
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/vmf_report_%j.log

set -euo pipefail
unset UNC_AI_API_KEY
DATA_ROOT="${SEMANTIC_DATA_ROOT:-/work/users/d/e/derekl/azure_semantic_corpus_20260914}"
EXPERIMENT_ROOT="${SEMANTIC_EXPERIMENT_ROOT:-$DATA_ROOT/experiments/semantic_threefold_converged_20260914}"
module load python/3.12.4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$EXPERIMENT_ROOT"
exec "$DATA_ROOT/.venv-azure/bin/python" code/run_vmf_convergence_extension.py \
  --source-results "$DATA_ROOT/experiments/semantic_threefold_20260914/results" \
  --data-root "$DATA_ROOT" \
  --output "$EXPERIMENT_ROOT/results" \
  --aggregate
