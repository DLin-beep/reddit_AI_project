#!/bin/bash
#SBATCH --job-name=vmf_prepare
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/vmf_prepare_%j.log

set -euo pipefail
unset UNC_AI_API_KEY
DATA_ROOT="${SEMANTIC_DATA_ROOT:-/work/users/d/e/derekl/azure_semantic_corpus_20260914}"
EXPERIMENT_ROOT="${SEMANTIC_EXPERIMENT_ROOT:-$DATA_ROOT/experiments/semantic_threefold_converged_20260914}"
module load python/3.12.4
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$EXPERIMENT_ROOT"
exec "$DATA_ROOT/.venv-azure/bin/python" code/run_vmf_convergence_extension.py \
  --source-results "$DATA_ROOT/experiments/semantic_threefold_20260914/results" \
  --data-root "$DATA_ROOT" \
  --output "$EXPERIMENT_ROOT/results" \
  --prepare
