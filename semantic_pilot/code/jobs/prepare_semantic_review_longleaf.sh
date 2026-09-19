#!/bin/bash
#SBATCH --job-name=semantic_review
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/semantic_review_%j.log

set -euo pipefail
unset UNC_AI_API_KEY
DATA_ROOT="/work/users/d/e/derekl/azure_semantic_corpus_20260914"
SOURCE_ROOT="$DATA_ROOT/experiments/semantic_threefold_converged_20260914"
EXPERIMENT_ROOT="$DATA_ROOT/experiments/semantic_readiness_20260915"
module load python/3.12.4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_ROOT/code"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$EXPERIMENT_ROOT"
exec "$DATA_ROOT/.venv-azure/bin/python" code/prepare_semantic_readiness_review.py \
  --results-dir "$SOURCE_ROOT/results" \
  --data-root "$DATA_ROOT" \
  --output "$EXPERIMENT_ROOT/review_packet" \
  --seed 20260915 --per-region 25
