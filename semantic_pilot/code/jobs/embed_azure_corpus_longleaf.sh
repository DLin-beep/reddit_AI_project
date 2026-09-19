#!/bin/bash
#SBATCH --job-name=azure_embed_60k
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=%x_%j.log

# Submit from the transferred project root with UNC_AI_API_KEY exported:
#   sbatch --export=ALL code/jobs/embed_azure_corpus_longleaf.sh
# The existing batch cache must travel with the prepared directory to resume.
# Azure requests require outbound HTTPS from the assigned compute node.

set -euo pipefail

fail() { printf '%s\n' "$1" >&2; exit 2; }

ROOT="${AZURE_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "$ROOT" ]] || fail 'Set AZURE_PROJECT_ROOT or submit with sbatch from the transferred project root.'
[[ -d "$ROOT" ]] || fail 'The configured project root does not exist.'
ROOT="$(cd "$ROOT" && pwd)"
cd "$ROOT"

PREPARED="$ROOT/output/azure_semantic_corpus_20260914_inputs"
PYTHON="${AZURE_VENV_DIR:-$ROOT/.venv-azure}/bin/python"
[[ -f "$ROOT/code/build_azure_semantic_pilot.py" ]] || fail 'The project root is missing code/build_azure_semantic_pilot.py.'
[[ -f "$PREPARED/plan.json" ]] || fail 'Transfer the full prepared corpus directory, including its completed Azure cache.'
[[ -x "$PYTHON" ]] || fail 'Create .venv-azure on Longleaf and install requirements-azure-semantic-pilot.txt first.'
[[ -n "${UNC_AI_API_KEY:-}" ]] || fail 'Export UNC_AI_API_KEY before submitting the job.'
[[ "$UNC_AI_API_KEY" != '123' ]] || fail 'Replace the example API key locally before submitting the job.'

module load python/3.12.4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Check the environment and transferred inputs without making an API request.
PYTHONPATH="$ROOT/code" "$PYTHON" - "$PREPARED" <<'PY'
import sys
from pathlib import Path
from build_azure_semantic_pilot import verify_prepared

if not (3, 9) <= sys.version_info < (3, 13):
    raise SystemExit('Python 3.9-3.12 is required by the frozen dependencies.')
plan = verify_prepared(Path(sys.argv[1]))
if plan.get('embedding_scope') != 'all_indexed_posts' or plan['unique_posts'] != 60000:
    raise SystemExit('Expected the prepared all-60,000-post corpus.')
print('Prepared corpus verified: 60,000 posts; completed batches will be reused.', flush=True)
PY

exec "$PYTHON" code/build_azure_semantic_pilot.py embed \
  --prepared "$PREPARED" \
  --model text-embedding-3-large \
  --request-timeout 120 \
  --data-root "$ROOT"
