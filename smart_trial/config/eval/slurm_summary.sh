#!/bin/bash
#SBATCH --job-name=smart_summary
#SBATCH --output=smart_trial/outputs/eval/logs/summary_%j.out
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=YOUR_CARC_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#
# Run after baseline + grid jobs complete.
#   sbatch smart_trial/config/eval/slurm_summary.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/carc_setup.sh"

BASELINE="smart_trial/outputs/eval/baseline/baseline_encounters.jsonl"
GRID="smart_trial/outputs/eval/grid/grid_encounters.jsonl"
OUT="smart_trial/outputs/eval/summary_metrics.csv"

if [[ ! -f "$BASELINE" ]]; then
  echo "ERROR: missing $BASELINE — run slurm_baseline.sh first" >&2
  exit 1
fi
if [[ ! -f "$GRID" ]]; then
  echo "ERROR: missing $GRID — run slurm_grid.sh first" >&2
  exit 1
fi

python -m smart_trial.eval.summary_metrics \
  --baseline "$BASELINE" \
  --grid "$GRID" \
  --out "$OUT"

echo "Wrote $OUT and by_category/*.jsonl under baseline/ and grid/"
