#!/bin/bash
#SBATCH --job-name=smart_grid
#SBATCH --output=smart_trial/outputs/eval/logs/grid_%j.out
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=YOUR_CARC_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#
# USC CARC — benchmark grid (50 cases × 9 paths, Claude + RAG).
# Build BM25 index first if missing: sbatch smart_trial/config/eval/slurm_build_bm25.sh
#   sbatch smart_trial/config/eval/slurm_grid.sh
#
# Default --resume skips finished (case_id, path_id) pairs in aggregate JSONL.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/carc_setup.sh"

python -m smart_trial.run_eval --config smart_trial/config/eval/config_grid.yaml
