#!/bin/bash
#SBATCH --job-name=smart_bm25
#SBATCH --output=smart_trial/outputs/eval/logs/build_bm25_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#
# One-time (or refresh): build smart_trial/data/bm25_index.pkl for grid RAG.
#   sbatch smart_trial/config/eval/slurm_build_bm25.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/carc_setup.sh"

python -m smart_trial.scripts.build_bm25_index
