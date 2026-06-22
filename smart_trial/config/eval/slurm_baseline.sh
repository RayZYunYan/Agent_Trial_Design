#!/bin/bash
#SBATCH --job-name=smart_baseline
#SBATCH --output=smart_trial/outputs/eval/logs/baseline_%j.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#
# USC CARC Discovery — benchmark baseline (50 cases, Claude).
#   sbatch smart_trial/config/eval/slurm_baseline.sh
#
# Requires: ANTHROPIC_API_KEY in repo-root .env (see carc_setup.sh).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/carc_setup.sh"

python -m smart_trial.run_eval --config smart_trial/config/eval/config_baseline.yaml
