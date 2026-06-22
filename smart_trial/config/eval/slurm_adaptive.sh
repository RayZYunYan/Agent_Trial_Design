#!/bin/bash
#SBATCH --job-name=smart_adaptive
#SBATCH --output=smart_trial/outputs/eval/logs/adaptive_%j.out
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=ruishanl_1185
#SBATCH --partition=oneweek
#
# USC CARC Discovery — Phase-2 closed-loop adaptive (medqa_0100..0149).
# Requires: grid_encounters.jsonl for initial_q_from (Phase 1).
#   sbatch smart_trial/config/eval/slurm_adaptive.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/carc_setup.sh"

python -m smart_trial.run_eval --config smart_trial/config/eval/config_adaptive.yaml
