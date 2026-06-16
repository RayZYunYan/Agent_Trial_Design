#!/bin/bash
#SBATCH --job-name=smart_baseline
#SBATCH --output=smart_trial/outputs/eval/logs/baseline_%j.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# USC CARC — adjust account/partition before submit.
#   sbatch smart_trial/config/eval/slurm_baseline.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
python -m smart_trial.run_eval --config smart_trial/config/eval/config_baseline.yaml
