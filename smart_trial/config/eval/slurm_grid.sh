#!/bin/bash
#SBATCH --job-name=smart_grid
#SBATCH --output=smart_trial/outputs/eval/logs/grid_%j.out
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export PYTHONUNBUFFERED=1
python -m smart_trial.run_eval --config smart_trial/config/eval/config_grid.yaml
