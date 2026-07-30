#!/bin/bash
#SBATCH --job-name=mediq_ae
#SBATCH --output=mediq_experiment/outputs/logs/mediq_ae_%j.out
#SBATCH --error=mediq_experiment/outputs/logs/mediq_ae_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=ruishanl_1185
#SBATCH --partition=main
#
# USC CARC — MediQ multi-doctor (A–E). Skips non-empty A/B artifacts by default.
#
# Before first run:
#   1. hf auth login && hf download the three local models into HF_HOME (see README)
#   2. Ensure .env has OPENAI_API_KEY + ANTHROPIC_API_KEY
# HF_HOME default:
#   /project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/model
#
#   sbatch mediq_experiment/slurm_mediq_ae.sh
#
# Interactive alternative (GPU node):
#   export HF_HOME=...
#   bash mediq_experiment/run.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p mediq_experiment/outputs/logs

# Model weights / Hugging Face cache (project disk)
export HF_HOME="${HF_HOME:-/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/model}"
mkdir -p "$HF_HOME"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY missing (patient/judge Haiku)." >&2
  exit 1
fi

echo "MediQ A–E | HF_HOME=$HF_HOME | cwd=$ROOT"
python -m mediq_experiment.run_pipeline --config mediq_experiment/config.yaml
