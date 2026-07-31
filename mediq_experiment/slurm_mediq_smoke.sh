#!/bin/bash
#SBATCH --job-name=mediq_smoke
#SBATCH --output=mediq_experiment/outputs_smoke/logs/mediq_smoke_%j.out
#SBATCH --error=mediq_experiment/outputs_smoke/logs/mediq_smoke_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=ruishanl_1185
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#
# Discovery GPU smoke (1 case → outputs_smoke/).
# If a40 is unavailable, change to: --gres=gpu:v100:1  or  --gres=gpu:a100:1
#   sbatch mediq_experiment/slurm_mediq_smoke.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p mediq_experiment/outputs_smoke/logs

export HF_HOME="${HF_HOME:-/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/hf_cache}"
mkdir -p "$HF_HOME"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

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

echo "SMOKE | HF_HOME=$HF_HOME | cwd=$ROOT | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python -m mediq_experiment.check_setup --config mediq_experiment/config_smoke.yaml --require-models
python -m mediq_experiment.run_pipeline --config mediq_experiment/config_smoke.yaml
