#!/usr/bin/env bash
# Mac mini full A–E run (100 cases → outputs/; skips non-empty existing artifacts)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
# Slightly more Metal allocator headroom for large MLX loads
export MLX_METAL_PATH_CACHE="${MLX_METAL_PATH_CACHE:-$HOME/.cache/mlx}"

echo "MAC FULL | HF_HOME=$HF_HOME | cwd=$ROOT | accelerator=mlx"
python3 -m mediq_experiment.check_setup --config mediq_experiment/config.yaml --require-models
python3 -m mediq_experiment.run_pipeline --config mediq_experiment/config.yaml "$@"
