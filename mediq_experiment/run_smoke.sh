#!/usr/bin/env bash
# Smoke run: 1 case → mediq_experiment/outputs_smoke (does not touch full outputs/)
# Usage (repo root, ideally on a GPU node):
#   bash mediq_experiment/run_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/hf_cache}"
mkdir -p "$HF_HOME"
echo "SMOKE | HF_HOME=$HF_HOME"

python -m mediq_experiment.check_setup --config mediq_experiment/config_smoke.yaml --require-models
python -m mediq_experiment.run_pipeline --config mediq_experiment/config_smoke.yaml "$@"
