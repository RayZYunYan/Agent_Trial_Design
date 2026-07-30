#!/usr/bin/env bash
# Full MediQ A–E run (100 cases, SC=3). Skips non-empty existing A/B artifacts.
# Usage (repo root):
#   bash mediq_experiment/run.sh
#   bash mediq_experiment/run.sh --skip-mediq
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
echo "HF_HOME=$HF_HOME"

python -m mediq_experiment.check_setup --config mediq_experiment/config.yaml || true
python -m mediq_experiment.run_pipeline --config mediq_experiment/config.yaml "$@"
