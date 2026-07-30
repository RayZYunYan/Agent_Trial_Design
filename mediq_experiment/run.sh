#!/usr/bin/env bash
# One-click MediQ A–E experiment (run from repo root or via this script).
# Usage:
#   export HF_HOME=/your/scratch/hf_cache   # required on CARC for C/D/E
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

# Default CARC model cache (override by exporting HF_HOME before calling this script)
export HF_HOME="${HF_HOME:-/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/model}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

python -m mediq_experiment.run_pipeline "$@"
