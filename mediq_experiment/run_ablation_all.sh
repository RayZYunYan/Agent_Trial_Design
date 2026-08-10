#!/usr/bin/env bash
# Run ALL one-shot ablation modes × all doctors (resume-safe).
# Usage (repo root):
#   bash mediq_experiment/run_ablation_all.sh
#   bash mediq_experiment/run_ablation_all.sh --dry-run
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

CONFIG="${ABLATION_CONFIG:-mediq_experiment/config_ablation.yaml}"
echo "ABLATION ALL | config=$CONFIG | cwd=$ROOT"
echo "Interrupt: Ctrl+C is safe. Re-run this command to resume unfinished case/doctor/mode."
python3 -m mediq_experiment.run_ablation --config "$CONFIG" --all-modes "$@"
