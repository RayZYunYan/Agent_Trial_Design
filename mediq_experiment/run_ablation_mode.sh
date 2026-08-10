#!/usr/bin/env bash
# Run ONE ablation mode × all doctors (resume-safe).
# Usage (repo root):
#   bash mediq_experiment/run_ablation_mode.sh base
#   bash mediq_experiment/run_ablation_mode.sh one_random
#   bash mediq_experiment/run_ablation_mode.sh two_most_important_claude
#   bash mediq_experiment/run_ablation_mode.sh two_most_important_gpt
#   bash mediq_experiment/run_ablation_mode.sh all_facts
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

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 <mode> [extra args...]"
  echo "Modes: base | one_random | two_most_important_claude | two_most_important_gpt | all_facts"
  exit 1
fi
shift || true

CONFIG="${ABLATION_CONFIG:-mediq_experiment/config_ablation.yaml}"
echo "ABLATION MODE=$MODE | config=$CONFIG | cwd=$ROOT"
echo "Interrupt: Ctrl+C is safe. Re-run the same command to resume."
python3 -m mediq_experiment.run_ablation --config "$CONFIG" --mode "$MODE" "$@"
