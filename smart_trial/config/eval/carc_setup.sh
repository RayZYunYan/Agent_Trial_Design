#!/bin/bash
# Shared setup for CARC SLURM eval jobs (Claude / Anthropic).
# Sourced from slurm_*.sh — do not submit directly.

set -euo pipefail

_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$_REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Load API key from repo-root .env if present (do not commit .env).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

unset SMART_TRIAL_MOCK_JUDGE SMART_TRIAL_USE_MOCK || true

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY is not set. Add it to .env on CARC or export before sbatch." >&2
  exit 1
fi

mkdir -p smart_trial/outputs/eval/logs

# Uncomment if your CARC module env lacks dependencies:
# pip install -q -r smart_trial/requirements.txt

echo "CARC setup OK | cwd=$(pwd) | models=haiku(patient,judge)+sonnet(doctor) | anthropic key=set"
