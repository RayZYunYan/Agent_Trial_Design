#!/usr/bin/env bash
# End-to-end smoke (1 case, RandomExpert, no API) for all ablation modes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG=mediq_experiment/config_ablation_smoke.yaml
echo "ABLATION SMOKE | config=$CONFIG"
python3 -m mediq_experiment.run_ablation --config "$CONFIG" --all-modes "$@"
echo "OK — see mediq_experiment/outputs_ablation_smoke/"
