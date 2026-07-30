# Smoke run (Windows). From repo root:
#   .\mediq_experiment\run_smoke.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m mediq_experiment.check_setup --config mediq_experiment/config_smoke.yaml
python -m mediq_experiment.run_pipeline --config mediq_experiment/config_smoke.yaml @args
