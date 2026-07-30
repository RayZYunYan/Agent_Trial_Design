# One-click full MediQ A–E (Windows). From repo root:
#   .\mediq_experiment\run.ps1
#   .\mediq_experiment\run.ps1 --skip-mediq
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m mediq_experiment.run_pipeline --config mediq_experiment/config.yaml @args
