# One-click MediQ two-model experiment (run from repo root)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m mediq_experiment.run_pipeline @args
