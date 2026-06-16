# Phase-1 benchmark eval (CARC)

Fixed cohort **medqa_0000–medqa_0049** (50 cases).

## Run locally (mock)

```bash
set SMART_TRIAL_USE_MOCK=1
python -m smart_trial.run_eval --config smart_trial/config/eval/config_baseline.yaml
python -m smart_trial.run_eval --config smart_trial/config/eval/config_grid.yaml --no-resume
```

## Run on CARC

```bash
sbatch smart_trial/config/eval/slurm_baseline.sh
sbatch smart_trial/config/eval/slurm_grid.sh
```

Outputs:

| Job | Aggregate file |
|-----|----------------|
| Baseline | `smart_trial/outputs/eval/baseline/baseline_encounters.jsonl` |
| Grid 50×9 | `smart_trial/outputs/eval/grid/grid_encounters.jsonl` |

Per-case append logs also land under each job's `output_dir` as `{case_id}.jsonl`.

## Summary metrics (after grid + baseline)

```bash
python -m smart_trial.eval.summary_metrics \
  --baseline smart_trial/outputs/eval/baseline/baseline_encounters.jsonl \
  --grid smart_trial/outputs/eval/grid/grid_encounters.jsonl \
  --out smart_trial/outputs/eval/summary_metrics.csv
```

Produces average correctness for:

- baseline
- each of 9 static `(A1,A2)` paths
- Q-learning adaptive policy (offline lookup on grid data)

## Pilot smoke test (3 Groq + mock judge)

Saves quota: 3 baseline + 3 grid (one path), judge mocked via `SMART_TRIAL_MOCK_JUDGE=1`.

```bash
python smart_trial/scripts/run_pilot_pipeline.py
```

Outputs under `smart_trial/outputs/eval/pilot/`. Or stepwise:

```bash
set SMART_TRIAL_MOCK_JUDGE=1
python -m smart_trial.run_eval --config smart_trial/config/eval/config_baseline_pilot.yaml --no-resume
python -m smart_trial.run_eval --config smart_trial/config/eval/config_grid_pilot.yaml --no-resume
python -m smart_trial.eval.summary_metrics --baseline smart_trial/outputs/eval/pilot/baseline/baseline_encounters.jsonl --grid smart_trial/outputs/eval/pilot/grid/grid_encounters.jsonl --out smart_trial/outputs/eval/pilot/summary_metrics.csv
```

Configs: `config_baseline_pilot.yaml`, `config_grid_pilot.yaml` (subset `eval.grid_paths`).


## Phase 2

Closed-loop on **medqa_0100–medqa_0149** uses `config_adaptive.yaml` (placeholder).
Cold-start Q from `grid_encounters.jsonl`, refit every 5 cases.
