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

Each eval job writes **one** aggregate JSONL in its `output_dir` (no per-case `{case_id}.jsonl` files).

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
- **by category** (8 `case_category` buckets): baseline, each path, and adaptive — 88 extra CSV rows

Also writes per-category encounter detail JSONL (when summary runs with default settings):

| Job | Category splits |
|-----|-----------------|
| Baseline | `baseline/by_category/{Cardiology,Neuro,...}.jsonl` |
| Grid | `grid/by_category/{Cardiology,Neuro,...}.jsonl` |

Empty categories are omitted from JSONL; CSV still lists all 8 with `n=0`.

## Smoke test (1 case × 9 grid — verify output structure)

Clears old `pilot/` / `grid/` / `baseline/` test JSONL where possible, then runs:

- `medqa_0000` baseline × 1
- `medqa_0000` grid × 9 paths (all unique `path_id`)

```bash
python smart_trial/scripts/run_smoke_eval.py
```

Outputs (canonical clean test):

| File | Expected lines |
|------|----------------|
| `smart_trial/outputs/eval/smoke/baseline/baseline_encounters.jsonl` | 1 |
| `smart_trial/outputs/eval/smoke/grid/grid_encounters.jsonl` | 9 |
| `smart_trial/outputs/eval/smoke/summary_metrics.csv` | metrics (~99 rows) |
| `smart_trial/outputs/eval/smoke/baseline/by_category/Other.jsonl` | 1 (smoke) |
| `smart_trial/outputs/eval/smoke/grid/by_category/Other.jsonl` | 9 (smoke) |

`--no-resume` now deletes **all** `*.jsonl` in the job output dir (aggregate + per-case).

## Claude smoke (1 case × 9 grid, real judge)

Requires `ANTHROPIC_API_KEY` in repo-root `.env`. Output: `smart_trial/outputs/eval/smoke_claude/`.

```bash
python smart_trial/scripts/run_smoke_claude_eval.py
```

Configs: `config_smoke_claude_baseline.yaml`, `config_smoke_claude_grid.yaml` (`claude-sonnet-4-6`).

## Pilot test (5 cases, API + real judge)

Requires `GROQ_API_KEY` or `ANTHROPIC_API_KEY` in repo-root `.env`. Clears `pilot/` and `smoke/` first.

5 baseline + 5 grid (one path `A1a→A2a` per case). Wipes prior `pilot/` and `smoke/` outputs.

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
