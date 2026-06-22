# Phase-1 benchmark eval (CARC + Claude)

Fixed cohort **medqa_0000–medqa_0049** (50 cases).  
Models: **Anthropic** — patient/judge `claude-haiku-4-5`, doctor `claude-sonnet-4-6`. See `models_claude.yaml`.

## CARC quick start

### 1. Clone / upload repo

Track via git (`.gitignore` allows `data/all_dev_good.jsonl`, `smart_trial/data/red_flag_cache.json`, `*.pkl` when built):

- `data/all_dev_good.jsonl` (~2.5 MB)
- `smart_trial/data/red_flag_cache.json`
- `smart_trial/data/bm25_index.pkl` (build on cluster or upload after local build)

### 2. Environment

```bash
cp .env.example .env   # edit: ANTHROPIC_API_KEY=...
pip install -r smart_trial/requirements.txt
```

Edit SLURM headers in `slurm_*.sh` if your account differs (default: `ruishanl_1185` on Discovery `main` / grid on `oneweek`).

### 3. Optional: verify Claude smoke (1 case)

```bash
python smart_trial/scripts/run_smoke_claude_eval.py
```

Output: `smart_trial/outputs/eval/smoke_claude/`

### 4. Submit jobs (Discovery cluster — `ssh discovery.usc.edu`)

```bash
# Once (grid RAG index; needs HuggingFace network on compute node):
sbatch smart_trial/config/eval/slurm_build_bm25.sh

# Benchmark (can run in parallel; both use --resume by default):
sbatch smart_trial/config/eval/slurm_baseline.sh
sbatch smart_trial/config/eval/slurm_grid.sh

# Phase 2 closed-loop adaptive (after grid exists for initial_q_from):
sbatch smart_trial/config/eval/slurm_adaptive.sh

# After baseline + grid finish (and optional adaptive):
sbatch smart_trial/config/eval/slurm_summary.sh
```

### 5. Outputs

| Job | Aggregate file |
|-----|----------------|
| Baseline | `smart_trial/outputs/eval/baseline/baseline_encounters.jsonl` (50 rows) |
| Grid | `smart_trial/outputs/eval/grid/grid_encounters.jsonl` (450 rows) |
| Summary | `smart_trial/outputs/eval/summary_metrics.csv` + `baseline/by_category/*.jsonl`, `grid/by_category/*.jsonl` |

Logs: `smart_trial/outputs/eval/logs/`

`carc_setup.sh` loads `.env`, checks `ANTHROPIC_API_KEY`, and clears mock env vars.

---

## Run locally (mock, no API)

```bash
set SMART_TRIAL_USE_MOCK=1
python -m smart_trial.run_eval --config smart_trial/config/eval/config_baseline.yaml
python -m smart_trial.run_eval --config smart_trial/config/eval/config_grid.yaml --no-resume
```

## Summary metrics (manual)

```bash
python -m smart_trial.eval.summary_metrics \
  --baseline smart_trial/outputs/eval/baseline/baseline_encounters.jsonl \
  --grid smart_trial/outputs/eval/grid/grid_encounters.jsonl \
  --out smart_trial/outputs/eval/summary_metrics.csv
```

Produces average correctness for baseline, 9 static paths, offline Q-learning adaptive,
closed-loop adaptive (optional), and category breakdown rows.

## Groq smoke (legacy local)

```bash
python smart_trial/scripts/run_smoke_eval.py
```

Uses `config_smoke_*.yaml` (Groq + mock judge if `GROQ_API_KEY` set).

## Pilot (5 cases, Claude)

```bash
python smart_trial/scripts/run_pilot_pipeline.py
```

Requires `ANTHROPIC_API_KEY`. Clears `pilot/` and `smoke/` first.

## Phase 2

Closed-loop **medqa_0100–0149**: `config_adaptive.yaml` + `slurm_adaptive.sh` (requires Phase-1 `grid_encounters.jsonl`).
