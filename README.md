# SMART Trial Simulator

Adaptive three-stage clinical encounter simulator with re-randomization and JSONL trajectory logging. Built for pilot experiments on MediQ-style cases.

## Requirements

- Python 3.10+
- API key in `.env` (copy from `.envExample`)

```bash
pip install -r smart_trial/requirements.txt
```

## Quick start

From the repository root:

```bash
# Single encounter (default: first case in dev set)
python run_encounter.py --case_id medqa_0000 --seed 42

# Three cases from different categories (Cardiology, Pediatrics, Neuro, ...)
python run_encounter.py --diverse 3 --seed 42

# Summarize all logged encounters
python -m smart_trial.scripts.summarize_encounters
```

Logs are written to `smart_trial/outputs/encounters/<case_id>.jsonl` (one JSON object per line per run).

## Configuration

Edit `smart_trial/config/trial_config.yaml`:

| Section | Purpose |
|---------|---------|
| `trial` | Turn limits, R1 responder threshold, R2 confidence threshold |
| `data` | Local JSONL vs HuggingFace `stellalisy/mediQ`, optional `red_flag_cache` |
| `models` | Provider and model per role (`patient_simulator`, `doctor_agent`, `judge`) |
| `randomization` | Seed and stratification field |
| `logging` | Output directory |

Supported model providers: `groq`, `openai`, `anthropic`, `gemini`, `mock`.

Arm-specific clinician behavior is defined under `smart_trial/config/arms/*.yaml`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Optional Groq key (legacy smoke scripts) |
| `ANTHROPIC_API_KEY` | **CARC benchmark** (Claude patient/doctor/judge) |
| `SMART_TRIAL_USE_MOCK=1` | Force mock models (tests, no API) |
| `SMART_TRIAL_LIVE_API=1` | Run pytest against real APIs (optional) |

## Red flags

Each case gets category-level default red flags at load time. For a persisted cache file:

```bash
# Deterministic cache from category defaults (no API)
python -m smart_trial.scripts.build_category_red_flag_cache --max-cases 100

# LLM-generated per-case cache (uses configured patient model)
python -m smart_trial.scripts.generate_red_flags --max-cases 50
```

Point `data.red_flag_cache` in `trial_config.yaml` to the JSON file, or pass `--red-flag-cache` on the CLI.

## CARC benchmark (Phase 1, Claude)

50-case baseline + 50×9 grid eval uses **Anthropic `claude-sonnet-4-6`**. Full steps:

```bash
cp .env.example .env   # ANTHROPIC_API_KEY=...
sbatch smart_trial/config/eval/slurm_build_bm25.sh   # once
sbatch smart_trial/config/eval/slurm_baseline.sh
sbatch smart_trial/config/eval/slurm_grid.sh
sbatch smart_trial/config/eval/slurm_summary.sh      # after both complete
```

See [smart_trial/config/eval/README.md](smart_trial/config/eval/README.md).

## Tests

```bash
python -m pytest smart_trial/tests/ -q
```

Tests use mock models by default (`smart_trial/tests/conftest.py`).

## Project layout

```
smart_trial/
  config/           trial_config.yaml, arms/
  core/             patient, doctor, judge, orchestrator, randomizer
  data/             loader, optional red_flag_cache.json
  models/           unified ModelClient (with rate-limit retry)
  trajectory_log/   trajectory JSONL writer
  scripts/          summarize, red-flag builders
  run_encounter.py  CLI entry
run_encounter.py    thin wrapper at repo root
```
