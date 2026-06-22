# SMART Trial Simulator

Adaptive two-stage clinical encounter simulator with re-randomization and JSONL trajectory logging. Built for pilot experiments on MediQ-style cases.

## Requirements

- Python 3.10+
- API key in `.env` (copy from `.envExample`)

```bash
pip install -r smart_trial/requirements.txt
```

## Quick start

From the repository root:

```bash
# Full dataset in file order: each case once, skip if already in the log
python run_encounter.py --seed 42

# Pilot: first 10 cases only
python run_encounter.py --n 10 --seed 42

# Single case
python run_encounter.py --case_id medqa_0000 --seed 42

# Re-run a case that is already logged
python run_encounter.py --case_id medqa_0000 --force

# Summarize all logged encounters
python -m smart_trial.scripts.summarize_encounters
```

Logs are appended to `smart_trial/outputs/encounters.jsonl` (one JSON object per line). A full run walks every case in the configured dataset in order; cases already present in the file are skipped unless you pass `--force`.

## Dataset

Default local file: `data/all_mediq_craft_merged.jsonl` (2685 cases: mediQ validation + test + CRAFT-MD).

Smaller dev subset: `data/all_dev_good.jsonl` (1272 cases) — set `data.local.path` in `trial_config.yaml`.

Rebuild the merged file (requires HuggingFace `datasets` and network):

```bash
python -m smart_trial.scripts.build_merged_dataset
```

## Configuration

Edit `smart_trial/config/trial_config.yaml`:

| Section | Purpose |
|---------|---------|
| `trial` | Turn limits, R1 responder threshold, R2 confidence threshold |
| `data` | Local JSONL vs HuggingFace `stellalisy/mediQ`, optional `red_flag_cache` |
| `models` | Provider and model per role (`patient_simulator`, `doctor_agent`, `judge`) |
| `randomization` | Seed and stratification field |
| `logging` | Consolidated JSONL output path (`output_file`) |

Supported model providers: `groq`, `openai`, `anthropic`, `gemini`, `cursor_sdk`, `mock`.

To use Cursor SDK, set each role's `provider: "cursor_sdk"` and `model_name` (e.g. `composer-2.5`), then set `CURSOR_API_KEY` in `.env`.

Arm-specific clinician behavior is defined under `smart_trial/config/arms/*.yaml`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (default pilot backend) |
| `CURSOR_API_KEY` | Cursor SDK API key (when using `cursor_sdk` provider) |
| `ANTHROPIC_API_KEY` | **CARC benchmark** (Claude patient/doctor/judge) |
| `GEMINI_API_KEY` | Optional Gemini key |
| `OPENAI_API_KEY` | If using OpenAI provider |
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

50-case baseline + 50×9 grid eval uses **Anthropic** (patient/judge `claude-haiku-4-5`, doctor `claude-sonnet-4-6`). Full steps:

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
data/               all_mediq_craft_merged.jsonl, all_dev_good.jsonl, all_craft_md.jsonl
smart_trial/
  config/           trial_config.yaml, arms/
  core/             patient, doctor, judge, orchestrator, randomizer
  data/             loader, optional red_flag_cache.json
  models/           unified ModelClient (groq, cursor_sdk, …)
  trajectory_log/   trajectory JSONL writer
  scripts/          summarize, dataset merge, red-flag builders
  run_encounter.py  CLI entry
run_encounter.py    thin wrapper at repo root
```
