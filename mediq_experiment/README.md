# MediQ two-model experiment (upstream MediQ + end-of-case coverage + cross finalize)

## What this does

1. Runs **upstream** `src/mediQ_benchmark.py` twice (doctor A, doctor B).
2. After each case dialogue ends, scores **Final Accuracy** + **Fact Coverage** (one judge call per case).
3. **Cross finalize**: A’s transcript → B answers MCQ once; B’s transcript → A answers once.

SMART arms / RAG / adaptive rounds are **not** used.

## Setup

1. Create a venv and install deps (API-only; no torch):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Put API keys in repo-root `.env` (copy from `.envExample`):

```
OPENAI_API_KEY=...                 # doctor A (GPT)
ANTHROPIC_API_KEY=...              # doctor B / patient / judge (Claude)
GROQ_API_KEY=...                   # optional local smoke
```

3. Edit `mediq_experiment/config.yaml` — current defaults:

```yaml
models:
  doctor_a:
    provider: "openai"
    name: "gpt-5.4"
  doctor_b:
    provider: "anthropic"
    name: "claude-sonnet-4-6"
  patient:
    provider: "anthropic"
    name: "claude-haiku-4-5"
  judge:
    provider: "anthropic"
    name: "claude-haiku-4-5"
```

Mixed OpenAI+Anthropic in one MediQ run is supported (`src/helper.py` infers backend from model id).

4. Optionally set `data.max_cases` (e.g. `5` for a smoke test; `null` for all).

## One-click run

From repo root:

```powershell
python -m mediq_experiment.run_pipeline
```

Or:

```powershell
.\mediq_experiment\run.ps1
```

Outputs land in `mediq_experiment/outputs/`:

| Path | Content |
|------|---------|
| `doctor_a/results.jsonl` | MediQ dialogues + letter accuracy (A) |
| `doctor_b/results.jsonl` | MediQ dialogues + letter accuracy (B) |
| `doctor_a/scored.jsonl` | + end-of-case fact coverage |
| `doctor_b/scored.jsonl` | + end-of-case fact coverage |
| `cross/a_transcript_b_answers.jsonl` | A dialogue → B final answer |
| `cross/b_transcript_a_answers.jsonl` | B dialogue → A final answer |
| `summary.json` | Aggregate metrics |

## Resume / re-score only

MediQ skips case ids already present in `results.jsonl`. To re-score without re-running dialogues:

```powershell
python -m mediq_experiment.run_pipeline --skip-mediq
```

## Notes

- Patient class: `FactSelectPatient` (MediQ). Precomputed `facts` in your JSONL are used when present.
- Fact coverage uses your existing `StageJudge` **once per finished case** (`review_mode`).
- Models are config-only; no code change needed when you pick final model names.
- `use_api: groq|openai|anthropic` supported; Claude/GPT mixed runs work via model-id inference.
- Default config: doctor A=`gpt-5.4`, doctor B=`claude-sonnet-4-6`, patient/judge=`claude-haiku-4-5`.
