# MediQ multi-doctor experiment (A–E + coverage + full cross)

## What this does

1. Runs upstream `src/mediQ_benchmark.py` for each `doctor_*` in config (A–E).
2. After each case dialogue ends, scores **Final Accuracy** + **Fact Coverage** (Haiku judge).
3. **Full cross finalize**: every doctor’s transcript → every *other* doctor answers the MCQ once.

SMART arms / RAG / adaptive rounds are **not** used.

Default doctors:

| Role | Provider | Model |
|------|----------|--------|
| A | OpenAI | `gpt-5.4` |
| B | Anthropic | `claude-sonnet-4-6` |
| C | Hugging Face (local/vLLM) | `meta-llama/Llama-3.1-8B-Instruct` |
| D | Hugging Face (local/vLLM) | `Qwen/Qwen2.5-7B-Instruct` |
| E | Hugging Face (local/vLLM) | `aaditya/Llama3-OpenBioLLM-8B` |
| Patient / Judge | Anthropic | `claude-haiku-4-5` |

## Setup

### API keys (repo-root `.env`)

```
OPENAI_API_KEY=...                 # doctor A
ANTHROPIC_API_KEY=...              # doctor B / patient / judge
```

### Local / CARC open-source doctors (C/D/E)

```bash
pip install -U "huggingface_hub" torch transformers accelerate
# CARC performance path:
pip install vllm
# optional transformers fallback on small GPUs:
pip install bitsandbytes
```

Login (new CLI):

```bash
hf auth login
hf auth whoami
```

In the browser, **Agree** to gated licenses for Llama and OpenBioLLM.

### Download models on CARC

Cache directory (already wired into `run.sh` / `slurm_mediq_ae.sh`):

```bash
export HF_HOME=/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/model
mkdir -p "$HF_HOME"

hf download meta-llama/Llama-3.1-8B-Instruct
hf download Qwen/Qwen2.5-7B-Instruct
hf download aaditya/Llama3-OpenBioLLM-8B

# optional size check
hf download meta-llama/Llama-3.1-8B-Instruct --dry-run
```

Jobs pick up the same `HF_HOME` automatically unless you override it.

## One-click run

From repo root:

```bash
# Linux / CARC
bash mediq_experiment/run.sh

# or
python -m mediq_experiment.run_pipeline

# Windows
.\mediq_experiment\run.ps1
```

**Skip existing artifacts** (`pipeline.skip_existing: true`):

- Non-empty `doctor_*/results.jsonl` → skip MediQ for that doctor (keeps prior A/B runs).
- Non-empty `doctor_*/scored.jsonl` → skip Haiku coverage.
- Non-empty `cross/{src}_transcript_{dst}_answers.jsonl` → skip that pair (legacy `a_transcript_b_answers.jsonl` still valid).

`summary.json` is **rewritten** each run with A–E self metrics + all cross pairs; raw JSONL files are not deleted.

Only re-score / re-cross:

```bash
python -m mediq_experiment.run_pipeline --skip-mediq
```

## CARC job

```bash
# Edit HF_HOME inside the script first, then:
sbatch mediq_experiment/slurm_mediq_ae.sh
```

Or interactively on a GPU node (after `export HF_HOME=...`):

```bash
bash mediq_experiment/run.sh
```

Single-GPU is enough for 7B/8B + vLLM; wall time can be long (SC=3, up to 30 turns, 100 cases × 3 local doctors + cross).

## Outputs

| Path | Content |
|------|---------|
| `doctor_a`…`doctor_e/results.jsonl` | MediQ dialogues + letter accuracy |
| `doctor_*/scored.jsonl` | + end-of-case fact coverage |
| `cross/{x}_transcript_{y}_answers.jsonl` | x dialogue → y final answer |
| `summary.json` | Aggregate self + cross metrics |

## Notes

- Patient class: `FactSelectPatient`. Precomputed `facts` in JSONL are used when present.
- Local doctors: `mediq.use_vllm: true` (default). If vLLM fails to load, MediQ helper falls back to transformers (`device_map=auto`; optional `load_in_4bit`).
- Fine-tuning later: reuse the same `HF_HOME` weights with transformers/PEFT (not vLLM).
- Data batch: `id_min`/`id_max` 100–199 (same as prior A/B).
