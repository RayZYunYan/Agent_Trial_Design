# MediQ multi-doctor experiment (A–E + coverage + full cross)

## Models

| Role | Provider | Model |
|------|----------|--------|
| A | OpenAI | `gpt-5.4` |
| B | Anthropic | `claude-sonnet-4-6` |
| C | Hugging Face (local/vLLM) | `google/gemma-2-9b-it` |
| D | Hugging Face (local/vLLM) | `Qwen/Qwen2.5-7B-Instruct` |
| E | Hugging Face (local/vLLM) | `aaditya/Llama3-OpenBioLLM-8B` |
| Patient / Judge | Anthropic | `claude-haiku-4-5` |

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | **Full** run (100 cases, SC=3) → `outputs/` |
| `config_smoke.yaml` | **Smoke** (1 case, SC=1) → `outputs_smoke/` |
| `check_setup.py` | Preflight: config, keys, HF_HOME, models |
| `tests/test_pipeline_config.py` | Offline unit tests (no GPU) |
| `run_smoke.sh` / `slurm_mediq_smoke.sh` | Start smoke |
| `run.sh` / `slurm_mediq_ae.sh` | Start full A–E |
| `run_smoke.ps1` / `run.ps1` | Windows equivalents |

SMART arms / RAG are **not** used. `pipeline.skip_existing: true` skips non-empty `results` / `scored` / `cross` files (keeps prior A/B work).

## Setup

```bash
# API keys in repo-root .env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...

pip install -U "huggingface_hub" torch transformers accelerate vllm pyyaml python-dotenv
# optional: bitsandbytes pytest
```

```bash
export HF_HOME=/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/model
mkdir -p "$HF_HOME"
hf auth login

# Agree on HF web for Gemma + OpenBioLLM, then:
hf download google/gemma-2-9b-it
hf download Qwen/Qwen2.5-7B-Instruct
hf download aaditya/Llama3-OpenBioLLM-8B
```

## Recommended order on CARC

```bash
cd /path/to/repo/root   # directory that contains mediq_experiment/

# 0) Offline unit tests (no GPU)
pytest mediq_experiment/tests/test_pipeline_config.py -q

# 1) Preflight
export HF_HOME=/project2/ruishanl_1185/proj-26su-agent-trial-design/Ray/Agent_Trial_Design/model
python -m mediq_experiment.check_setup --config mediq_experiment/config_smoke.yaml --require-models

# 2) Smoke (1 case → outputs_smoke/)
sbatch mediq_experiment/slurm_mediq_smoke.sh
# or: bash mediq_experiment/run_smoke.sh

# 3) After smoke looks good → full 100 cases (skips existing A/B)
sbatch mediq_experiment/slurm_mediq_ae.sh
# or: bash mediq_experiment/run.sh
```

## Outputs

| Path | Content |
|------|---------|
| `outputs/doctor_a`…`doctor_e/` | Full-run dialogues / scored |
| `outputs/cross/` | Full cross pairs |
| `outputs/summary.json` | Full aggregate (rewritten each run) |
| `outputs_smoke/` | Smoke-only tree (safe sandbox) |

## Notes

- Local doctors use `use_vllm: true`; helper falls back to transformers if vLLM fails.
- Fine-tune later: same `HF_HOME` weights + transformers/PEFT (not vLLM).
- Data: ids 100–199 (same batch as prior A/B).
