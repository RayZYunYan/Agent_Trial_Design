# MediQ multi-doctor experiment (A–E + coverage + full cross)

## Models (Mac mini / M4 Pro default)

| Role | Provider | Model |
|------|----------|--------|
| A | OpenAI | `gpt-5.4` |
| B | Anthropic | `claude-sonnet-4-6` |
| C | OpenAI | `gpt-5.6-luna` (API, no download) |
| D | Local (MLX) | `mlx-community/Qwen3.5-4B-OptiQ-4bit` (from `Qwen/Qwen3.5-4B`) |
| E | Local (MLX) | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Patient / Judge | Anthropic | `claude-haiku-4-5` |

**Accelerator on Apple Silicon:** `mediq.use_mlx: true` → [`mlx-lm`](https://github.com/ml-explore/mlx-lm) (Metal).  
vLLM is **not** used on Mac. Plain `transformers` + MPS is the fallback if MLX load fails.

48GB unified memory is plenty for 4B + 8B 4-bit; leave headroom for the OS and concurrent API calls.

## Mac mini quick start

```bash
# 1) deps (venv recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install mlx-lm                    # Apple Silicon accelerator
pip install torch transformers accelerate huggingface_hub   # MLX fallback / other tools

# 2) API keys in repo-root .env
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...

# 3) download D + E MLX 4-bit weights
chmod +x mediq_experiment/download_local_models.sh
bash mediq_experiment/download_local_models.sh
# default cache: ~/hf_cache  (override with HF_HOME=...)

# 4) smoke (1 case)
chmod +x mediq_experiment/run_mac_smoke.sh mediq_experiment/run_mac.sh
bash mediq_experiment/run_mac_smoke.sh

# 5) full 100 cases (skips non-empty existing doctor_*/cross files)
bash mediq_experiment/run_mac.sh
```

Offline unit tests:

```bash
pip install pytest
python -m pytest mediq_experiment/tests/test_pipeline_config.py -q
```

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | Full run → `outputs/` |
| `config_smoke.yaml` | Smoke → `outputs_smoke/` |
| `download_local_models.sh` | HF download for D + E (MLX 4-bit) |
| `run_mac_smoke.sh` / `run_mac.sh` | Mac launchers |
| `check_setup.py` | Preflight |
| `tests/test_pipeline_config.py` | Offline tests |

## Outputs

| Path | Content |
|------|---------|
| `outputs/doctor_a`…`doctor_e/` | Full-run dialogues / scored |
| `outputs/cross/` | Full cross pairs |
| `outputs/summary.json` | Aggregate metrics |
| `outputs_smoke/` | Smoke sandbox |

## Notes

- Doctor C is API-only; D/E need `HF_HOME` weights (`mlx_name` when `use_mlx: true`).
- Qwen3.5 thinking mode is disabled in `src/helper.py` for MediQ.
- Llama may require Meta license acceptance for the full HF id; the MLX 4-bit mirror is usually ungated.
- `pipeline.skip_existing: true` skips non-empty results/scored/cross files.
- To force transformers/MPS instead of MLX: set `mediq.use_mlx: false` and download full HF weights (`DOWNLOAD_HF_FULL=1`).
