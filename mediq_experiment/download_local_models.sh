#!/usr/bin/env bash
# Download local MediQ doctors D + E for Mac mini (Apple MLX 4-bit by default).
#
#   chmod +x mediq_experiment/download_local_models.sh
#   bash mediq_experiment/download_local_models.sh
#
# Llama: agree license on HF (MLX mirror is usually ungated; Meta full weights may be gated):
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# Prefer MLX 4-bit weights (fast on M4). Set DOWNLOAD_HF_FULL=1 to also pull transformers weights.
DOWNLOAD_MLX="${DOWNLOAD_MLX:-1}"
DOWNLOAD_HF_FULL="${DOWNLOAD_HF_FULL:-0}"

python3 -m pip install -U "huggingface_hub"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not logged in — running: hf auth login"
  hf auth login
fi
hf auth whoami

D_MLX="mlx-community/Qwen3.5-4B-OptiQ-4bit"
E_MLX="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
D_HF="Qwen/Qwen3.5-4B"
E_HF="meta-llama/Llama-3.1-8B-Instruct"

echo ""
echo "=== dry-run (sizes) ==="
if [[ "$DOWNLOAD_MLX" == "1" ]]; then
  hf download "$D_MLX" --dry-run || true
  hf download "$E_MLX" --dry-run || true
fi
if [[ "$DOWNLOAD_HF_FULL" == "1" ]]; then
  hf download "$D_HF" --dry-run || true
  hf download "$E_HF" --dry-run || true
fi

if [[ "$DOWNLOAD_MLX" == "1" ]]; then
  echo ""
  echo "=== download D (MLX): $D_MLX ==="
  hf download "$D_MLX"
  echo ""
  echo "=== download E (MLX): $E_MLX ==="
  hf download "$E_MLX"
fi

if [[ "$DOWNLOAD_HF_FULL" == "1" ]]; then
  echo ""
  echo "=== download D (HF full): $D_HF ==="
  hf download "$D_HF"
  echo ""
  echo "=== download E (HF full): $E_HF ==="
  hf download "$E_HF"
fi

echo ""
echo "Done. Models cached under: $HF_HOME"
echo "Install Apple accelerator:  pip install -U mlx-lm"
hf cache ls || true
