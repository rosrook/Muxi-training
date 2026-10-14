#!/usr/bin/env bash
# Launch Llama-3.2 LoRA SFT (TRL). MODEL / TRAIN_JSONL etc. overridable.
# Env knobs aligned with ms_swift_qwen35/train_sft_qwen35_4b_len8192.sh offline block.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- same offline / cache knobs as Qwen len8192 train script ----
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HOME="${HF_HOME:-/mnt/public/data/zhuxz/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/mnt/public/data/zhuxz/cache/modelscope}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

MODEL="${MODEL:-/mnt/public/model/huggingface/Llama-3.2-1B-Instruct}"
DATA_ROOT="${DATA_ROOT:-/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_ROOT}/val.jsonl}"
[[ -f "${VAL_JSONL}" ]] || VAL_JSONL=""

# calib 默认 2048（p99~1k）。长样本用 MAX_LENGTH=4096 + BS=1 + gradient_checkpointing
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"

export MODEL DATA_ROOT TRAIN_JSONL VAL_JSONL

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "ERROR: missing train JSONL: ${TRAIN_JSONL}" >&2
  echo "Read: ${ROOT}/DATA_FORMAT.md" >&2
  echo "  python3 ${ROOT}/src/prepare_calib_dataset.py --preset calib-v1" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))' 2>/dev/null || echo 1)}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python -c "import torch; print(','.join(map(str, range(torch.cuda.device_count() or 1))))" 2>/dev/null || echo 0)"
fi
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES

echo "[train] MODEL=${MODEL}"
echo "[train] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[train] VAL_JSONL=${VAL_JSONL:-<none>}"
echo "[train] MAX_LENGTH=${MAX_LENGTH}"
echo "[train] NPROC_PER_NODE=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train] HF_HOME=${HF_HOME} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

ARGS=(
  --model "${MODEL}"
  --train-jsonl "${TRAIN_JSONL}"
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --max-seq-length "${MAX_LENGTH}"
)
[[ -n "${VAL_JSONL}" ]] && ARGS+=(--val-jsonl "${VAL_JSONL}")
[[ -n "${OUTPUT_DIR:-}" ]] && ARGS+=(--output-dir "${OUTPUT_DIR}")
[[ "${SKIP_EVAL:-0}" == "1" ]] && ARGS+=(--skip-eval)
[[ "${MERGE_WEIGHTS:-0}" == "1" ]] && ARGS+=(--merge-weights)

cd "${ROOT}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    "${ROOT}/src/train_sft_trl.py" "${ARGS[@]}"
else
  python "${ROOT}/src/train_sft_trl.py" "${ARGS[@]}"
fi
