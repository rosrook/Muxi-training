#!/usr/bin/env bash
# Self-contained multi-GPU LoRA launch. Used by per-run run.sh (copied, not shared).
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${RUN_DIR}/run_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="${BPG_SRC:-/mnt/public/code/zhuxz/bench_probe_gen/src}${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "${SFT_DATASET}" ]]; then
  echo "[train_launch] ERROR: SFT_DATASET not found: ${SFT_DATASET}" >&2
  exit 1
fi
if [[ ! -f "${TRAINING_CONFIG}" ]]; then
  echo "[train_launch] ERROR: TRAINING_CONFIG not found: ${TRAINING_CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_SOURCE}" ]]; then
  echo "[train_launch] ERROR: base model not found: ${MODEL_SOURCE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${MODEL_LINK}")" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "${OUTPUT_DIR}"
ln -sfn "${MODEL_SOURCE}" "${MODEL_LINK}"

echo "[train_launch] RUN_ID=${RUN_ID}"
echo "[train_launch] SFT_DATASET=${SFT_DATASET}"
echo "[train_launch] TRAINING_CONFIG=${TRAINING_CONFIG}"
echo "[train_launch] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[train_launch] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[train_launch] TENSORBOARD_DIR=${TENSORBOARD_DIR}"
echo "[train_launch] nproc=${NPROC_PER_NODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if ! python3 -c "import torch, transformers, peft" 2>/dev/null; then
  echo "[train_launch] ERROR: missing GPU training packages (torch/transformers/peft)" >&2
  exit 1
fi

python3 - <<PY
import sys
from pathlib import Path
import torch
import transformers
from transformers import AutoConfig

visible = [x for x in "${CUDA_VISIBLE_DEVICES}".split(",") if x.strip()]
nproc = int("${NPROC_PER_NODE}")
if len(visible) < nproc:
    print(f"[train_launch] ERROR: need {nproc} CUDA ids, got {len(visible)}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("[train_launch] ERROR: torch.cuda.is_available() is False", file=sys.stderr)
    sys.exit(1)
if torch.cuda.device_count() < nproc:
    print(f"[train_launch] ERROR: need {nproc} devices, got {torch.cuda.device_count()}", file=sys.stderr)
    sys.exit(1)
for i in range(nproc):
    print(f"[train_launch] cuda:{i} name={torch.cuda.get_device_name(i)}", flush=True)
model = Path("${MODEL_LINK}")
cfg = AutoConfig.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
print(f"[train_launch] base ok model_type={getattr(cfg, 'model_type', '?')} transformers={transformers.__version__}", flush=True)
PY

export RUN_ID SFT_DATASET TRAINING_CONFIG OUTPUT_DIR CHECKPOINT_DIR TENSORBOARD_DIR
export MODEL_LINK TRAINING_BACKEND BPG_SRC

echo "[train_launch] torchrun..."
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${scripts_dir}/train_worker.py"

echo "[train_launch] done"
echo "[train_launch] checkpoint: ${CHECKPOINT_DIR}/final/"
echo "[train_launch] summary: ${OUTPUT_DIR}/train_summary.json"
