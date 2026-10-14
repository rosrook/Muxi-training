#!/usr/bin/env bash
# Qwen3.5-4B LoRA SFT with ms-swift (Dense path from official Best Practices).
# Doc: https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html
#
# Official notes we keep:
#   - transformers backend for Dense (4B)
#   - --tuner_type lora
#   - --add_non_thinking_prefix true
#   - --loss_scale ignore_empty_think
#   - --group_by_length true  (or swap to --packing true)
#   - bfloat16, deepspeed zero2, 4-GPU example footprint ~4x20GiB
#
# Usage (GPU node):
#   source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate
#   bash train_sft_qwen35_4b.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${DATA_ROOT}/run_${STAMP}}"
mkdir -p "${RUN_DIR}"

# Offline / no-xet (local model). Prevents hub network probes that look like hangs.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

# Lazy site patch for ms-swift + transformers>=5.14 (create_repo removed).
# Must NOT eagerly import transformers.trainer here.
SITE_SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -f "${ROOT}/patch_transformers_for_ms_swift.py" "${SITE_SP}/patch_transformers_for_ms_swift.py"
echo "import patch_transformers_for_ms_swift" > "${SITE_SP}/zz_ms_swift_tf_compat.pth"
echo "[compat] lazy create_repo hook installed at ${SITE_SP}/zz_ms_swift_tf_compat.pth"

# Local model (offline). Override to Hub id Qwen/Qwen3.5-4B if online.
MODEL="${MODEL:-/mnt/public/model/Qwen3.5-4B}"
# Fallback hub id
# MODEL="${MODEL:-Qwen/Qwen3.5-4B}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Dataset: pure messages JSONL
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_ROOT}/train.val.jsonl}"

# Rebuild dataset if missing or FORCE_PREPARE=1
if [[ ! -f "${TRAIN_JSONL}" || "${FORCE_PREPARE:-0}" == "1" ]]; then
  echo "[prepare] building ${TRAIN_JSONL}"
  PREP_ARGS=(--out "${TRAIN_JSONL}" --val-ratio 0.01)
  if [[ "${INCLUDE_FG:-0}" == "1" ]]; then
    PREP_ARGS+=(--include-fg)
  fi
  # point --nonfg at latest rewrite if you have a newer stamp
  if [[ -n "${NONFG_SFT:-}" ]]; then
    PREP_ARGS+=(--nonfg "${NONFG_SFT}")
  fi
  python3 "${ROOT}/prepare_dataset.py" "${PREP_ARGS[@]}"
fi

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "ERROR: train jsonl missing: ${TRAIN_JSONL}" >&2
  exit 2
fi

# Official env knobs for Qwen3.5 (vision-ready defaults; harmless for text-only)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"
export CUDA_VISIBLE_DEVICES
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/mnt/public/data/zhuxz/cache/modelscope}"
export HF_HOME="${HF_HOME:-/mnt/public/data/zhuxz/cache/huggingface}"

OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/output_Qwen3.5-4B}"
mkdir -p "${OUTPUT_DIR}"

# Persist launch config
cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "stamp": "${STAMP}",
  "model": "${MODEL}",
  "train_jsonl": "${TRAIN_JSONL}",
  "val_jsonl": "${VAL_JSONL}",
  "nproc": ${NPROC_PER_NODE},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "output_dir": "${OUTPUT_DIR}",
  "doc": "https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html"
}
EOF

echo "[train] model=${MODEL}"
echo "[train] train=${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") lines)"
[[ -f "${VAL_JSONL}" ]] && echo "[train] val=${VAL_JSONL} ($(wc -l < "${VAL_JSONL}") lines)"
echo "[train] output=${OUTPUT_DIR}"

# Dataset args: prefer explicit val file; else split_ratio as in official demo
DATASET_ARGS=(--dataset "${TRAIN_JSONL}")
if [[ -f "${VAL_JSONL}" ]]; then
  DATASET_ARGS+=(--val_dataset "${VAL_JSONL}")
else
  DATASET_ARGS+=(--split_dataset_ratio 0.01)
fi

# ---- Official Dense Qwen3.5-4B LoRA SFT skeleton ----
# Differences from the doc's multi-source alpine demo:
#   - local custom dataset (messages JSONL)
#   - local model path
#   - kept: add_non_thinking_prefix, loss_scale ignore_empty_think,
#           group_by_length, deepspeed zero2, lora r8/a32, bf16
#
set -x
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM}" \
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL}" \
  --tuner_type lora \
  "${DATASET_ARGS[@]}" \
  --load_from_cache_file true \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS:-4}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS:-4}" \
  --learning_rate "${LR:-1e-4}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_steps "${EVAL_STEPS:-100}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length "${MAX_LENGTH:-2048}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --deepspeed zero2 \
  --model_author zhuxz \
  --model_name qwen35-4b-benchgen-sft \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[done] logs: ${RUN_DIR}/train.log"
echo "[infer later]:"
echo "  swift infer --adapters ${OUTPUT_DIR}/vx-xxx/checkpoint-xxx --stream true --enable_thinking false --max_new_tokens 512"
