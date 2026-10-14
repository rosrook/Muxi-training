#!/usr/bin/env bash
# Qwen3.5-4B LoRA SFT with ms-swift, max_length 8192 variant.
# Doc: https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html
#
# Same recipe as train_sft_qwen35_4b.sh. The only change is the length cap:
#
#   max_length  2048 -> 8192
#
# Batch shape is deliberately left at 4 x 1 so this run stays directly
# comparable with the 2048 runs.
#
# Why the cap costs nothing. With --group_by_length true, batches are padded to
# the longest sample in the batch, not to max_length; the cap never allocates
# memory. Measured on the calibration-rewritten corpus (29,621 rows): p50 398,
# p99 1,051, max 3,518 tokens, with only 4 rows above 2,048 and none above
# 4,096. So raising the cap recovers those 4 rows and changes nothing else.
#
# The one real memory difference versus the 2048 runs is the single batch that
# group_by_length will assemble from those 4 long rows: 4 x ~3,500 tokens
# instead of 4 x 2,048. With gradient checkpointing on that is comfortable. If
# it does OOM, drop to PER_DEVICE_TRAIN_BS=2 GRAD_ACCUM=2 (global batch stays
# 16) rather than lowering max_length.
#
# Usage (GPU node):
#   source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate
#   bash train_sft_qwen35_4b_len8192.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${DATA_ROOT}/run_${STAMP}}"
mkdir -p "${RUN_DIR}"

# 8192-variant defaults. Override any of them from the environment.
MAX_LENGTH="${MAX_LENGTH:-8192}"
PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-4}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

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

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LR="${LR:-1e-4}"
GLOBAL_BATCH=$(( NPROC_PER_NODE * PER_DEVICE_TRAIN_BS * GRAD_ACCUM ))

# Persist launch config. The length/batch fields matter here: two runs of this
# corpus are only comparable if they share max_length and effective batch.
cat > "${RUN_DIR}/run_config.json" <<EOF
{
  "stamp": "${STAMP}",
  "variant": "len8192",
  "model": "${MODEL}",
  "train_jsonl": "${TRAIN_JSONL}",
  "val_jsonl": "${VAL_JSONL}",
  "nproc": ${NPROC_PER_NODE},
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "output_dir": "${OUTPUT_DIR}",
  "max_length": ${MAX_LENGTH},
  "per_device_train_batch_size": ${PER_DEVICE_TRAIN_BS},
  "gradient_accumulation_steps": ${GRAD_ACCUM},
  "global_batch_size": ${GLOBAL_BATCH},
  "num_train_epochs": ${NUM_TRAIN_EPOCHS},
  "learning_rate": "${LR}",
  "doc": "https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html"
}
EOF

echo "[train] model=${MODEL}"
echo "[train] train=${TRAIN_JSONL} ($(wc -l < "${TRAIN_JSONL}") lines)"
[[ -f "${VAL_JSONL}" ]] && echo "[train] val=${VAL_JSONL} ($(wc -l < "${VAL_JSONL}") lines)"
echo "[train] output=${OUTPUT_DIR}"
echo "[train] max_length=${MAX_LENGTH} per_device_bs=${PER_DEVICE_TRAIN_BS} grad_accum=${GRAD_ACCUM} global_batch=${GLOBAL_BATCH}"

# Dataset args: prefer explicit val file; else split_ratio as in official demo
DATASET_ARGS=(--dataset "${TRAIN_JSONL}")
if [[ -f "${VAL_JSONL}" ]]; then
  DATASET_ARGS+=(--val_dataset "${VAL_JSONL}")
else
  DATASET_ARGS+=(--split_dataset_ratio 0.01)
fi

# ---- Official Dense Qwen3.5-4B LoRA SFT skeleton ----
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
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BS}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BS}" \
  --learning_rate "${LR}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --group_by_length true \
  --output_dir "${OUTPUT_DIR}" \
  --eval_steps "${EVAL_STEPS:-100}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length "${MAX_LENGTH}" \
  --warmup_ratio 0.05 \
  --dataset_num_proc 4 \
  --dataloader_num_workers 4 \
  --deepspeed zero2 \
  --model_author zhuxz \
  --model_name qwen35-4b-benchgen-sft-len8192 \
  2>&1 | tee "${RUN_DIR}/train.log"

echo "[done] logs: ${RUN_DIR}/train.log"
echo "[infer later]:"
echo "  swift infer --adapters ${OUTPUT_DIR}/vx-xxx/checkpoint-xxx --stream true --enable_thinking false --max_new_tokens 512"
