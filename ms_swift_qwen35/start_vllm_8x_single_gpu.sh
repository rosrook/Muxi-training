#!/usr/bin/env bash
# 8× 单卡 vLLM 实例（老师建议：多实例 + 外层并行；实例内不靠大 batch）
# 镜像: cr.infini-ai.com/te-c7vnqlzlmvzffc2v/vllm-metax:0.21.0-maca.ai3.7.1.106-torch2.8-py310-ubuntu22.04-amd64
#
# 预算 8 卡：一次只服务「一个」变体（base / nonfg / full）。评完 A 再启 B。
#
# Usage:
#   VARIANT=base  bash start_vllm_8x_single_gpu.sh   # 训前 base
#   VARIANT=nonfg bash start_vllm_8x_single_gpu.sh
#   VARIANT=full  bash start_vllm_8x_single_gpu.sh
#
# 平台「1 卡 × 8 实例」表单：每个实例入口只跑下方「单实例」命令（见脚本末注释 / 控制台日志里的 MODEL_PATH）。
#
set -euo pipefail

VARIANT="${VARIANT:-nonfg}"   # base | nonfg | full
BASE_PORT="${BASE_PORT:-8000}"  # 实例端口 8000..8007；对外网关也可只暴露其中一个+负载，默认评测直连 8 端点
N_GPU="${N_GPU:-8}"
LOG_DIR="${LOG_DIR:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_vllm_8x/${VARIANT}}"
mkdir -p "${LOG_DIR}"

BASE_MODEL="${BASE_MODEL:-/mnt/public/model/Qwen3.5-4B}"
MODEL_PATH=""
SERVED_NAME=""
ADAPTER=""
case "${VARIANT}" in
  base)
    MODEL_PATH="${MODEL_PATH:-${BASE_MODEL}}"
    SERVED_NAME="${SERVED_NAME:-qwen35-4b-base}"
    ;;
  nonfg)
    MODEL_PATH="${MODEL_PATH:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/nonfg_ckpt1851}"
    SERVED_NAME="${SERVED_NAME:-qwen35-4b-nonfg-sft}"
    ADAPTER="${ADAPTER:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/run_20260806T074147Z/output_Qwen3.5-4B/v0-20260806-154220/checkpoint-1851}"
    ;;
  full)
    MODEL_PATH="${MODEL_PATH:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/full_ckpt2754}"
    SERVED_NAME="${SERVED_NAME:-qwen35-4b-full-sft}"
    ADAPTER="${ADAPTER:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/run_20260806T080723Z/output_Qwen3.5-4B/v0-20260806-160756/checkpoint-2754}"
    ;;
  *)
    echo "VARIANT must be base|nonfg|full" >&2
    exit 2
    ;;
esac

# 若尚未 merge LoRA（仅 nonfg/full），优先用本机已装的 swift；否则要求用户先 merge
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  if [[ "${VARIANT}" == "base" ]]; then
    echo "ERROR: base model missing config.json at ${MODEL_PATH}" >&2
    exit 3
  fi
  echo "[merge] missing ${MODEL_PATH}; try swift export (need ms-swift env) ..."
  if command -v swift >/dev/null 2>&1; then
    mkdir -p "$(dirname "${MODEL_PATH}")"
    swift export \
      --model "${BASE_MODEL}" \
      --adapters "${ADAPTER}" \
      --merge_lora true \
      --output_dir "${MODEL_PATH}"
  else
    echo "ERROR: merged weights not found and 'swift' unavailable in this vllm image." >&2
    echo "  Merge once in the ms-swift image, then re-run this script." >&2
    echo "  Expected: ${MODEL_PATH}" >&2
    exit 3
  fi
fi

echo "[info] model=${MODEL_PATH}"
echo "[info] served_model_name=${SERVED_NAME}"
echo "[info] ports=${BASE_PORT}..$((BASE_PORT + N_GPU - 1))  gpus=0..$((N_GPU - 1))"
echo "[info] per-instance: max-num-seqs=1 (no multi-sample batch inside vLLM)"

# 清旧 pid 文件
: > "${LOG_DIR}/workers.pid"
: > "${LOG_DIR}/endpoints.txt"

for i in $(seq 0 $((N_GPU - 1))); do
  port=$((BASE_PORT + i))
  logf="${LOG_DIR}/worker_gpu${i}_p${port}.log"
  echo "http://127.0.0.1:${port}/v1" >> "${LOG_DIR}/endpoints.txt"

  # 老师：单卡实例；max-num-seqs=1 → 实例内不做多样本 continuous batch
  # temperature 由客户端强制 0
  CUDA_VISIBLE_DEVICES="${i}" \
  nohup vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --served-model-name "${SERVED_NAME}" \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --max-num-seqs 1 \
    --disable-log-requests \
    > "${logf}" 2>&1 &
  echo $! >> "${LOG_DIR}/workers.pid"
  echo "[start] gpu=${i} port=${port} pid=$! log=${logf}"
done

echo "[wait] readiness (timeout 600s each)..."
for i in $(seq 0 $((N_GPU - 1))); do
  port=$((BASE_PORT + i))
  ok=0
  for t in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[ready] :${port}"
      ok=1
      break
    fi
    sleep 5
  done
  if [[ "${ok}" -ne 1 ]]; then
    echo "[ERROR] port ${port} not ready; see ${LOG_DIR}/worker_gpu${i}_p${port}.log" >&2
    exit 4
  fi
done

echo "[done] endpoints file: ${LOG_DIR}/endpoints.txt"
echo "       stop: bash $(dirname "$0")/stop_vllm_8x.sh LOG_DIR=${LOG_DIR}"
cat "${LOG_DIR}/endpoints.txt"
