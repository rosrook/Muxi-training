#!/usr/bin/env bash
# 8× single-GPU vLLM instances for EVAL (teacher layout):
#   - NO internal multi-request batch: --max-num-seqs 1
#   - Parallelism only via outer client queue across ports 8000..8007
#   - temperature=0 / timeout+retry are client-side
#
# Image:
#   cr.infini-ai.com/te-c7vnqlzlmvzffc2v/vllm-metax:0.21.0-maca.ai3.7.1.106-torch2.8-py310-ubuntu22.04-amd64
#
# Requirements: MODEL_PATH = already-merged HF dir (not raw LoRA adapter alone).
#   Merge once on ms-swift image, e.g.:
#     swift export --model /mnt/public/model/Qwen3.5-4B --adapters <ckpt> --merge_lora true --output_dir <MERGED>
#
# Usage:
#   MODEL_PATH=... SERVED_NAME=... bash deploy_vllm_multi_instance.sh start|stop|status
#
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to merged weights dir}"
SERVED_NAME="${SERVED_NAME:-qwen35-sft}"
BASE_PORT="${BASE_PORT:-8000}"
NUM_GPUS="${NUM_GPUS:-8}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
STAGGER_SEC="${STAGGER_SEC:-3}"

LOG_ROOT="${LOG_ROOT:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_vllm_logs/${SERVED_NAME}}"
PID_DIR="${LOG_ROOT}/pids"
CMD="${1:-start}"

mkdir -p "${LOG_ROOT}" "${PID_DIR}"

if [[ -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: ${MODEL_PATH} has no config.json (need merged HF weights, not empty dir)" >&2
  exit 2
fi

launch_instance() {
  local gpu="$1"
  local port=$((BASE_PORT + gpu))
  local log="${LOG_ROOT}/gpu${gpu}_port${port}.log"
  local pidf="${PID_DIR}/gpu${gpu}.pid"

  if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "[skip] gpu${gpu} already up pid=$(cat "${pidf}") :${port}"
    return 0
  fi

  echo "[start] CUDA_VISIBLE_DEVICES=${gpu} port=${port} name=${SERVED_NAME}"

  # vLLM ≥0.6 often ships `vllm serve`; Metax 0.21 should too.
  if command -v vllm >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    nohup vllm serve "${MODEL_PATH}" \
      --host "${HOST}" \
      --port "${port}" \
      --served-model-name "${SERVED_NAME}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --trust-remote-code \
      >"${log}" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    nohup python -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" \
      --host "${HOST}" \
      --port "${port}" \
      --served-model-name "${SERVED_NAME}" \
      --dtype "${DTYPE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM_UTIL}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --trust-remote-code \
      >"${log}" 2>&1 &
  fi
  echo $! >"${pidf}"
  echo "  pid=$(cat "${pidf}")  log=${log}"
}

stop_all() {
  local f pid
  for f in "${PID_DIR}"/gpu*.pid; do
    [[ -e "$f" ]] || continue
    pid="$(cat "$f" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] ${pid}"
      kill "${pid}" 2>/dev/null || true
      sleep 1
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "$f"
  done
  # also kill stragglers bound to our ports (best effort)
  if command -v fuser >/dev/null 2>&1; then
    local p
    for ((p = BASE_PORT; p < BASE_PORT + NUM_GPUS; p++)); do
      fuser -k "${p}/tcp" 2>/dev/null || true
    done
  fi
}

status_all() {
  echo "SERVED_NAME=${SERVED_NAME}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "layout: ${NUM_GPUS}×GPU  ports ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))  max_num_seqs=${MAX_NUM_SEQS}"
  local i port pidf pid st http
  for ((i = 0; i < NUM_GPUS; i++)); do
    port=$((BASE_PORT + i))
    pidf="${PID_DIR}/gpu${i}.pid"
    st="DOWN"; pid="-"; http="-"
    if [[ -f "${pidf}" ]]; then
      pid="$(cat "${pidf}")"
      kill -0 "${pid}" 2>/dev/null && st="UP"
    fi
    if curl -sf --max-time 2 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      http="READY"
    else
      http="WAIT/ERR"
    fi
    echo "  gpu${i} :${port}  proc=${st} pid=${pid}  http=${http}"
  done
}

case "${CMD}" in
  start)
    echo "[info] Teacher layout: 8 engines × max_num_seqs=1; outer queue for parallelism."
    local g
    for ((g = 0; g < NUM_GPUS; g++)); do
      launch_instance "${g}"
      sleep "${STAGGER_SEC}"
    done
    echo "[done] logs → ${LOG_ROOT}"
    echo "After load finishes: bash $0 status"
    echo "Endpoints: http://<this-host>:${BASE_PORT} .. $((BASE_PORT + NUM_GPUS - 1))"
    ;;
  stop) stop_all; echo "[done] stop" ;;
  status) status_all ;;
  *)
    echo "Usage: MODEL_PATH=... SERVED_NAME=... $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
