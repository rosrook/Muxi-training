#!/usr/bin/env bash
# Stop workers started by start_vllm_8x_single_gpu.sh
set -euo pipefail
LOG_DIR="${LOG_DIR:-}"
if [[ -z "${LOG_DIR}" ]]; then
  echo "Usage: LOG_DIR=.../deploy_vllm_8x/nonfg bash stop_vllm_8x.sh" >&2
  exit 2
fi
PIDF="${LOG_DIR}/workers.pid"
if [[ ! -f "${PIDF}" ]]; then
  echo "missing ${PIDF}" >&2
  exit 1
fi
while read -r pid; do
  [[ -z "${pid}" ]] && continue
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    echo "killed ${pid}"
  fi
done < "${PIDF}"
# best-effort cleanup orphaned vllm on ports
echo "done"
