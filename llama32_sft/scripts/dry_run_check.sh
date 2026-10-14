#!/usr/bin/env bash
# Offline dry-run: local model + chat_template (+ optional JSONL).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

MODEL="${MODEL:-/mnt/public/model/huggingface/Llama-3.2-1B-Instruct}"
JSONL="${JSONL:-${ROOT}/data/train.jsonl.example}"

python "${ROOT}/src/dry_run_check.py" --model "${MODEL}" --jsonl "${JSONL}"
