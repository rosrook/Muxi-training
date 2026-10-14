#!/usr/bin/env bash
# Llama-3.2-3B-Instruct LoRA SFT wrapper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL="${MODEL:-/mnt/public/model/huggingface/Llama-3.2-3B-Instruct}"
export PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-1}"
export PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
bash "${ROOT}/scripts/train_lora.sh"
