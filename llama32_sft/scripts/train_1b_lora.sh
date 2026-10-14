#!/usr/bin/env bash
# Llama-3.2-1B-Instruct LoRA SFT wrapper.
# Memory-safe defaults for MACA + eager attention (calib can spike to ~3.5k tokens).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL="${MODEL:-/mnt/public/model/huggingface/Llama-3.2-1B-Instruct}"
# global ≈ nproc * 1 * 16 — override freely
export PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-1}"
export PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
# calib: most rows <<2k; use 4096 only if you accept lower micro-batch
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
bash "${ROOT}/scripts/train_lora.sh"
