#!/usr/bin/env bash
# MACA 容器入口：检查环境 → 钉版本装依赖（不动 torch）→ dry-run → 训练。
# 镜像：cr.infini-ai.com/te-c7vnqlzlmvzffc2v/ms-swift:4.1.3-maca-torch2.8-py312
#
# 环境策略对齐 Qwen 的 run_in_container_len8192.sh：
#   - 离线 HF / 禁 xet（防 import 卡住）
#   - 共享 cache 路径
#   - 严格 pin 依赖 + 安装后 assert（不依赖镜像里残留的旧包）
#   - 绝不 pip install/upgrade torch、triton、liger-kernel
#
#   bash /mnt/public/code/zhuxz/training/llama32_sft/scripts/run_in_container.sh
#   SIZE=3b bash .../run_in_container.sh
#   DRY_RUN=1 bash .../run_in_container.sh
#
set -euo pipefail

CODE_DIR="${CODE_DIR:-/mnt/public/code/zhuxz/training/llama32_sft}"
# Default: MCQ calibration v1 → Llama messages (see data_prepare.md)
DATA_ROOT="${DATA_ROOT:-/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z}"
SIZE="${SIZE:-1b}"  # 1b | 3b

# ---- same offline / cache knobs as ms_swift_qwen35/run_in_container_len8192.sh ----
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HOME="${HF_HOME:-/mnt/public/data/zhuxz/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/mnt/public/data/zhuxz/cache/modelscope}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# calib 推理偏长；默认 2048（语料 p99~1k，仅极少行>2k）。要保留长样本再 export MAX_LENGTH=4096
export MAX_LENGTH="${MAX_LENGTH:-2048}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export DATA_ROOT

case "${SIZE}" in
  1b|1B)
    MODEL="${MODEL:-/mnt/public/model/huggingface/Llama-3.2-1B-Instruct}"
    TRAIN_SCRIPT="${CODE_DIR}/scripts/train_1b_lora.sh"
    ;;
  3b|3B)
    MODEL="${MODEL:-/mnt/public/model/huggingface/Llama-3.2-3B-Instruct}"
    TRAIN_SCRIPT="${CODE_DIR}/scripts/train_3b_lora.sh"
    ;;
  *)
    echo "ERROR: SIZE must be 1b or 3b, got ${SIZE}" >&2
    exit 2
    ;;
esac
export MODEL

echo "========== 0) code dir =========="
cd "${CODE_DIR}"
pwd
ls -la
echo "HF_HOME=${HF_HOME}"
echo "MODELSCOPE_CACHE=${MODELSCOPE_CACHE}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "MAX_LENGTH=${MAX_LENGTH}"

echo "========== 1) env probe (do NOT reinstall torch) =========="
python - <<'PY'
import torch
print("torch:", getattr(torch, "__version__", "?"))
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
try:
    import transformers
    print("transformers (pre-install):", transformers.__version__)
except Exception as e:
    print("transformers FAIL:", e)
try:
    import accelerate
    print("accelerate (pre-install):", accelerate.__version__)
except Exception as e:
    print("accelerate FAIL:", e)
PY

echo "========== 2) pin install (Qwen-style: explicit versions + assert) =========="
# 禁止：torch / liger-kernel / triton
# Llama 3.2 用 transformers 4.x 即可（镜像常见 4.57）；勿装到 5.14（Qwen 踩过的坑）
# accelerate 必须够新：transformers.Trainer.unwrap_model(..., keep_torch_compile=)
#   旧 0.33 会 TypeError — 对齐 Qwen「钉死版本 + 断言」的做法
python -m pip install -U \
  "accelerate>=1.1.0,<1.5" \
  "transformers>=4.46.0,<5.0" \
  "peft>=0.11.0,<0.20" \
  "trl>=0.12.0,<0.16" \
  "datasets>=2.19.0,<4.0" \
  "safetensors>=0.4.0"
# 与 Qwen 相同：去掉 hf-xet，避免受限网络下 hub import 假死
python -m pip uninstall -y hf-xet 2>/dev/null || true

python - <<'PY'
import inspect

import accelerate
import datasets
import peft
import transformers
import trl
from accelerate import Accelerator
from packaging.version import Version

print("peft", peft.__version__)
print("trl", trl.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)

assert Version(transformers.__version__) < Version("5.0"), (
    f"Llama path expects transformers 4.x, got {transformers.__version__}"
)
assert Version(accelerate.__version__) >= Version("1.1.0"), (
    f"need accelerate>=1.1.0 for keep_torch_compile, got {accelerate.__version__}"
)
assert Version(peft.__version__) < Version("0.20"), (
    f"need peft<0.20, got {peft.__version__}"
)

sig = inspect.signature(Accelerator.unwrap_model)
has_ktc = "keep_torch_compile" in sig.parameters
print("accelerate.unwrap_model keep_torch_compile:", has_ktc)
if not has_ktc:
    raise SystemExit(
        "ERROR: accelerate still lacks keep_torch_compile after pin-install; "
        "check pip target env / multiple pythons"
    )
print("[ok] package pins compatible with transformers.Trainer")
PY

echo "========== 3) dry-run (model + chat_template) =========="
bash "${CODE_DIR}/scripts/dry_run_check.sh"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] stop before training (DRY_RUN=1)"
  exit 0
fi

echo "========== 4) data check =========="
mkdir -p "${DATA_ROOT}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "ERROR: no training data at ${TRAIN_JSONL}" >&2
  echo "See ${CODE_DIR}/DATA_FORMAT.md / data_prepare.md" >&2
  echo "  python3 ${CODE_DIR}/src/prepare_calib_dataset.py --preset calib-v1" >&2
  exit 2
fi
export TRAIN_JSONL
echo "train lines: $(wc -l < "${TRAIN_JSONL}")"
[[ -f "${DATA_ROOT}/val.jsonl" ]] && echo "val lines: $(wc -l < "${DATA_ROOT}/val.jsonl")"

# Metax/MACA：卡数按环境可见 GPU（同 Qwen 入口）
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python -c "import torch; print(','.join(map(str, range(torch.cuda.device_count() or 1))))")"
fi
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "========== 5) train SIZE=${SIZE} MODEL=${MODEL} =========="
bash "${TRAIN_SCRIPT}"
