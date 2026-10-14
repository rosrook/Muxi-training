#!/usr/bin/env bash
# ms-swift 4.1.3 MACA 容器内一键：补依赖 → 跑 Qwen3.5-4B SFT（max_length 8192 版）
# 镜像：cr.infini-ai.com/te-c7vnqlzlmvzffc2v/ms-swift:4.1.3-maca-torch2.8-py312
#
# 与 run_in_container.sh 的差别只有两处：
#   1. 最后调用 train_sft_qwen35_4b_len8192.sh
#   2. 数据自检不再要求 train.jsonl 必须已存在。指定了 NONFG_SFT 时，
#      训练脚本会自己调 prepare_dataset.py 生成，原版在这一步会直接退出。
#
# 容器内用法：
#   export NONFG_SFT=/path/to/final/sft_dataset.jsonl
#   export DATA_ROOT=/path/to/new/06_ms_swift_sft
#   bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/run_in_container_len8192.sh
#
set -euo pipefail

CODE_DIR="${CODE_DIR:-/mnt/public/code/zhuxz/training/ms_swift_qwen35}"
DATA_ROOT="${DATA_ROOT:-/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft}"
MODEL="${MODEL:-/mnt/public/model/Qwen3.5-4B}"

# 本地权重 / 无外网：禁止 hub/xet 在 import 时探测网络（否则会表现为“卡住”）
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"

echo "========== 0) 进入训练目录 =========="
cd "${CODE_DIR}"
pwd
ls -la

echo "========== 1) 检查已预装组件（勿重装 torch） =========="
python - <<'PY'
import torch
print("python torch:", getattr(torch, "__version__", "?"))
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    print("device0:", torch.cuda.get_device_name(0))
try:
    import swift
    print("ms-swift:", getattr(swift, "__version__", swift.__file__))
except Exception as e:
    print("ms-swift import FAIL:", e)
try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers FAIL:", e)
PY
command -v swift || true

echo "========== 2) 补齐可能缺的包（不动 torch/maca/liger 栈） =========="
# 镜像: ms-swift 4.1.3 → 官方 v4.1 文档钉死 transformers==5.2.*
#   https://swift.readthedocs.io/en/v4.1/BestPractices/Qwen3_5-Best-Practice.html
# 禁止: pip -U torch / liger-kernel / triton（会冲掉 Metax MACA 轮子）
# 禁止: transformers>=5.9 / 5.14（与 ms-swift 4.1.3 冲突）
python -m pip install -U \
  "transformers==5.2.*" \
  "qwen_vl_utils>=0.0.14" \
  "peft>=0.11,<0.20"
# hf-xet 常在受限网络下让 hub import 挂起；训练用本地模型不需要它
python -m pip uninstall -y hf-xet 2>/dev/null || true

python - <<'PY'
import peft
import transformers
tv, pv = transformers.__version__, peft.__version__
print(f"transformers={tv} peft={pv}")
maj_min = tuple(int(x) for x in tv.split(".")[:2])
assert maj_min[:2] == (5, 2), f"need transformers 5.2.*, got {tv}"
assert tuple(int(x) for x in pv.split(".")[:2]) < (0, 20), f"need peft<0.20, got {pv}"
PY

echo "[check] deepspeed ..."
python - <<'PY' || python -m pip install -U deepspeed
import deepspeed
print("deepspeed ok", getattr(deepspeed, "__version__", "?"))
PY

echo "[check] liger-kernel ..."
python - <<'PY'
try:
    import liger_kernel
    print("liger-kernel keep as-is:", getattr(liger_kernel, "__version__", liger_kernel.__file__))
except Exception as e:
    print("[info] liger-kernel not present (optional):", e)
PY

# 懒加载补丁：.pth 只注册 import hook，绝不在启动时 import transformers.trainer
echo "[compat] install lazy create_repo patch (no eager trainer import)"
SITE_SP="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -f "${CODE_DIR}/patch_transformers_for_ms_swift.py" "${SITE_SP}/patch_transformers_for_ms_swift.py"
echo "import patch_transformers_for_ms_swift" > "${SITE_SP}/zz_ms_swift_tf_compat.pth"
python -c "import patch_transformers_for_ms_swift as p; print('[compat] lazy hook OK', p.__file__)"

echo "========== 3) 路径与数据自检 =========="
[[ -d "${MODEL}" ]] || { echo "ERROR: model missing: ${MODEL}"; exit 2; }

export TRAIN_JSONL="${TRAIN_JSONL:-${DATA_ROOT}/train.jsonl}"
export VAL_JSONL="${VAL_JSONL:-${DATA_ROOT}/train.val.jsonl}"

if [[ -f "${TRAIN_JSONL}" ]]; then
  echo "train=$(wc -l < "${TRAIN_JSONL}") lines  (${TRAIN_JSONL})"
  [[ -f "${VAL_JSONL}" ]] || echo "[warn] no val jsonl, script will split"
elif [[ -n "${NONFG_SFT:-}" ]]; then
  [[ -f "${NONFG_SFT}" ]] || { echo "ERROR: NONFG_SFT missing: ${NONFG_SFT}"; exit 2; }
  echo "[info] ${TRAIN_JSONL} not built yet; will be prepared from:"
  echo "       ${NONFG_SFT}"
else
  echo "ERROR: ${TRAIN_JSONL} missing and NONFG_SFT unset." >&2
  echo "       Set NONFG_SFT to the source sft_dataset.jsonl, or build it first." >&2
  exit 2
fi
echo "model=${MODEL}"

# Metax/MACA：勿覆盖官方 CUDA torch；卡数按环境可见 GPU
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(max(torch.cuda.device_count(), 1))')}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(python -c "import torch; print(','.join(map(str, range(torch.cuda.device_count() or 1))))")"
fi
export NPROC_PER_NODE CUDA_VISIBLE_DEVICES
export MODEL DATA_ROOT
export FORCE_PREPARE="${FORCE_PREPARE:-0}"

echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "========== 4) 启动 swift SFT（max_length 8192） =========="
bash "${CODE_DIR}/train_sft_qwen35_4b_len8192.sh"
