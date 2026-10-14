#!/usr/bin/env bash
# Install ms-swift stack for Qwen3.5 (official Best Practices env section).
# Ref: https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html
set -euo pipefail

ENV_DIR="${ENV_DIR:-/mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env}"
PYTHON="${PYTHON:-python3}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[info] creating conda/venv at ${ENV_DIR}"
  if command -v conda >/dev/null 2>&1; then
    conda create -y -p "${ENV_DIR}" python=3.12
  else
    "${PYTHON}" -m venv "${ENV_DIR}"
  fi
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel

# Core (official doc)
python -m pip install -U ms-swift
# MACA 镜像 / ms-swift 4.1.x：用官方 v4.1 钉死的 5.2.*（不要装 >=5.9，那是更新文档对应新版 ms-swift）
python -m pip install -U "transformers==5.2.*" "qwen_vl_utils>=0.0.14" "peft>=0.11,<0.20" liger-kernel

# Optional but recommended in the official guide (can be slow / build-heavy):
# python -m pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
# python -m pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
# python -m pip install "flash-attn==2.8.3" --no-build-isolation
python -m pip install -U deepspeed

python -c "import swift; print('ms-swift ok', getattr(swift, '__version__', 'unknown'))"
which swift
echo "[done] activate: source ${ENV_DIR}/bin/activate"
