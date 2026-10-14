# Llama-3.2 Instruct LoRA SFT（MACA / 本地 HF）

针对本地 checkpoint 的 **TRL + PEFT LoRA** 训练骨架（数据自备）。

## 模型（不下载）

```text
/mnt/public/model/huggingface/Llama-3.2-1B-Instruct
/mnt/public/model/huggingface/Llama-3.2-3B-Instruct
```

## 方案说明

| 优先级 | 方案 | 本仓库 |
|--|--|--|
| 1 | Meta [llama-cookbook](https://github.com/meta-llama/llama-cookbook) `finetuning.py` | 数据语义对齐 cookbook `dialog` + `apply_chat_template`；不强制 pip 整包 cookbook（MACA 易冲突） |
| 2 | **HF TRL SFTTrainer + PEFT**（默认实现） | `src/train_sft_trl.py` |

默认：**LoRA + bf16**，`ATTN_IMPLEMENTATION=eager`（Metax 更稳）。不做 4bit QLoRA（bitsandbytes 在 MACA 上不可靠）。

训练镜像（与 Qwen 同一类）：

```text
cr.infini-ai.com/te-c7vnqlzlmvzffc2v/ms-swift:4.1.3-maca-torch2.8-py312
```

Llama 3.2 可用镜像内 **transformers 4.5x**，**不要**为了本任务再装 transformers 5.14。

## 目录

```text
DATA_FORMAT.md          # 数据契约（必读）
requirements.txt        # 允许 pip 的包（禁 torch 升级）
data/*.example          # 样例行
src/data_messages.py    # schema 校验 + 加载
src/prepare_calib_dataset.py  # calib → Llama JSONL
src/train_sft_trl.py    # 训练主程序
src/dry_run_check.py    # 模型/template 检查
scripts/run_in_container.sh
scripts/train_1b_lora.sh
scripts/train_3b_lora.sh
scripts/train_lora.sh
scripts/dry_run_check.sh
```

## 数据

契约：[DATA_FORMAT.md](DATA_FORMAT.md)。运维说明：`运行指南/.../llama32/data_prepare.md`。

**默认 calib v1（已转换）：**

```text
/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z/
  train.jsonl   # 29325
  val.jsonl     # 296
```

来源：`mcq_calibration/20260808T152739Z`（与 Qwen calib-sft 同一 train/val 切分）。重转：

```bash
python3 src/prepare_calib_dataset.py --preset calib-v1
```

calib 建议 `MAX_LENGTH=4096` 或 `8192`（脚本默认 2048 会截断）。

## 在 MACA 容器内跑

```bash
cd /mnt/public/code/zhuxz/training/llama32_sft

# 仅检查模型 + chat_template（不需要数据）
DRY_RUN=1 SIZE=1b bash scripts/run_in_container.sh

# 默认 DATA_ROOT 已是 calib_20260808…；可显式写出：
export DATA_ROOT=/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z
export MAX_LENGTH=4096
SIZE=1b bash scripts/run_in_container.sh
SIZE=3b bash scripts/run_in_container.sh
```

或直接：

```bash
export MODEL=/mnt/public/model/huggingface/Llama-3.2-1B-Instruct
export DATA_ROOT=/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z
export MAX_LENGTH=4096
bash scripts/train_1b_lora.sh
```

产物默认：

```text
/mnt/public/data/zhuxz/training/llama32_sft/runs/<stamp>_Llama-3.2-*-Instruct_lora/
  adapter/           # PEFT 权重
  checkpoints/
  run_config.json
  train_metrics.json
```

合并全量（可选）：`MERGE_WEIGHTS=1`。

## 环境变量（常用）

| 变量 | 默认 / 含义 |
|--|--|
| `SIZE` | `1b` / `3b`（`run_in_container`） |
| `MODEL` | 本地模型目录 |
| `DATA_ROOT` | 数据根 |
| `TRAIN_JSONL` / `VAL_JSONL` | 覆盖 JSONL 路径 |
| `NPROC_PER_NODE` | 多卡数；默认可见 GPU 数 |
| `CUDA_VISIBLE_DEVICES` | 如 `0,1,2,3` |
| `PER_DEVICE_TRAIN_BS` / `GRAD_ACCUM` / `LR` / `MAX_LENGTH` | batch / lr / 长度 |
| `LORA_R` / `LORA_ALPHA` | 默认 16 / 32 |
| `ATTN_IMPLEMENTATION` | 默认 `eager` |
| `DRY_RUN` | `1` 则装依赖+检查后退出 |
| `SKIP_EVAL` | `1` 不做验证 |
| `MERGE_WEIGHTS` | `1` 写出 `merged/` |

## 禁止事项（MACA）

- 不要 `pip install -U torch` / `triton` / 替换 `liger-kernel`
- 不要默认用 bitsandbytes 4bit
- 离线训练保持 `HF_HUB_OFFLINE=1`

## cookbook 对照（可选，非本仓库默认入口）

在标准 NVIDIA + 已装 `llama-cookbook` 的环境，可参考：

```bash
# 单卡 LoRA 示意；数据集需按 cookbook custom_dataset 插件接入
python -m llama_cookbook.finetuning \
  --use_peft --peft_method lora \
  --model_name /mnt/public/model/huggingface/Llama-3.2-3B-Instruct \
  --output_dir /path/to/peft_out
```

本仓库数据格式与 cookbook 的 `role`/`content` dialog 同构，字段名用 `messages` 方便 TRL。

## 后续（未做）

- 部署：Transformers / vLLM serve adapter 或 merged
- 评测：lm-evaluation-harness（before/after）

运维向笔记：`bench_probe_discovery/运行指南/关于模型的训练部署与评测/llama32/train.md`
