# ms-swift · Qwen3.5-4B SFT（替代旧 create_run / lora_4gpu 脚本）

严格对齐官方文档：

https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html

## 官方要点（Dense 4B）

- 用 **ms-swift / transformers 后端**（Dense 推荐，不用 Megatron）
- LoRA：`--tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear`
- 思维模型友好：`--add_non_thinking_prefix true --loss_scale ignore_empty_think`
- 加速：`--group_by_length true`（也可用 `--packing true`）
- dtype / 并行：`bfloat16` + `deepspeed zero2` + 4 卡示例
- 自定义数据：`messages` JSONL

官方示例命令骨架（节选）：

```bash
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
  --model Qwen/Qwen3.5-4B \
  --tuner_type lora \
  --dataset ... \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-4 \
  --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --group_by_length true \
  --deepspeed zero2 \
  --max_length 2048
```

本目录 `train_sft_qwen35_4b.sh` 在上述参数上改为：

- `--model /mnt/public/model/Qwen3.5-4B`（本地权重，可覆盖）
- `--dataset` 为我们的 messages JSONL

## 数据流

```
Non-FG step rewrite sft_dataset.jsonl  ─┐
                                        ├─ prepare_dataset.py → train.jsonl (+ val)
FG knowledge sft (optional --include-fg)┘
```

- 重写中的 non-FG 可随时重跑 prepare（用最新 `sft_dataset.jsonl` 路径）。
- FG **不要**走 step-rewrite；需要混训时：`INCLUDE_FG=1`。

## 一步步

### 0. GPU 节点 + 环境

本机当前 **无 nvidia-smi / 未装 ms-swift**，请在 GPU 机执行：

```bash
bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/install_env.sh
source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate
# 可选重装 flash-attn / fla 见 install_env.sh 注释（与官方一致）
```

要求：`ms-swift>=4.3.1` 才完整支持 packing/padding_free。

**MACA 容器（ms-swift 4.1.3）**请用：

```bash
bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/run_in_container.sh
```

该入口会按 [v4.1 官方文档](https://swift.readthedocs.io/en/v4.1/BestPractices/Qwen3_5-Best-Practice.html) 安装 `transformers==5.2.*`、`peft<0.20`。  
不要装 `transformers>=5.9/5.14`（会和镜像里的 ms-swift 4.1.3 冲突）。

### 1. 准备 JSONL

```bash
# 仅 non-FG 重写结果（默认路径指向当前 rewrite 目录）
python3 /mnt/public/code/zhuxz/training/ms_swift_qwen35/prepare_dataset.py \
  --out /mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/train.jsonl \
  --val-ratio 0.01

# 混入 FG 知识 SFT
python3 .../prepare_dataset.py --include-fg --out ...
```

重写尚未跑完时，先用当前 accepted 子集开训；跑完后再 prepare 一次。

### 2. 启动 SFT

```bash
source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
# export NONFG_SFT=/path/to/latest/sft_dataset.jsonl
# export INCLUDE_FG=1
# export FORCE_PREPARE=1
bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/train_sft_qwen35_4b.sh
```

产物：`$RUN_DIR/output_Qwen3.5-4B/` + `train.log`。

### 3. 推理（官方同款）

```bash
swift infer \
  --adapters /path/to/output_Qwen3.5-4B/vx-xxx/checkpoint-xxx \
  --stream true \
  --enable_thinking false \
  --max_new_tokens 512
```

## 与旧管线差异

| | 旧 `training/create_run` | **本方案** |
|--|--|--|
| 框架 | 自研/自定义 | **ms-swift** |
| 入口 | `create_run.sh` + yaml | `swift sft` |
| 文档对齐 | 内部 | Qwen3.5 Best Practices |
| 数据 | 自有 pack 约定 | messages JSONL |

## 参数覆盖速查

| 环境变量 | 默认 |
|--|--|
| `MODEL` | `/mnt/public/model/Qwen3.5-4B` |
| `NPROC_PER_NODE` | 4 |
| `CUDA_VISIBLE_DEVICES` | 0,1,2,3 |
| `NUM_TRAIN_EPOCHS` | 1 |
| `PER_DEVICE_TRAIN_BS` | 4 |
| `LR` | 1e-4 |
| `MAX_LENGTH` | 2048 |
| `INCLUDE_FG` | 0 |
| `FORCE_PREPARE` | 0 |
