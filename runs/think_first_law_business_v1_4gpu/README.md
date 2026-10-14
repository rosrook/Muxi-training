# think_first_law_business_v1_4gpu

Law + Business **过程推理** LoRA 训练（~**1804** 条，**rewrite step-reasoning** 版）。

---

## 这事在干什么（通俗版）

全量 MMLU 上 23k think_first 拉不开；law/business 更吃过程。  
我们先对 law+business 子集做了 LLM 改写：去掉「标难点 + 讲错误选项」的阅卷体，改成：

```text
Let's reason step by step.
First, ...
Second, Knowledge: ... Application: ...
Therefore, the correct answer is C.
```

本 run 在 **Qwen3.5-4B + LoRA** 上吃这批 **过程监督** 数据，做领域小补丁。  
评测协议应与训练一致：**鼓励逐步推理后落到 letter**，不要再强制旧的 `<think>` / `Final answer:`。

---

## 数据格式（训练真正用的）

| | |
|--|--|
| 路径（run 入口） | `/mnt/public/data/zhuxz/training_runs/think_first_law_business_v1_4gpu/sft_dataset.jsonl` |
| 实际指向 | `.../rewrite_step_v1/full1804_20260803T091936Z/sft_dataset.jsonl` |
| n | **1804**（business 1376 / law 428） |
| messages | **仅** `user` + `assistant`（**无 system**） |
| user | 题干 + `A/B/C...` 选项（无 think 指令） |
| assistant | 逐步推理，末句锁 gold |

worker 会过滤非该格式样本；日志写到 `outputs/sft_load_stats.json`。

---

## 相对旧 1.8k think_first 改了什么

| 项 | 旧 | 现在 |
|--|--|--|
| SFT | 原 pack 讲解体 + think 标签 | **rewrite 过程版** |
| worker | 原样读 messages | 去 system、校验开收句、规范化 |
| native 拼序列 | prompt 模板 + 裸拼 assistant | **整轮 chat template**（含结束符，易学会停） |
| cutoff_len | 2048 | **3072**（护过程尾巴） |
| warmup | 0.06 | **0.08** |
| LR / epoch / LoRA | 3e-5 / 3 / r32 | 不变（小数据稳） |

---

## 启动

```bash
cd /mnt/public/code/zhuxz/bench_probe_gen
pip install -e ".[train]"
pip install -U "transformers>=4.52.0"
bash /mnt/public/code/zhuxz/training/runs/think_first_law_business_v1_4gpu/run.sh
```

换卡：

```bash
CUDA_DEVICES=4,5,6,7 MASTER_PORT=29542 \
  bash /mnt/public/code/zhuxz/training/runs/think_first_law_business_v1_4gpu/run.sh
```

产出：

- checkpoints: `.../training_runs/think_first_law_business_v1_4gpu/checkpoints/`
- `outputs/train_summary.json`、`outputs/sft_load_stats.json`

---

## 配置摘要

见本目录 `config.yaml`：

- global batch 16（1×4×4）
- LR 3e-5 · epochs 3 · warmup 0.08 · cutoff **3072**
- LoRA r32/α64 · save_steps 50 · val 10%

约 ~300 optimizer step。

---

## 评测提示

训完后不要用「只出一个字母」的协议与 23k think_first adapter 硬比；应用 **允许 CoT / necessary steps** 再抽答案，或统一 prompt：`reason step by step, end with Therefore, the correct answer is X`。
