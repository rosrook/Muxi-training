# 当前训练、MMLU-Pro 评测与模型部署主链路

更新时间：2026-08-07。

本文只记录目前最新、应继续使用的主链路：

1. Qwen3.5 官方最佳实践风格的 ms-swift LoRA SFT；
2. TIGER-Lab/MMLU-Pro 官方代码改造的 OpenAI API 评测；
3. LoRA 合并、严格校验，以及单卡单实例 vLLM 部署。

## 0. 当前有效路径

```text
基础模型
/mnt/public/model/Qwen3.5-4B

训练代码
/mnt/public/code/zhuxz/training/ms_swift_qwen35/

non-FG LoRA
/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/run_20260806T074147Z/output_Qwen3.5-4B/v0-20260806-154220/checkpoint-1851

non-FG 已合并模型
/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/nonfg_ckpt1851

non-FG+FG LoRA
/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/run_20260806T080723Z/output_Qwen3.5-4B/v0-20260806-160756/checkpoint-2754

non-FG+FG 已合并模型
/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/full_ckpt2754

官方 MMLU-Pro API 评测
/mnt/public/code/zhuxz/MMLUPro_eval_scripts/01_tiger_mmlu_pro/
```

## 1. 最新训练：ms-swift Qwen3.5-4B LoRA SFT

官方参考：

https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html

主入口：

```text
/mnt/public/code/zhuxz/training/ms_swift_qwen35/train_sft_qwen35_4b.sh
```

该脚本已经替代旧的 `training/create_run` / `lora_4gpu` 管线。

### 1.1 环境

普通 GPU 环境：

```bash
bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/install_env.sh
source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate
```

MACA 容器、镜像内为 ms-swift 4.1.3 时：

```bash
cd /mnt/public/code/zhuxz/training/ms_swift_qwen35

export MODEL=/mnt/public/model/Qwen3.5-4B
export DATA_ROOT=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft
export TRAIN_JSONL=$DATA_ROOT/train.jsonl
export VAL_JSONL=$DATA_ROOT/train.val.jsonl
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/run_in_container.sh
```

实际训练镜像：

```text
cr.infini-ai.com/te-c7vnqlzlmvzffc2v/ms-swift:4.1.3-maca-torch2.8-py312
```

不要在 ms-swift 4.1.3 镜像中随意升级到更新版 transformers。当前兼容组合使用
`transformers==5.2.*`、`peft<0.20`；训练脚本还会安装本目录中的
`patch_transformers_for_ms_swift.py` 兼容补丁。

### 1.2 数据准备

ms-swift 输入是标准 messages JSONL：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

仅 non-FG：

```bash
python3 /mnt/public/code/zhuxz/training/ms_swift_qwen35/prepare_dataset.py \
  --out /mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/train.jsonl \
  --val-ratio 0.01
```

non-FG+FG：

```bash
python3 /mnt/public/code/zhuxz/training/ms_swift_qwen35/prepare_dataset.py \
  --include-fg \
  --out /mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/train_with_fg.jsonl \
  --val-ratio 0.01
```

当前落盘数据规模：

- non-FG：29,607 train + 299 val，共 29,906；
- non-FG+FG：44,065 train + 445 val，共 44,510。

### 1.3 启动训练

默认 non-FG：

```bash
source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/train_sft_qwen35_4b.sh
```

指定完整混训数据：

```bash
export TRAIN_JSONL=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/train_with_fg.jsonl
export VAL_JSONL=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/train_with_fg.val.jsonl

bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/train_sft_qwen35_4b.sh
```

关键默认超参数：

- base：`/mnt/public/model/Qwen3.5-4B`
- LoRA：rank 8、alpha 32、dropout 0.05、`all-linear`
- epoch：1
- 每卡 train/eval batch size：4/4
- gradient accumulation：1
- 4 卡全局有效 batch size：`4 × 4 × 1 = 16`
- learning rate：`1e-4`
- scheduler：cosine，warmup ratio 0.05
- max length：2048
- bf16、gradient checkpointing、DeepSpeed ZeRO-2
- 每 100 step eval/save，最多保留 2 个 checkpoint
- `add_non_thinking_prefix=true`
- `loss_scale=ignore_empty_think`
- `group_by_length=true`

以上均可用脚本声明的环境变量覆盖，例如：

```bash
NUM_TRAIN_EPOCHS=2 \
PER_DEVICE_TRAIN_BS=2 \
GRAD_ACCUM=2 \
LR=5e-5 \
MAX_LENGTH=4096 \
bash /mnt/public/code/zhuxz/training/ms_swift_qwen35/train_sft_qwen35_4b.sh
```

每次运行会写：

```text
06_ms_swift_sft/run_<UTC时间>/
├── run_config.json
├── train.log
└── output_Qwen3.5-4B/v*/checkpoint-*/
```

## 2. 最新评测：TIGER-Lab 官方 MMLU-Pro API 评测

上游说明：

```text
/mnt/public/code/zhuxz/MMLUPro_eval_scripts/01_tiger_mmlu_pro/README.md
```

实际评测入口：

```text
/mnt/public/code/zhuxz/MMLUPro_eval_scripts/01_tiger_mmlu_pro/evaluate_from_apiX.py
```

这是 TIGER-Lab/MMLU-Pro 的 universal OpenAI `chat.completions` 多并发评测方式，不是内部
`general_mcq_eval`。

### 2.1 当前启动包装脚本

```text
/mnt/public/code/zhuxz/MMLUPro_eval_scripts/01_tiger_mmlu_pro/scripts/local/run_apiX_nonfg_sft_n64.sh
```

先做 dry-run：

```bash
cd /mnt/public/code/zhuxz/MMLUPro_eval_scripts/01_tiger_mmlu_pro

DRY_RUN=1 \
NUM_WORKERS=128 \
MAX_TOKENS=4096 \
bash scripts/local/run_apiX_nonfg_sft_n64.sh
```

正式后台启动：

```bash
cd /mnt/public/code/zhuxz/MMLUPro_eval_scripts/01_tiger_mmlu_pro

URL=http://if-ddetqh2ufsmgbrmd-service:80/v1 \
MODEL_NAME=qwen35-4b-nonfg-sft \
NUM_WORKERS=128 \
RETRY=2 \
MAX_TOKENS=4096 \
RETRY_WRONG=0 \
SUBJECTS=all \
bash scripts/local/run_apiX_nonfg_sft_n64.sh
```

包装脚本会先请求 `${URL}/models`，再用 `nohup` 后台启动，并写：

```text
<OUT_DIR>/
├── run_config.json
├── logs/launch_<时间>.log
├── logs/launch.pid
├── <subject>_result.json
└── <subject>_summary.json
```

注意：

1. 文件名虽然仍叫 `n64`，最新成功口径及当前脚本默认值是
   `NUM_WORKERS=128, MAX_TOKENS=4096`；正式运行仍建议显式写出。
2. 评测代码实际发送 `temperature=0.1, top_p=0.95`，并使用 streaming；这是当前“官方脚本口径”。部署侧建议的 `temperature=0` 不会自动覆盖它。
3. `MAX_TOKENS=4096` 是每题生成上限；脚本单请求超时为 3600 秒。
   不能在 `--max-model-len=8192` 的服务上把生成上限也设成 8192，因为
   MMLU-Pro few-shot prompt 本身已占上下文，实际会收到 HTTP 400。
4. `RETRY` 用于 max-token/请求失败重试；`RETRY_WRONG=0` 表示不要为了提高分数而重跑答错题。
5. 数据从本地官方 Arrow cache 读取，默认路径：

```text
/root/.cache/huggingface/datasets/TIGER-Lab___mmlu-pro/default/0.0.0/b189ec765aa7ed75c8acfea42df31fdae71f97be
```

该 Arrow cache 包含官方 test 12,032 条及 validation 70 条。当前评测不读取
`/mnt/public/data/zhuxz/benchmarks/mmlu_pro/validation.jsonl`；后者缺少官方
few-shot 所需的 `cot_content`。

运行依赖包括 `openai`、`datasets`、`tiktoken`、`rich` 和 `pyarrow`。

### 2.2 当前已完成基准

最新完整 non-FG 运行：

```text
/mnt/public/data/zhuxz/results/mmlu_pro_official_apiX_20260806T134409Z_qwen35-4b-nonfg-sft/
```

口径：128 workers、max_tokens 4096、retry 2、retry_wrong 0、14 科全量
12,032 题。结果为 7,944/12,032，准确率 66.02%；其中 479 题达到
max-token 上限。该分数使用正则抽取 A–J 字母并与 gold 比较，不使用 LLM judge。

### 2.3 完成性检查

评测覆盖 14 个 subject。一次完整运行应同时满足：

- 14 个 `*_result.json`；
- 14 个 `*_summary.json`；
- 每个 summary 的 `total.corr + total.wrong` 等于该学科官方 test 样本数；
- 日志中没有仍在重试的 worker；
- 不能只根据“目录存在”判定完成。

汇总结果：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/path/to/mmlu_pro_output")
corr = wrong = 0
for path in sorted(root.glob("*_summary.json")):
    total = json.loads(path.read_text())["total"]
    corr += int(total["corr"])
    wrong += int(total["wrong"])
    print(path.stem, f'{100 * total["acc"]:.2f}%', int(total["corr"] + total["wrong"]))
print("subjects", len(list(root.glob("*_summary.json"))))
print("overall", corr / (corr + wrong), corr, wrong)
PY
```

## 3. LoRA 合并为可部署整模

vLLM 的 `MODEL_PATH` 必须指向完整 Hugging Face 模型目录，不能直接指向只有
`adapter_model.safetensors` 的 LoRA checkpoint。

### 3.1 标准导出命令

始终输出到一个新的临时目录，不要直接覆盖当前已验证目录：

```bash
source /mnt/public/code/zhuxz/dev_envs/xuzhou_ms_swift/env/bin/activate

BASE=/mnt/public/model/Qwen3.5-4B
ADAPTER=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/run_20260806T074147Z/output_Qwen3.5-4B/v0-20260806-154220/checkpoint-1851
MERGED=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/nonfg_ckpt1851_candidate

swift export \
  --model "$BASE" \
  --adapters "$ADAPTER" \
  --merge_lora true \
  --output_dir "$MERGED"
```

当前生产目录不是只做了“文件存在”检查。首次导出时发现：

- 使用普通 `AutoModelForCausalLM` 不适配 Qwen3.5 多模态架构；
- 正确 HF 类是 `AutoModelForImageTextToText` / `Qwen3_5ForConditionalGeneration`；
- 默认重新保存会遗漏 15 个冻结的 `mtp.*` 张量；
- 新版 transformers 重新序列化的 config/processor 文件可能与部署镜像不兼容。

因此当前两个已验证目录还完成了：

1. 从 base 原样补回 15 个未训练的 `mtp.*` 张量；
2. 用 base 原件覆盖 config、tokenizer、chat template、preprocessor 等辅助文件；
3. 保留合并后的 `model.safetensors`；
4. 完成数学校验、tokenizer 校验和 GPU 推理冒烟。

重新导出新模型时，必须重复下面的严格校验；不能只看 `swift export` 返回 0。

补充：2026-08-06 当前两个生产目录实际是用
`AutoModelForImageTextToText + PeftModel.merge_and_unload()` 生成，再完成 MTP
恢复和辅助文件对齐；`swift export` 是后续重新导出时的标准入口，不应把两者误记成
同一次操作。

### 3.2 合并结果的最低文件检查

```bash
MERGED=/path/to/candidate

test -s "$MERGED/config.json"
test -s "$MERGED/model.safetensors"
test -s "$MERGED/tokenizer.json"
test -s "$MERGED/tokenizer_config.json"
test -s "$MERGED/chat_template.jinja"
test -s "$MERGED/preprocessor_config.json"
test -s "$MERGED/video_preprocessor_config.json"
```

当前 non-FG 目录约 8.8 GiB，其中 `model.safetensors` 约 8.7 GiB。

### 3.3 严格权重检查

合并验收必须覆盖以下内容：

1. merged 与 base 的权重 key 集合完全一致；
2. merged 中不存在 `lora_A`、`lora_B`、`base_model.*` 残留；
3. 对全部 LoRA 命中模块验证：

```text
merged_W ≈ base_W + (lora_alpha / r) × (B @ A)
         ≈ base_W + 4 × (B @ A)
```

4. 非 LoRA 目标权重与 base 一致；允许的唯一 dtype 变化必须满足
   `merged == base.to(bfloat16)`；
5. `config.json` 架构、vocab size、特殊 token、chat template 与 base 一致；
6. tokenizer 对中英文、数学、代码、特殊 token 的 encode 结果一致；
7. 用真实 GPU 加载并做 deterministic generation 冒烟；
8. 最后再用目标 vLLM 镜像拉起服务并请求 `/v1/models` 和 chat completion。

已将当时临时验证脚本固化为：

```text
/mnt/public/code/zhuxz/training/ms_swift_qwen35/verify_lora_merge.py
```

non-FG 验证命令：

```bash
python3 /mnt/public/code/zhuxz/training/ms_swift_qwen35/verify_lora_merge.py \
  --base /mnt/public/model/Qwen3.5-4B \
  --adapter /mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/run_20260806T074147Z/output_Qwen3.5-4B/v0-20260806-154220/checkpoint-1851 \
  --merged /mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/nonfg_ckpt1851 \
  --tag nonfg
```

只有输出 `VERDICT: PASS` 且进程退出码为 0 才算通过。该脚本要求 base
使用 `model.safetensors.index.json` 分片索引、merged 使用单文件
`model.safetensors`，与当前 Qwen3.5-4B 目录布局一致。

当前两个生产 merged 的已完成验证：

- base/merged 均为 738 个张量；
- 248 个 LoRA 目标模块全部验证；
- 无 LoRA 残留键；
- 490 个非目标张量验证通过；
- tokenizer/chat template 功能等价；
- Metax GPU 推理 `17 × 23` 均正确输出 391；
- 所有辅助文件已与 base 原件逐字节对齐。

因此当前生产 merged 不应被未验证的新导出结果覆盖。

另一个 checkpoint 选择事实：full 训练的 trainer state 曾记录
`checkpoint-2700` 为 best checkpoint，但当前部署采用最后一步
`checkpoint-2754`。比较实验时必须明确写“best”还是“last”，不能只写 full。

## 4. 当前模型加载：单卡单实例 vLLM

目标部署形态是 `1 卡 × 8 个实例`。下面这段是“每个实例”的启动命令；平台复制为
8 个副本。每个副本都只看到本地 GPU 0，并都可在自己的网络空间监听 8000。

```bash
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/nonfg_ckpt1851

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen35-4b-nonfg-sft \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code
```

镜像：

```text
cr.infini-ai.com/te-c7vnqlzlmvzffc2v/vllm-metax:0.21.0-maca.ai3.7.1.106-torch2.8-py310-ubuntu22.04-amd64
```

参数含义：

- `max-num-seqs=1`：实例内部不做多请求 continuous batching；
- 8 路并行来自 8 个独立实例，由评测客户端/Service 在外层调度；
- `temperature` 是请求参数，不是服务启动参数；
- 8192 是 prompt+generation 的总上下文上限；若正式 CoT 超过该长度，再评估改为 16384；
- 不要加 TP=8；当前资源形态不是“一实例八卡”。

### 4.1 启动前检查

```bash
MODEL_PATH=/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/06_ms_swift_sft/deploy_merged/nonfg_ckpt1851

test -s "$MODEL_PATH/config.json"
test -s "$MODEL_PATH/model.safetensors"
ls -lh "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors"
```

如果目录不存在或缺 `config.json`，transformers 可能把绝对路径误当作 Hugging Face
repo id，并报 `HFValidationError` / `Can't load the configuration`。

### 4.2 服务就绪与推理冒烟

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen35-4b-nonfg-sft",
    "messages":[{"role":"user","content":"Compute 17 * 23 and give the final answer."}],
    "temperature":0,
    "max_tokens":256
  }'
```

验收：

- `/health` 为 HTTP 200；
- `/v1/models` 返回 `qwen35-4b-nonfg-sft`；
- completion 为 HTTP 200、非空、无 NaN/乱码，并正确回答 391；
- 服务日志无 missing/unexpected weight、OOM 或 config 解析错误。

full 模型只需替换：

```text
MODEL_PATH=.../deploy_merged/full_ckpt2754
served-model-name=qwen35-4b-full-sft
```

## 5. 统一执行顺序

```text
prepare_dataset.py
  → train_sft_qwen35_4b.sh
  → 选择 checkpoint
  → LoRA 导出到 candidate 目录
  → 权重公式/key/config/tokenizer/GPU 冒烟严格校验
  → 验证通过后才晋升为 deploy_merged/<name>
  → 1 卡 × 8 实例启动 vLLM
  → /health + /v1/models + completion 冒烟
  → evaluate_from_apiX.py 跑官方 MMLU-Pro
  → 检查 14 科 result/summary 与样本数
```
