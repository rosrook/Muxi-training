# Qwen3.5-4B 训练—评测经验总览（2026-06 ~ 2026-08）

> **工作量总账（给老师）**：`/mnt/public/code/zhuxz/bench_probe_gen/docs/WORK_MASTER_LEDGER.md`  
> 本文档保留工程细节、踩坑与超参经验；与总账冲突时以更新的 E8 官方跑次为准。

> **文档性质**：自包含总结。可单独交给另一位 AI / 同事，不依赖外链也能理解全过程。  
> **重点**：训练数据、训练格式、训练管线演进、失败与踩坑、以及评测如何反过来校正训练判断。  
> **基座模型（除非另注）**：`/mnt/public/model/Qwen3.5-4B`，训练方式几乎全程为 **LoRA SFT**（4 卡）。  
> **整理日期**：2026-08-11。

---

## 0. 给读者的速览（先读这 1 页）

### 0.1 一句话主线

6–7 月用自研 HF/PEFT LoRA 把「合成数据 → 训练 → 回测」链路打通，并做了大量**数据量 / 答题格式**消融；8 月初发现**旧评测打分偏松 + 推理协议不一致**，导致此前不少「涨分」不可靠，主对照下约 23k 合成数据相对 Base 几乎无增益（MMLU-Pro necessary_steps@8k 约 **+0.4pp**）。8/6 起切换到 **ms-swift 官方最佳实践式 LoRA** + **TIGER-Lab MMLU-Pro 官方 apiX 评测**；用错题驱动的 non-FG 合成 + step rewrite（约 30k messages）把官方协议下分数从 Base **43.04%** 拉到 **66.02%**——这是当前可写进结论的主结果。随后做的 MCQ 选项标定（calib）在官方协议下为 **64.15%**，略低于 nonFG，说明「修字母偏置 / 扩 10 选项」本身不是免费午餐。

### 0.2 代际对照（记忆锚点）

| 代际 | 时间 | 训练栈 | 数据量级 | 训练侧在干什么 | 今天该怎么用这些结论 |
|------|------|--------|----------|----------------|----------------------|
| Gen1 | 06/11–06/30 | 自研 PEFT LoRA | 1k / 3k / test3k | 打通远程训练与落盘 | 工程历史；数字仅作链路验证 |
| Gen2 | 07/09–07/16 | 自研 PEFT LoRA | 9k→18k + CoT 衍生 | mixed_10k 闭环；短答 vs CoT | **格式对能力分布的影响**仍有效 |
| Gen3 | 07/25–07/31 | 自研 PEFT LoRA | 池~23.6k；earlytrain~11k；英译格式消融 | generator_v5 + 多种答题模板 | 训练完成度高；**多数表观涨分需重读** |
| 校准窗 | 08/02–08/05 | 复测为主 | 仍用 23k think-first 等 | 统一协议重测；law/biz 专项 | **负结论**：大规模合成未稳定吃到红利 |
| Gen4 | 08/06–08/10 | **ms-swift LoRA** | nonFG~30k；nonFG+FG~44k；calib~30k | 换训练栈 + 官方评测 + 合并校验 | **当前主线与主结论** |

### 0.3 当前主结论数字（官方 apiX）

| 模型 | max_tokens | Acc | 触顶(MaxTokensExceeded) | 备注 |
|------|------------|-----|-------------------------|------|
| Base（未训） | 4096 | **43.04%**（5179/12032） | **53.17%** | 大量回答被截断 |
| nonFG SFT（ckpt-1851 last） | 4096 | **66.02%**（7944/12032） | **3.98%** | 当前主部署 |
| calib SFT（ckpt-1833 last） | 8192 | **64.15%**（7719/12032） | **1.84%** | 数据与预算同时变了，勿直接说「预算更大更好」 |

打分方式：正则抽 A–J 字母 vs gold；**不用 LLM judge**；14 科全量 12,032 题；temperature=0.1，top_p=0.95（官方脚本口径）。

### 0.4 三条不可再踩的红线

1. **不要用旧 MCQ rule 松散字母匹配当结论**——相对严格字母可虚高约 **16–20pp**。  
2. **训练格式必须对齐评测契约**——短答、`####` CoT、`<think>` think-first、necessary_steps 收尾，换协议分数量级会变，禁止横比绝对值。  
3. **Qwen3.5 LoRA 合并不是 `swift export` 返回 0 就完事**——必须补回冻结的 `mtp.*`、对齐 config/tokenizer，并用权重公式校验 PASS 后才能部署。

---

## 1. 仓库与落盘结构（训练视角）

整条业务可理解为三条代码仓 + 一块数据盘 + 一个训练中枢：

| 角色 | 路径 | 训练相关用途 |
|------|------|--------------|
| 早期完整仓 | `code/zhuxz/bench_probe_gen` | 最初 pipeline、早期 `scripts/training`、大量 personal_scaffold 实验脚本 |
| 主工作仓 | `code/zhuxz/bench_probe_gen_clean` | generator_v5、step rewrite、mcq_calibration_rewrite、评测 scaffold、周报/进度文档 |
| 难点发现仓 | `code/zhuxz/bench_probe_discovery` | 从 clean 切出的加速 discovery；后置「训练部署评测运行指南」与官方评测实验记录 |
| 训练中枢（现行） | `code/zhuxz/training/` | `ms_swift_qwen35/`（主）、`llama32_sft/`（对照）、`runs/<RUN_ID>/`（自包含旧 run 包）、`REGISTRY.md` |
| 数据盘 | `data/zhuxz/` | `checkpoints/`、`training_runs/`、`results/`、`runtime_logs/training/` |

**约定（8/3 起强化）**：每个训练 job 尽量自包含——配置与 launcher 复制进 `code/.../training/runs/<RUN_ID>/`，产物进 `data/.../training_runs/<RUN_ID>/`（旧权重常以软链指向 `checkpoints/`）。**不要**为单次实验去改 `personal_scaffold` 里的共享脚本；改 run 目录内的拷贝。

**现行 Gen4 入口**（替代旧 `create_run` / `lora_4gpu` 作为主路径）：

- 训练脚本：`code/zhuxz/training/ms_swift_qwen35/train_sft_qwen35_4b.sh`（默认 `MAX_LENGTH=2048`）  
- 长上下文版：`train_sft_qwen35_4b_len8192.sh` / `run_in_container_len8192.sh`  
- 数据准备：`prepare_dataset.py` → 标准 `messages` JSONL  
- 合并与校验：`merge_lora_qwen35.py`、`verify_lora_merge.py`

---

## 2. 训练环境与通用超参经验

### 2.1 硬件 / 镜像现实

训练发生在 **Metax / MACA** 环境，不是普通 NVIDIA CUDA 桌面环境。关键约束：

- **禁止**在容器里 `pip install` 覆盖官方/镜像自带的 `torch`（会直接废掉 MACA 轮子）。  
- 自研 LoRA 阶段曾用 megatron/maca 类镜像；Gen4 主用：  
  `cr.infini-ai.com/te-c7vnqlzlmvzffc2v/ms-swift:4.1.3-maca-torch2.8-py312`  
- 注意力：Metax 上更稳的是 `attn_implementation=eager`（曾踩过 SDPA/`enable_gqa` 问题）。  
- ms-swift 4.1.3 兼容组合实践：**transformers==5.2.\***、**peft\<0.20**，并安装目录内 `patch_transformers_for_ms_swift.py`（处理新版 transformers 与 ms-swift 的 API 裂口，例如 `create_repo`）。不要为「追上最新 transformers」盲目升级。

### 2.2 Gen1–Gen3 自研 LoRA 的典型形态

- 基座只读 + PEFT adapter；多卡 `torchrun`。  
- 数据多为 ShareGPT / messages；早期短答、中期 `####` 答案标记、后期英译 think-first 等。  
- run_id 与 checkpoint / results / runtime_logs 三方对齐（`paths.py` 写死根目录）。  
- 大量实验脚本落在 `personal_scaffold/runtime_error_case_backtest/`（yaml + 4gpu shell）。

### 2.3 Gen4 ms-swift 默认超参（nonFG 主跑）

这些是当前应继续沿用的默认，而不是「曾经试过」的值：

| 项 | 值 |
|----|-----|
| 基座 | `/mnt/public/model/Qwen3.5-4B` |
| LoRA | rank 8，alpha 32，dropout 0.05，`all-linear` |
| epochs | 1 |
| 每卡 train/eval batch | 4 / 4 |
| grad accum | 1 |
| 全局有效 batch | `4 × 4 × 1 = 16`（4 卡） |
| lr | `1e-4`，cosine，warmup ratio 0.05 |
| max length | **2048**（calib 用 **8192**） |
| 精度 / 并行 | bf16，gradient checkpointing，DeepSpeed ZeRO-2 |
| Qwen3.5 专项 | `add_non_thinking_prefix=true`，`loss_scale=ignore_empty_think`，`group_by_length=true` |
| 保存 | 约每 100 step eval/save，最多保留 2 个 ckpt |

数据格式必须是：

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### 2.4 部署侧与训练强相关的事实

- vLLM 的 `MODEL_PATH` 必须是**合并后的完整 HF 目录**，不能直接指只有 `adapter_model.safetensors` 的 LoRA 目录。  
- 目标形态：**1 卡 × 8 独立实例**（实例内 `max-num-seqs=1`），不是单实例 TP=8。  
- nonFG / Base 常用 `max-model-len=8192`；calib 因生成预算 8192 + few-shot，服务侧用 **16384**。  
- **不能**在 `max-model-len=8192` 的服务上把评测 `MAX_TOKENS` 也设成 8192：MMLU-Pro few-shot prompt 已占上下文，会 HTTP 400。

---

## 3. Gen1：打通链路（2026-06）

### 3.1 目标

证明：合成题 → 打包 SFT → 多卡 LoRA → checkpoint 落盘 →（可选）挂 adapter 部署，这条链在 Metax 上能跑通。

### 3.2 实际训练

| 时间(约) | run_id | 数据量 | 格式 | 状态 |
|----------|--------|--------|------|------|
| 06-12 | `v1_40_*` 多次 | 极小/冒烟（曾有 42 条级） | smoke | 大量 partial；可归档 |
| 06-21 | `v1_remote_1000_02` | 总约 1.2k（训 ~1092） | 短答（字母/短答案） | 完成 |
| 06-25 | `v1_remote_3000_01` | 总约 4.4k（训 ~3998） | 短答 | 完成 |
| 06-30 | `test3k_run` | 总约 3.0k（训 ~2691） | 短答（v2 generator pilot） | 完成 |

### 3.3 评测印象（不要当能力终局）

- 在「原错题修复集」上，从 Base 接近 0 拉到可观 fix rate（1k 约 15%，3k 约 34% 量级）——说明**监督信号能写进权重**。  
- 在小 MMLU val / 子集上常常接近打平或小波动——说明**还没证明泛化红利**。  
- 部署曾踩 **peft 与 torchao 版本不兼容**：加载 LoRA 时 `ImportError: Found an incompatible version of torchao... only versions above 0.16.0 are supported`（见早期错误日志）。教训：训练环境能训完 ≠ 推理容器能加载同一 adapter。

### 3.4 Gen1 训练教训

1. 先把 run_id、checkpoint、日志三件套对齐，否则后面无法复盘。  
2. 冒烟 run（`v1_40_*`）不要混进「正式代际表」。  
3. adapter 部署环境的 peft/torchao/transformers 要单独验收。

---

## 4. Gen2：mixed_10k 闭环与「格式杀伤力」（2026-07 中旬）

### 4.1 目标

在更完整的数据闭环上回答：  
（1）加数据量是否抬分？  
（2）短答-only 会不会牺牲某些能力？  
（3）CoT / 续训能否补救？

生成侧大致到 **~18k** 量级；训练上最关键的几次：

| 时间 | run_id | 数据量 | 训练格式 | 意图 |
|------|--------|--------|----------|------|
| 07-13 | `mixed10k_backtest_20260713T041155Z` | ~9.1k（训 ~8228） | 短答 only（无 rationale） | 9k 回测 |
| 07-14 | `mixed10k_fulltrain_20260714T071006Z` | ~18.0k（训 ~16193） | 短答 only | 全量短答 |
| 07-14 | `mixed10k_gsm8k_cot_20260714T114313Z` | ~1.8k（训 ~1644） | CoT：rationale + `####` 数字 | 专救 GSM8K |
| 07-15 | `mixed10k_stage2_mmlu_replay_20260715T030751Z` | ~20.2k（训 ~18188） | stage2：MMLU 短 rationale+`####`字母 + GSM CoT replay | 续训混合 |
| 07-16 | `fulltrain_vb_4gpu_20260716T064800Z` | ~15.5k（训 ~13987） | letter-only vb | 极简终答 |

### 4.2 关键评测（同源全量约 13351 = MMLU+GSM8K）

| 模型 | overall | MMLU | GSM8K | 解读 |
|------|---------|------|-------|------|
| Base | 64.41% | 64.35% | 64.90% | 基线 |
| 9k short | 63.02% | 66.91% | **27.52%** | MMLU 略升，GSM 崩 |
| 18k short | **65.25%** | **69.39%** | 27.45% | 量上来 overall 略超 Base，但 GSM 仍崩 |

其它可记数字：

- GSM-CoT 专用训后，GSM8K **83.93%**（相对 Base 约 +19pp）——**格式对了就能把被短答打死的能力拉回来**。  
- stage2 在 MMLU `####` 协议上曾到约 **75.11%**；vb letter-only 约 **69.08%**（协议不同，勿与 short 表横比）。  
- OOD：18k 在 SuperGPQA 约 +1.35pp，AGIEval 接近打平。

### 4.3 Gen2 对后续训练的核心经验

1. **「只优化终答字母/短答」会系统性伤害需要中间推理的任务**（GSM8K 是最狠的证人）。  
2. **数据量从 9k→18k 有帮助，但救不了错误的答案契约**。  
3. 需要「能力保持」时，应用 **CoT / 逐步推理格式** 做专项或混训，而不是指望短答规模魔法。  
4. Gen2 是整段历程里**训练—评测闭环最完整**的一代，适合当作「格式消融」教科书，而不是当作最终 SOTA。

---

## 5. Gen3：generator_v5 大规模合成与格式大爆炸（2026-07 末）

### 5.1 数据从哪来

主生成池（量级）：

- `generated.jsonl` ≈ **23,654**  
- earlytrain 冻结可用约 **11,295**  
- 英译 `generated.en.jsonl` ≈ **23,468**

上游逻辑：错题 / 难点发现（discovery）→ generator_v5 合成 →（可选）英译 → 按模板打包 SFT。

### 5.2 训练清单（一行一次）

| 时间 | 数据量 | 训练格式 | run_id（缩写） | 状态 |
|------|--------|----------|----------------|------|
| 07-28 | 11.3k（训 ~10163） | 中文 CoT：rationale + `####` | `earlytrain_v5_cot_…` | 完成 |
| 07-28 | 9k / 6k | 同上，子采样 scale | `earlytrain_v5_scale_9k/6k_…` | 完成，**未统一 OOD 评** |
| 07-29 | 3k / 1k | 同上 | `…_3k/1k_…` | 完成，**未统一 OOD 评** |
| 07-29 | ~23.5k | 英译 CoT `####` | `en_translated_cot_4gpu_…` | 完成；首轮评测异常偏低 |
| 07-29 | ~23.4k | 英译 letter-only | `en_translated_letter_only_…` | 完成；评测仍显著弱于主线预期 |
| 07-31 | ~23.4k | 英译 answer-first（先答后 think） | `en_translated_answer_first_…` | **启动失败 ×2** |
| 07-31 | 训 21031 + eval 2336 | 英译 think-first + `Final answer:` | `en_translated_think_first_final_…` | 完成：global_step=2630，train_loss≈0.912，4 卡；ckpt 含 2600/2630/final |
| （打包未训/未完） | ~23.5k | `cot_budget_v2` 限长 rationale | — | 有数据包装，主线未收口 |

另：08-03 左右对 law+business 子集做了 **1804** 条 step-reasoning 重写训练（`think_first_law_business_v1_4gpu`，completed）——见第 6 章校准窗。

### 5.3 曾被认为「主结果」的 earlytrain

在**旧口径**下：

| 评测 | 口径 | earlytrain 11k | Base | Δ |
|------|------|----------------|------|---|
| MMLU-Pro | **strict_letter** | **54.79%** | 45.89% | **+8.9pp** |
| MMLU-Pro | rule（偏松） | 74.90% | （更高虚高） | **勿单独引用** |
| 覆盖原错 ~2935 | `####` CoT | ~47.6% | — | 说明对训练分布内错题有修复 |

当时判断：Gen3 earlytrain 是主胜利。  
**后来校准证明：这个胜利高度依赖打分器与协议；不能无条件外推为「23k 合成已经成功」。**

### 5.4 英译格式消融学到什么

意图：同一英译池，只改 assistant 模板，看哪种答题格式更好训。

实际发生：

1. **answer_first** 连续两次 launch 失败——说明「新模板 + 多卡 launcher」本身就要先 dry-run。  
2. **EN CoT / letter-only 首轮评测数字异常低**（例如曾见 MMLU ~39%、SuperGPQA 个位数到 20% 段）——事后归因高度怀疑 **max_tokens 过小（如 1024）、协议与训练模板不一致、未跑满** 等，属于「无效评测」，不能用来否定英译数据。  
3. think-first 模板（先 `<think>…</think>`，再 `Final answer: X`）成为后续 23k 全量复测的主模型之一。

**训练格式示例（think-first）**：

```text
system: 先在 <think>...</think> 内推理；</think> 后单独一行 Final answer: <LETTER>
user: 题干 + Options A.. 
assistant:
  <think>
  ...推理...
  </think>
  Final answer: B
```

### 5.5 Gen3 训练教训

1. **先做规模 scale（1/3/6/9k）却不做统一评测 = 半废弃实验**；算力花了但决策用不上。  
2. 格式消融必须 **同一评测协议、同一 max_tokens、同一抽答案规则**；否则表格全是噪声。  
3. 「生成池 23k」≠「有效监督 23k」：英译、过滤、模板转换、val 切分都会掉量。  
4. launcher 失败要记进 registry（answer_first），避免后人当「训过」。

---

## 6. 校准窗：推翻「表观成功」，重定训练价值（2026-08-02 ~ 08-05）

这是整段历程里**对训练判断冲击最大**的窗口。重点不是又训了多少新权重，而是：**用更诚实的评测，重新定价 Gen3 数据与格式。**

### 6.1 发现 1：旧 MCQ rule 虚高

周报级别结论：

- Base 曾报 rule ~64.35%，严格字母仅 ~**45.89%**  
- earlytrain CoT 曾报 ~74.9%，严格仅 ~**54.82%**  
- 虚高幅度约 **16–20pp**

因此：任何「rule 涨了 10 个点」若未伴随 strict / 官方协议，**不能指导下一轮该不该加数据**。

### 6.2 发现 2：协议决定分数量级

同一 23k think-first LoRA vs Base：

**协议 A — 强制 think，总预算 ≈1024；严格读 Final answer + GPT judge（多轮一致）**

| 模型 | strict | GPT judge |
|------|--------|-----------|
| Base | ~29.6% | ~57.8% |
| 23k | ~29.4% | ~57.8% |

附加：几乎整集 `hit_think_budget_rate ≈ 1.0`（思考配额打满）。  
解读：分数测的是「预算锁死 + 收尾格式」，不是能力崩盘；**23k 与 Base 打平**。

**协议 B — necessary_steps @ max_tokens=8192（主对照）**

| 模型 | MMLU-Pro full |
|------|---------------|
| Base | **70.36%**（8466/12032） |
| 23k | **70.80%**（8519/12032） |

Δ = **+0.44pp**。  
这是校准后的**主负结论**：在公平、偏日常答题的契约下，这批全量 think-first 合成数据**没有带来可用的稳定红利**。

### 6.3 发现 3：外推与专项也不支持「已成功」

全量 23k 在专业集（同 nec@8k）：

| 集 | Base | 23k | Δ |
|----|------|-----|---|
| MedQA | 82.33% | 83.82% | +1.49pp |
| SciBench | 37.76% | 38.79% | +1.03pp |
| LegalBench | 11.60% | 12.38% | +0.78pp |

方向弱正，幅度约 1pp，不能当主证据。

**law + business 专项**：从 23k 筛约 **1804** 条（business~1376 / law~428），并做 step rewrite（把长 think-first 改成 `Let's reason step by step.` … `Therefore, the answer is X`）。在 MMLU-Pro law+business 子集 1890 题 nec@8k 上：

| 设置 | Acc |
|------|-----|
| law/biz step-LoRA | ~52.01% |
| Base 两波 | ~52.22% / 51.27% |
| law/biz LoRA 两波 | ~51.59% / 52.22% |

与 Base **落在同一方差带**。多波交叉还看到约 **12%** 题对错翻转——提示输出不稳（截断/长推理循环）会吞噬小幅训练增益。

### 6.4 归因（指导下一轮训练该改什么）

当时归纳为三类，对后续 Gen4 直接有用：

1. **评测 + 模型回答倾向**  
   Qwen 爱长推理、自检循环；训练数据若没把收尾压住，严格字母分与真实能力脱节。触顶 max_tokens 是不稳来源之一。

2. **合成策略错位**  
   为降低泄漏常用「旁敲侧击」语料；样本多为**单难点**，而真实错题常是**多机制叠加**。在 ~23k 量级下，对原错题结构纠错拉力不足。

3. **难点链路损耗**  
   提取出虚难点；检索不到语料就丢难点 → 有效监督变稀。只加量重复同一管线，难换稳定收益。

### 6.5 校准窗对训练决策的含义

- **停止**把 earlytrain 的 +8.9pp strict 当「合成策略已验证成功」。  
- **停止**在未改合成/契约的前提下，对同构 23k 再做同质扩量训练。  
- **开始**两条新动作：（a）换更稳的训练栈与官方评测；（b）改写数据契约（step rewrite、选项标定）后再训。

---

## 7. Gen4：ms-swift 主线与数据改写（2026-08-06 ~ 08-10）

### 7.1 为什么换训练栈

自研 LoRA 能跑，但：

- 与 Qwen3.5 官方最佳实践（non-thinking 前缀、empty think loss、group_by_length 等）不一致；  
- 合并/部署踩坑成本高；  
- 需要把「训练成功」和「可复现官方基准」绑在同一套工具链上。

于是主入口改为 ms-swift LoRA SFT（见第 2.3 节超参）。

### 7.2 数据主链：错题 → discovery → generator_v5 dual → step rewrite → SFT

根 run（概念上）：

`data/zhuxz/results/mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/`

要点：

1. **错题池**：Base 在相关评测上的 wrong → `02_wrong/wrong_pool.jsonl`  
2. **Discovery**：compass_gen_v2 / compass_exec_v2 / gene_extract_v3  
3. **生成**：generator_v5「MMLU dual simple」  
   - dual：FG knowledge 与 expression/item 物化并行  
   - simple：简化 call345、限检索  
   - 题型 mix：MCQ / T/F / short_answer / matching / …  
   - 注意：dual 会抽 FG knowledge，但 **nonFG 训练集最终不并入 FG-only 段**  
4. **step rewrite（nonFG）**：教师 gpt-5.5；保留题目与金标，改写 assistant 为分步推理；accept≈0.9987；产出 **29,906** 行  
5. **prepare_dataset**：`train.jsonl` **29,607** + `train.val.jsonl` **299**（`include_fg=false`）

**nonFG+FG 混训包**（另一条）：约 **44,065** train + **445** val，对应 full 训练 `run_20260806T080723Z`（ckpt-2754）。当前主部署**不是**这条，而是 nonFG-only。

### 7.3 训练 run（Gen4）

| 时间 | run | 数据 | max_length | 选用 checkpoint | 角色 |
|------|-----|------|------------|-----------------|------|
| 08-06 | `run_20260806T074147Z` | nonFG ~29.9k | 2048 | **last `checkpoint-1851`** | **主线** |
| 08-06 | `run_20260806T080723Z` | nonFG+FG ~44.5k | 2048 | last `checkpoint-2754`（trainer 曾记 best=2700） | 对照/非主部署 |
| 08-09 | `run_20260809T113846Z` | calib ~29.6k | **8192** | **last `checkpoint-1833`** | 选项标定实验 |

**checkpoint 语义教训**：必须写清 **last vs best**。full 跑曾出现 best≠last；比较实验时不能只写「full」。

### 7.4 官方评测结果（再强调一遍）

| 模型 | Acc | 触顶率 | 训练侧解释 |
|------|-----|--------|------------|
| Base@4096 | 43.04% | 53.17% | 未训；极高触顶 |
| nonFG@4096 | **66.02%** | 3.98% | step-rewrite 后的 messages SFT；触顶骤降与分数同向改善 |
| calib@8192 | 64.15% | 1.84% | 在 nonFG 语料上做选项标定后再训；字母更均匀、10 选项大增 |

相对 Base，nonFG **+22.98pp**。这与校准窗里「23k think-first +0.4pp」不矛盾——**数据工序、训练栈、评测协议三者都变了**。

### 7.5 calib：为了修训练数据缺陷而做的第二次训练

#### 7.5.1 nonFG 语料被确认的训练缺陷

1. 金标字母严重偏 **A–D**（E–J 近 0）→ 模型容易学到「不会就选 A」。  
2. **10 选项题极少**（改写前约 4 条）→ 与 MMLU-Pro 10 选一评测分布错位。  
3. 推理较少按**选项内容**消元；部分 reasoning 与最终字母绑定脆弱。

#### 7.5.2 标定改写策略（v1，实际训练用的版本）

对 nonFG 的 29,906 行做分 tier：

| Tier | 量级 | 做什么 | 是否 LLM |
|------|------|--------|----------|
| T0 | ~10 | 已有 ≥8 选项 → 置换 | 否 |
| T1 | ~1.6k | 现有选项内置换 | 否 |
| T2A | ~5k 目标 | 扩到 10 选项，不动原推理 | 是（gpt-5.5） |
| T2B | ~4k 目标 | 开放题建成 10 选一 | 是 |
| T3 | ~2k 目标 | 重写 stem+选项+消元推理 | 是 |
| KEEP | ~17k | 原样保留 | 否 |

硬约束摘要：

- 置换不改选项文本；若推理用「第 K 项/字母」指称则避免危险置换。  
- 生成的 reasoning **按选项内容**指称，不靠字母（因为字母在全局 assemble 时才分配）。  
- 泄漏扫描丢弃与 MMLU-Pro 种子过像的样本（本 run 丢 **284** 条）。  
- 全库金标字母在 merge 时调度到近似均匀 A–J（max 偏离约 2.3%）。

最终：`sft_dataset.jsonl` **29,621** 行；10 选项题升至 **10,713**；长度 p99~1050、max~3518 token → **必须 `MAX_LENGTH≥8192`**（2048 会截断）。

#### 7.5.3 calib 训练结果怎么读

Acc 64.15% **低于** nonFG 66.02%，但：

- 评测预算不同（8192 vs 4096）；  
- 数据分布（字母先验、10 选项比例）变了；  
- v1 追求**近似均匀 A–J**，可能伤害真实 MMLU-Pro 的 A–D 边际（因此文档里已动机化 v2：改为贴真实字母边际，而不是均匀）。

**训练启示**：修数据偏置是必要的，但「均匀字母」是建模选择不是真理；要以**目标 benchmark 的真实先验**为准，并用同一官方协议对比。

### 7.6 LoRA 合并：Gen4 最值钱的工程课

Qwen3.5-4B 实际是多模态架构权重布局。合并时踩过：

1. 普通 `AutoModelForCausalLM` 不适配；应用 `AutoModelForImageTextToText` / `Qwen3_5ForConditionalGeneration`。  
2. 默认重新保存会**漏掉约 15 个冻结 `mtp.*` 张量**。  
3. 新版 transformers 重序列化的 config/processor 可能与部署镜像不兼容 → 应用 base **原件覆盖**辅助文件，只保留合并后的 `model.safetensors`。  
4. 验收必须包括：  
   - merged 与 base 的 key 集合一致；无 `lora_*` 残留；  
   - 对全部 LoRA 模块验证 `merged_W ≈ base_W + (alpha/r) * (B@A)`（本设置 alpha/r=4）；  
   - tokenizer / chat template 行为一致；  
   - GPU 冒烟（如 `17*23 → 391`）；  
   - 目标 vLLM 镜像真正拉起。

生产 merged（nonFG）约 8.8 GiB；验证过 738 张量、248 个 LoRA 模块等。  
**规则：未 `VERDICT: PASS` 的 candidate 不得覆盖生产目录。**

### 7.7 Gen4 为什么「训得动」而 Gen3 校准后「几乎没动」？

不宜简单说「ms-swift 魔法」。更可检验的差异是：

1. **数据工序**：错题驱动 + dual 生成 + **高接受率 step rewrite**，把 assistant 推到「分步推理再给答案」，比单纯 think-first 模板堆量更对齐官方 CoT few-shot 评测。  
2. **评测**：官方 apiX + 固定抽字母，结束了 rule 虚高。  
3. **训练栈**：Qwen3.5 官方推荐的 non-thinking / empty-think loss 等，减少「空 think / 乱 think」与 loss 信号打架。  
4. **触顶率**：Base 53% → nonFG 4%，说明模型学会在预算内收尾——这本身就是可训行为，不只是「知识灌进权重」。

仍未证明的问题：FG 知识段是否值得混入（full 44k 未成主部署）；calib v2 是否超过 nonFG；Llama 对照是否减弱「长推理纠缠」。

---

## 8. 完整训练表（6–8 月，一行一次）

> 数据量为 SFT 打包量级（含 val 时按当时记录）；时间为北京时间附近。冒烟 `v1_40_*` 不逐条展开。

| 时间 | 代 | 数据量 | 训练格式 / 栈 | run | 状态 |
|------|----|--------|----------------|-----|------|
| 06-21 | 1 | 1.2k | 短答 · 自研 LoRA | `v1_remote_1000_02` | 完成 |
| 06-25 | 1 | 4.4k | 短答 · 自研 LoRA | `v1_remote_3000_01` | 完成 |
| 06-30 | 1→2 | 3.0k | 短答 v2 pilot | `test3k_run` | 完成 |
| 07-13 | 2 | 9.1k | 短答 only | `mixed10k_backtest_…` | 完成 |
| 07-14 | 2 | 18.0k | 短答 only | `mixed10k_fulltrain_…` | 完成 |
| 07-14 | 2 | 1.8k | GSM CoT `####` | `mixed10k_gsm8k_cot_…` | 完成 |
| 07-15 | 2 | 20.2k | stage2 MMLU+GSM replay | `mixed10k_stage2_…` | 完成 |
| 07-16 | 2 | 15.5k | letter-only vb | `fulltrain_vb_4gpu_…` | 完成 |
| 07-28 | 3 | 11.3k | 中文 CoT `####` | `earlytrain_v5_cot_…` | 完成 |
| 07-28/29 | 3 | 1/3/6/9k | 同上 scale | `earlytrain_v5_scale_*` | 完成·未统一评 |
| 07-29 | 3 | ~23.5k | EN CoT | `en_translated_cot_…` | 完成·评异常 |
| 07-29 | 3 | ~23.4k | EN letter-only | `en_translated_letter_only_…` | 完成·评偏弱 |
| 07-31 | 3 | ~23.4k | EN answer_first | `en_translated_answer_first_*` | **失败×2** |
| 07-31 | 3 | ~23.4k | EN think_first | `en_translated_think_first_…` | 完成 |
| 08-03 | 3 | 1.8k | law/biz step rewrite | `think_first_law_business_v1` | 完成·无增益 |
| 08-06 | 4 | ~29.9k | ms-swift messages nonFG | `run_20260806T074147Z` | **主线完成** |
| 08-06 | 4 | ~44.5k | ms-swift nonFG+FG | `run_20260806T080723Z` | 完成·非主部署 |
| 08-09 | 4 | ~29.6k | ms-swift calib · len8192 | `run_20260809T113846Z` | 完成 |

---

## 9. 踩坑全集（按主题，可当 checklist）

### 9.1 评测误导训练

| 坑 | 现象 | 正确做法 |
|----|------|----------|
| rule 松散打分 | 虚高 16–20pp | strict_letter 或官方 apiX 抽字母 |
| 协议混用 | think1k~29% vs nec8k~70% | 表格必须带协议名；禁止横比绝对值 |
| 首轮英译评测 | 分数异常低 | 先查 max_tokens、是否跑满、模板是否匹配 |
| 未跑满 / discarded | 目录在但无效 | 查 14 科 summary、样本数、日志；`_discarded_*` 勿引用 |
| 多波翻转 ~12% | 小增益被噪声吞 | 固定种子/预算；看触顶率；多波再下结论 |

### 9.2 数据与格式

| 坑 | 现象 | 正确做法 |
|----|------|----------|
| 短答 only | GSM 从 ~65% 砸到 ~27% | 推理型任务保留 CoT / 逐步格式 |
| 字母偏 A | 模型投机选 A | 统计金标边际；标定或重采样 |
| 几乎无 10 选项 | 与 MMLU-Pro 错位 | 扩选项 tier（T2A/T2B） |
| 旁敲侧击 + 单难点 | 原错题修不动 | 改合成对齐原错结构；或错题重生 |
| 检索丢难点 | 有效监督变稀 | 修 retrieval；统计入库率再扩量 |
| 均匀字母先验 | calib 未必涨点 | 对齐真实 benchmark 字母边际（v2 方向） |
| 长样本 + 短 max_length | 静默截断 | calib 用 8192；先做长度分位检查 |

### 9.3 训练工程

| 坑 | 现象 | 正确做法 |
|----|------|----------|
| 覆盖 torch | MACA 环境坏掉 | 只装 peft/transformers 等，不动 torch |
| 乱升 transformers | ms-swift 裂开 | 钉 5.2.\* + patch；容器内固化 |
| SDPA/GQA | 训练不稳 | `attn_implementation=eager` |
| answer_first launch | 失败×2 | 新模板先单卡 dry-run |
| 改共享 scaffold | 实验互相污染 | 只改 `training/runs/<id>/` 拷贝 |
| scale 只训不评 | 无法决策 | 要么评，要么标成「仅损失曲线」 |
| last vs best | 部署与论文口径打架 | 记录里写死选用哪一个 |
| HF offline | 看似挂起在拉模型 | `HF_HUB_OFFLINE=1` 等；本地 MODEL 路径 |

### 9.4 合并与部署

| 坑 | 现象 | 正确做法 |
|----|------|----------|
| 漏 mtp.* | 权重不完整 | 从 base 补回冻结张量 |
| 重写 config | vLLM/transformers 拒载 | 用 base 原件覆盖辅助文件 |
| 直接指 adapter 目录 | 配置当 repo id / 加载失败 | 先 merge 成整模 |
| peft/torchao 版本 | 推理容器 ImportError | 推理环境单独验收 |
| MAX_TOKENS=模型长 | HTTP 400 | 给 few-shot 留上下文；calib 服务用 16384 |
| 未 PASS 覆盖生产 | 静默坏模型 | candidate 目录 + verify_lora_merge |

---

## 10. 「什么有效 / 什么无效」决策表（训练视角）

| 做法 | 有效性 | 依据 |
|------|--------|------|
| 短答规模从 9k→18k | 部分有效 | MMLU 升、GSM 仍崩 |
| GSM 专项 CoT | 有效 | GSM → 83.9% |
| earlytrain 11k + 旧 strict | 条件有效 | +8.9pp，但打分/协议受限 |
| 23k think-first 同质扩量 | **无效（主对照）** | nec@8k +0.4pp |
| law/biz 1.8k 缩圈 + step 改写 | **无效** | 与 Base 方差重叠 |
| 1/3/6/9k scale 只训 | **决策无效** | 无统一评 |
| answer_first 模板 | 工程失败 | launch×2 |
| EN 格式消融首轮评测 | 评测无效 | 异常低分 |
| ms-swift + nonFG step-rewrite ~30k | **当前最有效** | 官方 43%→66% |
| calib v1 均匀字母 | 有工程价值、分数未超 nonFG | 64.15% vs 66.02% |
| nonFG+FG 44k | 训完但非主结论 | 未取代 nonFG 部署 |

---

## 11. 若下一位 AI 要接手：建议动作顺序

1. **默认复现主线**：nonFG `checkpoint-1851` 合并模型 + 官方 apiX@4096，确认仍接近 66%。  
2. **新训练一律走** `training/ms_swift_qwen35/`，不要回到自研 lora_4gpu，除非做对照实验。  
3. **改数据先于改 lr**：优先检查金标字母边际、选项数、长度分位、训练模板与评测收尾是否一致。  
4. **任何声称涨点的实验**必须带：协议名、max_tokens、触顶率、last/best、题量。  
5. **不要**在未改合成策略的情况下重复「再堆 20k 同构 think-first」。  
6. Llama-3.2 对照与 calib v2（贴真实字母边际）是已铺开但未在本总结中给出终局分数的方向——接手时先查是否已有新的 apiX 目录。

---

## 12. 叙事收束

六到八月的训练史，不是一条平滑涨分曲线，而是三次认知切换：

1. **Gen1–2**：证明链路与「格式 > 蛮力加数据」。  
2. **Gen3→校准窗**：证明「能训完 + 表观涨分」仍可能几乎没有真实泛化；评测纪律决定训练决策质量。  
3. **Gen4**：在更对齐的数据契约（step rewrite）+ 官方训练栈 + 官方评测下，重新得到 **+23pp 级**的可引用增益；同时用 calib 提醒——修偏置的方式本身也是一种归纳偏置，必须用同一标尺回测。

**当前训练立场**：以 ms-swift nonFG 主线为默认；以官方 MMLU-Pro apiX 为默认尺子；以「数据契约与评测契约对齐」为第一超参；其余学习率与 batch 的微调，优先级低于数据与协议。

---

*本文数字来自 2026-06 至 2026-08 期间落盘的 train/eval 记录与进度整理；若与某一目录内更新结果冲突，以更新的官方 apiX 跑次与 `training/REGISTRY.md` 为准，并应回写本节。*
