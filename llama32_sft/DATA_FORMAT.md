# Llama-3.2 SFT 数据契约

本目录训练脚本**只接受**下列格式，不自动使用 Qwen / ms-swift 侧已有 JSONL。

对齐 Meta llama-cookbook 对 Llama 3+ 的对话语义：`role` + `content`，训练时用 tokenizer 的 **`apply_chat_template`**（禁止手写 `User:/Assistant:` 拼接）。

## 文件约定

默认数据根目录（可被环境变量 `DATA_ROOT` 覆盖）——当前为 **calib v1 转换结果**：

```text
/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z
```

生成方式见 `src/prepare_calib_dataset.py --preset calib-v1` 与运行指南 `llama32/data_prepare.md`。

| 文件 | 必需 | 说明 |
|--|--|--|
| `train.jsonl` | **是** | 一行一个 JSON 样本 |
| `val.jsonl` | 否 | 有则做 eval；没有则按 `VAL_RATIO` 从 train 切，或 `SKIP_EVAL=1` |

仓库内示例（**不能**直接当 train，需复制并扩成真实数据）：

```text
data/train.jsonl.example
data/val.jsonl.example
```

快速放样（空目录冒烟；正式训练请用 calib 转换）：

```bash
# 推荐：calib → Llama
python3 /mnt/public/code/zhuxz/training/llama32_sft/src/prepare_calib_dataset.py --preset calib-v1

# 或最小 example：
DATA_ROOT=/tmp/llama32_smoke
mkdir -p "$DATA_ROOT"
cp /mnt/public/code/zhuxz/training/llama32_sft/data/train.jsonl.example "$DATA_ROOT/train.jsonl"
```

## 行 schema

每行必须是合法 JSON object，且**仅**使用字段 `messages`（允许额外元数据字段，训练时会忽略）：

```json
{
  "messages": [
    {"role": "system", "content": "可选。系统指令。"},
    {"role": "user", "content": "用户问题或指令。"},
    {"role": "assistant", "content": "模型应学习的回复。"}
  ]
}
```

### 硬规则

1. `messages` 必须是 **非空 list**。
2. 每条 message：`role` ∈ `{system, user, assistant}`，`content` 为**非空字符串**。
3. 至少出现 **1 个** `user` 与 **1 个** `assistant`。
4. `system` 可选；若有，建议放在列表最前。
5. 多轮对话允许：`user/assistant` 交替；最后一轮建议以 `assistant` 结束（SFT 学完整回复）。
6. **禁止**纯文本字段 `prompt`/`response`/`instruction`/`input`/`output` 作为唯一格式（不会被读）。
7. **禁止**在 `content` 里手工包一层 `<|start_header_id|>` 等 special token；由 chat template 负责。

### 正例（单轮，无 system）

```json
{"messages":[{"role":"user","content":"2+2=?"},{"role":"assistant","content":"4"}]}
```

### 正例（多轮 + system）

```json
{"messages":[{"role":"system","content":"You are a helpful math tutor."},{"role":"user","content":"What is 3*3?"},{"role":"assistant","content":"9"},{"role":"user","content":"And 3*4?"},{"role":"assistant","content":"12"}]}
```

### 反例

- 只有 `user` 没有 `assistant`
- `role: "human"` / `"gpt"`（需改成 `user` / `assistant`）
- 手写：

  ```text
  User: ...
  Assistant: ...
  ```

  应拆成两条 message。

## 训练时如何编码

脚本使用本地模型自带的 chat template（`tokenizer_config.json` 内）：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
)
```

Loss 只在 **assistant** 段上计算（completion-only；语义对齐 cookbook `tokenize_dialog` 中 mask system/user 的做法）。

## 与 Meta cookbook 的关系

| 项目 | 说明 |
|--|--|
| 对话字段 | 与 cookbook `custom_dataset.py` 中 `dialog` / `role`+`content` 同构；我们用 JSONL 字段名 **`messages`** |
| Template | 依赖模型内建 Llama 3 / 3.2 Instruct template（`vocab≥128000`） |
| 官方入口对照 | `python -m llama_cookbook.finetuning --use_peft --peft_method lora --model_name <local> ...`；本仓库主路径为 **TRL + PEFT** 以便 MACA 镜像少装依赖 |

## 规模建议（参考，非强制）

| 场景 | 粗略量级 |
|--|--|
| 冒烟 | 几十～几百条 |
| 领域适配 | 数千～数万条 |
| `max_length` 默认 | 2048 tokens（可环境变量改） |

超长样本会被截断；组装数据时尽量保证关键答案在前 2048 token 内。
