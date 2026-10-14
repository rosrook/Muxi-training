"""TRL + PEFT LoRA SFT for local Meta Llama-3.2 Instruct HF checkpoints.

Uses tokenizer.apply_chat_template (no hand-written User:/Assistant: formats).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `python src/train_sft_trl.py` without installing a package.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_messages import load_messages_dataset, summarize_jsonl  # noqa: E402


def _patch_accelerator_unwrap_model() -> None:
    """Old accelerate rejects keep_torch_compile= used by newer transformers.Trainer."""
    try:
        import inspect

        from accelerate import Accelerator
    except Exception as e:  # noqa: BLE001
        print(f"[compat] accelerate import skip: {e}")
        return
    try:
        sig = inspect.signature(Accelerator.unwrap_model)
        if "keep_torch_compile" in sig.parameters:
            print(
                "[compat] accelerate.unwrap_model supports keep_torch_compile "
                f"(accelerate import ok)"
            )
            return
    except (TypeError, ValueError):
        pass

    _orig = Accelerator.unwrap_model

    def _unwrap(self, model, *args, keep_torch_compile=None, **kwargs):  # noqa: ANN001
        return _orig(self, model, *args, **kwargs)

    Accelerator.unwrap_model = _unwrap  # type: ignore[method-assign]
    print("[compat] patched Accelerator.unwrap_model to ignore keep_torch_compile")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-3.2 LoRA SFT via TRL")
    p.add_argument(
        "--model",
        default=os.environ.get(
            "MODEL", "/mnt/public/model/huggingface/Llama-3.2-1B-Instruct"
        ),
        help="Local HF model directory",
    )
    p.add_argument(
        "--train-jsonl",
        default=os.environ.get("TRAIN_JSONL", ""),
        help="Path to train.jsonl (messages schema)",
    )
    p.add_argument(
        "--val-jsonl",
        default=os.environ.get("VAL_JSONL", ""),
        help="Optional path to val.jsonl",
    )
    p.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", ""),
        help="Run output directory",
    )
    p.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1")))
    p.add_argument("--per-device-train-batch-size", type=int, default=int(os.environ.get("PER_DEVICE_TRAIN_BS", "1")))
    p.add_argument("--per-device-eval-batch-size", type=int, default=int(os.environ.get("PER_DEVICE_EVAL_BS", "1")))
    p.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("GRAD_ACCUM", "16")))
    p.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "2e-4")))
    p.add_argument("--max-seq-length", type=int, default=int(os.environ.get("MAX_LENGTH", "2048")))
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=_env_bool("GRADIENT_CHECKPOINTING", True),
        help="Trade compute for activation memory (needed for long eager attention)",
    )
    p.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        default=False,
        help="Disable gradient checkpointing",
    )
    p.add_argument("--lora-r", type=int, default=int(os.environ.get("LORA_R", "16")))
    p.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    p.add_argument("--lora-dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    p.add_argument("--logging-steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "10")))
    p.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "200")))
    p.add_argument("--eval-steps", type=int, default=int(os.environ.get("EVAL_STEPS", "200")))
    p.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("WARMUP_RATIO", "0.03")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument(
        "--val-ratio",
        type=float,
        default=float(os.environ.get("VAL_RATIO", "0.02")),
        help="If no val file, optional fraction split from train (0 disables)",
    )
    p.add_argument(
        "--skip-eval",
        action="store_true",
        default=_env_bool("SKIP_EVAL", False),
    )
    p.add_argument(
        "--merge-weights",
        action="store_true",
        default=_env_bool("MERGE_WEIGHTS", False),
        help="Also write merged full weights under output_dir/merged",
    )
    p.add_argument(
        "--attn-implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION", "eager"),
        help="eager|sdpa|flash_attention_2 (MACA often needs eager)",
    )
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> None:
    data_root = Path(
        os.environ.get(
            "DATA_ROOT",
            "/mnt/public/data/zhuxz/training/llama32_sft/calib_20260808T152739Z",
        )
    )
    if not args.train_jsonl:
        args.train_jsonl = str(data_root / "train.jsonl")
    if not args.val_jsonl:
        cand = data_root / "val.jsonl"
        args.val_jsonl = str(cand) if cand.is_file() else ""
    if not args.output_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_tag = Path(args.model).name.replace(" ", "_")
        args.output_dir = str(
            Path("/mnt/public/data/zhuxz/training/llama32_sft/runs")
            / f"{stamp}_{model_tag}_lora"
        )


def _build_train_eval(args: argparse.Namespace):
    train_path = Path(args.train_jsonl)
    if not train_path.is_file():
        raise SystemExit(
            f"ERROR: train JSONL missing: {train_path}\n"
            f"See DATA_FORMAT.md and put messages JSONL at DATA_ROOT "
            f"(default .../llama32_sft/calib_20260808T152739Z/train.jsonl).\n"
            f"Example:\n"
            f"  cp .../llama32_sft/data/train.jsonl.example {train_path}"
        )

    train_ds = load_messages_dataset(train_path)
    print("[data]", summarize_jsonl(train_path))

    eval_ds = None
    if not args.skip_eval:
        if args.val_jsonl and Path(args.val_jsonl).is_file():
            eval_ds = load_messages_dataset(args.val_jsonl)
            print("[data]", summarize_jsonl(args.val_jsonl))
        elif args.val_ratio > 0 and len(train_ds) >= 10:
            split = train_ds.train_test_split(test_size=args.val_ratio, seed=args.seed)
            train_ds, eval_ds = split["train"], split["test"]
            print(f"[data] split val_ratio={args.val_ratio} -> train={len(train_ds)} eval={len(eval_ds)}")
        else:
            print("[data] no val; training without evaluation")
    return train_ds, eval_ds


def main() -> None:
    args = parse_args()
    _resolve_paths(args)

    # Offline / cache defaults (same knobs as Qwen train_sft_*_len8192.sh)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HOME", "/mnt/public/data/zhuxz/cache/huggingface")
    os.environ.setdefault("MODELSCOPE_CACHE", "/mnt/public/data/zhuxz/cache/modelscope")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    model_path = Path(args.model)
    if not model_path.is_dir():
        raise SystemExit(f"ERROR: model directory missing: {model_path}")

    train_ds, eval_ds = _build_train_eval(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "model": str(model_path),
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl or None,
        "output_dir": str(out_dir),
        "epochs": args.num_train_epochs,
        "lr": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "attn_implementation": args.attn_implementation,
        "stack": "trl+peft+transformers",
        "note": "chat_template via tokenizer; not hand-rolled prompts",
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[run]", json.dumps(run_cfg, ensure_ascii=False))

    # Patch first so any later Transformers Trainer import path sees it
    _patch_accelerator_unwrap_model()

    import torch
    from peft import LoraConfig, TaskType, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    try:
        import accelerate

        print(f"[accelerate] {accelerate.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"[accelerate] import fail: {e}")

    print(f"[torch] {torch.__version__} cuda={torch.cuda.is_available()} n={torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("ERROR: tokenizer has no chat_template; refuse to train without Llama template")

    # Sanity: template renders
    sample = train_ds[0]["messages"]
    rendered = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
    print("[template sample head]", rendered[:240].replace("\n", "\\n"))

    def _messages_to_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    # Map to plain text via official chat_template — widest TRL compatibility
    # (still sourced from messages; never hand-rolled User:/Assistant:).
    # num_proc>1 can be flaky with small JSONL / container IPC; default 1
    num_proc = int(os.environ.get("DATASET_NUM_PROC", "1"))
    map_kw = {"remove_columns": train_ds.column_names}
    if num_proc > 1:
        map_kw["num_proc"] = num_proc
    train_ds = train_ds.map(_messages_to_text, **map_kw)
    if eval_ds is not None:
        map_kw_e = {"remove_columns": eval_ds.column_names}
        if num_proc > 1:
            map_kw_e["num_proc"] = num_proc
        eval_ds = eval_ds.map(_messages_to_text, **map_kw_e)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    model.config.use_cache = False

    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    if use_gc:
        # Required for PEFT + checkpointing (else grad None on inputs)
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("[mem] gradient_checkpointing=ON (eager attn long seq needs this)")
    else:
        print("[mem] gradient_checkpointing=OFF")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_base = dict(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        seed=args.seed,
        report_to=[],
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=use_gc,
        optim="adamw_torch",
        dataloader_num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "2")),
        ddp_find_unused_parameters=False,
    )
    # Optional: only set if user asks (needs token lengths; can misbehave on raw text datasets)
    if _env_bool("GROUP_BY_LENGTH", False):
        sft_base["group_by_length"] = True
    if eval_ds is not None:
        sft_base["eval_steps"] = args.eval_steps

    print(
        f"[mem] micro_bs={args.per_device_train_batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} "
        f"max_len={args.max_seq_length} attn={args.attn_implementation}"
    )

    training_args = None
    last_err: Exception | None = None
    # Drop optional keys that older TRL rejects
    optional_drop_order = (
        ("gradient_checkpointing",),
        ("group_by_length",),
        ("ddp_find_unused_parameters",),
        ("dataloader_num_workers",),
    )
    for length_key in ("max_length", "max_seq_length"):
        for eval_key in ("eval_strategy", "evaluation_strategy"):
            kw = dict(sft_base)
            kw[length_key] = args.max_seq_length
            kw[eval_key] = "steps" if eval_ds is not None else "no"
            try:
                training_args = SFTConfig(**kw)
                last_err = None
                break
            except TypeError as e:
                last_err = e
                training_args = None
                for drop_keys in optional_drop_order:
                    for dk in drop_keys:
                        kw.pop(dk, None)
                    try:
                        training_args = SFTConfig(**kw)
                        last_err = None
                        print(f"[compat] SFTConfig ok after dropping {drop_keys}: {e}")
                        break
                    except TypeError as e2:
                        last_err = e2
                        training_args = None
                if training_args is not None:
                    break
        if training_args is not None:
            break
    if training_args is None:
        raise SystemExit(f"SFTConfig incompatible with installed TRL/transformers: {last_err}")

    trainer_common = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    try:
        trainer = SFTTrainer(**trainer_common)
    except TypeError:
        trainer_common.pop("processing_class", None)
        trainer_common["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_common)

    train_result = trainer.train()
    metrics = {k: float(v) if hasattr(v, "item") else v for k, v in train_result.metrics.items()}
    (out_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[done] adapter -> {adapter_dir}")

    if args.merge_weights:
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"[merge] writing full weights to {merged_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            str(model_path), trust_remote_code=True, torch_dtype=dtype
        )
        merged = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[done] merged -> {merged_dir}")

    print(f"[done] run dir: {out_dir}")


if __name__ == "__main__":
    main()
