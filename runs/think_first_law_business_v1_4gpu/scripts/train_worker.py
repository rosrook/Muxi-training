#!/usr/bin/env python3
"""torchrun worker: LoRA SFT for **step-reasoning** ShareGPT rows (this run only).

Data contract (rewrite_step_v1):
  messages = [user, assistant]
  assistant starts with "Let's reason step by step."
  ends with "Therefore, the correct answer is <GOLD>."

Prep steps for best learning:
  - drop any system turn (process style is learned from the gold assistant)
  - drop train rows that are not user→assistant process traces
  - light whitespace normalize; keep full multi-step answers under higher cutoff
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


def _ensure_src_on_path() -> None:
    candidates = [
        Path(os.environ.get("BPG_SRC", "")),
        Path("/mnt/public/code/zhuxz/bench_probe_gen/src"),
        Path("/mnt/public/code/zhuxz/bench_probe_gen_clean/src"),
    ]
    for src in candidates:
        if src and (src / "bench_probe_gen").is_dir():
            sys.path.insert(0, str(src))
            return


_ensure_src_on_path()

from bench_probe_gen.trainer.config import load_training_config_file  # noqa: E402
from bench_probe_gen.trainer.dataset import TrainingExample  # noqa: E402
from bench_probe_gen.trainer.native_sft import run_native_lora_sft  # noqa: E402

_OPEN = re.compile(r"^Let's reason step by step\.?\s+", re.I)
_CLOSE = re.compile(r"Therefore,\s+the correct answer is\s+.+\s*$", re.I | re.S)


def _norm_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _prepare_messages(raw: list[Any]) -> list[dict[str, str]] | None:
    """Return cleaned [user, assistant] or None if row should be skipped."""
    cleaned: list[dict[str, str]] = []
    for m in raw or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = _norm_text(str(m.get("content") or ""))
        if not content:
            continue
        # Process SFT deliberately omits system: format lives in the gold assistant.
        if role == "system":
            continue
        if role not in {"user", "assistant"}:
            continue
        cleaned.append({"role": role, "content": content})

    if len(cleaned) < 2:
        return None
    # Keep last user before last assistant (tolerate stray extra turns by collapsing).
    # Expected: exactly one user then one assistant.
    if cleaned[-1]["role"] != "assistant":
        return None
    # find last user before assistant
    user_idx = None
    for i in range(len(cleaned) - 2, -1, -1):
        if cleaned[i]["role"] == "user":
            user_idx = i
            break
    if user_idx is None:
        return None
    user = cleaned[user_idx]
    asst = cleaned[-1]
    if not _OPEN.match(asst["content"]):
        return None
    if not _CLOSE.search(asst["content"]):
        return None
    # Prefer trailing newline on assistant so tokenization / end tokens sit cleanly.
    if not asst["content"].endswith("\n"):
        asst = {"role": "assistant", "content": asst["content"] + "\n"}
    return [user, asst]


def load_step_reasoning_examples(sft_path: Path) -> tuple[list[TrainingExample], dict[str, int]]:
    examples: list[TrainingExample] = []
    stats = {
        "rows_read": 0,
        "rows_kept": 0,
        "rows_skipped_schema": 0,
        "rows_skipped_format": 0,
        "assistant_chars_sum": 0,
        "user_chars_sum": 0,
    }
    with sft_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["rows_read"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["rows_skipped_schema"] += 1
                continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                stats["rows_skipped_schema"] += 1
                continue
            prepared = _prepare_messages(messages)
            if prepared is None:
                stats["rows_skipped_format"] += 1
                continue
            meta = dict(row.get("metadata") or {})
            meta["sft_format"] = "step_reasoning_v1"
            key = str(
                meta.get("generation_key")
                or meta.get("rewrite_record_id")
                or meta.get("bundle_id")
                or meta.get("item_id")
                or len(examples)
            )
            examples.append(
                TrainingExample(generation_key=key, messages=prepared, metadata=meta)
            )
            stats["rows_kept"] += 1
            stats["user_chars_sum"] += len(prepared[0]["content"])
            stats["assistant_chars_sum"] += len(prepared[1]["content"])
    return examples, stats


def main() -> int:
    sft_path = Path(os.environ["SFT_DATASET"]).expanduser()
    train_cfg = Path(os.environ["TRAINING_CONFIG"]).expanduser()
    run_id = os.environ["RUN_ID"]
    output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser()
    ckpt = Path(os.environ["CHECKPOINT_DIR"]).expanduser()
    tb = Path(os.environ["TENSORBOARD_DIR"]).expanduser()
    model_link = Path(os.environ["MODEL_LINK"]).expanduser()
    backend = os.environ.get("TRAINING_BACKEND", "native")

    if not sft_path.is_file():
        raise SystemExit(f"SFT_DATASET not found: {sft_path}")

    examples, load_stats = load_step_reasoning_examples(sft_path)
    if not examples:
        raise SystemExit(
            f"no valid step-reasoning examples in {sft_path} "
            f"(read={load_stats['rows_read']} kept=0)"
        )

    config = load_training_config_file(train_cfg)
    config.enabled = True
    config.backend = backend
    config.base_model_path = str(model_link)
    config.checkpoint_dir = ckpt
    config.tensorboard_log_dir = tb

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        n = max(load_stats["rows_kept"], 1)
        print(
            f"[train_worker] sft_format=step_reasoning_v1 path={sft_path}",
            flush=True,
        )
        print(
            f"[train_worker] load kept={load_stats['rows_kept']} "
            f"read={load_stats['rows_read']} "
            f"skip_schema={load_stats['rows_skipped_schema']} "
            f"skip_format={load_stats['rows_skipped_format']} "
            f"mean_user_chars={load_stats['user_chars_sum'] // n} "
            f"mean_asst_chars={load_stats['assistant_chars_sum'] // n}",
            flush=True,
        )
        print(
            f"[train_worker] examples={len(examples)} "
            f"cutoff={config.cutoff_len} epochs={config.num_train_epochs} "
            f"lr={config.learning_rate} accum={config.gradient_accumulation_steps} "
            f"per_device={config.per_device_train_batch_size} "
            f"base_model_path={config.base_model_path}",
            flush=True,
        )
        # Persist load stats for this run
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sft_load_stats.json").write_text(
            json.dumps(
                {
                    "sft_path": str(sft_path.resolve()),
                    "sft_format": "step_reasoning_v1",
                    **load_stats,
                    "n_examples": len(examples),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    summary = run_native_lora_sft(config, examples, output_dir=output_dir, run_id=run_id)
    if rank == 0:
        summary = dict(summary)
        summary["sft_format"] = "step_reasoning_v1"
        summary["sft_path"] = str(sft_path.resolve())
        summary["sft_load_stats"] = load_stats
        out = output_dir / "train_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    k: summary.get(k)
                    for k in (
                        "status",
                        "checkpoint_dir",
                        "train_examples",
                        "eval_examples",
                        "world_size",
                        "sft_format",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
        print(f"[train_worker] summary -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
