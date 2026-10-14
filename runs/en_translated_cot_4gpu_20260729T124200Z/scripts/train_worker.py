#!/usr/bin/env python3
"""torchrun worker: LoRA SFT from prebuilt ShareGPT sft_dataset.jsonl (env-driven).

Self-contained entry for a single training run package. Depends on installed
``bench_probe_gen`` (repo ``pip install -e``), not on sibling scaffold paths.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    # Prefer installed package; fall back to checkout next to /training.
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


def main() -> int:
    sft_path = Path(os.environ["SFT_DATASET"]).expanduser()
    train_cfg = Path(os.environ["TRAINING_CONFIG"]).expanduser()
    run_id = os.environ["RUN_ID"]
    output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser()
    ckpt = Path(os.environ["CHECKPOINT_DIR"]).expanduser()
    tb = Path(os.environ["TENSORBOARD_DIR"]).expanduser()
    model_link = Path(os.environ["MODEL_LINK"]).expanduser()
    backend = os.environ.get("TRAINING_BACKEND", "native")

    examples: list[TrainingExample] = []
    with sft_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            meta = dict(row.get("metadata") or {})
            key = str(meta.get("generation_key") or meta.get("bundle_id") or len(examples))
            examples.append(
                TrainingExample(generation_key=key, messages=messages, metadata=meta)
            )
    if not examples:
        raise SystemExit(f"no examples in {sft_path}")

    config = load_training_config_file(train_cfg)
    config.enabled = True
    config.backend = backend
    config.base_model_path = str(model_link)
    config.checkpoint_dir = ckpt
    config.tensorboard_log_dir = tb

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        print(
            f"[train_worker] examples={len(examples)} "
            f"cutoff={config.cutoff_len} epochs={config.num_train_epochs} "
            f"lr={config.learning_rate} accum={config.gradient_accumulation_steps} "
            f"per_device={config.per_device_train_batch_size} "
            f"base_model_path={config.base_model_path}",
            flush=True,
        )

    summary = run_native_lora_sft(config, examples, output_dir=output_dir, run_id=run_id)
    if rank == 0:
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
