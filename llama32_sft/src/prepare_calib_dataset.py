#!/usr/bin/env python3
"""Convert calib / ms-swift SFT JSONL into Llama-3.2 TRL messages train/val.

Accepts rows with:
  {"messages": [{"role": "...", "content": "..."}, ...], ...optional meta...}

Writes lean Llama DATA_FORMAT rows (messages only).

Examples:
  # From already-split ms-swift calib pack (recommended)
  python prepare_calib_dataset.py --preset calib-v1

  # From final merged sft_dataset (re-split)
  python prepare_calib_dataset.py \\
    --src .../final/sft_dataset.jsonl \\
    --out-dir .../llama32_sft/from_final \\
    --val-ratio 0.01
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_messages import MessagesSchemaError, validate_messages  # noqa: E402

# Qwen calib run used for official MMLU-Pro calib-sft record
PRESET_CALIB_V1 = {
    "name": "calib-v1-20260808T152739Z",
    "train": Path(
        "/mnt/public/data/zhuxz/results/mcq_calibration/"
        "20260808T152739Z/06_ms_swift_sft/train.jsonl"
    ),
    "val": Path(
        "/mnt/public/data/zhuxz/results/mcq_calibration/"
        "20260808T152739Z/06_ms_swift_sft/train.val.jsonl"
    ),
    "source_sft": Path(
        "/mnt/public/data/zhuxz/results/mcq_calibration/"
        "20260808T152739Z/final/sft_dataset.jsonl"
    ),
    "out_dir": Path(
        "/mnt/public/data/zhuxz/training/llama32_sft/"
        "calib_20260808T152739Z"
    ),
}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: bad JSON: {e}") from e
            if isinstance(obj, dict):
                yield obj


def _row_to_messages(obj: dict[str, Any], *, line_no: int, path: Path) -> list[dict[str, str]] | None:
    try:
        return validate_messages(obj.get("messages"), line_no=line_no)
    except MessagesSchemaError as e:
        # skip bad rows rather than abort whole corpus
        print(f"[skip] {path.name}:{line_no}: {e}")
        return None


def convert_file(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    out: list[dict[str, Any]] = []
    n_in = 0
    for line_no, obj in enumerate(_iter_jsonl(path), start=1):
        n_in += 1
        messages = _row_to_messages(obj, line_no=line_no, path=path)
        if messages is None:
            continue
        out.append({"messages": messages})
    print(f"[convert] {path} in={n_in} ok={len(out)}")
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {path} n={len(rows)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        choices=["calib-v1", "none"],
        default="none",
        help="calib-v1: official mcq_calibration 20260808 pack split",
    )
    ap.add_argument("--src", type=Path, default=None, help="Single source sft_dataset.jsonl")
    ap.add_argument("--train-src", type=Path, default=None)
    ap.add_argument("--val-src", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.preset == "calib-v1":
        p = PRESET_CALIB_V1
        out_dir = args.out_dir or p["out_dir"]
        train_rows = convert_file(p["train"])
        val_rows = convert_file(p["val"])
        meta = {
            "preset": p["name"],
            "train_src": str(p["train"]),
            "val_src": str(p["val"]),
            "source_sft": str(p["source_sft"]),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "note": "Lean messages-only from Qwen calib ms-swift split; Llama TRL ready",
        }
    elif args.train_src:
        out_dir = args.out_dir or Path(
            "/mnt/public/data/zhuxz/training/llama32_sft/from_custom"
        )
        train_rows = convert_file(args.train_src)
        val_rows = convert_file(args.val_src) if args.val_src else []
        if not val_rows and args.val_ratio > 0 and len(train_rows) >= 10:
            rng = random.Random(args.seed)
            rng.shuffle(train_rows)
            n_val = max(1, int(len(train_rows) * args.val_ratio))
            val_rows = train_rows[:n_val]
            train_rows = train_rows[n_val:]
        meta = {
            "train_src": str(args.train_src),
            "val_src": str(args.val_src) if args.val_src else None,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
        }
    elif args.src:
        out_dir = args.out_dir or Path(
            "/mnt/public/data/zhuxz/training/llama32_sft/from_src"
        )
        all_rows = convert_file(args.src)
        rng = random.Random(args.seed)
        rng.shuffle(all_rows)
        n_val = max(1, int(len(all_rows) * max(0.0, args.val_ratio))) if args.val_ratio > 0 else 0
        if n_val >= len(all_rows):
            n_val = max(1, len(all_rows) // 100)
        val_rows = all_rows[:n_val]
        train_rows = all_rows[n_val:]
        meta = {
            "src": str(args.src),
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "n_train": len(train_rows),
            "n_val": len(val_rows),
        }
    else:
        ap.error("use --preset calib-v1, or --train-src, or --src")

    if not train_rows:
        print("ERROR: no train rows")
        return 1

    out_dir = Path(out_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    if val_rows:
        write_jsonl(out_dir / "val.jsonl", val_rows)
    (out_dir / "prepare_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[meta]", json.dumps(meta, ensure_ascii=False))
    print(f"[done] DATA_ROOT={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
