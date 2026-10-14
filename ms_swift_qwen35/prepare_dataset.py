#!/usr/bin/env python3
"""Prepare ms-swift messages JSONL from our generator rewrite outputs.

Strict format used by ms-swift custom dataset (see Qwen3.5 Best Practices):
  {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

Sources (default):
  - Non-FG rewritten SFT (step reasoning) — may be incomplete if rewrite still running
  - FG knowledge SFT (optional) — knowledge-style, no step-rewrite

Usage:
  python prepare_dataset.py --out /path/to/train.jsonl
  python prepare_dataset.py --nonfg ... --fg ... --out ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_NONFG = Path(
    "/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/"
    "discovery_v2v2v3_full_20260725T211503Z/05_step_rewrite_dual_nonfg/"
    "rewrite_step_v1_20260806T030314Z/sft_dataset.jsonl"
)
DEFAULT_FG = Path(
    "/mnt/public/data/zhuxz/results/mixed_mmlu_aime_math/"
    "discovery_v2v2v3_full_20260725T211503Z/04_generator_v5_mmlu_fg_only_20260805T030835Z/"
    "00_fg_knowledge/sft_dataset.jsonl"
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _normalize_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    out: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip().lower()
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        out.append({"role": role, "content": content.strip()})
    # need at least user + assistant
    roles = {m["role"] for m in out}
    if "user" not in roles or "assistant" not in roles:
        return None
    return out


def convert_file(path: Path, source_tag: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        print(f"[skip] missing: {path}")
        return rows
    n_in = n_ok = 0
    for row in _iter_jsonl(path):
        n_in += 1
        messages = _normalize_messages(row)
        if not messages:
            continue
        n_ok += 1
        # ms-swift only requires messages; optional meta is fine but keep lean
        rec: dict[str, Any] = {"messages": messages}
        rows.append(rec)
    print(f"[{source_tag}] in={n_in} ok={n_ok} path={path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonfg", type=Path, default=DEFAULT_NONFG)
    ap.add_argument("--fg", type=Path, default=DEFAULT_FG)
    ap.add_argument("--include-fg", action="store_true", help="Mix FG knowledge SFT")
    ap.add_argument("--no-nonfg", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_rows: list[dict[str, Any]] = []
    if not args.no_nonfg:
        all_rows.extend(convert_file(args.nonfg, "nonfg_rewrite"))
    if args.include_fg:
        all_rows.extend(convert_file(args.fg, "fg_knowledge"))

    if not all_rows:
        print("ERROR: no rows")
        return 1

    # stable shuffle
    import random

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    n_val = max(1, int(len(all_rows) * max(0.0, args.val_ratio))) if args.val_ratio > 0 else 0
    if n_val >= len(all_rows):
        n_val = max(1, len(all_rows) // 100)
    val_rows = all_rows[:n_val] if n_val else []
    train_rows = all_rows[n_val:] if n_val else all_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.out
    val_path = args.out.with_name(args.out.stem + ".val" + args.out.suffix)

    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if val_rows:
        with val_path.open("w", encoding="utf-8") as f:
            for r in val_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "train": str(train_path),
        "val": str(val_path) if val_rows else None,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_total": len(all_rows),
        "include_fg": bool(args.include_fg),
        "nonfg": str(args.nonfg),
        "fg": str(args.fg) if args.include_fg else None,
    }
    meta_path = train_path.with_suffix(train_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
