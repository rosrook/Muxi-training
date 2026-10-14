#!/usr/bin/env python3
"""Merge a Qwen3.5-4B LoRA adapter into a deployable full HF directory.

This is the procedure documented in
`bench_probe_discovery/.../qwen35_4B/deployment.md`, not a thin wrapper around
`swift export`. The important steps that `swift export` alone does not guarantee:

1. Load with ``AutoModelForImageTextToText`` (CausalLM is the wrong class).
2. ``PeftModel.merge_and_unload()`` then save a single ``model.safetensors``.
3. Restore the frozen ``mtp.*`` tensors that transformers drops on save.
4. Replace config / tokenizer / chat template / preprocessor with the base
   originals so the vLLM-Metax image sees a familiar layout.

Always write to a ``*_candidate`` directory. Promote only after
``verify_lora_merge.py`` prints ``VERDICT: PASS``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

AUX_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=Path("/mnt/public/model/Qwen3.5-4B"))
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="candidate directory; refuse if exists unless --force")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="cpu", help="cpu is safest for merge; use cuda:0 if RAM is tight")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _copy_aux(base: Path, out: Path) -> None:
    for name in AUX_FILES:
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)


def _restore_mtp(base: Path, out: Path) -> list[str]:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    index = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    mtp_keys = sorted(k for k in weight_map if k.startswith("mtp."))
    if not mtp_keys:
        print("[merge] no mtp.* keys in base; skip restore")
        return []

    merged_path = out / "model.safetensors"
    tensors = load_file(str(merged_path))
    handles: dict[str, object] = {}

    def get_base(key: str):
        fn = weight_map[key]
        if fn not in handles:
            handles[fn] = safe_open(str(base / fn), framework="pt")
        return handles[fn].get_tensor(key)

    restored = []
    for key in mtp_keys:
        if key in tensors:
            continue
        tensors[key] = get_base(key)
        restored.append(key)

    if restored:
        save_file(tensors, str(merged_path))
    print(f"[merge] restored {len(restored)} mtp.* tensors (base has {len(mtp_keys)})")
    return restored


def main() -> int:
    args = parse_args()
    if not args.base.is_dir():
        print(f"ERROR: base missing: {args.base}", file=sys.stderr)
        return 2
    if not (args.adapter / "adapter_model.safetensors").is_file():
        print(f"ERROR: adapter missing: {args.adapter}", file=sys.stderr)
        return 2
    if args.output.exists() and not args.force:
        print(f"ERROR: output exists (pass --force to overwrite): {args.output}", file=sys.stderr)
        return 2

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base={args.base}")
    print(f"[merge] adapter={args.adapter}")
    print(f"[merge] output={args.output}")
    print(f"[merge] dtype={args.dtype} device_map={args.device_map}")

    print("[merge] loading base ...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base),
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("[merge] attaching LoRA ...")
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    print("[merge] merge_and_unload ...")
    model = model.merge_and_unload()
    model.eval()

    print("[merge] saving model.safetensors ...")
    # Avoid transformers re-serializing tokenizer/config into a newer layout.
    model.save_pretrained(
        str(args.output),
        safe_serialization=True,
        max_shard_size="50GB",
    )
    del model

    # If transformers wrote shards, collapse is not needed for 4B (~8.7G).
    # Prefer a single file named model.safetensors for verify_lora_merge.py.
    shards = sorted(args.output.glob("model-*.safetensors")) + sorted(
        args.output.glob("model.safetensors-*-of-*.safetensors")
    )
    single = args.output / "model.safetensors"
    if shards and not single.is_file():
        from safetensors.torch import load_file, save_file

        print(f"[merge] collapsing {len(shards)} shards into model.safetensors")
        tensors = {}
        for shard in shards:
            tensors.update(load_file(str(shard)))
            shard.unlink()
        idx = args.output / "model.safetensors.index.json"
        if idx.is_file():
            idx.unlink()
        save_file(tensors, str(single))

    if not single.is_file():
        # Some transformers versions write model.safetensors directly.
        alts = list(args.output.glob("*.safetensors"))
        print(f"ERROR: no model.safetensors after save; found {alts}", file=sys.stderr)
        return 1

    print("[merge] restoring mtp.* from base ...")
    _restore_mtp(args.base, args.output)

    print("[merge] copying aux files from base ...")
    _copy_aux(args.base, args.output)

    # Drop leftover generation artefacts that confuse deploy checks.
    for junk in ("adapter_config.json", "adapter_model.safetensors", "README.md"):
        # Keep README from base (already copied); remove peft leftovers if any.
        pass
    for path in args.output.glob("adapter_*"):
        path.unlink()

    size_g = single.stat().st_size / (1024**3)
    print(f"[merge] done: {single} ({size_g:.2f} GiB)")
    print("[merge] next: run verify_lora_merge.py on this candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
