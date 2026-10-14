"""Lightweight offline checks: local model + chat_template + optional example JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_messages import load_messages_dataset, validate_messages  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "MODEL", "/mnt/public/model/huggingface/Llama-3.2-1B-Instruct"
        ),
    )
    p.add_argument(
        "--jsonl",
        default="",
        help="Optional messages JSONL to validate (defaults to repo example)",
    )
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    model = Path(args.model)
    assert model.is_dir(), f"model missing: {model}"
    for name in ("config.json", "tokenizer_config.json"):
        assert (model / name).is_file(), f"missing {model / name}"

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    assert getattr(tok, "chat_template", None), "tokenizer.chat_template is empty"
    print(f"[ok] model={model}")
    print(f"[ok] transformers tokenizer class={tok.__class__.__name__}")

    jsonl = Path(args.jsonl) if args.jsonl else (
        Path(__file__).resolve().parents[1] / "data" / "train.jsonl.example"
    )
    if jsonl.is_file():
        ds = load_messages_dataset(jsonl)
        sample = ds[0]["messages"]
        validate_messages(sample)
        text = tok.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        ids = tok.apply_chat_template(sample, tokenize=True, add_generation_prompt=False)
        print(f"[ok] jsonl={jsonl} samples={len(ds)} example_tokens={len(ids)}")
        print("[ok] rendered_head:", text[:200].replace("\n", "\\n"))
    else:
        demo = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        text = tok.apply_chat_template(demo, tokenize=False, add_generation_prompt=False)
        print("[ok] no jsonl; demo template head:", text[:200].replace("\n", "\\n"))

    # Optional torch/cuda peek
    try:
        import torch

        print(
            f"[ok] torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"n={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    except Exception as e:  # noqa: BLE001
        print("[warn] torch not importable:", e)

    print(json.dumps({"status": "ok", "model": str(model)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
