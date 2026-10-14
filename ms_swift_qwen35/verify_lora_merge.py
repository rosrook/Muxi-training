#!/usr/bin/env python3
"""Numerically verify that a merged model exactly represents a LoRA adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--tag", default="merge")
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--delta-norm-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter_config = json.loads(
        (args.adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    scaling = adapter_config["lora_alpha"] / adapter_config["r"]

    if adapter_config.get("use_rslora") or adapter_config.get("use_dora"):
        raise ValueError("This verifier only supports standard LoRA scaling")
    if adapter_config.get("modules_to_save"):
        raise ValueError("modules_to_save requires separate verification")

    index = json.loads(
        (args.base / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    base_map: dict[str, str] = index["weight_map"]
    base_handles = {}

    def get_base(key: str) -> torch.Tensor:
        filename = base_map[key]
        if filename not in base_handles:
            base_handles[filename] = safe_open(
                args.base / filename, framework="pt"
            )
        return base_handles[filename].get_tensor(key)

    merged_file = safe_open(args.merged / "model.safetensors", framework="pt")
    merged_keys = set(merged_file.keys())
    base_keys = set(base_map)
    only_base = sorted(base_keys - merged_keys)
    only_merged = sorted(merged_keys - base_keys)
    leftovers = sorted(
        key
        for key in merged_keys
        if "lora" in key.lower() or "base_model" in key
    )

    print(
        f"[{args.tag}] base tensors={len(base_keys)} "
        f"merged tensors={len(merged_keys)}"
    )
    print(f"[{args.tag}] keys only in base={len(only_base)} {only_base[:10]}")
    print(
        f"[{args.tag}] keys only in merged={len(only_merged)} "
        f"{only_merged[:10]}"
    )
    print(f"[{args.tag}] LoRA/base_model leftovers={leftovers}")

    adapter_file = safe_open(
        args.adapter / "adapter_model.safetensors", framework="pt"
    )
    modules: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"base_model\.model\.(.+)\.lora_(A|B)\.weight")
    for key in adapter_file.keys():
        match = pattern.fullmatch(key)
        if match is None:
            raise ValueError(f"Unexpected adapter tensor: {key}")
        modules.setdefault(match.group(1), {})[match.group(2)] = key

    targeted_weights = set()
    failures = []
    max_relative_error = 0.0
    worst_norm_ratio = (1.0, None)

    for module, matrices in sorted(modules.items()):
        if set(matrices) != {"A", "B"}:
            failures.append((module, "missing A or B matrix"))
            continue

        weight_key = f"{module}.weight"
        targeted_weights.add(weight_key)
        matrix_a = adapter_file.get_tensor(matrices["A"]).float()
        matrix_b = adapter_file.get_tensor(matrices["B"]).float()
        base_weight = get_base(weight_key).float()
        merged_weight = merged_file.get_tensor(weight_key).float()

        expected_delta = scaling * (matrix_b @ matrix_a)
        if expected_delta.norm().item() == 0:
            failures.append((module, "zero adapter delta"))
            continue

        expected_weight = base_weight + expected_delta
        absolute_error = (merged_weight - expected_weight).abs().max().item()
        relative_error = absolute_error / max(
            expected_weight.abs().max().item(), 1e-8
        )
        max_relative_error = max(max_relative_error, relative_error)

        norm_ratio = (
            (merged_weight - base_weight).norm().item()
            / expected_delta.norm().item()
        )
        if abs(norm_ratio - 1.0) > abs(worst_norm_ratio[0] - 1.0):
            worst_norm_ratio = (norm_ratio, module)
        if (
            relative_error > args.relative_tolerance
            or abs(norm_ratio - 1.0) > args.delta_norm_tolerance
        ):
            failures.append((module, relative_error, norm_ratio))

    print(
        f"[{args.tag}] checked {len(modules)} LoRA modules; "
        f"max relative error={max_relative_error:.3e}; "
        f"worst delta-norm ratio={worst_norm_ratio[0]:.6f} "
        f"at {worst_norm_ratio[1]}"
    )

    non_target_mismatches = []
    non_target_count = 0
    for key in sorted(base_keys & merged_keys):
        if key in targeted_weights:
            continue
        base_tensor = get_base(key)
        merged_tensor = merged_file.get_tensor(key)
        non_target_count += 1
        is_equal = base_tensor.shape == merged_tensor.shape and (
            torch.equal(base_tensor, merged_tensor)
            or (
                base_tensor.dtype == torch.float32
                and merged_tensor.dtype == torch.bfloat16
                and torch.equal(base_tensor.to(torch.bfloat16), merged_tensor)
            )
        )
        if not is_equal:
            non_target_mismatches.append(key)

    print(
        f"[{args.tag}] non-target tensors checked={non_target_count}; "
        f"mismatched={len(non_target_mismatches)}"
    )
    for failure in failures[:10]:
        print("  BAD LoRA:", failure)
    for key in non_target_mismatches[:10]:
        print("  BAD non-target:", key)

    passed = not (
        only_base
        or only_merged
        or leftovers
        or failures
        or non_target_mismatches
    )
    print(f"[{args.tag}] VERDICT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
