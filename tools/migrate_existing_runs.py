#!/usr/bin/env python3
"""Migrate historical LoRA runs into self-contained training packages.

Creates for each known RUN_ID:
  /mnt/public/code/zhuxz/training/runs/<RUN_ID>/   (scripts + config copies)
  /mnt/public/data/zhuxz/training_runs/<RUN_ID>/  (symlinks to legacy artifacts)

Does not move multi-GB checkpoints; links them.
"""

from __future__ import annotations

import json
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CODE_TRAIN = Path("/mnt/public/code/zhuxz/training")
TEMPLATE = CODE_TRAIN / "templates" / "lora_4gpu"
RUNS_CODE = CODE_TRAIN / "runs"
DATA_RUNS = Path("/mnt/public/data/zhuxz/training_runs")
CKPT_ROOT = Path("/mnt/public/data/zhuxz/checkpoints")
LOG_ROOT = Path("/mnt/public/data/zhuxz/runtime_logs/training")
SCAFFOLD = Path(
    "/mnt/public/code/zhuxz/bench_probe_gen/personal_scaffold/runtime_error_case_backtest"
)
BPG_CFG = Path("/mnt/public/code/zhuxz/bench_probe_gen/configs/training")
RES = Path("/mnt/public/data/zhuxz/results")

# Catalog entries (from user generation table + disk). sft/config best-effort.
RUNS: list[dict[str, Any]] = [
    {
        "run_id": "v1_remote_1000_02",
        "gen": 1,
        "format": "short_answer",
        "sft": RES / "v1_remote_1000_02/training/sft_dataset.jsonl",
        "config": BPG_CFG / "qwen35_4b_lora.yaml",
        "status": "completed",
    },
    {
        "run_id": "v1_remote_3000_01",
        "gen": 1,
        "format": "short_answer",
        "sft": RES / "v1_remote_3000_01/training/sft_dataset.jsonl",
        "config": BPG_CFG / "qwen35_4b_lora_5k.yaml",
        "status": "completed",
    },
    {
        "run_id": "test3k_run",
        "gen": 2,
        "format": "short_answer",
        "sft": RES / "v2_generator/test3k_run/training/sft_dataset.jsonl",
        "config": BPG_CFG / "qwen35_4b_lora.yaml",
        "status": "completed",
    },
    {
        "run_id": "mixed10k_backtest_20260713T041155Z",
        "gen": 2,
        "format": "short_answer_only",
        "sft": RES
        / "mixed_10k/mixed_10k_20260709T095800Z/loop_snapshots/backtest_20260713T041155Z/training/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_18k.yaml",
        "status": "completed",
    },
    {
        "run_id": "mixed10k_fulltrain_20260714T071006Z",
        "gen": 2,
        "format": "short_answer_only",
        "sft": RES
        / "mixed_10k/mixed_10k_20260709T095800Z/loop_snapshots/fulltrain_20260714T071006Z/training/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_18k_4gpu.yaml",
        "status": "completed",
    },
    {
        "run_id": "mixed10k_gsm8k_cot_20260714T114313Z",
        "gen": 2,
        "format": "cot_hash_gsm8k",
        "sft": RES
        / "mixed_10k/mixed_10k_20260709T095800Z/sft_split_cot_20260714T114313Z/gsm8k/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_gsm8k_cot.yaml",
        "status": "completed",
    },
    {
        "run_id": "mixed10k_stage2_mmlu_replay_20260715T030751Z",
        "gen": 2,
        "format": "stage2_mmlu_cot_replay",
        "sft": RES
        / "mixed_10k/mixed_10k_20260709T095800Z/sft_stage2_mmlu_replay_20260715T030751Z/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_stage2_mmlu.yaml",
        "status": "completed",
    },
    {
        "run_id": "fulltrain_vb_4gpu_20260716T064800Z",
        "gen": 2,
        "format": "letter_only_vb",
        "sft": RES
        / "mixed_10k/mixed_10k_20260709T095800Z/loop_snapshots/fulltrain_vb_20260716T064800Z/training/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_18k_4gpu.yaml",
        "status": "completed",
    },
    {
        "run_id": "earlytrain_v5_cot_20260728T022500Z",
        "gen": 3,
        "format": "cot_rationale_v1_zh",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/loop_snapshots/earlytrain_v5_20260728T022500Z/training/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_earlytrain_v5_cot.yaml",
        "outputs": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/loop_snapshots/earlytrain_v5_20260728T022500Z",
        "status": "completed",
    },
    {
        "run_id": "earlytrain_v5_scale_9k_4gpu_20260728T065935Z",
        "gen": 3,
        "format": "cot_rationale_v1_zh_scale",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/loop_snapshots/earlytrain_v5_20260728T022500Z/training/scale_subsets/sft_dataset_9k.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_earlytrain_v5_cot_4gpu_9k.yaml",
        "outputs": RES / "scale_ablation_earlytrain_v5/earlytrain_v5_scale_9k_4gpu_20260728T065935Z",
        "status": "completed",
    },
    {
        "run_id": "earlytrain_v5_scale_6k_4gpu_20260728T103731Z",
        "gen": 3,
        "format": "cot_rationale_v1_zh_scale",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/loop_snapshots/earlytrain_v5_20260728T022500Z/training/scale_subsets/sft_dataset_6k.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_earlytrain_v5_cot_4gpu_6k.yaml",
        "outputs": RES / "scale_ablation_earlytrain_v5/earlytrain_v5_scale_6k_4gpu_20260728T103731Z",
        "status": "completed",
    },
    {
        "run_id": "earlytrain_v5_scale_3k_4gpu_20260728T172041Z",
        "gen": 3,
        "format": "cot_rationale_v1_zh_scale",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/loop_snapshots/earlytrain_v5_20260728T022500Z/training/scale_subsets/sft_dataset_3k.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_earlytrain_v5_cot_4gpu_3k.yaml",
        "outputs": RES / "scale_ablation_earlytrain_v5/earlytrain_v5_scale_3k_4gpu_20260728T172041Z",
        "status": "completed",
    },
    {
        "run_id": "earlytrain_v5_scale_1k_4gpu_20260729T063523Z",
        "gen": 3,
        "format": "cot_rationale_v1_zh_scale",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/loop_snapshots/earlytrain_v5_20260728T022500Z/training/scale_subsets/sft_dataset_1k.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_earlytrain_v5_cot_4gpu_1k.yaml",
        "outputs": RES / "scale_ablation_earlytrain_v5/earlytrain_v5_scale_1k_4gpu_20260729T063523Z",
        "status": "completed",
    },
    {
        "run_id": "en_translated_cot_4gpu_20260729T124200Z",
        "gen": 3,
        "format": "en_cot_hash",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/04_generator_v5/09_items_en_translated/training_cot/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_en_translated_cot_4gpu.yaml",
        "outputs": RES / "en_translated_cot/en_translated_cot_4gpu_20260729T124200Z",
        "status": "completed",
    },
    {
        "run_id": "en_translated_letter_only_4gpu_20260729T124358Z",
        "gen": 3,
        "format": "en_letter_only",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/04_generator_v5/09_items_en_translated/training_letter_only/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_en_translated_letter_only_4gpu.yaml",
        "outputs": RES / "en_translated_letter_only/en_translated_letter_only_4gpu_20260729T124358Z",
        "status": "completed",
    },
    {
        "run_id": "en_translated_answer_first_think_4gpu_20260731T064450Z",
        "gen": 3,
        "format": "en_answer_first_think",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/04_generator_v5/09_items_en_translated/training_answer_first_think_v1/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_en_translated_cot_4gpu.yaml",
        "status": "failed_launch",
    },
    {
        "run_id": "en_translated_answer_first_think_4gpu_20260731T064622Z",
        "gen": 3,
        "format": "en_answer_first_think",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/04_generator_v5/09_items_en_translated/training_answer_first_think_v1/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_en_translated_cot_4gpu.yaml",
        "status": "failed_launch",
    },
    {
        "run_id": "en_translated_think_first_final_4gpu_20260731T081559Z",
        "gen": 3,
        "format": "en_think_first_final",
        "sft": RES
        / "mixed_mmlu_aime_math/discovery_v2v2v3_full_20260725T211503Z/04_generator_v5/09_items_en_translated/training_think_first_final_v1/sft_dataset.jsonl",
        "config": SCAFFOLD / "qwen35_4b_lora_en_translated_cot_4gpu.yaml",
        "outputs": RES
        / "en_translated_think_first_final/en_translated_think_first_final_4gpu_20260731T081559Z",
        "status": "completed",
    },
]


def _chmod_x(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _link(src: Path | None, dst: Path) -> str:
    if src is None:
        return "missing_src"
    if not src.exists():
        return f"missing:{src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            # do not rm trees
            return f"skip_exists_dir:{dst}"
    dst.symlink_to(src.resolve())
    return f"link:{src}"


def migrate_one(entry: dict[str, Any]) -> dict[str, Any]:
    rid = entry["run_id"]
    code_dir = RUNS_CODE / rid
    data_dir = DATA_RUNS / rid
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "scripts").mkdir(exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # copy scripts template (independent copies)
    for name in ("train_worker.py", "train_launch.sh"):
        shutil.copy2(TEMPLATE / "scripts" / name, code_dir / "scripts" / name)
    shutil.copy2(TEMPLATE / "run.sh", code_dir / "run.sh")
    _chmod_x(code_dir / "run.sh")
    _chmod_x(code_dir / "scripts" / "train_launch.sh")
    _chmod_x(code_dir / "scripts" / "train_worker.py")

    # config: prefer catalog, else template
    cfg_src = entry.get("config")
    if isinstance(cfg_src, Path) and cfg_src.is_file():
        shutil.copy2(cfg_src, code_dir / "config.yaml")
        cfg_note = str(cfg_src)
    else:
        shutil.copy2(TEMPLATE / "config.yaml", code_dir / "config.yaml")
        cfg_note = "template_default"

    sft = entry.get("sft")
    sft_path = Path(sft) if sft else None

    # data layout with symlinks
    ckpt_legacy = CKPT_ROOT / rid
    log_legacy = LOG_ROOT / rid
    tb_legacy = log_legacy / "tensorboard"
    if not tb_legacy.exists() and log_legacy.exists():
        tb_legacy = log_legacy

    outs = entry.get("outputs")
    if outs is None:
        # common places
        candidates = [
            RES / "en_translated_cot" / rid,
            RES / "en_translated_letter_only" / rid,
            RES / "en_translated_think_first_final" / rid,
            RES / "en_translated_answer_first_think" / rid,
            RES / "scale_ablation_earlytrain_v5" / rid,
            RES / rid,
        ]
        outs = next((p for p in candidates if p.exists()), None)
    else:
        outs = Path(outs)

    report: dict[str, Any] = {
        "run_id": rid,
        "status": entry.get("status"),
        "gen": entry.get("gen"),
        "format": entry.get("format"),
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "config_source": cfg_note,
        "links": {},
    }
    report["links"]["checkpoints"] = _link(
        ckpt_legacy if ckpt_legacy.exists() else None, data_dir / "checkpoints"
    )
    report["links"]["tensorboard"] = _link(
        tb_legacy if tb_legacy.exists() else None, data_dir / "tensorboard"
    )
    report["links"]["outputs"] = _link(outs if outs and outs.exists() else None, data_dir / "outputs")
    report["links"]["sft_dataset"] = _link(
        sft_path if sft_path and sft_path.is_file() else None, data_dir / "sft_dataset.jsonl"
    )
    report["links"]["legacy_logs_dir"] = _link(
        log_legacy if log_legacy.exists() else None, data_dir / "legacy_runtime_logs"
    )

    # env for re-train / re-finetune (points into data_dir)
    env_text = f"""# Owned by run {rid}. Historical migration snapshot.
export RUN_ID="{rid}"
export SFT_DATASET="{data_dir / 'sft_dataset.jsonl'}"
export TRAINING_CONFIG="{code_dir / 'config.yaml'}"
export CHECKPOINT_DIR="{data_dir / 'checkpoints'}"
export TENSORBOARD_DIR="{data_dir / 'tensorboard'}"
export OUTPUT_DIR="{data_dir / 'outputs'}"
export MODEL_SOURCE="${{MODEL_SOURCE:-/mnt/public/model/Qwen3.5-4B}}"
export MODEL_LINK="${{MODEL_LINK:-/mnt/public/data/zhuxz/models/Qwen3.5-4B}}"
export BPG_SRC="${{BPG_SRC:-/mnt/public/code/zhuxz/bench_probe_gen/src}}"
export TRAINING_BACKEND="${{TRAINING_BACKEND:-native}}"
export NPROC_PER_NODE="${{NPROC_PER_NODE:-4}}"
export CUDA_DEVICES="${{CUDA_DEVICES:-0,1,2,3}}"
export MASTER_ADDR="${{MASTER_ADDR:-127.0.0.1}}"
export MASTER_PORT="${{MASTER_PORT:-29531}}"
"""
    (code_dir / "run_env.sh").write_text(env_text, encoding="utf-8")

    meta = {
        "run_id": rid,
        "migrated_at": datetime.now(timezone.utc).isoformat(),
        "status": entry.get("status"),
        "generation": entry.get("gen"),
        "format": entry.get("format"),
        "legacy": {
            "checkpoints": str(ckpt_legacy) if ckpt_legacy.exists() else None,
            "runtime_logs": str(log_legacy) if log_legacy.exists() else None,
            "outputs": str(outs) if outs and Path(outs).exists() else None,
            "sft": str(sft_path) if sft_path and sft_path.is_file() else None,
            "config_source": cfg_note,
        },
        "code_dir": str(code_dir),
        "data_dir": str(data_dir),
        "note": "Scripts/config are local copies. Artifacts are symlinked from legacy paths.",
    }
    (code_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (data_dir / "manifest.json").write_text(
        json.dumps({**meta, "links": report["links"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# Training run `{rid}`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | {entry.get("status")} |
| gen | {entry.get("gen")} |
| format | {entry.get("format")} |
| code | `{code_dir}` |
| data | `{data_dir}` |
| config | `{code_dir / "config.yaml"}` (local **copy**) |
| sft | `{data_dir / "sft_dataset.jsonl"}` (symlink) |
| checkpoints | `{data_dir / "checkpoints"}` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash {code_dir}/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
"""
    (code_dir / "README.md").write_text(readme, encoding="utf-8")
    (data_dir / "README.md").write_text(readme, encoding="utf-8")
    return report


def write_registry(reports: list[dict[str, Any]]) -> None:
    lines = [
        "# Training REGISTRY",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Self-contained packages live under `code/zhuxz/training/runs/<run_id>/`;",
        "artifacts under `data/zhuxz/training_runs/<run_id>/`.",
        "",
        "| gen | run_id | status | format | code | data | sft_link | ckpt_link |",
        "|-----|--------|--------|--------|------|------|----------|-----------|",
    ]
    for r in reports:
        sft_ok = "yes" if str(r["links"].get("sft_dataset", "")).startswith("link:") else "no"
        ck_ok = "yes" if str(r["links"].get("checkpoints", "")).startswith("link:") else "no"
        lines.append(
            f"| {r.get('gen')} | `{r['run_id']}` | {r.get('status')} | {r.get('format')} | "
            f"`training/runs/{r['run_id']}` | `training_runs/{r['run_id']}` | {sft_ok} | {ck_ok} |"
        )
    lines.append("")
    lines.append("## Create a new independent run")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "bash /mnt/public/code/zhuxz/training/tools/create_run.sh \\"
    )
    lines.append("  --run-id my_run_$(date -u +%Y%m%dT%H%M%SZ) \\")
    lines.append("  --sft /path/to/sft_dataset.jsonl \\")
    lines.append("  --config /path/to/config.yaml")
    lines.append("```")
    lines.append("")
    (CODE_TRAIN / "REGISTRY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (CODE_TRAIN / "registry.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    RUNS_CODE.mkdir(parents=True, exist_ok=True)
    DATA_RUNS.mkdir(parents=True, exist_ok=True)
    reports = []
    for entry in RUNS:
        # also auto-include scale runs that exist on disk but listed
        reports.append(migrate_one(entry))
        print(f"migrated {entry['run_id']}")

    # any other checkpoint dirs not in catalog (minor smoke) — light stub
    known = {e["run_id"] for e in RUNS}
    for d in sorted(CKPT_ROOT.iterdir()):
        if not d.is_dir() or d.name in known:
            continue
        if d.name.startswith("v1_40_"):
            reports.append(
                migrate_one(
                    {
                        "run_id": d.name,
                        "gen": 1,
                        "format": "smoke",
                        "status": "smoke_or_partial",
                        "config": TEMPLATE / "config.yaml",
                    }
                )
            )
            print(f"migrated smoke {d.name}")

    write_registry(reports)
    print(f"registry -> {CODE_TRAIN / 'REGISTRY.md'} ({len(reports)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
