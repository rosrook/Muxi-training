# Training layout (self-contained per run)

**经验总结（6–8 月，自包含）**：[`TRAINING_EXPERIENCE_202606_202608.md`](./TRAINING_EXPERIENCE_202606_202608.md)

Each training job is independent: scripts and configs are **copied** into the run
directory (no shared mutable launchers). Artifacts live under a single data tree.

## Paths

| | Path |
|--|--|
| Code (run packages) | `/mnt/public/code/zhuxz/training/runs/<RUN_ID>/` |
| Data (artifacts) | `/mnt/public/data/zhuxz/training_runs/<RUN_ID>/` |
| Templates (seed for *new* runs) | `/mnt/public/code/zhuxz/training/templates/` |
| Tools | `/mnt/public/code/zhuxz/training/tools/` |
| Registry | `/mnt/public/code/zhuxz/training/REGISTRY.md` |

Legacy paths (`data/zhuxz/checkpoints/<RUN_ID>`, old `results/<family>/<RUN_ID>`,
`runtime_logs/training/<RUN_ID>`) remain for history. Migrated runs **symlink** into
`training_runs/` instead of moving multi‑GB weights.

## Per-run code package

```text
code/zhuxz/training/runs/<RUN_ID>/
  README.md
  run_meta.json          # id, format, status, pointers
  config.yaml            # copied training HP (self-owned)
  run.sh                 # launch (only needs this dir + meta paths)
  scripts/
    train_worker.py      # copy; calls bench_probe_gen.trainer (pkg install)
    train_launch.sh      # copy; torchrun wrapper
```

## Per-run data package

```text
data/zhuxz/training_runs/<RUN_ID>/
  README.md
  manifest.json
  checkpoints/           # dir or symlink → legacy ckpt
  tensorboard/           # dir or symlink
  outputs/               # train_summary etc. (or symlink)
  sft_dataset.jsonl      # symlink to training data (not copied)
```

## New run

```bash
bash /mnt/public/code/zhuxz/training/tools/create_run.sh \
  --run-id my_lora_$(date -u +%Y%m%dT%H%M%SZ) \
  --sft /path/to/sft_dataset.jsonl \
  --config /path/to/config.yaml   # will be copied into the run dir
# then:
bash /mnt/public/code/zhuxz/training/runs/<RUN_ID>/run.sh
```

## Re-migrate / refresh registry

```bash
python3 /mnt/public/code/zhuxz/training/tools/migrate_existing_runs.py
```

## Notes

- Training still imports `bench_probe_gen` from the package (`pip install -e` repo).
  Only **launcher + config + worker entry** are per-run copies.
- Do **not** edit shared files under `bench_probe_gen/personal_scaffold/...`
  for a specific experiment; edit the copy inside `training/runs/<RUN_ID>/`.
