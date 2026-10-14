# Training run `mixed10k_stage2_mmlu_replay_20260715T030751Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 2 |
| format | stage2_mmlu_cot_replay |
| code | `/mnt/public/code/zhuxz/training/runs/mixed10k_stage2_mmlu_replay_20260715T030751Z` |
| data | `/mnt/public/data/zhuxz/training_runs/mixed10k_stage2_mmlu_replay_20260715T030751Z` |
| config | `/mnt/public/code/zhuxz/training/runs/mixed10k_stage2_mmlu_replay_20260715T030751Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/mixed10k_stage2_mmlu_replay_20260715T030751Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/mixed10k_stage2_mmlu_replay_20260715T030751Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/mixed10k_stage2_mmlu_replay_20260715T030751Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
