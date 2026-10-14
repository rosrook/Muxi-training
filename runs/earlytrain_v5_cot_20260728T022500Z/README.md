# Training run `earlytrain_v5_cot_20260728T022500Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 3 |
| format | cot_rationale_v1_zh |
| code | `/mnt/public/code/zhuxz/training/runs/earlytrain_v5_cot_20260728T022500Z` |
| data | `/mnt/public/data/zhuxz/training_runs/earlytrain_v5_cot_20260728T022500Z` |
| config | `/mnt/public/code/zhuxz/training/runs/earlytrain_v5_cot_20260728T022500Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/earlytrain_v5_cot_20260728T022500Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/earlytrain_v5_cot_20260728T022500Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/earlytrain_v5_cot_20260728T022500Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
