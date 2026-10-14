# Training run `v1_40_20260611T091729Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | smoke_or_partial |
| gen | 1 |
| format | smoke |
| code | `/mnt/public/code/zhuxz/training/runs/v1_40_20260611T091729Z` |
| data | `/mnt/public/data/zhuxz/training_runs/v1_40_20260611T091729Z` |
| config | `/mnt/public/code/zhuxz/training/runs/v1_40_20260611T091729Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/v1_40_20260611T091729Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/v1_40_20260611T091729Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/v1_40_20260611T091729Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
