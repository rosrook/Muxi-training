# Training run `earlytrain_v5_scale_9k_4gpu_20260728T065935Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 3 |
| format | cot_rationale_v1_zh_scale |
| code | `/mnt/public/code/zhuxz/training/runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z` |
| data | `/mnt/public/data/zhuxz/training_runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z` |
| config | `/mnt/public/code/zhuxz/training/runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
