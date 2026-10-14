# Training run `fulltrain_vb_4gpu_20260716T064800Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 2 |
| format | letter_only_vb |
| code | `/mnt/public/code/zhuxz/training/runs/fulltrain_vb_4gpu_20260716T064800Z` |
| data | `/mnt/public/data/zhuxz/training_runs/fulltrain_vb_4gpu_20260716T064800Z` |
| config | `/mnt/public/code/zhuxz/training/runs/fulltrain_vb_4gpu_20260716T064800Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/fulltrain_vb_4gpu_20260716T064800Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/fulltrain_vb_4gpu_20260716T064800Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/fulltrain_vb_4gpu_20260716T064800Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
