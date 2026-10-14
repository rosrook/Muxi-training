# Training run `test3k_run`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 2 |
| format | short_answer |
| code | `/mnt/public/code/zhuxz/training/runs/test3k_run` |
| data | `/mnt/public/data/zhuxz/training_runs/test3k_run` |
| config | `/mnt/public/code/zhuxz/training/runs/test3k_run/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/test3k_run/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/test3k_run/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/test3k_run/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
