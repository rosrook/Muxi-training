# Training run `v1_remote_1000_02`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 1 |
| format | short_answer |
| code | `/mnt/public/code/zhuxz/training/runs/v1_remote_1000_02` |
| data | `/mnt/public/data/zhuxz/training_runs/v1_remote_1000_02` |
| config | `/mnt/public/code/zhuxz/training/runs/v1_remote_1000_02/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/v1_remote_1000_02/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/v1_remote_1000_02/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/v1_remote_1000_02/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
