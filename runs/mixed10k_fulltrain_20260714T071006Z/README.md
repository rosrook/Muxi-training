# Training run `mixed10k_fulltrain_20260714T071006Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 2 |
| format | short_answer_only |
| code | `/mnt/public/code/zhuxz/training/runs/mixed10k_fulltrain_20260714T071006Z` |
| data | `/mnt/public/data/zhuxz/training_runs/mixed10k_fulltrain_20260714T071006Z` |
| config | `/mnt/public/code/zhuxz/training/runs/mixed10k_fulltrain_20260714T071006Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/mixed10k_fulltrain_20260714T071006Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/mixed10k_fulltrain_20260714T071006Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/mixed10k_fulltrain_20260714T071006Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
