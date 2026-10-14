# Training run `en_translated_answer_first_think_4gpu_20260731T064622Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | failed_launch |
| gen | 3 |
| format | en_answer_first_think |
| code | `/mnt/public/code/zhuxz/training/runs/en_translated_answer_first_think_4gpu_20260731T064622Z` |
| data | `/mnt/public/data/zhuxz/training_runs/en_translated_answer_first_think_4gpu_20260731T064622Z` |
| config | `/mnt/public/code/zhuxz/training/runs/en_translated_answer_first_think_4gpu_20260731T064622Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/en_translated_answer_first_think_4gpu_20260731T064622Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/en_translated_answer_first_think_4gpu_20260731T064622Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/en_translated_answer_first_think_4gpu_20260731T064622Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
