# Training run `en_translated_letter_only_4gpu_20260729T124358Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 3 |
| format | en_letter_only |
| code | `/mnt/public/code/zhuxz/training/runs/en_translated_letter_only_4gpu_20260729T124358Z` |
| data | `/mnt/public/data/zhuxz/training_runs/en_translated_letter_only_4gpu_20260729T124358Z` |
| config | `/mnt/public/code/zhuxz/training/runs/en_translated_letter_only_4gpu_20260729T124358Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/en_translated_letter_only_4gpu_20260729T124358Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/en_translated_letter_only_4gpu_20260729T124358Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/en_translated_letter_only_4gpu_20260729T124358Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
