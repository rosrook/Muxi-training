# Training run `mixed10k_gsm8k_cot_20260714T114313Z`

Migrated into the self-contained training layout.

| | |
|--|--|
| status | completed |
| gen | 2 |
| format | cot_hash_gsm8k |
| code | `/mnt/public/code/zhuxz/training/runs/mixed10k_gsm8k_cot_20260714T114313Z` |
| data | `/mnt/public/data/zhuxz/training_runs/mixed10k_gsm8k_cot_20260714T114313Z` |
| config | `/mnt/public/code/zhuxz/training/runs/mixed10k_gsm8k_cot_20260714T114313Z/config.yaml` (local **copy**) |
| sft | `/mnt/public/data/zhuxz/training_runs/mixed10k_gsm8k_cot_20260714T114313Z/sft_dataset.jsonl` (symlink) |
| checkpoints | `/mnt/public/data/zhuxz/training_runs/mixed10k_gsm8k_cot_20260714T114313Z/checkpoints` (symlink → legacy) |

## Re-launch (fresh train into current data dirs — careful)

```bash
# Only if you intend to train again; may overwrite linked dirs if not careful.
bash /mnt/public/code/zhuxz/training/runs/mixed10k_gsm8k_cot_20260714T114313Z/run.sh
```

For historical weights, use `checkpoints/final` under the data dir (symlink).
