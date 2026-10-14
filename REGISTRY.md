# Training REGISTRY

Updated: 2026-08-03T07:49:43.693105+00:00

Self-contained packages live under `code/zhuxz/training/runs/<run_id>/`;
artifacts under `data/zhuxz/training_runs/<run_id>/`.

| gen | run_id | status | format | code | data | sft_link | ckpt_link |
|-----|--------|--------|--------|------|------|----------|-----------|
| 1 | `v1_remote_1000_02` | completed | short_answer | `training/runs/v1_remote_1000_02` | `training_runs/v1_remote_1000_02` | yes | yes |
| 1 | `v1_remote_3000_01` | completed | short_answer | `training/runs/v1_remote_3000_01` | `training_runs/v1_remote_3000_01` | yes | yes |
| 2 | `test3k_run` | completed | short_answer | `training/runs/test3k_run` | `training_runs/test3k_run` | yes | yes |
| 2 | `mixed10k_backtest_20260713T041155Z` | completed | short_answer_only | `training/runs/mixed10k_backtest_20260713T041155Z` | `training_runs/mixed10k_backtest_20260713T041155Z` | yes | yes |
| 2 | `mixed10k_fulltrain_20260714T071006Z` | completed | short_answer_only | `training/runs/mixed10k_fulltrain_20260714T071006Z` | `training_runs/mixed10k_fulltrain_20260714T071006Z` | yes | yes |
| 2 | `mixed10k_gsm8k_cot_20260714T114313Z` | completed | cot_hash_gsm8k | `training/runs/mixed10k_gsm8k_cot_20260714T114313Z` | `training_runs/mixed10k_gsm8k_cot_20260714T114313Z` | yes | yes |
| 2 | `mixed10k_stage2_mmlu_replay_20260715T030751Z` | completed | stage2_mmlu_cot_replay | `training/runs/mixed10k_stage2_mmlu_replay_20260715T030751Z` | `training_runs/mixed10k_stage2_mmlu_replay_20260715T030751Z` | yes | yes |
| 2 | `fulltrain_vb_4gpu_20260716T064800Z` | completed | letter_only_vb | `training/runs/fulltrain_vb_4gpu_20260716T064800Z` | `training_runs/fulltrain_vb_4gpu_20260716T064800Z` | yes | yes |
| 3 | `earlytrain_v5_cot_20260728T022500Z` | completed | cot_rationale_v1_zh | `training/runs/earlytrain_v5_cot_20260728T022500Z` | `training_runs/earlytrain_v5_cot_20260728T022500Z` | yes | yes |
| 3 | `earlytrain_v5_scale_9k_4gpu_20260728T065935Z` | completed | cot_rationale_v1_zh_scale | `training/runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z` | `training_runs/earlytrain_v5_scale_9k_4gpu_20260728T065935Z` | yes | yes |
| 3 | `earlytrain_v5_scale_6k_4gpu_20260728T103731Z` | completed | cot_rationale_v1_zh_scale | `training/runs/earlytrain_v5_scale_6k_4gpu_20260728T103731Z` | `training_runs/earlytrain_v5_scale_6k_4gpu_20260728T103731Z` | yes | yes |
| 3 | `earlytrain_v5_scale_3k_4gpu_20260728T172041Z` | completed | cot_rationale_v1_zh_scale | `training/runs/earlytrain_v5_scale_3k_4gpu_20260728T172041Z` | `training_runs/earlytrain_v5_scale_3k_4gpu_20260728T172041Z` | yes | yes |
| 3 | `earlytrain_v5_scale_1k_4gpu_20260729T063523Z` | completed | cot_rationale_v1_zh_scale | `training/runs/earlytrain_v5_scale_1k_4gpu_20260729T063523Z` | `training_runs/earlytrain_v5_scale_1k_4gpu_20260729T063523Z` | yes | yes |
| 3 | `en_translated_cot_4gpu_20260729T124200Z` | completed | en_cot_hash | `training/runs/en_translated_cot_4gpu_20260729T124200Z` | `training_runs/en_translated_cot_4gpu_20260729T124200Z` | yes | yes |
| 3 | `en_translated_letter_only_4gpu_20260729T124358Z` | completed | en_letter_only | `training/runs/en_translated_letter_only_4gpu_20260729T124358Z` | `training_runs/en_translated_letter_only_4gpu_20260729T124358Z` | yes | yes |
| 3 | `en_translated_answer_first_think_4gpu_20260731T064450Z` | failed_launch | en_answer_first_think | `training/runs/en_translated_answer_first_think_4gpu_20260731T064450Z` | `training_runs/en_translated_answer_first_think_4gpu_20260731T064450Z` | yes | yes |
| 3 | `en_translated_answer_first_think_4gpu_20260731T064622Z` | failed_launch | en_answer_first_think | `training/runs/en_translated_answer_first_think_4gpu_20260731T064622Z` | `training_runs/en_translated_answer_first_think_4gpu_20260731T064622Z` | yes | yes |
| 3 | `en_translated_think_first_final_4gpu_20260731T081559Z` | completed | en_think_first_final | `training/runs/en_translated_think_first_final_4gpu_20260731T081559Z` | `training_runs/en_translated_think_first_final_4gpu_20260731T081559Z` | yes | yes |
| 1 | `v1_40_20260611T085454Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260611T085454Z` | `training_runs/v1_40_20260611T085454Z` | no | yes |
| 1 | `v1_40_20260611T091448Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260611T091448Z` | `training_runs/v1_40_20260611T091448Z` | no | yes |
| 1 | `v1_40_20260611T091729Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260611T091729Z` | `training_runs/v1_40_20260611T091729Z` | no | yes |
| 1 | `v1_40_20260611T122728Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260611T122728Z` | `training_runs/v1_40_20260611T122728Z` | no | yes |
| 1 | `v1_40_20260611T123044Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260611T123044Z` | `training_runs/v1_40_20260611T123044Z` | no | yes |
| 1 | `v1_40_20260612T023109Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260612T023109Z` | `training_runs/v1_40_20260612T023109Z` | no | yes |
| 1 | `v1_40_20260612T023601Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260612T023601Z` | `training_runs/v1_40_20260612T023601Z` | no | yes |
| 1 | `v1_40_20260612T070030Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260612T070030Z` | `training_runs/v1_40_20260612T070030Z` | no | yes |
| 1 | `v1_40_20260612T070136Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260612T070136Z` | `training_runs/v1_40_20260612T070136Z` | no | yes |
| 1 | `v1_40_20260612T090300Z` | smoke_or_partial | smoke | `training/runs/v1_40_20260612T090300Z` | `training_runs/v1_40_20260612T090300Z` | no | yes |

| 3 | `think_first_law_business_v1_4gpu` | ready_to_train | en_think_first_law_business | `training/runs/think_first_law_business_v1_4gpu` | `training_runs/think_first_law_business_v1_4gpu` | yes | pending |

## Create a new independent run

```bash
bash /mnt/public/code/zhuxz/training/tools/create_run.sh \
  --run-id my_run_$(date -u +%Y%m%dT%H%M%SZ) \
  --sft /path/to/sft_dataset.jsonl \
  --config /path/to/config.yaml
```

