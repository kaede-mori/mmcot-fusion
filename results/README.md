# Result artifacts

Raw evaluation artifacts for every experiment reported in the paper, sufficient
to recompute all tables. Each `stage2/predictions_ans_test.json` contains the
per-question predictions, the per-question correctness map
(`per_question_correct`), and the aggregated accuracy breakdown
(`scores.answer`); training runs also include `train_log.json` (loss history)
and `run_config.json` (exact configuration).

| Directory | Paper reference | Headline number |
|---|---|---|
| `checkpoint_eval/` | Table 2, "released checkpoint + our code" | 85.95 |
| `scienceqa/baseline_sigmoid_1h/` | Table 2 "retrained, official recipe"; Tables 3-5 baseline | 85.48 |
| `scienceqa/tanh_mh/` | Table 3 | 85.43 |
| `scienceqa/sigmoid_mh/` | Table 3 | 85.92 |
| `scienceqa/tanh_1h/` | Table 3 | 85.26 |
| `scienceqa/distilled_gpt4o/` | Table 4 | 78.76 |
| `scienceqa/image_blind/` | Sec. "image-blind evaluation" | 82.88 |
| `m3cot_zeroshot/*` | Table 5 (zero-shot rows) | 33.26 / 31.32 / 30.93 |
| `m3cot_finetune/*` | Table 5 (fine-tuned rows) | 44.22 / 45.04 / 44.26 |

Recompute any pairwise comparison (accuracy breakdown + McNemar test):

```bash
uv run python scripts/aggregate_results.py \
  base=results/scienceqa/baseline_sigmoid_1h/stage2 \
  tanh_mh=results/scienceqa/tanh_mh/stage2
```

The `labels` fields reproduce ScienceQA / M3CoT reference text and are
therefore covered by those datasets' licenses (CC BY-NC-SA 4.0 for ScienceQA);
they are included solely to make the evaluation verifiable.
