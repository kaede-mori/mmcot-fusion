#!/usr/bin/env bash
# E0 (Track A): evaluate the official MM-CoT base checkpoints with this
# reimplementation. Reproduces the ScienceQA test accuracy reported for
# the current official release (flan-alpaca-base + ViT + captions).
set -euo pipefail
cd "$(dirname "$0")/.."

uv run mmcot-fusion run \
  --user-msg e0-rationale \
  --evaluate-dir models/mm-cot-base-rationale \
  --orig-ckpt --img-type vit \
  --caption-file data/instruct_captions.json \
  --prompt-format QCM-E --output-len 512 \
  --eval-bs 16

uv run mmcot-fusion run \
  --user-msg e0-answer \
  --evaluate-dir models/mm-cot-base-ans \
  --orig-ckpt --img-type vit \
  --caption-file data/instruct_captions.json \
  --prompt-format QCMG-A --output-len 64 \
  --eval-bs 16 \
  --eval-le experiments/eval_e0-rationale_mm-cot-base-rationale_QCM-E/predictions_ans_eval.json \
  --test-le experiments/eval_e0-rationale_mm-cot-base-rationale_QCM-E/predictions_ans_test.json
