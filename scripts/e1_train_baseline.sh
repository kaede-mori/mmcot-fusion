#!/usr/bin/env bash
# E1 (Track B): retrain the baseline (original sigmoid_1h fusion) under the
# reduced compute budget, matching the official recipe otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${SEED:-42}"
EPOCH="${EPOCH:-3}"
FUSION="${FUSION:-sigmoid_1h}"
TAG="${TAG:-e1}"

uv run mmcot-fusion run \
  --user-msg ${TAG}-rationale --fusion "$FUSION" \
  --model declare-lab/flan-alpaca-base --img-type vit \
  --caption-file data/instruct_captions.json \
  --prompt-format QCM-E --output-len 512 --lr 8e-5 --skip-val-generation \
  --epoch "$EPOCH" --seed "$SEED" --bs 2 --grad-accum 4 --gradient-checkpointing --eval-bs 16

RAT_DIR=$(ls -dt experiments/${TAG}-rationale_*seed${SEED} | head -1)

uv run mmcot-fusion run \
  --user-msg ${TAG}-answer --fusion "$FUSION" \
  --model declare-lab/flan-alpaca-base --img-type vit \
  --caption-file data/instruct_captions.json \
  --prompt-format QCMG-A --output-len 64 --lr 8e-5 \
  --epoch "$EPOCH" --seed "$SEED" --bs 2 --grad-accum 4 --gradient-checkpointing --eval-bs 16 \
  --test-le "$RAT_DIR/predictions_ans_test.json"
