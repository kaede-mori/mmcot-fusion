#!/usr/bin/env bash
# E4 (ablation): 8-head attention with the original sigmoid gate, isolating
# the contribution of head count from gate design.
set -euo pipefail
cd "$(dirname "$0")/.."
FUSION=sigmoid_mh TAG=e4 SEED="${SEED:-42}" EPOCH="${EPOCH:-3}" bash "$(dirname "$0")/e1_train_baseline.sh"
