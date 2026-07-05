#!/usr/bin/env bash
# E2 (Track B): train the proposed tanh-gated multi-head fusion under the
# identical budget as E1.
set -euo pipefail
cd "$(dirname "$0")/.."
FUSION=tanh_mh TAG=e2 SEED="${SEED:-42}" EPOCH="${EPOCH:-3}" bash "$(dirname "$0")/e1_train_baseline.sh"
