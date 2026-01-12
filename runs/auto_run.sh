#!/bin/bash

GPU="1 2 3"
MODE="test"  # train or test

JSON_PATH="outputs_auto_run/2026-01-12_12-33-37_xgboost_0.0_to_0.8_0.1_step_multi_mcar_gain_auto.json"

# test 모드일 때 JSON path 세 번째 인자로 넘김
if [ "$MODE" = "test" ]; then
    python -m scripts.auto_run \
        --mode test \
        --gpus $GPU \
        --json_path "$JSON_PATH"
    exit 0
fi

# (9851454 56547867 69897412 96799887 4311324 6035877)
# (42 567 2025 6652 87654)

SEEDS=(9851454 56547867 69897412 96799887 4311324 6035877)

python -m scripts.auto_run \
    --mode train \
    --gpus $GPU \
    --seeds ${SEEDS[@]}
