#!/bin/bash

GPU="1 2 3"
MODE="test"  # train or test

JSON_PATH="outputs_auto_run/2026-01-16_12-43-23_lightgbm__mcar_median_auto.json"

# test 모드일 때 JSON path 세 번째 인자로 넘김
if [ "$MODE" = "test" ]; then
    python -m scripts.auto_run \
        --mode test \
        --gpus $GPU \
        --json_path "$JSON_PATH"
    exit 0
fi


SEEDS=(69897412 6652 676423 100992 7875544 2025)

python -m scripts.auto_run \
    --mode train \
    --gpus $GPU \
    --seeds ${SEEDS[@]}
