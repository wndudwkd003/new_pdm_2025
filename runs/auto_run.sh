#!/bin/bash

GPU="0 1 2"
MODE="train"  # train or test

JSON_PATH="outputs_auto_run/2025-12-31_05-49-32_saint_0.0_to_0.5_0.1_step_multi_mcar_zero_auto.json"

# test 모드일 때 JSON path 세 번째 인자로 넘김
if [ "$MODE" = "test" ]; then
    python -m scripts.auto_run \
        --mode test \
        --gpus $GPU \
        --json_path "$JSON_PATH"
    exit 0
fi

# train 모드일 때 seeds 정의 75342
SEEDS=(42 567 2025 6652 87654)

python -m scripts.auto_run \
    --mode train \
    --gpus $GPU \
    --seeds ${SEEDS[@]}
