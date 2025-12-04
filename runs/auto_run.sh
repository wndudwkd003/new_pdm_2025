#!/bin/bash

GPU="0 1 2"
MODE="test"  # train or test

JSON_PATH="outputs_auto_run/2025-12-04_11-45-10_mlp_0.0_to_0.0_0.0_step_single_mcar_zero_auto.json"

# test 모드일 때 JSON path 세 번째 인자로 넘김
if [ "$MODE" = "test" ]; then
    python -m scripts.auto_run \
        --mode test \
        --gpus $GPU \
        --json_path "$JSON_PATH"
    exit 0
fi

# train 모드일 때 seeds 정의
SEEDS=(42 234 2025 6652 45321) # (42 234 2025 6652 45321)

python -m scripts.auto_run \
    --mode train \
    --gpus $GPU \
    --seeds ${SEEDS[@]}
