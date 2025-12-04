#!/bin/bash

GPU=$1
MODE=$2

JSON_PATH="outputs_auto_run/2025-12-04_10-08-12_lightgbm_0.0_to_0.0_0.0_step_single_mcar_mean_auto.json"

# test 모드일 때 JSON path 세 번째 인자로 넘김
if [ "$MODE" = "test" ]; then
    python -m scripts.auto_run \
        --mode test \
        --gpu $GPU \
        --json_path "$JSON_PATH"
    exit 0
fi

# train 모드일 때 seeds 정의
SEEDS=(42 234 2025 6652 45321) # (42 234 2025 6652 45321)

python -m scripts.auto_run \
    --mode train \
    --gpu $GPU \
    --seeds ${SEEDS[@]}
