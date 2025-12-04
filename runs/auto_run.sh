#!/bin/bash

GPU="0 1 2"
MODE="train"  # train or test

JSON_PATH=""

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
