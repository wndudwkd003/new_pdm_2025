#!/bin/bash

MODE=$1
CUDA=${2:-0}    # GPU 번호 없으면 기본값 0 사용

if [ -z "$MODE" ]; then
    echo "사용법: ./run.sh <mode> [gpu]"
    exit 1
fi

echo "모드: $MODE"
echo "GPU: $CUDA"

CUDA_VISIBLE_DEVICES=$CUDA python -m scripts.run --mode "$MODE"
