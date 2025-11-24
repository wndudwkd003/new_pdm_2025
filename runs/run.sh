#!/bin/bash


CUDA=$1
MODE=$2

echo "GPU: $CUDA"
echo "모드: $MODE"

CUDA_VISIBLE_DEVICES=$CUDA python -m scripts.run --mode "$MODE"
