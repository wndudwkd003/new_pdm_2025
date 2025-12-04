#!/bin/bash

CUDA=$1
MODE=$2
SEED=$3
LOAD_DIR=$4

echo "GPU: $CUDA"
echo "모드: $MODE"

# ============================
# TEST 모드
# ============================
if [ "$MODE" = "test" ]; then
    echo "시드: ${SEED:-없음}"
    echo "로드 디렉토리: ${LOAD_DIR:-없음}"

    # 옵션 배열 만들기 (있는 것만 추가)
    ARGS=( "--mode" "test" )

    if [ -n "$SEED" ]; then
        ARGS+=( "--seed" "$SEED" )
    fi

    if [ -n "$LOAD_DIR" ]; then
        ARGS+=( "--load_dir" "$LOAD_DIR" )
    fi

    CUDA_VISIBLE_DEVICES=$CUDA python -m scripts.run "${ARGS[@]}"
    exit 0
fi

# ============================
# TRAIN 모드
# ============================
if [ "$MODE" = "train" ]; then
    echo "시드: ${SEED:-없음}"

    ARGS=( "--mode" "train" )

    if [ -n "$SEED" ]; then
        ARGS+=( "--seed" "$SEED" )
    fi

    CUDA_VISIBLE_DEVICES=$CUDA python -m scripts.run "${ARGS[@]}"
    exit 0
fi

echo "[ERROR] 지원하지 않는 MODE: $MODE"
exit 1
