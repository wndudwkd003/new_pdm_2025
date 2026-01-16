#!/bin/bash
# "HYGATE"
# chmod +x runs/agg_seeds.sh

RUN_JSON="outputs_auto_run/2026-01-16_12-43-23_lightgbm__mcar_median_auto.json"

TEST_MODE="0"

export MODEL="LIGHTGBM (L)"
python -m scripts.agg_seeds "$RUN_JSON" "$TEST_MODE"
