#!/bin/bash
# "HYGATE"
# chmod +x runs/agg_seeds.sh

RUN_JSON="outputs_auto_run/2026-01-12_12-10-55_xgboost_0.0_to_0.8_0.1_step_multi_mcar_gain_auto.json"
TEST_MODE="0"

export MODEL="XGBOOST"

python -m scripts.agg_seeds "$RUN_JSON" "$TEST_MODE"
