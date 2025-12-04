#!/bin/bash

# chmod +x runs/agg_seeds.sh

RUN_JSON="outputs_auto_run/2025-12-04_08-31-10_lightgbm_0.0_to_0.0_0.0_step_single_mcar_zero_auto.json"
TEST_MODE="0"

python -m scripts.agg_seeds "$RUN_JSON" "$TEST_MODE"
