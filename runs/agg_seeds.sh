#!/bin/bash

# chmod +x runs/agg_seeds.sh

RUN_JSON="outputs_auto_run/2026-01-09_06-56-01_hybrid_seed42_0.0_to_0.5_0.1_step_multi_mcar_zero_auto.json"
TEST_MODE="0"

export MODEL="HYGATE"

python -m scripts.agg_seeds "$RUN_JSON" "$TEST_MODE"
