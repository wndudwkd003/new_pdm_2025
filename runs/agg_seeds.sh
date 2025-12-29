#!/bin/bash

# chmod +x runs/agg_seeds.sh

RUN_JSON="outputs_auto_run/2025-12-28_15-27-52_regvae_0.0_to_0.5_0.1_step_multi_mcar_zero_auto.json"
TEST_MODE="0"

export MODEL="TabPFN"

python -m scripts.agg_seeds "$RUN_JSON" "$TEST_MODE"
