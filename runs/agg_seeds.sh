#!/bin/bash

# chmod +x runs/agg_seeds.sh

T1="outputs/2025-12-03_09-04-06_xgboost_seed42_0.0_to_0.5_0.1_step_multi_mcar_zero"
T2="outputs/2025-12-03_09-35-22_xgboost_seed6652_0.0_to_0.5_0.1_step_multi_mcar_zero"

echo "$T1"
echo "$T2"

python -m scripts.agg_seeds "$T1" "$T2"
