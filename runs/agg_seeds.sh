#!/bin/bash

# chmod +x runs/agg_seeds.sh

T1="outputs/2025-12-03_10-15-50_lightgbm_seed42_0.0_to_0.5_0.1_step_multi_mcar_zero"
T2="outputs/2025-12-03_10-17-12_lightgbm_seed6652_0.0_to_0.5_0.1_step_multi_mcar_zero"

python -m scripts.agg_seeds "$T1" "$T2"
