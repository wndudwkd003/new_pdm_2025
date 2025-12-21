
TEST="0"

RUN_DIR="outputs_seeds/2025-12-20_13-42-29_tabpfn_0.0_to_0.5_0.1_step_multi_mcar_zero"

MODEL="TabPFN"


FINAL_DIR="$RUN_DIR/test_$TEST"



python3 scripts/latext_py.py --run_dir "$FINAL_DIR" --model "$MODEL"
