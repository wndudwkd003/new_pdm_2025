import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────
# Root / Path
# ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

OUT_ROOT = ROOT / "processed_data"
TRAIN_OUT = OUT_ROOT / "train"
VALID_OUT = OUT_ROOT / "valid"
TEST_OUT = OUT_ROOT / "test"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 원인(5개): 정확히 하나만 1인 경우만 클래스 부여, 다중이면 drop
FAIL_COL = "Machine failure"
CAUSE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# 입력 특성(연속)
CONTINUOUS_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

DROP_ID_OR_CAT = ["UDI", "Product ID", "Type"]

# 클래스: 0=No_Cause(정상), 1..5 = 각 원인 단일 선택
NUM_CLASS = 1 + len(CAUSE_COLS)  # 6
TASK = "multiclass_classification"


def save_xy_singlelabel(
    save_dir: Path,
    feature_cols: list[str],
    X: np.ndarray,
    y: np.ndarray,
):
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X={X.shape[0]} rows, y={y.shape[0]} rows")
    if X.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature dim mismatch: X has {X.shape[1]} cols but feature_cols has {len(feature_cols)}"
        )

    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        w = csv.writer(fx)
        w.writerow(feature_cols)
        w.writerows(X.tolist())

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        w = csv.writer(fy)
        w.writerow(["label"])
        w.writerows([[int(v)] for v in y.tolist()])


def _dist_int(arr: np.ndarray) -> dict:
    vals, cnts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    if abs((TRAIN_RATIO + VALID_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN/VALID/TEST ratios must sum to 1.0")

    df = pd.read_csv(DATA_PATH)

    required_cols = [FAIL_COL] + CONTINUOUS_COLS + CAUSE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) != 0:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 결측 검증: 있으면 즉시 에러
    if df[required_cols].isna().any().any():
        na_cols = df[required_cols].columns[df[required_cols].isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {na_cols}")

    # 원인 플래그
    flags_all = df[CAUSE_COLS].astype(int).to_numpy()
    sum_all = flags_all.sum(axis=1)

    # keep 조건:
    # - sum==0 : 정상(라벨 0)
    # - sum==1 : 단일 원인(라벨 1..5)
    # - sum>=2 : 제거
    keep_mask = (sum_all == 0) | (sum_all == 1)
    df_f = df.loc[keep_mask].reset_index(drop=True)
    flags_f = df_f[CAUSE_COLS].astype(int).to_numpy()
    sum_f = flags_f.sum(axis=1)

    dropped_multi = int((sum_all >= 2).sum())
    kept_total = int(len(df_f))
    if kept_total == 0:
        raise ValueError("No rows left after filtering (sum(flags) in {0,1}).")

    # X
    X = df_f[CONTINUOUS_COLS].astype(float).to_numpy()

    # y 생성:
    # - sum==0 -> 0
    # - sum==1 -> argmax(flags)+1  (CAUSE_COLS 순서 기준)
    y = np.zeros((kept_total,), dtype=np.int64)
    one_mask = sum_f == 1
    if int(one_mask.sum()) > 0:
        y[one_mask] = flags_f[one_mask].argmax(axis=1).astype(np.int64) + 1

    # 데이터 품질: Machine failure vs flags OR 불일치 기록
    mf_all = df[FAIL_COL].astype(int).to_numpy()
    mf_from_flags_or = (sum_all > 0).astype(int)
    mismatch_mf = int((mf_all != mf_from_flags_or).sum())

    # 분포
    y_dist = _dist_int(y)
    sum_dist_kept = _dist_int(sum_f.astype(np.int64))

    # shuffle -> split
    n = kept_total
    idx = np.arange(n)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(idx)

    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    n_test = n - n_train - n_valid

    if n_train <= 0 or n_valid <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, valid={n_valid}, test={n_test}"
        )

    idx_train = idx[:n_train]
    idx_valid = idx[n_train : n_train + n_valid]
    idx_test = idx[n_train + n_valid :]

    X_train, y_train = X[idx_train], y[idx_train]
    X_valid, y_valid = X[idx_valid], y[idx_valid]
    X_test, y_test = X[idx_test], y[idx_test]

    save_xy_singlelabel(TRAIN_OUT, CONTINUOUS_COLS, X_train, y_train)
    save_xy_singlelabel(VALID_OUT, CONTINUOUS_COLS, X_valid, y_valid)
    save_xy_singlelabel(TEST_OUT, CONTINUOUS_COLS, X_test, y_test)

    # label map: 0=No_Cause, 1..5=각 원인
    label_map = {"0": "No_Cause"}
    for i, c in enumerate(CAUSE_COLS, start=1):
        label_map[str(i)] = c

    meta = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "task": TASK,
        "num_class": int(NUM_CLASS),
        "features": {
            "continuous_cols": CONTINUOUS_COLS,
            "categorical_cols": [],
            "input_dim": int(X.shape[1]),
        },
        "labeling": {
            "rule": "if sum(TWF,HDF,PWF,OSF,RNF)==0 -> label 0; if exactly one==1 -> label=(argmax+1) in CAUSE_COLS order; if sum>=2 -> dropped",
            "cause_cols_order": CAUSE_COLS,
            "label_map": label_map,
            "dropped_multi_cause_rows": int(dropped_multi),
            "kept_sum_flags_distribution": sum_dist_kept,
            "kept_label_distribution": y_dist,
        },
        "data_quality_notes": {
            "machine_failure_mismatch_count_vs_OR(flags)": int(mismatch_mf),
        },
        "split": {
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
            "method": "shuffle_then_slice",
        },
        "dropped_cols": {
            "identifiers_or_categorical": DROP_ID_OR_CAT,
            "binary_failure_col": FAIL_COL,
        },
        "num_samples": {
            "train": int(len(X_train)),
            "valid": int(len(X_valid)),
            "test": int(len(X_test)),
            "after_filter": int(n),
            "original": int(len(df)),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"csv": str(DATA_PATH)},
    }

    with open(OUT_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
