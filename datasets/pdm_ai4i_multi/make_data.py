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

# 멀티라벨 원인(5개)
FAIL_COL = "Machine failure"
CAUSE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NUM_LABELS = 5

CONTINUOUS_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

DROP_ID_OR_CAT = ["UDI", "Product ID", "Type"]


def save_xy_multilabel(
    save_dir: Path,
    feature_cols: list[str],
    label_cols: list[str],
    X: np.ndarray,
    Y: np.ndarray,
):
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D array, got shape={Y.shape}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Row mismatch: X={X.shape[0]} rows, Y={Y.shape[0]} rows")
    if X.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature dim mismatch: X has {X.shape[1]} cols but feature_cols has {len(feature_cols)}"
        )
    if Y.shape[1] != len(label_cols):
        raise ValueError(
            f"Label dim mismatch: Y has {Y.shape[1]} cols but label_cols has {len(label_cols)}"
        )

    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        w = csv.writer(fx)
        w.writerow(feature_cols)
        w.writerows(X.tolist())

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        w = csv.writer(fy)
        w.writerow(label_cols)  # 멀티라벨이므로 5개 컬럼 헤더
        w.writerows(Y.astype(int).tolist())


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

    # flags (라벨의 본체)
    flags_all = df[CAUSE_COLS].astype(int).to_numpy()
    sum_all = flags_all.sum(axis=1)

    # 멀티라벨 데이터에서는 '비정상'을 OR(flags)>0으로 정의
    keep_mask = sum_all > 0
    df_f = df.loc[keep_mask].reset_index(drop=True)
    if len(df_f) == 0:
        raise ValueError("No rows with any active cause flag (sum(flags) > 0).")

    # X / Y
    X = df_f[CONTINUOUS_COLS].astype(float).to_numpy()
    Y = df_f[CAUSE_COLS].astype(int).to_numpy()

    # 요약 통계(데이터가 깨진 정도 기록)
    mf_all = df[FAIL_COL].astype(int).to_numpy()
    mf_from_flags = (sum_all > 0).astype(int)
    mismatch_mf = int((mf_all != mf_from_flags).sum())

    # 멀티라벨에서 중요한 분포: sum(flags)가 1/2/3... 얼마나 있나
    sum_dist_kept = _dist_int(Y.sum(axis=1))

    # shuffle -> split
    n = len(df_f)
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

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_valid, Y_valid = X[idx_valid], Y[idx_valid]
    X_test, Y_test = X[idx_test], Y[idx_test]

    save_xy_multilabel(TRAIN_OUT, CONTINUOUS_COLS, CAUSE_COLS, X_train, Y_train)
    save_xy_multilabel(VALID_OUT, CONTINUOUS_COLS, CAUSE_COLS, X_valid, Y_valid)
    save_xy_multilabel(TEST_OUT, CONTINUOUS_COLS, CAUSE_COLS, X_test, Y_test)

    # 라벨별 양성 개수도 기록(각 원인별 imbalance 확인용)
    def pos_counts(Ym: np.ndarray) -> dict:
        return {CAUSE_COLS[i]: int(Ym[:, i].sum()) for i in range(len(CAUSE_COLS))}

    meta = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "task": "multilabel_failure_cause_detection",
        "labels": CAUSE_COLS,
        "num_labels": int(NUM_LABELS),
        "features": {
            "continuous_cols": CONTINUOUS_COLS,
            "categorical_cols": [],
            "input_dim": int(X.shape[1]),
        },
        "sample_selection": {
            "kept_condition": "OR(TWF,HDF,PWF,OSF,RNF) == 1  (i.e., sum(flags) > 0)",
            "num_samples_total_original": int(len(df)),
            "num_samples_after_filter": int(n),
        },
        "data_quality_notes": {
            "machine_failure_mismatch_count": mismatch_mf,
            "cause_sum_distribution_after_filter": sum_dist_kept,
        },
        "label_positive_counts": {
            "train": pos_counts(Y_train),
            "valid": pos_counts(Y_valid),
            "test": pos_counts(Y_test),
            "all_filtered": pos_counts(Y),
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
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"csv": str(DATA_PATH)},
    }

    with open(OUT_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
