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

LABEL_COL = "Machine failure"
NUM_CLASS = 2

# 연속형만 사용
CONTINUOUS_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# 식별자/범주형(참고용)
DROP_ID_OR_CAT = ["UDI", "Product ID", "Type"]

# 누수 컬럼: Machine failure를 구성하는 모드 플래그들 (X에서 제외)
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def save_xy(save_dir: Path, feature_cols: list[str], X: np.ndarray, y: np.ndarray):
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


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {DATA_PATH}")

    if abs((TRAIN_RATIO + VALID_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN/VALID/TEST ratios must sum to 1.0")

    df = pd.read_csv(DATA_PATH)

    # 컬럼 검증
    required_cols = [LABEL_COL] + CONTINUOUS_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) != 0:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # label: 0/1
    y = df[LABEL_COL].astype(int).to_numpy()
    uniq_y = np.unique(y)
    if not set(uniq_y.tolist()).issubset({0, 1}):
        raise ValueError(f"Unexpected label values: {sorted(uniq_y.tolist())}")

    # X: 연속형 5개만
    X = df[CONTINUOUS_COLS].astype(float).to_numpy()
    feature_cols = CONTINUOUS_COLS

    # ─────────────────────────────────────────────
    # shuffle -> split (seed=42)
    # ─────────────────────────────────────────────
    n = len(df)
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

    save_xy(TRAIN_OUT, feature_cols, X_train, y_train)
    save_xy(VALID_OUT, feature_cols, X_valid, y_valid)
    save_xy(TEST_OUT, feature_cols, X_test, y_test)

    meta = {
        "dataset": "AI4I 2020 Predictive Maintenance",
        "task": "binary_classification",
        "label": LABEL_COL,
        "num_class": int(NUM_CLASS),
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": int(X.shape[1]),
        "num_samples_total": int(n),
        "num_samples": {
            "train": int(X_train.shape[0]),
            "valid": int(X_valid.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "split": {
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
            "method": "shuffle_then_slice",
        },
        "dropped_cols": {
            "identifiers_or_categorical": DROP_ID_OR_CAT,
            "leakage_cols": LEAK_COLS,
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"csv": str(DATA_PATH)},
    }

    with open(OUT_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
