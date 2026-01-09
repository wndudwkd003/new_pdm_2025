import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_NAME = "IFDD"

DATASET_DIR = Path(f"/ws/new_pdm_2025/datasets/{DATA_NAME}")
RAW_PATH = DATASET_DIR / "raw" / "Industrial_fault_detection.csv"

SAVE_ROOT = DATASET_DIR / "processed_data"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

LABEL_COL = "Fault_Type"


def split_indices_stratified(labels: np.ndarray):
    if abs((TRAIN_RATIO + VALID_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VALID_RATIO + TEST_RATIO must be 1.0")

    rng = np.random.RandomState(RANDOM_SEED)

    idxs_by_class = {}
    for i, y in enumerate(labels.tolist()):
        idxs_by_class.setdefault(int(y), []).append(i)

    train_idx = []
    valid_idx = []
    test_idx = []

    for c, idxs in idxs_by_class.items():
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)

        c_train = idxs[:n_train]
        c_valid = idxs[n_train : n_train + n_valid]
        c_test = idxs[n_train + n_valid :]

        if len(c_train) == 0 or len(c_valid) == 0 or len(c_test) == 0:
            raise ValueError(
                f"Class {c} split empty: train={len(c_train)}, valid={len(c_valid)}, test={len(c_test)}"
            )

        train_idx.extend(c_train)
        valid_idx.extend(c_valid)
        test_idx.extend(c_test)

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)

    return train_idx, valid_idx, test_idx


def save_xy(save_dir: Path, feature_cols, X: np.ndarray, y: np.ndarray):
    save_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved X: {X.shape[0]} rows -> {x_path}")
    print(f"Saved y: {y.shape[0]} rows -> {y_path}")


def main():
    if RAW_PATH.exists() is False:
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column not found: {LABEL_COL} in {RAW_PATH}")

    # 결측 체크(원하면 여기서 바로 에러)
    if df.isna().any().any():
        na_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {na_cols}")

    # feature / label
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")

    X = df[feature_cols].to_numpy(dtype=np.float64)

    # 라벨은 int로 강제
    y_raw = df[LABEL_COL].to_numpy()
    y = y_raw.astype(np.int64)

    uniq = sorted(set(y.tolist()))
    if uniq != [0, 1, 2, 3]:
        # Kaggle 설명은 0~3인데, 실제 파일이 다르면 여기서 바로 잡아야 함
        raise ValueError(f"Unexpected label set: {uniq} (expected [0,1,2,3])")

    num_class = len(uniq)
    task = "binary_classification" if num_class == 2 else "multiclass_classification"

    # stratified split
    random.seed(RANDOM_SEED)
    train_idx, valid_idx, test_idx = split_indices_stratified(y)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    save_xy(SAVE_ROOT / "train", feature_cols, X_train, y_train)
    save_xy(SAVE_ROOT / "valid", feature_cols, X_valid, y_valid)
    save_xy(SAVE_ROOT / "test", feature_cols, X_test, y_test)

    meta = {
        "dataset": DATA_NAME,
        "task": task,
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": len(feature_cols),
        "num_class": num_class,
        "num_samples": {
            "train": int(X_train.shape[0]),
            "valid": int(X_valid.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_path": str(RAW_PATH),
        "label_col": LABEL_COL,
        "label_map": {
            "0": "No Fault (Normal Operation)",
            "1": "Overheating Fault",
            "2": "Leakage Fault",
            "3": "Power Fluctuation Fault",
        },
        "split": {
            "type": "stratified by label",
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
        },
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Meta saved:", SAVE_ROOT / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
