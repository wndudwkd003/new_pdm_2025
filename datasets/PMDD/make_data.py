import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────
DATASET_DIR = Path("/ws/new_pdm_2025/datasets/PMDD")
RAW_PATH = DATASET_DIR / "raw" / "manufacturing_defect_dataset.csv"

OUT_ROOT = DATASET_DIR / "processed_data"
TRAIN_OUT = OUT_ROOT / "train"
VALID_OUT = OUT_ROOT / "valid"
TEST_OUT = OUT_ROOT / "test"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

LABEL_COL = "DefectStatus"
DROP_FEATURE_CANDIDATES = ["ProductionVolume", "ProductionVolum"]  # 오타 가능성 포함


def split_indices_stratified(y: np.ndarray):
    rng = np.random.RandomState(RANDOM_SEED)

    idxs0 = np.where(y == 0)[0].tolist()
    idxs1 = np.where(y == 1)[0].tolist()

    rng.shuffle(idxs0)
    rng.shuffle(idxs1)

    def split_one(idxs):
        n = len(idxs)
        n_train = int(n * TRAIN_RATIO)
        n_valid = int(n * VALID_RATIO)
        train = idxs[:n_train]
        valid = idxs[n_train : n_train + n_valid]
        test = idxs[n_train + n_valid :]
        return train, valid, test

    tr0, va0, te0 = split_one(idxs0)
    tr1, va1, te1 = split_one(idxs1)

    train_idx = tr0 + tr1
    valid_idx = va0 + va1
    test_idx = te0 + te1

    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)

    return train_idx, valid_idx, test_idx


def save_xy(save_dir: Path, feature_cols, X: np.ndarray, y: np.ndarray):
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


def dist(y: np.ndarray):
    vals, cnts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}


def main():
    df = pd.read_csv(RAW_PATH)

    # 라벨
    y = df[LABEL_COL].astype(int).to_numpy()
    uniq = sorted(set(y.tolist()))
    if uniq != [0, 1]:
        raise ValueError(f"Unexpected label set: {uniq} (expected [0,1])")

    # 피처 컬럼 결정
    drop_cols = [LABEL_COL]
    for c in DROP_FEATURE_CANDIDATES:
        if c in df.columns:
            drop_cols.append(c)

    feature_cols = [c for c in df.columns if c not in drop_cols]

    # X
    X = df[feature_cols].astype(float).to_numpy()

    # stratified split
    train_idx, valid_idx, test_idx = split_indices_stratified(y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    save_xy(TRAIN_OUT, feature_cols, X_train, y_train)
    save_xy(VALID_OUT, feature_cols, X_valid, y_valid)
    save_xy(TEST_OUT, feature_cols, X_test, y_test)

    meta = {
        "dataset": "PMDD",
        "task": "binary_classification",
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": int(len(feature_cols)),
        "num_class": 2,
        "label_col": LABEL_COL,
        "label_map": {"0": "Low Defects", "1": "High Defects"},
        "dropped_feature_cols": [c for c in DROP_FEATURE_CANDIDATES if c in df.columns],
        "num_samples": {
            "train": int(len(train_idx)),
            "valid": int(len(valid_idx)),
            "test": int(len(test_idx)),
            "total": int(len(df)),
        },
        "label_distribution_total": dist(y),
        "label_distribution": {
            "train": dist(y_train),
            "valid": dist(y_valid),
            "test": dist(y_test),
        },
        "split": {
            "type": "stratified by label",
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_path": str(RAW_PATH),
    }

    with open(OUT_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Meta saved:", OUT_ROOT / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
