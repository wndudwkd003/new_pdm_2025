import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_NAME = "CBM"

DATASET_DIR = Path(f"/ws/new_pdm_2025/datasets/{DATA_NAME}")
RAW_PATH = DATASET_DIR / "raw" / "data.txt"
FEATURES_PATH = DATASET_DIR / "raw" / "Features.txt"

SAVE_ROOT = DATASET_DIR / "processed_data"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

X_COLS = 16  # 1~16
Y_COEF_COLS = 2  # 17~18
TOTAL_COLS = 18

NUM_CLASS = 4
TASK = "multiclass_classification"


# ─────────────────────────────────────────────
# 로더
# ─────────────────────────────────────────────
def read_feature_names(path: Path):
    if path.exists() is False:
        raise FileNotFoundError(f"Features file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    names = [ln.strip() for ln in lines if ln.strip() != ""]
    if len(names) == 0:
        raise ValueError(f"No feature names in {path}")

    # Features.txt가 18개(16+2)인지 확인
    if len(names) != TOTAL_COLS:
        raise ValueError(
            f"Expected {TOTAL_COLS} names in Features.txt, got {len(names)}"
        )

    return names


def read_whitespace_table(path: Path) -> np.ndarray:
    if path.exists() is False:
        raise FileNotFoundError(f"Raw file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == 0:
        raise ValueError(f"Empty file: {path}")

    rows = []
    ncols = None
    for line in lines:
        s = line.strip()
        if s == "":
            continue
        parts = s.split()
        if ncols is None:
            ncols = len(parts)
        if len(parts) != ncols:
            raise ValueError(f"Inconsistent column count: {len(parts)} vs {ncols}")
        rows.append([float(v) for v in parts])

    if ncols is None:
        raise ValueError(f"No valid rows in {path}")

    if ncols != TOTAL_COLS:
        raise ValueError(f"Expected {TOTAL_COLS} columns, got {ncols}")

    return np.asarray(rows, dtype=np.float64)


# ─────────────────────────────────────────────
# split
# ─────────────────────────────────────────────
def split_indices(n: int):
    if abs((TRAIN_RATIO + VALID_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError("TRAIN_RATIO + VALID_RATIO + TEST_RATIO must be 1.0")

    idxs = list(range(n))
    random.shuffle(idxs)

    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_idx = idxs[:n_train]
    valid_idx = idxs[n_train : n_train + n_valid]
    test_idx = idxs[n_train + n_valid :]

    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split produced empty set: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
        )

    return train_idx, valid_idx, test_idx


# ─────────────────────────────────────────────
# 라벨링: score=min(kMc,kMt) -> train quantile 4-class
# ─────────────────────────────────────────────
def compute_score_min_coeff(coeff: np.ndarray) -> np.ndarray:
    # coeff shape (N,2): [kMc, kMt]
    return coeff.min(axis=1)


def compute_quantiles(train_score: np.ndarray):
    q25, q50, q75 = np.quantile(train_score, [0.25, 0.5, 0.75])
    return float(q25), float(q50), float(q75)


def score_to_label(score: np.ndarray, q25: float, q50: float, q75: float) -> np.ndarray:
    y = np.empty(score.shape[0], dtype=np.int64)
    y[score >= q75] = 0
    y[(score >= q50) & (score < q75)] = 1
    y[(score >= q25) & (score < q50)] = 2
    y[score < q25] = 3
    return y


# ─────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────
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
    random.seed(RANDOM_SEED)

    names = read_feature_names(FEATURES_PATH)
    arr = read_whitespace_table(RAW_PATH)

    n = arr.shape[0]
    train_idx, valid_idx, test_idx = split_indices(n)

    # X: 16개 측정값
    X_all = arr[:, :X_COLS]
    x_cols = names[:X_COLS]

    # coeff: kMc, kMt (마지막 2개)
    coeff_all = arr[:, X_COLS : X_COLS + Y_COEF_COLS]
    kMc_name = names[16]
    kMt_name = names[17]

    score_all = compute_score_min_coeff(coeff_all)

    # threshold는 train에서만 계산 (누수 방지)
    q25, q50, q75 = compute_quantiles(score_all[train_idx])
    y_all = score_to_label(score_all, q25=q25, q50=q50, q75=q75)

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_valid = X_all[valid_idx]
    y_valid = y_all[valid_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    save_xy(SAVE_ROOT / "train", x_cols, X_train, y_train)
    save_xy(SAVE_ROOT / "valid", x_cols, X_valid, y_valid)
    save_xy(SAVE_ROOT / "test", x_cols, X_test, y_test)

    meta = {
        "dataset": DATA_NAME,
        "task": TASK,
        "continuous_cols": x_cols,
        "categorical_cols": [],
        "input_dim": len(x_cols),
        "num_class": NUM_CLASS,
        "num_samples": {
            "train": int(X_train.shape[0]),
            "valid": int(X_valid.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_path": str(RAW_PATH),
        "features_path": str(FEATURES_PATH),
        "raw_total_cols": TOTAL_COLS,
        "x_cols_range": [0, X_COLS - 1],
        "coef_cols": {"kMc": kMc_name, "kMt": kMt_name, "col_indices_0based": [16, 17]},
        "labeling": {
            "y_source": ["kMc", "kMt"],
            "score_def": "score = min(kMc, kMt)",
            "binning": "quantile(4)",
            "thresholds_from": "train_only",
            "thresholds": {"q25": q25, "q50": q50, "q75": q75},
            "label_def": {
                "0": "score >= q75 (healthiest)",
                "1": "q50 <= score < q75",
                "2": "q25 <= score < q50",
                "3": "score < q25 (most degraded)",
            },
        },
        "split": {
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
            "note": "random row split (no unit/cycle id available in this dataset)",
        },
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Meta saved:", SAVE_ROOT / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
