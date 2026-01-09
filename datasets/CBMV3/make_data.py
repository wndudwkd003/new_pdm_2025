import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_NAME = "CBMV3"
DATASET_DIR = Path(f"/ws/new_pdm_2025/datasets/{DATA_NAME}")

RAW_PATH = DATASET_DIR / "raw" / "data.txt"
SAVE_ROOT = DATASET_DIR / "processed_data"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

TOTAL_COLS = 30
X_COLS = 25
COEF_COLS = 5  # last 5 cols
NUM_CLASS = 4  # 4-class labeling


# ─────────────────────────────────────────────
# 로더: 공백/탭 구분 모두 대응
# ─────────────────────────────────────────────
def read_whitespace_table(path: Path):
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
        parts = s.split()  # 모든 공백(스페이스/탭) 처리
        if ncols is None:
            ncols = len(parts)
        if len(parts) != ncols:
            raise ValueError(
                f"Inconsistent column count: {len(parts)} vs {ncols} in line: {s[:80]}..."
            )
        rows.append([float(v) for v in parts])

    if ncols is None:
        raise ValueError(f"No valid rows in {path}")

    if ncols != TOTAL_COLS:
        raise ValueError(f"Expected {TOTAL_COLS} columns, got {ncols} in {path}")

    arr = np.asarray(rows, dtype=np.float64)
    return arr


# ─────────────────────────────────────────────
# 라벨링: score = min(last 5 coeffs), train 분위수로 4등급
# ─────────────────────────────────────────────
def compute_score_min_coeff(arr: np.ndarray) -> np.ndarray:
    coeff = arr[:, -COEF_COLS:]  # shape (N,5)
    score = coeff.min(axis=1)
    return score


def compute_quantile_thresholds(train_score: np.ndarray):
    # q25, q50, q75
    q25, q50, q75 = np.quantile(train_score, [0.25, 0.50, 0.75])
    return float(q25), float(q50), float(q75)


def score_to_label(score: np.ndarray, q25: float, q50: float, q75: float) -> np.ndarray:
    # label 0: >= q75
    # label 1: [q50, q75)
    # label 2: [q25, q50)
    # label 3: < q25
    y = np.empty(score.shape[0], dtype=np.int64)
    y[score >= q75] = 0
    y[(score >= q50) & (score < q75)] = 1
    y[(score >= q25) & (score < q50)] = 2
    y[score < q25] = 3
    return y


# ─────────────────────────────────────────────
# Split
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

    arr = read_whitespace_table(RAW_PATH)
    n = arr.shape[0]
    print(f"Loaded: {RAW_PATH} | rows={n}, cols={arr.shape[1]}")

    train_idx, valid_idx, test_idx = split_indices(n)

    # X는 앞 25개만 사용(문서 기준 vessel relevant features)
    X_all = arr[:, :X_COLS]

    # score는 last 5 coeffs에서 생성
    score_all = compute_score_min_coeff(arr)

    # threshold는 train에서만 추정(누수 방지)
    score_train = score_all[train_idx]
    q25, q50, q75 = compute_quantile_thresholds(score_train)

    y_all = score_to_label(score_all, q25=q25, q50=q50, q75=q75)

    # split
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]

    X_valid = X_all[valid_idx]
    y_valid = y_all[valid_idx]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    feature_cols = [f"f{i}" for i in range(X_COLS)]

    save_xy(SAVE_ROOT / "train", feature_cols, X_train, y_train)
    save_xy(SAVE_ROOT / "valid", feature_cols, X_valid, y_valid)
    save_xy(SAVE_ROOT / "test", feature_cols, X_test, y_test)

    meta = {
        "dataset": DATA_NAME,
        "task": "multiclass_classification",
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": X_COLS,
        "num_class": NUM_CLASS,
        "num_samples": {
            "train": int(X_train.shape[0]),
            "valid": int(X_valid.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_path": str(RAW_PATH),
        "raw_format": "whitespace-separated text, 30 float columns per row",
        "raw_total_cols": TOTAL_COLS,
        "x_cols_range": [0, X_COLS - 1],
        "coeff_cols_range": [TOTAL_COLS - COEF_COLS, TOTAL_COLS - 1],
        "labeling": {
            "y_source": "last 5 decay coefficients",
            "score_def": "score = min(decay_coeff_5)",
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
            "unit_leakage_note": "no unit/cycle id provided; random row split",
        },
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Meta saved:", SAVE_ROOT / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
