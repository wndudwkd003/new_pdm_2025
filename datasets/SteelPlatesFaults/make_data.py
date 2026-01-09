import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATA_NAME = "SteelPlatesFaults"

DATASET_DIR = Path(f"/ws/new_pdm_2025/datasets/{DATA_NAME}")
RAW_DIR = DATASET_DIR / "raw"

# 보통 UCI 원본은 Faults.NNA에 데이터가 들어있습니다.
# (Faults27x7_var는 변수명 파일인 경우가 많음)
RAW_DATA_PATH = RAW_DIR / "Faults.NNA"
RAW_VAR_PATH = RAW_DIR / "Faults27x7_var"

SAVE_ROOT = DATASET_DIR / "processed_data"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# 27 features + 7 fault targets = 34 columns
NUM_FEATURES = 27
FAULT_TARGETS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",  # 원본에 K_Scatch로 들어있는 경우가 많음(오타 포함)
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]
NO_FAULT_CLASS_NAME = "No_Fault"
NUM_CLASS = 8  # 7 faults + 1 no-fault


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def read_header_lines(path: Path):
    if path.exists() is False:
        raise FileNotFoundError(f"Header file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    cols = [ln.strip() for ln in lines if ln.strip() != ""]
    if len(cols) == 0:
        raise ValueError(f"No columns found in {path}")
    return cols


def read_whitespace_table(path: Path):
    if path.exists() is False:
        raise FileNotFoundError(f"Raw data file not found: {path}")

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
            raise ValueError(
                f"Inconsistent column count in {path}: {len(parts)} vs {ncols}"
            )
        rows.append([float(v) for v in parts])

    if ncols is None:
        raise ValueError(f"No valid rows in {path}")

    arr = np.asarray(rows, dtype=np.float64)
    return arr


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


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    # 컬럼명 로드
    cols = read_header_lines(RAW_VAR_PATH)

    if len(cols) != (NUM_FEATURES + len(FAULT_TARGETS)):
        raise ValueError(
            f"Expected {NUM_FEATURES + len(FAULT_TARGETS)} columns in var file, got {len(cols)}"
        )

    feature_cols = cols[:NUM_FEATURES]
    target_cols = cols[NUM_FEATURES:]

    # target 컬럼이 우리가 기대한 7개인지 확인 (이름/순서 확인용)
    # 원본에는 K_Scatch 오타가 있을 수 있어, 여기서는 존재 여부로 검증
    missing = [t for t in FAULT_TARGETS if t not in target_cols]
    if len(missing) != 0:
        raise ValueError(f"Target columns missing in var file: {missing}")

    # 데이터 로드
    arr = read_whitespace_table(RAW_DATA_PATH)
    if arr.shape[1] != len(cols):
        raise ValueError(
            f"Column mismatch: data has {arr.shape[1]} cols, var has {len(cols)} cols"
        )

    X_all = arr[:, :NUM_FEATURES].astype(np.float64)

    # fault targets는 마지막 7개(순서는 var 파일 기준)
    fault_mat = arr[:, NUM_FEATURES:].astype(np.int64)

    # 8-class 라벨 생성:
    # - fault가 정확히 1개면 그 index(0..6)
    # - fault가 0개면 No_Fault(7)
    # - fault가 2개 이상이면 제거(요구사항: 하나만 있거나 전부 없음)
    fault_sum = fault_mat.sum(axis=1)

    keep_mask = (fault_sum == 0) | (fault_sum == 1)
    dropped = int((~keep_mask).sum())

    X_all = X_all[keep_mask]
    fault_mat = fault_mat[keep_mask]
    fault_sum = fault_sum[keep_mask]

    y_all = np.full((X_all.shape[0],), 7, dtype=np.int64)  # default No_Fault=7
    one_mask = fault_sum == 1
    if int(one_mask.sum()) > 0:
        y_all[one_mask] = fault_mat[one_mask].argmax(axis=1).astype(np.int64)

    # split
    n = X_all.shape[0]
    train_idx, valid_idx, test_idx = split_indices(n)

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_valid = X_all[valid_idx]
    y_valid = y_all[valid_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    save_xy(SAVE_ROOT / "train", feature_cols, X_train, y_train)
    save_xy(SAVE_ROOT / "valid", feature_cols, X_valid, y_valid)
    save_xy(SAVE_ROOT / "test", feature_cols, X_test, y_test)

    label_map = {i: FAULT_TARGETS[i] for i in range(7)}
    label_map[7] = NO_FAULT_CLASS_NAME

    meta = {
        "dataset": DATA_NAME,
        "task": "multiclass_classification",
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": len(feature_cols),
        "num_class": NUM_CLASS,
        "num_samples": {
            "train": int(X_train.shape[0]),
            "valid": int(X_valid.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_paths": {"data": str(RAW_DATA_PATH), "var": str(RAW_VAR_PATH)},
        "raw_format": "whitespace-separated, var file provides column names",
        "labeling": {
            "type": "8-class (7 faults + No_Fault)",
            "rule": "if exactly one fault=1 -> class index of that fault; if all zero -> No_Fault; if multiple faults=1 -> dropped",
            "dropped_multi_fault_rows": dropped,
            "fault_targets": FAULT_TARGETS,
            "no_fault_class": {"id": 7, "name": NO_FAULT_CLASS_NAME},
            "label_map": label_map,
        },
        "split": {
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
            "note": "random row split (no group id available)",
        },
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Meta saved:", SAVE_ROOT / "meta.json")
    print(f"Kept rows: {n} | Dropped multi-fault rows: {dropped}")
    print("Done.")


if __name__ == "__main__":
    main()
