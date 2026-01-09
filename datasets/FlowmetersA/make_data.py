import csv
import json
import random
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# 사용자 설정
# ─────────────────────────────────────────────
DATA_NAME = "FlowmetersA"  # FlowmetersA | FlowmetersB | FlowmetersC | FlowmetersD
DATA_ALPHA = DATA_NAME[-1]  # A/B/C/D
print(f"Processing dataset: {DATA_NAME} (Meter {DATA_ALPHA})")

DATASET_DIR = Path(f"/ws/new_pdm_2025/datasets/{DATA_NAME}")
RAW_PATH = DATASET_DIR / f"Meter {DATA_ALPHA}"

SAVE_ROOT = Path(f"{DATASET_DIR}/processed_data")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1


def read_tab_file(path: Path):
    if path.exists() is False:
        raise FileNotFoundError(f"Raw file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == 0:
        raise ValueError(f"Empty file: {path}")

    rows = []
    ncols = None

    for line in lines:
        parts = line.strip().split("\t")
        if ncols is None:
            ncols = len(parts)
        if len(parts) != ncols:
            raise ValueError(
                f"Inconsistent column count in {path}: {len(parts)} vs {ncols}"
            )
        rows.append(parts)

    if ncols is None or ncols < 2:
        raise ValueError(f"Invalid column count (need >=2) in {path}: {ncols}")

    return rows, ncols


def infer_label_mapping(rows):
    """
    마지막 컬럼이 라벨이라고 가정.
    라벨 유니크를 자동 수집하고, 정렬 후 0..K-1로 매핑.
    """
    raw_labels = []
    for r in rows:
        y_raw = r[-1]
        y_int = int(float(y_raw))
        raw_labels.append(y_int)

    uniq = sorted(set(raw_labels))
    if len(uniq) < 2:
        raise ValueError(f"Need at least 2 classes, got: {uniq}")

    label_to_index = {lab: i for i, lab in enumerate(uniq)}
    index_to_label = {i: lab for lab, i in label_to_index.items()}

    num_class = len(uniq)
    task = "binary_classification" if num_class == 2 else "multiclass_classification"

    return num_class, task, label_to_index, index_to_label


def build_xy(rows, label_to_index):
    # last column is label, others are features
    ncols = len(rows[0])
    n_feat = ncols - 1

    feature_cols = [f"f{i}" for i in range(n_feat)]
    X_list = []
    y_list = []

    for r in rows:
        x_raw = r[:n_feat]
        y_raw = r[n_feat]

        x = [float(v) for v in x_raw]
        y_int = int(float(y_raw))

        if y_int not in label_to_index:
            raise ValueError(f"Unknown label {y_int} in {RAW_PATH}")

        y = label_to_index[y_int]  # 0-based remap
        X_list.append(x)
        y_list.append([y])

    return feature_cols, X_list, y_list


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


def save_xy(save_dir: Path, feature_cols, X_list, y_list):
    save_dir.mkdir(parents=True, exist_ok=True)

    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        w = csv.writer(fx)
        w.writerow(feature_cols)
        w.writerows(X_list)

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        w = csv.writer(fy)
        w.writerow(["label"])
        w.writerows(y_list)

    print(f"Saved X: {len(X_list)} rows -> {x_path}")
    print(f"Saved y: {len(y_list)} rows -> {y_path}")


def main():
    random.seed(RANDOM_SEED)

    rows, ncols = read_tab_file(RAW_PATH)

    num_class, task, label_to_index, index_to_label = infer_label_mapping(rows)
    print(f"Detected classes: {num_class} -> task={task}")
    print(f"Label mapping (raw -> saved): {label_to_index}")

    feature_cols, X_all, y_all = build_xy(rows, label_to_index)

    train_idx, valid_idx, test_idx = split_indices(len(X_all))

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_valid = [X_all[i] for i in valid_idx]
    y_valid = [y_all[i] for i in valid_idx]
    X_test = [X_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]

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
            "train": len(X_train),
            "valid": len(X_valid),
            "test": len(X_test),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_path": str(RAW_PATH),
        "total_cols_in_raw": ncols,
        "label_col_index": ncols - 1,
        "label_mapping_raw_to_saved": label_to_index,
        "label_mapping_saved_to_raw": index_to_label,
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Meta saved:", SAVE_ROOT / "meta.json")


if __name__ == "__main__":
    main()
