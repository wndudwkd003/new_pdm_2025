from scipy.io import arff
import pandas as pd
import numpy as np
from pathlib import Path
import csv
import json
from datetime import datetime
from sklearn.model_selection import train_test_split


# ────────────────────────────────────────────────────
# Path 설정
# ────────────────────────────────────────────────────
DATA_DIR = Path("datasets/fordengine/data")

TRAIN_PATH = DATA_DIR / "FordA_TRAIN.arff"
TEST_PATH = DATA_DIR / "FordA_TEST.arff"

PROCESSED_DIR = Path("datasets/fordengine/processed_data")
TRAIN_OUT = PROCESSED_DIR / "train"
VALID_OUT = PROCESSED_DIR / "valid"
TEST_OUT = PROCESSED_DIR / "test"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────
NUM_CLASS = 2
VALID_RATIO = 0.2
RANDOM_SEED = 42


# ────────────────────────────────────────────────────
# 데이터 로드 (ARFF → numpy)
# ────────────────────────────────────────────────────
def load_arff_as_table(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_data, meta = arff.loadarff(str(path))
    df = pd.DataFrame(raw_data)

    X = df.iloc[:, :-1].astype(float).to_numpy()
    y = df.iloc[:, -1].astype(int).to_numpy()

    # FordA가 {-1, 1}이면 0/1로 변환
    uniq = np.unique(y)
    if set(uniq.tolist()) == set([-1, 1]):
        y = (y == 1).astype(int)

    uniq2 = np.unique(y)
    if NUM_CLASS == 2:
        if not set(uniq2.tolist()).issubset(set([0, 1])):
            raise ValueError(
                f"Unexpected labels for binary task: {sorted(uniq2.tolist())}"
            )

    return X, y


# ────────────────────────────────────────────────────
# CSV 저장 (CMAPSS/MPTMS 스타일: X.csv, y.csv)
# ────────────────────────────────────────────────────
def save_xy(save_dir: Path, feature_cols: list[str], X: np.ndarray, y: np.ndarray):
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D array, got shape={y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch: X has {X.shape[0]} rows but y has {y.shape[0]} rows"
        )
    if X.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature dim mismatch: X has {X.shape[1]} cols but feature_cols has {len(feature_cols)}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)

    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        writer = csv.writer(fx)
        writer.writerow(feature_cols)
        writer.writerows(X.tolist())

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        writer = csv.writer(fy)
        writer.writerow(["label"])
        writer.writerows([[int(v)] for v in y.tolist()])

    print(f"Saved X: {X.shape[0]} rows → {x_path}")
    print(f"Saved y: {y.shape[0]} rows → {y_path}")


# ────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────
def main():
    # 1) 로드
    X_train_all, y_train_all = load_arff_as_table(TRAIN_PATH)
    X_test, y_test = load_arff_as_table(TEST_PATH)

    n_features = int(X_train_all.shape[1])

    # CMAPSS 스타일처럼 s1..sN으로 컬럼명 부여
    feature_cols = [f"s{i}" for i in range(1, n_features + 1)]

    # 2) train/valid split (TRAIN 내부에서만)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_all,
        y_train_all,
        test_size=VALID_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_train_all if NUM_CLASS == 2 else None,
    )

    # 3) 저장 (경로는 fordengine 유지)
    save_xy(TRAIN_OUT, feature_cols, X_train, y_train)
    save_xy(VALID_OUT, feature_cols, X_valid, y_valid)
    save_xy(TEST_OUT, feature_cols, X_test, y_test)

    # 4) meta.json (형식은 CMAPSS meta와 유사하게)
    meta = {
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": int(n_features),
        "num_class": int(NUM_CLASS),
        "num_samples": {
            "train": int(X_train.shape[0]),
            "valid": int(X_valid.shape[0]),
            "test": int(X_test.shape[0]),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets_used": ["FordA"],
        "split": {
            "rule": "valid split from TRAIN only (no leakage into TEST)",
            "ratios": {
                "train": float(1.0 - VALID_RATIO),
                "valid": float(VALID_RATIO),
                "test": "official",
            },
            "seed": int(RANDOM_SEED),
            "stratify": bool(NUM_CLASS == 2),
        },
        "source": {
            "train": str(TRAIN_PATH),
            "test": str(TEST_PATH),
            "label_mapping": "if labels are {-1,1}, map -1->0 and 1->1",
        },
        "output_root": str(PROCESSED_DIR),
    }

    with open(PROCESSED_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nMeta saved →", PROCESSED_DIR / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
