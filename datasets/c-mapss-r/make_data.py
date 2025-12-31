"""
C-MAPSS 회귀(RUL) 버전 전처리 스크립트.

- train_FDxxx.txt: unit별 run-to-failure → 행 단위 RUL = max_cycle - cycle
- test_FDxxx.txt: run-to-failure 이전에 끊김
- RUL_FDxxx.txt: test 각 unit의 "마지막 cycle에서 남은 RUL" 정답 벡터 제공

출력(MPTMS 스타일):
datasets/c-mapss-r/processed_data/
  - train/X.csv, y.csv
  - valid/X.csv, y.csv
  - test/X.csv, y.csv
  - meta.json

주의:
- y.csv의 헤더는 항상 "label"로 저장합니다. (회귀여도 label 고정)
- meta.json의 task는 영어로 "regression" 입니다.
"""

import json
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
RANDOM_SEED = 42

# train 내 unit 기준 split
TRAIN_RATIO = 0.9
VALID_RATIO = 0.1

BASE_DIR = Path("datasets/c-mapss-r/data/CMaps")
SAVE_ROOT = Path("datasets/c-mapss-r/processed_data")
TRAIN_OUT = SAVE_ROOT / "train"
VALID_OUT = SAVE_ROOT / "valid"
TEST_OUT = SAVE_ROOT / "test"

TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)

DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]

# 내부 계산용 컬럼명(저장 헤더는 label로 고정)
TARGET_COL_INTERNAL = "RUL"
TARGET_COL_SAVED = "label"


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def compute_train_rul(df_train: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df_train.groupby("unit")["cycle"].max()
    df = df_train.copy()
    df[TARGET_COL_INTERNAL] = (df["unit"].map(max_cycle) - df["cycle"]).astype(int)
    return df


def load_rul_vector(path: Path) -> np.ndarray:
    arr = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=int)
    return arr


def compute_test_rul(df_test: pd.DataFrame, rul_end_vec: np.ndarray) -> pd.DataFrame:
    units = sorted(df_test["unit"].unique().tolist())
    if len(units) != int(len(rul_end_vec)):
        raise ValueError(
            f"Mismatch: num_units_in_test={len(units)} vs len(RUL_vector)={len(rul_end_vec)}"
        )

    rul_end_by_unit = {u: int(r) for u, r in zip(units, rul_end_vec.tolist())}

    cmax_test = df_test.groupby("unit")["cycle"].max()
    df = df_test.copy()

    # RUL = (Cmax_test + RUL_end) - cycle
    df[TARGET_COL_INTERNAL] = (
        df["unit"].map(cmax_test) + df["unit"].map(rul_end_by_unit) - df["cycle"]
    ).astype(int)

    return df


def split_train_valid_by_unit(
    df_train_labeled: pd.DataFrame, train_ratio: float, seed: int
):
    rng = np.random.RandomState(seed)
    units = sorted(df_train_labeled["unit"].unique().tolist())
    rng.shuffle(units)

    n_train = int(len(units) * train_ratio)
    train_units = set(units[:n_train])
    valid_units = set(units[n_train:])

    df_tr = df_train_labeled[df_train_labeled["unit"].isin(train_units)].reset_index(
        drop=True
    )
    df_va = df_train_labeled[df_train_labeled["unit"].isin(valid_units)].reset_index(
        drop=True
    )
    return df_tr, df_va


def save_xy(save_dir: Path, feature_cols: list[str], df: pd.DataFrame):
    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    X = df[feature_cols].to_numpy()
    y = df[TARGET_COL_INTERNAL].astype(int).to_numpy()

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        writer = csv.writer(fx)
        writer.writerow(feature_cols)
        writer.writerows(X.tolist())

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        writer = csv.writer(fy)
        writer.writerow([TARGET_COL_SAVED])  # 항상 label
        writer.writerows([[int(v)] for v in y.tolist()])

    print(f"Saved {save_dir.name}: {len(df)} rows")
    print("  X ->", x_path)
    print("  y ->", y_path)


def main():
    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]

    all_train_parts = []
    all_test_parts = []

    for fd in DATASETS:
        train_path = BASE_DIR / f"train_{fd}.txt"
        test_path = BASE_DIR / f"test_{fd}.txt"
        rul_path = BASE_DIR / f"RUL_{fd}.txt"

        print(f"\n=== {fd} ===")
        df_train_raw = load_table(train_path)
        df_test_raw = load_table(test_path)
        rul_end_vec = load_rul_vector(rul_path)

        df_train_lab = compute_train_rul(df_train_raw)
        df_test_lab = compute_test_rul(df_test_raw, rul_end_vec)

        df_train_lab["dataset"] = fd
        df_test_lab["dataset"] = fd

        all_train_parts.append(df_train_lab)
        all_test_parts.append(df_test_lab)

        print(f"train rows: {len(df_train_lab)}, test rows: {len(df_test_lab)}")

    df_train_all = pd.concat(all_train_parts, ignore_index=True)
    df_test_all = pd.concat(all_test_parts, ignore_index=True)

    # train에서만 train/valid 분리 (unit 누수 방지)
    df_train, df_valid = split_train_valid_by_unit(
        df_train_labeled=df_train_all,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED,
    )

    save_xy(TRAIN_OUT, sensor_cols, df_train)
    save_xy(VALID_OUT, sensor_cols, df_valid)
    save_xy(TEST_OUT, sensor_cols, df_test_all)

    meta = {
        "task": "regression",  # 영어로 고정
        "y_col": TARGET_COL_SAVED,  # 저장 헤더 기준으로 label 고정
        "continuous_cols": sensor_cols,
        "categorical_cols": [],
        "input_dim": len(sensor_cols),
        "num_samples": {
            "train": int(len(df_train)),
            "valid": int(len(df_valid)),
            "test": int(len(df_test_all)),
        },
        "datasets_used": DATASETS,
        "drop_sensors": DROP,
        "split": {
            "train_valid_source": "train_FDxxx.txt only",
            "train_valid_rule": "unit-level split (no leakage)",
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO},
            "seed": RANDOM_SEED,
            "test_source": "official test_FDxxx.txt",
            "test_labels": "RUL_FDxxx.txt provides RUL at last cycle; expanded to per-row RUL",
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nMeta saved ->", SAVE_ROOT / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
