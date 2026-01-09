import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 사용자 설정
# ─────────────────────────────────────────────
# Pre-failure window W (시간)
PREFAIL_HOURS = 48

DATASET_DIR = Path(f"/ws/new_pdm_2025/datasets/MetroPT3_{PREFAIL_HOURS}")
RAW_PATH = DATASET_DIR / "raw" / "data.csv"  # 실제 파일명에 맞게 수정
OUT_ROOT = DATASET_DIR / "processed_data"

TRAIN_OUT = OUT_ROOT / "train"
VALID_OUT = OUT_ROOT / "valid"
TEST_OUT = OUT_ROOT / "test"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)

TIME_COL = "timestamp"


# 고정 월 분할
VALID_MONTH = "2020-06"
TEST_MONTH = "2020-07"

# failure report (Air leak)
FAILURE_INTERVALS = [
    {
        "id": 1,
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:59",
        "failure": "Air leak",
        "severity": "High stress",
    },
    {
        "id": 2,
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
        "failure": "Air leak",
        "severity": "High stress",
    },
    {
        "id": 3,
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
        "failure": "Air leak",
        "severity": "High stress",
    },
    {
        "id": 4,
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
        "failure": "Air leak",
        "severity": "High stress",
    },
]


def save_xy(save_dir: Path, feature_cols: list[str], X: np.ndarray, y: np.ndarray):
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


def dist_int(y: np.ndarray):
    vals, cnts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}


def build_labels(t: pd.Series, prefail_hours: int, intervals: list[dict]) -> np.ndarray:
    y = np.zeros(len(t), dtype=np.int64)
    prefail_delta = pd.Timedelta(hours=int(prefail_hours))

    # failure 먼저
    for itv in intervals:
        s = pd.to_datetime(itv["start"])
        e = pd.to_datetime(itv["end"])
        mask_fail = (t >= s) & (t <= e)
        y[mask_fail.to_numpy()] = 2

    # pre-failure (failure보다 우선순위 낮음)
    for itv in intervals:
        s = pd.to_datetime(itv["start"])
        pre_s = s - prefail_delta
        pre_e = s
        mask_pre = (t >= pre_s) & (t < pre_e)
        idx = mask_pre.to_numpy()
        y[idx & (y != 2)] = 1

    return y


def month_mask(ts: pd.Series, ym: str) -> np.ndarray:
    # ym: "YYYY-MM"
    m_start = pd.to_datetime(ym + "-01 00:00:00")
    m_end = m_start + pd.DateOffset(months=1)
    return ((ts >= m_start) & (ts < m_end)).to_numpy()


def split_by_months(df: pd.DataFrame, time_col: str, valid_month: str, test_month: str):
    df = df.sort_values(time_col).reset_index(drop=True)

    m_valid = month_mask(df[time_col], valid_month)
    m_test = month_mask(df[time_col], test_month)

    if m_valid.sum() == 0:
        raise ValueError(f"No rows found for valid_month={valid_month}")
    if m_test.sum() == 0:
        raise ValueError(f"No rows found for test_month={test_month}")

    df_valid = df[m_valid].reset_index(drop=True)
    df_test = df[m_test].reset_index(drop=True)
    df_train = df[~(m_valid | m_test)].reset_index(drop=True)

    if len(df_train) == 0:
        raise ValueError("Train split is empty after excluding valid/test months.")

    return df_train, df_valid, df_test


def auto_drop_index_like_numeric_cols(
    df: pd.DataFrame, numeric_cols: list[str], always_drop: set[str]
):
    auto_drop = []
    for c in numeric_cols:
        if c in always_drop:
            continue
        v = df[c].to_numpy()
        if v.ndim != 1 or len(v) < 3:
            continue
        dv = np.diff(v)
        if np.all(dv > 0) and np.allclose(dv, dv[0]):
            auto_drop.append(c)
    return auto_drop


def main():
    df = pd.read_csv(RAW_PATH)

    if TIME_COL not in df.columns:
        raise ValueError(f"TIME_COL not found: {TIME_COL}")

    # timestamp 파싱/정렬
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # 라벨 생성
    df["label"] = build_labels(df[TIME_COL], PREFAIL_HOURS, FAILURE_INTERVALS).astype(
        int
    )

    # feature 컬럼 구성: timestamp 제거 + 인덱스성 숫자열 제거
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    always_drop = set(["label", "Unnamed: 0", "", "index", "id"])
    auto_drop = auto_drop_index_like_numeric_cols(df, numeric_cols, always_drop)

    feature_cols = [
        c for c in numeric_cols if (c not in always_drop) and (c not in auto_drop)
    ]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns selected. Check drop rules.")

    # split: valid=6월, test=7월, 나머지 train
    df_keep = df[[TIME_COL] + feature_cols + ["label"]].copy()
    df_train, df_valid, df_test = split_by_months(
        df_keep, TIME_COL, VALID_MONTH, TEST_MONTH
    )

    X_train = df_train[feature_cols].to_numpy(dtype=np.float64)
    y_train = df_train["label"].to_numpy(dtype=np.int64)

    X_valid = df_valid[feature_cols].to_numpy(dtype=np.float64)
    y_valid = df_valid["label"].to_numpy(dtype=np.int64)

    X_test = df_test[feature_cols].to_numpy(dtype=np.float64)
    y_test = df_test["label"].to_numpy(dtype=np.int64)

    save_xy(TRAIN_OUT, feature_cols, X_train, y_train)
    save_xy(VALID_OUT, feature_cols, X_valid, y_valid)
    save_xy(TEST_OUT, feature_cols, X_test, y_test)

    meta = {
        "dataset": "MetroPT-3",
        "task": "multiclass_classification",
        "num_class": 3,
        "label_definition": {
            "0": "normal",
            "1": f"pre-failure (within {int(PREFAIL_HOURS)}h before failure start)",
            "2": "failure (within failure report start~end)",
            "priority": "failure(2) > pre-failure(1) > normal(0)",
        },
        "prefail_hours": int(PREFAIL_HOURS),
        "failure_intervals": FAILURE_INTERVALS,
        "split": {
            "method": "fixed-month split (no leakage)",
            "train": f"all months except {VALID_MONTH} and {TEST_MONTH}",
            "valid": VALID_MONTH,
            "test": TEST_MONTH,
        },
        "features": {
            "continuous_cols": feature_cols,
            "categorical_cols": [],
            "input_dim": int(len(feature_cols)),
            "dropped_from_X": [TIME_COL, "label", "Unnamed: 0", "", "index", "id"]
            + auto_drop,
            "auto_dropped_monotonic_cols": auto_drop,
        },
        "num_samples": {
            "train": int(len(df_train)),
            "valid": int(len(df_valid)),
            "test": int(len(df_test)),
            "total": int(len(df)),
        },
        "label_distribution": {
            "train": dist_int(y_train),
            "valid": dist_int(y_valid),
            "test": dist_int(y_test),
            "total": dist_int(df["label"].to_numpy(dtype=np.int64)),
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
