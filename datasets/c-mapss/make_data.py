"""
CMAPSS train_FDxxx.txt만 사용해서 라벨링 후,
MPTMS 스타일 탭울러(train/valid/test)로 저장하는 스크립트.

출력:
datasets/c-mapss/processed_data/
  - train/X.csv, y.csv
  - valid/X.csv, y.csv
  - test/X.csv, y.csv
  - meta.json
  - engine_knee_plots_tabular/<FDxxx>/<FDxxx>_engine_<id>.png

핵심:
- tag(u/d/c/o)별로 state(0~3)를 모두 계산한 뒤,
  (unit, cycle)마다 최종 state를 max(state_tag들)로 통합합니다.
- 저장 라벨(label)은 state를 다음처럼 재매핑합니다:
    state 0,1 -> label 0
    state 2   -> label 1
    state 3   -> label 2
- X.csv에는 센서(s1~s21, DROP 제외)만 저장합니다.
- test_FDxxx.txt는 쓰지 않고, train_FDxxx.txt 내부에서 unit 기준으로
  train/valid/test = 8:1:1 split 합니다(누수 방지).
"""

import json
import csv
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
DATASETS = ["FD001", "FD003"]
SHIFT = 0

BASE_DIR = Path("datasets/c-mapss/data/CMaps")
MAP_FILE = Path("datasets/c-mapss/data/sensor_udc_map.json")

SAVE_ROOT = Path("datasets/c-mapss/processed_data")
TRAIN_OUT = SAVE_ROOT / "train"
VALID_OUT = SAVE_ROOT / "valid"
TEST_OUT = SAVE_ROOT / "test"
PLOT_ROOT = SAVE_ROOT / "engine_knee_plots_tabular"

TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VALID_OUT.mkdir(parents=True, exist_ok=True)
TEST_OUT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

SAVE_PLOTS = True

RANDOM_SEED = 42

# 8:1:1
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# state(0~3) -> label(0~2) 재매핑
LABEL_MAP = {0: 0, 1: 0, 2: 1, 3: 2}
NUM_CLASS = 3  # label 기준 (0~2)

DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
TAGS = ["u", "d", "c", "o"]

COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]

COLORS = ["#8fd175", "#fff07e", "#f6b08c", "#d9534f"]  # state 0~3
ALPHA = 0.15


# ─────────────────────────────────────────────
# knee util
# ─────────────────────────────────────────────
def edges_10(y: np.ndarray, tag: str) -> list[int]:
    x = np.arange(len(y))

    if tag == "u":
        y1 = y
        curve = "concave"
        direction = "increasing"
    elif tag == "d":
        y1 = y
        curve = "convex"
        direction = "decreasing"
    else:
        y1 = np.abs(y - y.mean())
        curve = "concave"
        direction = "increasing"

    k = KneeLocator(x, y1, curve=curve, direction=direction, S=2.0)

    knees = sorted(list(k.all_knees))
    idx = knees[:9]

    while len(idx) < 9:
        q = int(len(y) * (len(idx) + 1) / 10)
        if q not in idx:
            idx.append(q)

    idx = sorted(idx)[:9]
    return [0] + idx + [len(y) - 1]


def smooth(v: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return pd.Series(v).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def load_raw_train_only(fd: str) -> pd.DataFrame:
    fpath = BASE_DIR / f"train_{fd}.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    return pd.read_csv(fpath, sep=r"\s+", header=None, names=COLS)


# ─────────────────────────────────────────────
# tag 전체로 state 계산 후, cycle별 최종 state = max(state_tag들)
# ─────────────────────────────────────────────
def label_states_and_plot_integrated(
    fd: str,
    df_raw: pd.DataFrame,
    fd_map: dict,
    shift: int,
    plot_root: Path,
    save_plots: bool,
) -> pd.DataFrame:
    sensors = [
        s
        for s in df_raw.columns
        if s.startswith("s") and s[1:].isdigit() and (s not in DROP) and (s in fd_map)
    ]
    if len(sensors) == 0:
        raise ValueError(
            f"[{fd}] usable sensors가 없습니다. mapping / DROP를 확인하세요."
        )

    # 정규화(플롯과 knee에 사용)
    df_norm = df_raw.copy()
    for s in sensors:
        df_norm[s] = MinMaxScaler().fit_transform(df_raw[[s]])

    # tag별 센서 그룹
    groups = {k: [s for s in sensors if fd_map[s] == k] for k in TAGS}

    out_dir = plot_root / fd
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled_parts: list[pd.DataFrame] = []

    for unit_id, g_raw in tqdm(list(df_raw.groupby("unit")), desc=f"{fd} engines"):
        g_raw = g_raw.sort_values("cycle").reset_index(drop=True)
        g_norm = (
            df_norm[df_norm["unit"] == unit_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        cyc = g_raw["cycle"].to_numpy()
        state_by_tag: dict[str, np.ndarray] = {}

        if save_plots:
            fig, ax = plt.subplots(figsize=(14, 4))
            for s in sensors:
                ax.plot(cyc, g_norm[s], color="grey", alpha=0.3, lw=0.35)

        for tag, cols in groups.items():
            if len(cols) != 0:
                m_norm = g_norm[cols].mean(axis=1).to_numpy()
                m_line = smooth(m_norm)
                edges = edges_10(m_norm, tag)

                base = [
                    min(i, 10) for i in [shift, shift + 3, shift + 6, shift + 9, 10]
                ]
                seg_idx = [edges[i] for i in base]
                seg_cyc = [int(cyc[i]) for i in seg_idx]
                seg_cyc[-1] = seg_cyc[-1] + 1

                state_label = np.zeros(len(cyc), dtype=int)
                for i in range(len(seg_cyc) - 1):
                    l = seg_cyc[i]
                    r = seg_cyc[i + 1]
                    mask = (cyc >= l) & (cyc < r)
                    state_label[mask] = i

                state_by_tag[tag] = state_label

                if save_plots:
                    for (l, r), c in zip(zip(seg_cyc[:-1], seg_cyc[1:]), COLORS):
                        ax.axvspan(l, r, color=c, alpha=ALPHA)
                    ax.plot(cyc, m_line, lw=2, label=f"{tag} mean")

        if len(state_by_tag) == 0:
            raise ValueError(
                f"[{fd}] unit={int(unit_id)}: tag별 state를 하나도 만들지 못했습니다."
            )

        states_stack = np.stack([state_by_tag[t] for t in state_by_tag.keys()], axis=0)
        state_integrated = states_stack.max(axis=0).astype(int)

        g_out = g_raw.copy()
        g_out["state"] = state_integrated
        g_out["dataset"] = fd
        labeled_parts.append(g_out)

        if save_plots:
            ax.set_xlabel("Cycle")
            ax.set_ylabel("Scaled Value")
            ax.set_title(
                f"{fd} – Engine {int(unit_id)}  (shift={shift}, integrated=max)"
            )
            ax.legend(loc="upper left", fontsize="small")
            fig.tight_layout()
            fig.savefig(out_dir / f"{fd}_engine_{int(unit_id)}.png", dpi=150)
            plt.close(fig)

    return pd.concat(labeled_parts, ignore_index=True)


# ─────────────────────────────────────────────
# unit 단위 stratified(근사) 선택: 목표 label 히스토그램에 가깝게 unit을 그리디 선택
# ─────────────────────────────────────────────
def greedy_select_units(
    units: list[int],
    unit_counts: pd.DataFrame,  # index=unit, columns=0..num_class-1
    target_units: int,
    target_counts: np.ndarray,
) -> list[int]:
    selected = []
    cur_counts = np.zeros_like(target_counts, dtype=np.int64)

    remaining_units = units.copy()
    for u in units:
        remaining_units.pop(0)

        need = target_units - len(selected)
        if need != 0:
            u_counts = unit_counts.loc[u].to_numpy(dtype=np.int64)

            force_take = len(remaining_units) == need - 1
            if force_take:
                selected.append(u)
                cur_counts += u_counts
            else:
                dist_if_take = np.abs((cur_counts + u_counts) - target_counts).sum()
                dist_if_skip = np.abs(cur_counts - target_counts).sum()

                if dist_if_take <= dist_if_skip:
                    selected.append(u)
                    cur_counts += u_counts

    return selected


# ─────────────────────────────────────────────
# train/valid/test = 8:1:1, unit 기준 + 라벨 분포 근사 균형
# ─────────────────────────────────────────────
def split_train_valid_test_by_unit_stratified(
    df_labeled: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    num_class: int,
    label_col: str,
):
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + valid_ratio + test_ratio must be 1.0")

    rng = np.random.RandomState(seed)

    train_parts = []
    valid_parts = []
    test_parts = []

    for fd, df_fd in df_labeled.groupby("dataset"):
        units = sorted(df_fd["unit"].unique().tolist())
        rng.shuffle(units)

        unit_counts = df_fd.groupby(["unit", label_col]).size().unstack(fill_value=0)

        for s in range(num_class):
            if s not in unit_counts.columns:
                unit_counts[s] = 0
        unit_counts = unit_counts[[s for s in range(num_class)]]

        total_counts = unit_counts.sum(axis=0).to_numpy(dtype=np.int64)

        # 1) train 선택
        target_train_units = int(len(units) * train_ratio)
        target_train_counts = (total_counts * train_ratio).astype(np.int64)

        selected_train_units = greedy_select_units(
            units=units,
            unit_counts=unit_counts,
            target_units=target_train_units,
            target_counts=target_train_counts,
        )
        train_set = set(selected_train_units)

        remaining_units = [u for u in units if u not in train_set]

        # 2) remaining에서 valid 선택 (valid:test = valid_ratio:test_ratio 비율로)
        rem_total_counts = (
            unit_counts.loc[remaining_units].sum(axis=0).to_numpy(dtype=np.int64)
        )

        denom = valid_ratio + test_ratio
        valid_share_in_remaining = valid_ratio / denom

        target_valid_units = int(len(remaining_units) * valid_share_in_remaining)
        target_valid_counts = (rem_total_counts * valid_share_in_remaining).astype(
            np.int64
        )

        selected_valid_units = greedy_select_units(
            units=remaining_units,
            unit_counts=unit_counts,
            target_units=target_valid_units,
            target_counts=target_valid_counts,
        )
        valid_set = set(selected_valid_units)

        test_units = [u for u in remaining_units if u not in valid_set]

        df_train_fd = df_fd[df_fd["unit"].isin(train_set)]
        df_valid_fd = df_fd[df_fd["unit"].isin(valid_set)]
        df_test_fd = df_fd[df_fd["unit"].isin(set(test_units))]

        train_parts.append(df_train_fd)
        valid_parts.append(df_valid_fd)
        test_parts.append(df_test_fd)

    df_train = pd.concat(train_parts, ignore_index=True)
    df_valid = pd.concat(valid_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True)
    return df_train, df_valid, df_test


# ─────────────────────────────────────────────
# CSV 저장 (MPTMS 스타일) - X는 센서만, y는 label_col
# ─────────────────────────────────────────────
def save_xy(save_dir: Path, feature_cols: list[str], df: pd.DataFrame, label_col: str):
    save_dir.mkdir(parents=True, exist_ok=True)

    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    X = df[feature_cols].to_numpy()
    y = df[label_col].astype(int).to_numpy()

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        writer = csv.writer(fx)
        writer.writerow(feature_cols)
        writer.writerows(X.tolist())

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        writer = csv.writer(fy)
        writer.writerow(["label"])
        writer.writerows([[int(v)] for v in y.tolist()])

    print(f"Saved X: {len(df)} rows → {x_path}")
    print(f"Saved y: {len(df)} rows → {y_path}")


def main():
    random.seed(RANDOM_SEED)

    if not MAP_FILE.exists():
        raise FileNotFoundError(f"sensor_udc_map.json not found: {MAP_FILE}")

    mapping_all = json.loads(MAP_FILE.read_text(encoding="utf-8"))

    sensor_cols = [f"s{i}" for i in range(1, 22) if f"s{i}" not in DROP]

    labeled_all = []

    for fd in DATASETS:
        if fd not in mapping_all:
            raise ValueError(f"Mapping not found for dataset key: {fd}")

        fd_map = mapping_all[fd]

        print(f"\n=== Processing {fd} (train only) ===")
        df_raw = load_raw_train_only(fd)

        df_lab = label_states_and_plot_integrated(
            fd=fd,
            df_raw=df_raw,
            fd_map=fd_map,
            shift=SHIFT,
            plot_root=PLOT_ROOT,
            save_plots=SAVE_PLOTS,
        )
        labeled_all.append(df_lab)
        print(f"[{fd}] labeled rows: {len(df_lab)}")

    df_labeled = pd.concat(labeled_all, ignore_index=True)

    # state -> label 재매핑 (저장 및 split은 label 기준으로)
    uniq_states = sorted(df_labeled["state"].astype(int).unique().tolist())
    for s in uniq_states:
        if s not in LABEL_MAP:
            raise ValueError(
                f"Unexpected state value: {s} (expected keys={sorted(LABEL_MAP.keys())})"
            )

    df_labeled["label"] = (
        df_labeled["state"].astype(int).apply(lambda x: LABEL_MAP[int(x)]).astype(int)
    )

    # unit 기준 train/valid/test = 8:1:1 (라벨 분포 근사 균형) - label 기준
    df_train, df_valid, df_test = split_train_valid_test_by_unit_stratified(
        df_labeled=df_labeled,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED,
        num_class=NUM_CLASS,
        label_col="label",
    )

    save_xy(TRAIN_OUT, sensor_cols, df_train, label_col="label")
    save_xy(VALID_OUT, sensor_cols, df_valid, label_col="label")
    save_xy(TEST_OUT, sensor_cols, df_test, label_col="label")

    meta = {
        "continuous_cols": sensor_cols,
        "categorical_cols": [],
        "input_dim": len(sensor_cols),
        "num_class": NUM_CLASS,
        "num_samples": {
            "train": int(len(df_train)),
            "valid": int(len(df_valid)),
            "test": int(len(df_test)),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets_used": DATASETS,
        "shift": SHIFT,
        "drop_sensors": DROP,
        "split": {
            "source": "train_FDxxx.txt only (no official test used)",
            "rule": "unit-level split within each FD (no leakage)",
            "ratios": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
            "seed": RANDOM_SEED,
            "balance": "greedy unit assignment to match label histogram (approx)",
            "stratify_on": "label",
        },
        "labeling": {
            "tags_used": TAGS,
            "integrate_rule": "state_per_cycle = max(state_tag) over available tags",
            "state_range": [0, 1, 2, 3],
            "label_map": LABEL_MAP,
            "y_saved_as": "label",
        },
    }

    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nMeta saved →", SAVE_ROOT / "meta.json")
    print("Done.")


if __name__ == "__main__":
    main()
