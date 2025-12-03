from scipy.io import arff
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm

from xgboost import XGBClassifier  # ← 추가

# ────────────────────────────────────────────────────
# Path 설정
# ────────────────────────────────────────────────────
DATA_DIR = Path("datasets/fordengine/data")

TRAIN_PATH = DATA_DIR / "FordA_TRAIN.arff"
TEST_PATH  = DATA_DIR / "FordA_TEST.arff"

PROCESSED_DIR = Path("datasets/fordengine/processed_data")
PLOT_DIR = PROCESSED_DIR / "plots"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────
# 하이퍼파라미터
# ────────────────────────────────────────────────────
FORWARD = 30
BACKWARD = 10
WINDOW = 3
FEATURE_DIM = 500      # 원본 feature 개수 (FordA 기준)
NUM_CLASS = 2

SELECT_K = 100         # ← XGBoost로 뽑을 상위 feature 개수 (원하시는 값으로 조정)


# ────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────
def load_arff(path: str):
    raw_data, meta = arff.loadarff(path)
    df = pd.DataFrame(raw_data)

    X = df.iloc[:, :-1].astype(float).values

    # 원본 라벨
    y = df.iloc[:, -1].astype(int).values      # 예: [-1, 1, -1, 1, ...]
    uniq = np.unique(y)

    # 라벨이 {-1, 1}이면 {0, 1}로 매핑
    if set(uniq.tolist()) == set([-1, 1]):
        y = (y == 1).astype(int)              # -1 → 0, 1 → 1

    return X, y


# ────────────────────────────────────────────────────
# XGBoost 기반 feature selection (테이블 수준에서 실행)
# ────────────────────────────────────────────────────
def select_top_k_features_xgb(X: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    - X: (N, F)
    - y: (N,)
    - k: 선택할 feature 개수

    반환:
    - selected_indices: 선택된 feature 인덱스 (오름차순)
    - selected_ratio:  선택된 feature 개수 / 전체 feature 개수
    - dropped_ratio:   제외된 feature 개수 / 전체 feature 개수
    - importances:     feature importance (원본 인덱스 순서, shape (F,))
    """
    n_features = X.shape[1]

    if NUM_CLASS == 2:
        objective = "binary:logistic"
        num_class = None
    else:
        objective = "multi:softprob"
        num_class = NUM_CLASS

    model_kwargs = {
        "n_estimators": 1000,
        "objective": objective,
        "device": "cuda",
        "tree_method": "hist",
    }
    if num_class is not None:
        model_kwargs["num_class"] = num_class

    model = XGBClassifier(**model_kwargs)
    model.fit(X, y)

    importances = model.feature_importances_   # (F,)
    # 중요도 내림차순 정렬
    sorted_idx_desc = np.argsort(importances)[::-1]
    selected_indices = sorted_idx_desc[:k]
    selected_indices = np.sort(selected_indices)  # 오름차순 정렬

    selected_ratio = float(len(selected_indices)) / float(n_features)
    dropped_ratio  = 1.0 - selected_ratio

    return selected_indices, selected_ratio, dropped_ratio, importances


# ────────────────────────────────────────────────────
# Sliding Window
# ────────────────────────────────────────────────────
def sliding_window_with_time(X: np.ndarray, y: np.ndarray):
    """
    X: (L, F)   - 시간축 L, feature F
    y: (L,)     - 각 시점 라벨
    """
    samples_X = []
    samples_y = []

    total_length = X.shape[0]

    for start in range(0, total_length - FORWARD - BACKWARD + 1, WINDOW):
        end = start + FORWARD
        X_win = X[start:end]              # (30, F)
        y_win = y[end:end + BACKWARD]     # (10,)
        samples_X.append(X_win)
        samples_y.append(y_win)

    return np.array(samples_X), np.array(samples_y)


def save_feature_selection_info_json(
    save_path: Path,
    selected_indices: np.ndarray,
    dropped_indices: np.ndarray,
    selected_ratio: float,
    dropped_ratio: float,
    importances: np.ndarray,
):
    """
    선택된 feature / 선택되지 않은 feature 를 별도 JSON 파일로 저장.
    - selected_indices, dropped_indices: 원본 feature 인덱스 기준
    - importances: 원본 인덱스 순서대로의 feature importance (shape: (F,))
    """
    obj = {
        "total_features": int(importances.shape[0]),
        "select_k": int(len(selected_indices)),
        "selected_indices": selected_indices.tolist(),
        "dropped_indices": dropped_indices.tolist(),
        "selected_ratio": float(selected_ratio),
        "dropped_ratio": float(dropped_ratio),
        # 중요도는 원본 index 순서대로 기록
        "feature_importances": importances.tolist(),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ────────────────────────────────────────────────────
# Step-wise label distribution (t+1~t+10)
# ────────────────────────────────────────────────────
def plot_stepwise_label_distribution(name: str, y_sw: np.ndarray):
    num_steps = y_sw.shape[1]

    step_labels = {step: Counter(y_sw[:, step]) for step in range(num_steps)}

    plt.figure(figsize=(12, 6))

    steps = np.arange(1, num_steps + 1)
    zeros = [step_labels[s].get(0, 0) for s in range(num_steps)]
    ones  = [step_labels[s].get(1, 0) for s in range(num_steps)]

    width = 0.35
    plt.bar(steps - width/2, zeros, width=width, label="class 0")
    plt.bar(steps + width/2, ones,  width=width, label="class 1")

    plt.xticks(steps)
    plt.xlabel("Step (t+1 ~ t+10)")
    plt.ylabel("Count")
    plt.title(f"Step-wise Label Distribution - {name}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_DIR / f"label_step_dist_{name}.png", dpi=200)
    plt.close()


# ────────────────────────────────────────────────────
# 전체 label 분포 + example sequences
# ────────────────────────────────────────────────────
def plot_label_distribution(name: str, y_sw: np.ndarray):
    flat_labels = y_sw.reshape(-1)
    counts = Counter(flat_labels)

    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title(f"Label Distribution - {name}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_DIR / f"label_dist_{name}.png", dpi=200)
    plt.close()

    # Sequence 예시
    num_examples = min(5, y_sw.shape[0])
    example_indices = np.linspace(0, y_sw.shape[0] - 1, num_examples, dtype=int)

    plt.figure(figsize=(10, 6))
    for idx in example_indices:
        plt.plot(y_sw[idx], label=f"sample {idx}")
    plt.title(f"Example Label Sequences - {name}")
    plt.xlabel("t")
    plt.ylabel("label")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_DIR / f"label_sequences_{name}.png", dpi=200)
    plt.close()


# ────────────────────────────────────────────────────
# JSONL 저장
# ────────────────────────────────────────────────────
def save_jsonl_sample(
    save_path: Path,
    sample_id: str,
    X: np.ndarray,
    y: np.ndarray,
    phase: str,
    selected_feature_indices: np.ndarray,
    selected_ratio: float,
    dropped_ratio: float,
):
    # 실제 feature 차원에 맞춰서 정의
    feature_dim = X.shape[1]
    continuous_cols = [f"f{i}" for i in range(feature_dim)]
    categorical_cols: list[str] = []

    obj = {
        "sample_id": sample_id,
        "input": {"X": X.tolist()},
        "target": {"y": y.tolist()},
        "metadata": {
            # Datasets.load_data() 가 기대하는 필드들
            "continuous_cols": continuous_cols,
            "categorical_cols": categorical_cols,
            "forward": FORWARD,
            "backward": BACKWARD,
            "num_class": NUM_CLASS,

            # 부가 정보
            "feature_dim": feature_dim,                 # 선택된 feature 개수
            "original_feature_dim": FEATURE_DIM,        # 원본 feature 개수
            "selected_feature_indices": selected_feature_indices.tolist(),
            "selected_feature_ratio": float(selected_ratio),     # 선택된 컬럼 비율
            "dropped_feature_ratio": float(dropped_ratio),       # 선택되지 않은 컬럼 비율
            "data_phase": phase,
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def export_split(
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    selected_feature_indices: np.ndarray,
    selected_ratio: float,
    dropped_ratio: float,
):
    out_dir = PROCESSED_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(X)), desc=f"Exporting {split_name}"):
        sample_id = f"FORD_{split_name}_{idx:06d}"
        filepath = out_dir / f"{sample_id}.jsonl"
        save_jsonl_sample(
            filepath,
            sample_id,
            X[idx],
            y[idx],
            split_name,
            selected_feature_indices,
            selected_ratio,
            dropped_ratio,
        )


# ────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────
def main():
    # 1) 원본 로드 (테이블 형태)
    X_train_raw, y_train_raw = load_arff(str(TRAIN_PATH))
    X_test_raw,  y_test_raw  = load_arff(str(TEST_PATH))

    # 2) 테이블 상태에서 XGBoost로 feature importance → 상위 SELECT_K개 선택
    (
        selected_feature_indices,
        selected_ratio,
        dropped_ratio,
        importances,
    ) = select_top_k_features_xgb(
        X_train_raw,
        y_train_raw,
        SELECT_K,
    )

    print(f"Total features: {X_train_raw.shape[1]}")
    print(f"Selected top-{SELECT_K} features indices: {selected_feature_indices}")
    print(f"Selected ratio: {selected_ratio * 100:.2f} %, Dropped ratio: {dropped_ratio * 100:.2f} %")

    # ---- 여기서 별도 JSON 파일로 저장 ----
    all_indices = np.arange(X_train_raw.shape[1])
    dropped_indices = np.setdiff1d(all_indices, selected_feature_indices)

    feature_info_path = PROCESSED_DIR / "ford_feature_selection.json"
    save_feature_selection_info_json(
        feature_info_path,
        selected_feature_indices,
        dropped_indices,
        selected_ratio,
        dropped_ratio,
        importances,
    )
    # -----------------------------------

    # 3) 선택된 feature만 남기고 나머지는 버림
    X_train_raw_sel = X_train_raw[:, selected_feature_indices]
    X_test_raw_sel  = X_test_raw[:,  selected_feature_indices]

    # 4) Sliding window (선택된 feature subset에 대해 수행)
    X_train_sw, y_train_sw = sliding_window_with_time(X_train_raw_sel, y_train_raw)
    X_test_sw,  y_test_sw  = sliding_window_with_time(X_test_raw_sel,  y_test_raw)

    # Train → Train/Valid (8:2)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sw, y_train_sw,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # (이하 그대로)
    plot_label_distribution("train", y_tr)
    plot_stepwise_label_distribution("train", y_tr)

    plot_label_distribution("valid", y_val)
    plot_stepwise_label_distribution("valid", y_val)

    plot_label_distribution("test", y_test_sw)
    plot_stepwise_label_distribution("test", y_test_sw)

    export_split("train", X_tr,   y_tr,   selected_feature_indices, selected_ratio, dropped_ratio)
    export_split("valid", X_val,  y_val,  selected_feature_indices, selected_ratio, dropped_ratio)
    export_split("test",  X_test_sw, y_test_sw, selected_feature_indices, selected_ratio, dropped_ratio)



# 실행
if __name__ == "__main__":
    main()
