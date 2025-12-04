"""
MPTMS 데이터를 단순 병합하여 train/valid/test 폴더별로
X.csv (features)와 y.csv (labels)를 생성하는 스크립트.

⚠ 라벨 JSON 구조는 다음과 같다고 가정:
annotations → [ { "tagging": [ { "state": "0" } ] } ]
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import random


# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
DATA_DIR = Path("datasets/MPTMS/data")
SAVE_ROOT = Path("datasets/MPTMS/processed_data")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

TRAINING_DATA = DATA_DIR / "Training"
VALIDATION_DATA = DATA_DIR / "Validation"

SOURCE_DATA = "01.원천데이터"
LABELING_DATA = "02.라벨링데이터"

NUM_CLASS = 4
CSV_HAS_HEADER = True
TRAIN_VALID_SPLIT = 0.8
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# 라벨 JSON 파싱 (MPTMS 전용)
# ─────────────────────────────────────────────
def parse_label_json(json_path: Path) -> int:
    """
    라벨은 다음 구조에 항상 존재한다고 가정한다.
    {
      ...
      "annotations": [
        {
          "tagging": [
            { "annotation_type": "tagging", "state": "0" }
          ]
        }
      ]
    }
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Label json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        state = data["annotations"][0]["tagging"][0]["state"]
    except Exception as e:
        raise ValueError(f"Cannot parse label from {json_path}") from e

    if not isinstance(state, str) or not state.isdigit():
        raise ValueError(f"Invalid label value in {json_path}: {state}")

    y = int(state)
    if y < 0 or y >= NUM_CLASS:
        raise ValueError(f"Label {y} out of range in: {json_path}")

    return y


# ─────────────────────────────────────────────
# CSV feature row 읽기
# ─────────────────────────────────────────────
def read_feature_row(csv_path: Path, has_header: bool = True):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if has_header:
        rows = rows[1:]

    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    return [v.strip() for v in rows[0]]


# ─────────────────────────────────────────────
# 전체 데이터 수집 (Training 전체 / Validation 전체)
# ─────────────────────────────────────────────
def collect_all(base_dir: Path, label_phase_key: str):
    """
    base_dir/SOURCE_DATA 아래의 모든 device(TS_agv_01_...) 폴더에서
    feature + label을 수집하여 리스트로 반환.

    label_phase_key:
      "train_all" → Training → TS_ → TL_
      "test"      → Validation → VS_ → VL_
    """

    print(f"\nCollecting from {base_dir} ...")
    source_root = base_dir / SOURCE_DATA
    device_dirs = sorted(source_root.glob("*"))

    if not device_dirs:
        raise ValueError(f"No device dirs found under {source_root}")

    X_list = []
    y_list = []
    feature_cols = None

    for d in tqdm(device_dirs, desc="devices"):
        bin_files = sorted(d.glob("*.bin"))
        if not bin_files:
            continue

        # 첫 CSV에서 feature 컬럼 결정
        if feature_cols is None:
            first_csv = bin_files[0].with_suffix(".csv")
            with open(first_csv, "r", encoding="utf-8", newline="") as f:
                first_row = next(csv.reader(f))
            feature_cols = [c.strip() for c in first_row]

        for bin_file in tqdm(bin_files, desc=d.name, leave=False):
            csv_path = bin_file.with_suffix(".csv")

            # 라벨 JSON Path
            if label_phase_key == "train_all":
                # TS_ → TL_
                label_dir = Path(str(bin_file.parent).replace(SOURCE_DATA, LABELING_DATA).replace("TS_", "TL_"))
            else:
                # VS_ → VL_
                label_dir = Path(str(bin_file.parent).replace(SOURCE_DATA, LABELING_DATA).replace("VS_", "VL_"))

            label_path = label_dir / (bin_file.stem + ".json")

            # 읽기
            X = read_feature_row(csv_path, has_header=CSV_HAS_HEADER)
            y = parse_label_json(label_path)

            X_list.append(X)
            y_list.append([y])

    return feature_cols, X_list, y_list


# ─────────────────────────────────────────────
# CSV로 저장
# ─────────────────────────────────────────────
def save_xy(save_dir: Path, feature_cols, X_list, y_list):
    save_dir.mkdir(parents=True, exist_ok=True)

    x_path = save_dir / "X.csv"
    y_path = save_dir / "y.csv"

    with open(x_path, "w", encoding="utf-8", newline="") as fx:
        writer = csv.writer(fx)
        writer.writerow(feature_cols)
        writer.writerows(X_list)

    with open(y_path, "w", encoding="utf-8", newline="") as fy:
        writer = csv.writer(fy)
        writer.writerow(["label"])
        writer.writerows(y_list)

    print(f"Saved X: {len(X_list)} rows → {x_path}")
    print(f"Saved y: {len(y_list)} rows → {y_path}")


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────
def main():
    random.seed(RANDOM_SEED)

    # 1) Training 전체 수집
    feature_cols, X_all, y_all = collect_all(TRAINING_DATA, label_phase_key="train_all")

    # 2) train/valid split
    total = len(X_all)
    idxs = list(range(total))
    random.shuffle(idxs)

    split = int(total * TRAIN_VALID_SPLIT)
    train_idx = idxs[:split]
    valid_idx = idxs[split:]

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_valid = [X_all[i] for i in valid_idx]
    y_valid = [y_all[i] for i in valid_idx]

    save_xy(SAVE_ROOT / "train", feature_cols, X_train, y_train)
    save_xy(SAVE_ROOT / "valid", feature_cols, X_valid, y_valid)

    # 3) Validation → test
    _, X_test, y_test = collect_all(VALIDATION_DATA, label_phase_key="test")
    save_xy(SAVE_ROOT / "test", feature_cols, X_test, y_test)

    # 4) meta.json
    meta = {
        "continuous_cols": feature_cols,
        "categorical_cols": [],
        "input_dim": len(feature_cols),
        "num_class": NUM_CLASS,
        "num_samples": {
            "train": len(X_train),
            "valid": len(X_valid),
            "test": len(X_test),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(SAVE_ROOT / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nMeta saved.")


if __name__ == "__main__":
    main()
