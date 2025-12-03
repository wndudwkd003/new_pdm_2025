"""
MPTMS 데이터를 장치에 따라 분류하고, 시계열 데이터로 변환하는 스크립트.
데이터는 forward=앞의 데이터, backward=뒤의 예측 데이터, interval=각 데이터의 간격(초단위)로 설정
예를들어 forward=12, backward=2, interval_sec=300이면, 5분(300초) 간격의 데이터 12개를 입력으로하여,
다음 2개의 상태를 예측하는 데이터로 변환

===> 데이터를 슬라이딩 윈도우로 만들어야 함
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import csv

# --- 전역 변수 설정 ---
DATA_DIR = Path("datasets/MPTMS/data")
SAVE_DIR = Path("datasets/MPTMS")

SAVE_DIR = SAVE_DIR / "processed_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATA = DATA_DIR / "Training"
VALIDATION_DATA = DATA_DIR / "Validation"

SOURCE_DATA = "01.원천데이터"
LABELING_DATA = "02.라벨링데이터"

FORWARD = 30 # 90초-> 1분 30초
BACKWARD = 10 # 30초
INTERVAL_SEC = 3 # 초

def infer_columns_from_csv(csv_path: Path, has_header: bool = True) -> list[str]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)
    if has_header:
        # CSV 첫 행이 헤더라고 가정
        return [c.strip() for c in first_row]
    # 헤더가 없다면 열 개수만큼 이름 생성
    return [f"col_{i}" for i in range(len(first_row))]

def extract_datetime(file_path):
    """파일명에서 datetime 추출"""
    stem = file_path.stem
    parts = stem.split('_')
    date_part = parts[-2]
    time_part = parts[-1]
    datetime_str = f"2024{date_part}{time_part}"
    return datetime.strptime(datetime_str, "%Y%m%d%H%M%S")

def filter_by_interval(files, interval_seconds): # <-- 변경 (interval_minutes -> interval_seconds)
    """interval 간격(초)으로 파일 필터링"""
    if not files:
        return []

    files_with_time = sorted([(f, extract_datetime(f)) for f in files], key=lambda x: x[1])

    filtered = []
    last_selected_file, last_selected_time = files_with_time[0]
    filtered.append(last_selected_file)

    # 초 단위로 다음 목표 시간 설정
    next_target_time = last_selected_time + timedelta(seconds=interval_seconds) # <-- 변경 (minutes -> seconds)

    for file_path, file_time in files_with_time[1:]:
        if file_time >= next_target_time:
            filtered.append(file_path)
            last_selected_time = file_time
            # 초 단위로 다음 목표 시간 갱신
            next_target_time = last_selected_time + timedelta(seconds=interval_seconds) # <-- 변경 (minutes -> seconds)

    return [Path(f) for f in filtered]

def sliding_window_with_time(data, forward=FORWARD, backward=BACKWARD, interval_sec=INTERVAL_SEC): # <-- 변경
    """슬라이딩 윈도우로 데이터 생성"""
    print(f"    - 원본 파일 개수: {len(data)}")

    filtered_files = filter_by_interval(data, interval_sec) # <-- 변경

    print(f"    - 필터링 후 파일 개수: {len(filtered_files)} (윈도우 생성에 최소 {forward + backward}개 필요)")

    window_size = forward + backward
    windows = []

    for i in range(len(filtered_files) - window_size + 1):
        window = filtered_files[i:i + window_size]
        windows.append({
            'input_files': window[:forward],
            'target_files': window[forward:]
        })
    print(f"    - 생성된 샘플 개수: {len(windows)}") # <-- 생성된 샘플 개수 확인
    return windows

def get_label_path(source_path, data_type):
    """소스 경로에서 라벨 경로 생성"""
    source_str = str(source_path)

    if data_type == "train":
        label_str = source_str.replace(SOURCE_DATA, LABELING_DATA).replace("TS_", "TL_")
    else:  # validation
        label_str = source_str.replace(SOURCE_DATA, LABELING_DATA).replace("VS_", "VL_")

    return Path(label_str)

def save_processed_data(
    data,
    base_name,
    phase,
    save_dir=SAVE_DIR,
    csv_has_header: bool = True,
):
    """처리된 데이터 경로만 저장 (JSONL 형식으로 수정)"""
    if not data:
        print(f"Saved {base_name} metadata for 0 samples.")
        return

    # 저장 경로의 확장자를 .jsonl로 설정
    save_phase_dir = save_dir / phase
    save_phase_dir.mkdir(parents=True, exist_ok=True)
    save_path = (save_phase_dir / base_name).with_suffix('.jsonl')

    # CSV 헤더 추출
    first_csv = Path(str(data[0]['input_files'][0])).with_suffix(".csv")
    continuous_cols = infer_columns_from_csv(first_csv, has_header=csv_has_header)
    categorical_cols: list[str] = []

    target_names = [f"y_t+{i+1}" for i in range(BACKWARD)]

    common_meta = {
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "target_names": target_names,
        "forward": FORWARD,
        "backward": BACKWARD,
        "interval_sec": INTERVAL_SEC,
        "base_name": base_name,
        "data_phase": phase,
    }

    # with 문을 사용하여 파일을 열고, 각 샘플을 한 줄씩 씁니다.
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, window_data in enumerate(data):
            label_paths = []
            for bin_file in window_data['target_files']:
                label_dir = get_label_path(bin_file.parent, phase)
                json_file = label_dir / (bin_file.stem + '.json')
                label_paths.append(str(json_file))

            sample_metadata = {
                'sample_id': f"{base_name}_sample_{idx:04d}",
                'input_files': {
                    'images': [str(f) for f in window_data['input_files']],
                    'csvs': [str(f.with_suffix('.csv')) for f in window_data['input_files']]
                },
                'target_files': {'labels': label_paths},
                'metadata': common_meta,
            }

            json_string = json.dumps(sample_metadata, ensure_ascii=False)
            f.write(json_string + '\n')

    print(f"Saved {len(data)} samples to {save_path}")

def main():
    target_dirs = {
        "train": TRAINING_DATA,
        "test": VALIDATION_DATA
    }

    # 전체 샘플 개수를 저장할 딕셔너리 초기화
    total_samples_summary = {key: 0 for key in target_dirs.keys()}

    for phase, target_dir in target_dirs.items():
        print(f"\n--- Processing {phase} data ---")
        source_dir = target_dir / SOURCE_DATA

        target_device_dirs = sorted(list(source_dir.glob("*")))

        print(f"Total directories: {len(target_device_dirs)}")
        for target_device_dir in target_device_dirs:
            base_name = target_device_dir.name
            print(f"Processing directory: {base_name}")

            sample_files = list(target_device_dir.glob("*.bin"))
            processed_samples = sliding_window_with_time(sample_files)

            # 해당 phase의 전체 샘플 개수에 현재 생성된 샘플 개수를 더함
            total_samples_summary[phase] += len(processed_samples) # <-- 추가

            save_processed_data(processed_samples, base_name, phase)

    # --- 최종 요약 출력 ---
    print("\n" + "="*30)
    print("      Total Summary")
    print("="*30)
    for phase, count in total_samples_summary.items():
        print(f"Total {phase.upper()} samples created: {count}")
    print("="*30)

if __name__ == "__main__":
    main()
