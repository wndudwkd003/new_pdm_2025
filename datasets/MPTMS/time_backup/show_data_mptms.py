import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path

import os

DATA_DIR = Path("/home/juyoung-lab/ws/dev_ws/pi3/datasets/MPTMS/data")

TRAINING_DATA = DATA_DIR / "Training"
VALIDATION_DATA = DATA_DIR / "Validation"

SOURCE_DATA = "01.원천데이터"
LABELING_DATA = "02.라벨링데이터"

TEST_DATA = "datasets/MPTMS/data/Training/01.원천데이터/TS_agv_02_agv02_0902_1306"
TEST_DATA = Path(TEST_DATA)



# 라벨링 데이터 경로
label_path_str = str(TEST_DATA).replace(SOURCE_DATA, LABELING_DATA).replace("TS", "TL")
LABEL_DATA = Path(label_path_str)

# 저장 폴더 준비 및 안내
IMAGES_DIR = TEST_DATA / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
print(f"이미지 저장 폴더: {IMAGES_DIR}")

# GUI 가능 여부(창 표시 조건)
HAS_GUI = (os.environ.get("DISPLAY") is not None) and hasattr(cv2, "imshow")

def extract_time(file_path):
    stem = file_path.stem
    return stem.split('_')[-1]

def get_corresponding_json(bin_file):
    json_name = bin_file.stem + ".json"
    json_path = LABEL_DATA / json_name
    return json_path if json_path.exists() else None

def get_corresponding_csv(bin_file):
    csv_name = bin_file.stem + ".csv"
    csv_path = TEST_DATA / csv_name
    return csv_path if csv_path.exists() else None

def print_annotation_colored(state):
    """어노테이션 상태에 따라 색깔로 출력"""
    colors = {
        '0': '\033[92m',  # 녹색
        '1': '\033[93m',  # 노란색
        '2': '\033[94m',  # 파란색
        '3': '\033[91m'   # 빨간색
    }
    reset = '\033[0m'
    color = colors.get(str(state), '\033[0m')
    print(f"Annotation state: {color}{state}{reset}")

def main():
    bin_files = list(TEST_DATA.glob("*.bin"))
    bin_files = sorted(bin_files, key=extract_time)

    print(f"총 {len(bin_files)}개의 bin 파일을 찾았습니다.")

    for i, bin_file in enumerate(bin_files, 1):
        print(f"\n[{i}/{len(bin_files)}] {bin_file.name}")

        # CSV 파일 읽기
        csv_path = get_corresponding_csv(bin_file)
        if csv_path:
            df = pd.read_csv(csv_path)
            print("센서 데이터:")
            for col in df.columns:
                value = df[col].iloc[0]
                print(f"  {col}: {value}")
        else:
            print("해당 CSV 파일 없음")

        # JSON 파일 읽기
        json_path = get_corresponding_json(bin_file)
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                annotation = data.get('annotations', [])
                if annotation:
                    state = annotation[0]['tagging'][0]['state']
                    print_annotation_colored(state)
        else:
            print("해당 JSON 파일 없음")

        # IR → 컬러맵 → 리사이즈
        ir_image = np.load(bin_file)
        image_normalized = cv2.normalize(ir_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_colored = cv2.applyColorMap(image_normalized, cv2.COLORMAP_JET)

        scale = 3
        h, w = img_colored.shape[:2]
        print(f"원본 이미지 크기: {w}x{h}") # 160 120

        exit()
        img_resized = cv2.resize(img_colored, (int(w * scale), int(h * scale)))

        # JPEG 저장 (항상 저장)
        save_path = IMAGES_DIR / f"{bin_file.stem}.jpg"
        cv2.imwrite(str(save_path), img_resized)
        print(f"이미지 저장: {save_path}")

        # GUI 가능하면 창 표시도 수행
        if HAS_GUI:
            cv2.imshow("IR Image", img_resized)
            key = cv2.waitKey(33)
            if key == 27:  # ESC
                break

    if HAS_GUI:
        cv2.destroyAllWindows()

    print(f"{TEST_DATA}")
if __name__ == "__main__":
    main()
