from pathlib import Path
import json
import csv
import numpy as np

SAVE_DIR = Path("datasets/MPTMS/processed_data")
CSV_HAS_HEADER = True

def infer_columns_from_csv(csv_path: Path, has_header: bool = True) -> int:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader)
    if has_header:
        return len(first_row)
    return len(first_row)

def load_first_row(csv_path: Path, has_header: bool = True) -> np.ndarray:
    skip_header = 1 if has_header else 0
    arr = np.genfromtxt(str(csv_path), delimiter=",", dtype=np.float32, skip_header=skip_header)
    if arr.ndim == 0:
        arr = np.array([arr], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    row = arr[0]
    return row

def check_jsonl(jsonl_path: Path):
    total_samples = 0
    total_csv_files = 0
    total_nan_csv_files = 0
    total_all_nan_rows = 0
    total_empty_rows = 0
    col_nan_counts = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            total_samples += 1
            csv_paths = d["input_files"]["csvs"]
            for p in csv_paths:
                csv_path = Path(p)
                if not csv_path.exists():
                    print(f"missing csv file: {csv_path}")
                    continue

                row = load_first_row(csv_path, has_header=CSV_HAS_HEADER)
                total_csv_files += 1

                if row.size == 0:
                    total_empty_rows += 1
                    print(f"empty row in csv: {csv_path}")
                    continue

                nan_mask = np.isnan(row)
                if nan_mask.any():
                    total_nan_csv_files += 1
                    if col_nan_counts is None:
                        col_nan_counts = nan_mask.astype(np.int64)
                    else:
                        col_nan_counts += nan_mask.astype(np.int64)
                    if nan_mask.all():
                        total_all_nan_rows += 1
                        print(f"all-NaN row in csv: {csv_path}")

    print(f"=== {jsonl_path.name} ===")
    print(f"total_samples        : {total_samples}")
    print(f"total_csv_files      : {total_csv_files}")
    print(f"csv_files_with_nan   : {total_nan_csv_files}")
    print(f"rows_all_nan         : {total_all_nan_rows}")
    print(f"rows_empty           : {total_empty_rows}")
    if col_nan_counts is not None:
        print(f"per-column NaN counts: {col_nan_counts.tolist()}")

def main():
    for phase in ["train", "test"]:
        phase_dir = SAVE_DIR / phase
        if not phase_dir.exists():
            continue
        jsonl_files = sorted(phase_dir.glob("*.jsonl"))
        for jf in jsonl_files:
            check_jsonl(jf)

if __name__ == "__main__":
    main()
