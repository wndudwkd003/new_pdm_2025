import pandas as pd
from pathlib import Path
import re
from tqdm.auto import tqdm

BASE = Path("/ws/new_pdm_2025/datasets/PHM15-master/data/original")
TARGET = Path("/ws/new_pdm_2025/datasets/PHM15-master/data/merged")
TARGET.mkdir(parents=True, exist_ok=True)

files = list(BASE.glob("site*"))
site_ids = sorted(
    set(int(re.findall(r"site(\d+)[abc]\.csv", f.name)[0]) for f in files)
)

for site in tqdm(site_ids):
    file_a = BASE / f"site{site}a.csv"
    file_b = BASE / f"site{site}b.csv"
    file_c = BASE / f"site{site}c.csv"

    if file_a.stat().st_size == 0 or file_b.stat().st_size == 0 or file_c.stat().st_size == 0:
        print(f"Skip site{site} (empty csv)")
        continue

    # ─────────────────────────────────────────────
    # 1) A 파일 (HVAC + Occupancy + 기타 센서)
    # ─────────────────────────────────────────────
    df_a = pd.read_csv(file_a, header=None)
    df_a.columns = [
        "component_id", "time",
        "a1","a2","a3","a4","a5","a6","a7",
        "occupancy"
    ]

    df_a["time"] = pd.to_datetime(df_a["time"], errors="coerce")
    df_a = df_a.dropna(subset=["time"])

    # ─────────────────────────────────────────────
    # 2) B 파일 (DEMAND / METER 등 전력 계측)
    #    구조: meter_id, time, b1, b2 (두 값은 사이트마다 의미 다름)
    # ─────────────────────────────────────────────
    df_b = pd.read_csv(file_b, header=None)
    df_b.columns = ["meter_id", "time", "b1", "b2"]

    df_b["time"] = pd.to_datetime(df_b["time"], errors="coerce")
    df_b = df_b.dropna(subset=["time"])

    # ─────────────────────────────────────────────
    # 3) C 파일 (fault logs)
    # ─────────────────────────────────────────────
    df_c = pd.read_csv(file_c, header=None)
    df_c.columns = ["fault_start", "fault_end", "fault_type"]

    df_c["fault_start"] = pd.to_datetime(df_c["fault_start"], errors="coerce")
    df_c["fault_end"]   = pd.to_datetime(df_c["fault_end"], errors="coerce")

    df_c = df_c.dropna(subset=["fault_start", "fault_end"]).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 4) A + B 시간 기준 outer merge
    #    → HVAC + DEMAND/METER 를 시간축으로 역순 정렬
    # ─────────────────────────────────────────────
    df = pd.merge(df_a, df_b, on="time", how="outer")
    df = df.sort_values("time").reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 5) Fault 레이블링
    # ─────────────────────────────────────────────
    df["fault_active"] = 0
    df["fault_type_active"] = None

    for i in range(len(df_c)):
        t1 = df_c.loc[i, "fault_start"]
        t2 = df_c.loc[i, "fault_end"]
        ftype = df_c.loc[i, "fault_type"]

        mask = (df["time"] >= t1) & (df["time"] <= t2)
        df.loc[mask, "fault_active"] = 1
        df.loc[mask, "fault_type_active"] = ftype

    # ─────────────────────────────────────────────
    # 6) 사이트 정보 추가
    # ─────────────────────────────────────────────
    df["site_id"] = site

    # ─────────────────────────────────────────────
    # 7) 저장
    # ─────────────────────────────────────────────
    out_path = TARGET / f"site{site}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
