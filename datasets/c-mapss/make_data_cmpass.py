import pandas as pd, json, sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np

# ── 설정 ────────────────────────────────────────────
DATASETS  = ['FD001', 'FD003']   # 처리할 파일
SHIFT     = 0                                      # 0 = 그대로, 1·2 … = 뒤로 밀기

BASE_DIR  = Path('datasets/c-mapss/data/CMaps')
MAP_FILE  = Path('datasets/c-mapss/data/sensor_udc_map.json')

PLOT_ROOT = Path('datasets/c-mapss/processed_data/engine_knee_plots_multi_no_normal')
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

# 슬라이딩 윈도우 설정 (MPTMS 스크립트와 맞추기)
FORWARD      = 30
BACKWARD     = 10
WINDOW_SIZE  = FORWARD + BACKWARD

JSON_OUTROOT = Path('datasets/c-mapss/processed_data')
JSON_OUTROOT.mkdir(parents=True, exist_ok=True)

TRAIN_OUTROOT = JSON_OUTROOT / "train"
TRAIN_OUTROOT.mkdir(parents=True, exist_ok=True)

TEST_OUTROOT = JSON_OUTROOT / "test"
TEST_OUTROOT.mkdir(parents=True, exist_ok=True)

# step 단위 CSV / JSON 라벨 저장 루트
STEP_ROOT = JSON_OUTROOT / "steps"
STEP_ROOT.mkdir(parents=True, exist_ok=True)

# 플롯 저장 여부
SAVE_PLOTS = True  # False 로 바꾸면 플롯 안 만들고 데이터만 생성

# train/test split 설정
RANDOM_SEED = 42
TRAIN_RATIO = 0.9

# ── 매핑 로드 ───────────────────────────────────────
if not MAP_FILE.exists():
    sys.exit('sensor_udc_map.json not found')
mapping_all = json.loads(MAP_FILE.read_text())      # {FDxxx: {sensor: tag}}

# ── 공통 정보 ──────────────────────────────────────
DROP   = ['s1','s5','s6','s10','s16','s18','s19']
COLS   = ['unit','cycle','set1','set2','set3'] + [f's{i}' for i in range(1,22)]
FEATURE_COLS = [c for c in COLS if c not in ['unit', 'cycle']]  # CSV에 찍을 피처들

COLORS = ['#8fd175', '#fff07e', '#f6b08c', '#d9534f']  # normal→danger
ALPHA  = 0.15


def edges_10(y, tag):
    x = np.arange(len(y))
    if   tag == 'u':
        y1, curve, d = y, 'concave', 'increasing'
    elif tag == 'd':
        y1, curve, d = y, 'convex',  'decreasing'
    else:
        y1, curve, d = np.abs(y - y.mean()), 'concave', 'increasing'

    k = KneeLocator(x, y1, curve=curve, direction=d, S=2.0)
    idx = sorted(k.all_knees)[:9]
    while len(idx) < 9:
        q = int(len(y) * (len(idx) + 1) / 10)
        if q not in idx:
            idx.append(q)
    idx = sorted(idx)[:9]
    return [0] + idx + [len(y) - 1]                   # 11 경계


def smooth(v, alpha=0.05):
    return pd.Series(v).ewm(alpha=alpha, adjust=False).mean().to_numpy()


all_labeled = []  # ← 전체 결과를 모을 리스트

# ── 1단계: 원본 코드 그대로 상태 라벨링 + 플롯 생성 ─────────────
for fd in DATASETS:
    print(f'\n=== Processing {fd} ===')
    fd_map = mapping_all.get(fd)
    if not fd_map:
        print('  ↳ 매핑이 없습니다. 건너뜀')
        continue

    fpath = BASE_DIR / f'train_{fd}.txt'
    df_raw = pd.read_csv(fpath, sep=r'\s+', header=None, names=COLS)

    sensors = [
        s for s in df_raw.columns
        if s.startswith('s') and s[1:].isdigit() and s not in DROP and s in fd_map
    ]

    df = df_raw.copy()
    df_norm = df_raw.copy()
    for s in sensors:
        df_norm[s] = MinMaxScaler().fit_transform(df_raw[[s]])

    groups = {k: [s for s in sensors if fd_map[s] == k] for k in ['u', 'd', 'c', 'o']}
    out_dir = PLOT_ROOT / fd
    out_dir.mkdir(parents=True, exist_ok=True)

    for eid, g_raw in df.groupby('unit'):             # g_raw = 원본
        g_raw = g_raw.sort_values('cycle')
        g_norm = df_norm[df_norm.unit == eid].sort_values('cycle')  # 정규화-본
        cyc = g_raw.cycle.to_numpy()

        # ── (1) 이 엔진(unit)에 대한 step 단위 CSV 생성 ─────────────
        unit_dir_csv = STEP_ROOT / fd / f"unit_{int(eid):03d}"
        unit_dir_csv.mkdir(parents=True, exist_ok=True)

        for _, row in g_raw.iterrows():
            cyc_val = int(row["cycle"])
            csv_path = unit_dir_csv / f"cycle_{cyc_val:05d}.csv"
            if not csv_path.exists():
                row_feat = row[FEATURE_COLS].to_numpy(dtype=np.float32)[None, :]  # (1, F)
                np.savetxt(csv_path, row_feat, delimiter=',')

        # ── (2) 플롯 및 태그별 state 생성 ───────────────────────────
        if SAVE_PLOTS:
            fig, ax = plt.subplots(figsize=(14, 4))

            # 회색 센서 궤적 (정규화 값)
            for s in sensors:
                ax.plot(cyc, g_norm[s], color='grey', alpha=.3, lw=.35)

        for tag, cols in groups.items():
            if not cols:
                continue

            m_norm = g_norm[cols].mean(axis=1).values
            m_line = smooth(m_norm)
            edges  = edges_10(m_norm, tag)

            base  = [min(i, 10) for i in [SHIFT, SHIFT + 3, SHIFT + 6, SHIFT + 9, 10]]
            seg_idx = [edges[i] for i in base]
            seg_cyc = [cyc[i] for i in seg_idx]
            seg_cyc[-1] = seg_cyc[-1] + 1

            if SAVE_PLOTS:
                for (l, r), c in zip(zip(seg_cyc[:-1], seg_cyc[1:]), COLORS):
                    ax.axvspan(l, r, color=c, alpha=ALPHA)

                ax.plot(cyc, m_line, lw=2, label=f'{tag} mean')

            # ── 상태 라벨링 (원본과 동일) ──
            state_label = np.zeros(len(cyc), dtype=int)
            for i in range(len(seg_cyc) - 1):
                mask = (cyc >= seg_cyc[i]) & (cyc < seg_cyc[i + 1])
                state_label[mask] = i

            g_out = g_raw.copy()
            g_out['state'] = state_label
            g_out['dataset'] = fd
            g_out['tag'] = tag          # 태그 정보 추가
            all_labeled.append(g_out)

            # ── (3) 이 tag에 대한 라벨 JSON 생성 ───────────────────
            label_unit_dir = STEP_ROOT / fd / f"tag_{tag}" / f"unit_{int(eid):03d}"
            label_unit_dir.mkdir(parents=True, exist_ok=True)

            for cyc_val, st in zip(cyc, state_label):
                cyc_int = int(cyc_val)
                label_path = label_unit_dir / f"label_{cyc_int:05d}.json"
                if not label_path.exists():
                    payload = {
                        "annotations": [
                            {
                                "tagging": [
                                    {"state": str(int(st))}
                                ]
                            }
                        ]
                    }
                    label_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        if SAVE_PLOTS:
            ax.set_xlabel('Cycle')
            ax.set_ylabel('Scaled Value')
            ax.set_title(f'{fd} – Engine {eid}  (shift={SHIFT})')
            ax.legend(loc='upper left', fontsize='small')
            fig.tight_layout()
            fig.savefig(out_dir / f'{fd}_engine_{eid}.png', dpi=150)
            plt.close(fig)
            print(f'  ↳ {fd} engine {eid} → {out_dir / f"{fd}_engine_{eid}.png"}')

# ── 통합 CSV 저장 ─────────────────────────────────
df_all = pd.concat(all_labeled, ignore_index=True)
csv_out = PLOT_ROOT / 'all_engines_labeled.csv'
df_all.to_csv(csv_out, index=False)
print('\n✔ 통합 CSV 저장 완료 →', csv_out)

# ── 2단계: 슬라이딩 윈도우(JSONL) 생성 (MPTMS 형식 상위 구조) ─────────

# feature(연속형) 컬럼 자동 추론 (메타데이터용)
continuous_cols = [
    c for c in df_all.columns
    if c not in ['unit', 'cycle', 'dataset', 'state', 'tag']
]
categorical_cols = []
target_names = [f"state_t+{i+1}" for i in range(BACKWARD)]

common_meta_base = {
    "continuous_cols": continuous_cols,
    "categorical_cols": categorical_cols,
    "target_names": target_names,
    "forward": FORWARD,
    "backward": BACKWARD,
    "interval_sec": None,        # CMAPSS는 시간 간격이 없어 None으로 둠
    "data_phase": "train",
}

print("\n=== Sliding-window JSONL 생성 시작 ===")

rng = np.random.default_rng(RANDOM_SEED)

# dataset(FD001~4) × tag(u/d/c/o) 별로 따로 파일 생성
for fd, df_fd in df_all.groupby('dataset'):
    for tag, df_tag in df_fd.groupby('tag'):
        # 샘플을 먼저 전부 메모리에 모은 뒤 train/test로 나눔
        samples: list[dict] = []

        meta = dict(common_meta_base)
        meta["base_name"] = fd
        meta["tag"] = tag

        for unit_id, df_eng in df_tag.groupby('unit'):
            df_eng = df_eng.sort_values('cycle').reset_index(drop=True)
            n = len(df_eng)

            if n >= WINDOW_SIZE:
                unit_int = int(unit_id)
                unit_dir_csv  = STEP_ROOT / fd / f"unit_{unit_int:03d}"
                label_unit_dir = STEP_ROOT / fd / f"tag_{tag}" / f"unit_{unit_int:03d}"

                for start in range(0, n - WINDOW_SIZE + 1):
                    in_slice  = df_eng.iloc[start : start + FORWARD]
                    tgt_slice = df_eng.iloc[start + FORWARD : start + WINDOW_SIZE]

                    # forward 개수만큼 CSV 경로
                    csv_paths = []
                    for cyc_val in in_slice["cycle"].tolist():
                        cyc_int = int(cyc_val)
                        csv_path = unit_dir_csv / f"cycle_{cyc_int:05d}.csv"
                        csv_paths.append(str(csv_path))

                    # backward 개수만큼 라벨 JSON 경로
                    label_paths = []
                    for cyc_val in tgt_slice["cycle"].tolist():
                        cyc_int = int(cyc_val)
                        label_path = label_unit_dir / f"label_{cyc_int:05d}.json"
                        label_paths.append(str(label_path))

                    sample_metadata = {
                        "sample_id": f"{fd}_tag{tag}_u{unit_int:03d}_s{len(samples):05d}",
                        "input_files": {
                            "images": [],        # CMAPSS는 실제 이미지 없음
                            "csvs":   csv_paths  # ← forward 개수만큼
                        },
                        "target_files": {
                            "labels": label_paths  # ← backward 개수만큼
                        },
                        "metadata": {
                            **meta,
                            "dataset": fd,
                            "unit": unit_int,
                            "cycles_input":  in_slice["cycle"].tolist(),
                            "cycles_target": tgt_slice["cycle"].tolist(),
                            "states_target": tgt_slice["state"].astype(int).tolist(),
                        },
                    }

                    samples.append(sample_metadata)
            else:
                print(f"  - {fd}, tag={tag}, unit {int(unit_id)}: length {n} < {WINDOW_SIZE}, 건너뜀")

        # 이 fd/tag 조합에서 생성된 샘플을 9:1로 split
        n_samples = len(samples)
        print(f"\n[{fd}][tag={tag}] total samples: {n_samples}")

        if n_samples == 0:
            continue

        indices = np.arange(n_samples)
        rng.shuffle(indices)

        split_idx = int(n_samples * TRAIN_RATIO)
        train_idx = indices[:split_idx]
        test_idx  = indices[split_idx:]

        train_path = TRAIN_OUTROOT / f"{fd}_{tag}_fw{FORWARD}_bw{BACKWARD}.jsonl"
        test_path  = TEST_OUTROOT  / f"{fd}_{tag}_fw{FORWARD}_bw{BACKWARD}.jsonl"

        # train 저장
        with open(train_path, "w", encoding="utf-8") as f_tr:
            for i in train_idx:
                line = json.dumps(samples[int(i)], ensure_ascii=False)
                f_tr.write(line + "\n")

        # test 저장
        with open(test_path, "w", encoding="utf-8") as f_te:
            for i in test_idx:
                line = json.dumps(samples[int(i)], ensure_ascii=False)
                f_te.write(line + "\n")

        print(f"  ↳ train: {len(train_idx)} → {train_path}")
        print(f"  ↳ test : {len(test_idx)} → {test_path}")

print("\n✔ 모든 FD 데이터에 대한 슬라이딩 윈도우 JSONL 생성 및 train/test 분할 완료")
