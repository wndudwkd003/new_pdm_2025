import pandas as pd, json, sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np

# ── 설정 ────────────────────────────────────────────
DATASETS  = ['FD001', 'FD003']   # 처리할 파일
SHIFT     = 0                    # 0 = 그대로, 1·2 … = 뒤로 밀기

BASE_DIR  = Path('datasets/c-mapss/data/CMaps')
MAP_FILE  = Path('datasets/c-mapss/data/sensor_udc_map.json')

PLOT_ROOT = Path('datasets/c-mapss/processed_data/engine_knee_plots_multi_no_normal')
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

# 슬라이딩 윈도우 설정
FORWARD      = 30
BACKWARD     = 10
WINDOW_SIZE  = FORWARD + BACKWARD

JSON_OUTROOT = Path('datasets/c-mapss/processed_data')
JSON_OUTROOT.mkdir(parents=True, exist_ok=True)

TRAIN_OUTROOT = JSON_OUTROOT / "train"
TRAIN_OUTROOT.mkdir(parents=True, exist_ok=True)

VALID_OUTROOT = JSON_OUTROOT / "valid"
VALID_OUTROOT.mkdir(parents=True, exist_ok=True)

TEST_OUTROOT = JSON_OUTROOT / "test"
TEST_OUTROOT.mkdir(parents=True, exist_ok=True)

# 플롯 저장 여부
SAVE_PLOTS = True

# split 설정
RANDOM_SEED = 42
TRAINVAL_RATIO = 0.9   # train+valid : test = 9 : 1
TRAIN_RATIO_IN_TV = 0.8  # (train / (train+valid)) = 8 : 2

# ── 매핑 로드 ───────────────────────────────────────
if not MAP_FILE.exists():
    sys.exit('sensor_udc_map.json not found')
mapping_all = json.loads(MAP_FILE.read_text())      # {FDxxx: {sensor: tag}}

# ── 공통 정보 ──────────────────────────────────────
DROP   = ['s1','s5','s6','s10','s16','s18','s19']
COLS   = ['unit','cycle','set1','set2','set3'] + [f's{i}' for i in range(1,22)]
FEATURE_COLS = [c for c in COLS if c not in ['unit', 'cycle']]

COLORS = ['#8fd175', '#fff07e', '#f6b08c', '#d9534f']  # normal→danger (4개 상태)
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

all_labeled = []  # 전체 (unit × tag) 라벨링 결과 모음

# ── 1단계: 상태 라벨링 + 엔진별 플롯 (step 파일 생성 X) ─────────────
for fd in DATASETS:
    print(f'\n=== Processing {fd} ===')
    fd_map = mapping_all.get(fd)
    if not fd_map:
        print('  ↳ 매핑이 없습니다. 건너뜀')
        # 기존 코드 유지 (여기서만 pass되는 형태)
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

    for eid, g_raw in df.groupby('unit'):
        g_raw = g_raw.sort_values('cycle')
        g_norm = df_norm[df_norm.unit == eid].sort_values('cycle')
        cyc = g_raw.cycle.to_numpy()

        if SAVE_PLOTS:
            fig, ax = plt.subplots(figsize=(14, 4))
            for s in sensors:
                ax.plot(cyc, g_norm[s], color='grey', alpha=.3, lw=.35)

        for tag, cols in groups.items():
            if not cols:
                # 기존 코드 유지
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

            # 상태 라벨링
            state_label = np.zeros(len(cyc), dtype=int)
            for i in range(len(seg_cyc) - 1):
                mask = (cyc >= seg_cyc[i]) & (cyc < seg_cyc[i + 1])
                state_label[mask] = i

            g_out = g_raw.copy()
            g_out['state'] = state_label
            g_out['dataset'] = fd
            g_out['tag'] = tag
            all_labeled.append(g_out)

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

# ── 2단계: 슬라이딩 윈도우(JSONL) 생성 (X, y를 직접 저장) ─────────

# 메타데이터용 연속형 컬럼
continuous_cols = [
    c for c in df_all.columns
    if c not in ['unit', 'cycle', 'dataset', 'state', 'tag'] and c not in DROP
]

categorical_cols = []

target_names = [f"state_t+{i+1}" for i in range(BACKWARD)]

common_meta_base = {
    "continuous_cols": continuous_cols,
    "categorical_cols": categorical_cols,
    "target_names": target_names,
    "forward": FORWARD,
    "backward": BACKWARD,
    "interval_sec": None,         # CMAPSS는 시간 간격 미정
    "num_class": len(COLORS),     # 상태 개수 (0~3)
}

print("\n=== Sliding-window JSONL 생성 시작 ===")

rng = np.random.default_rng(RANDOM_SEED)

# split/라벨 통계 + 시퀀스 저장
split_counts = {"train": 0, "valid": 0, "test": 0}
label_counts = {"train": {}, "valid": {}, "test": {}}
split_sequences = {"train": [], "valid": [], "test": []}  # 각 split에 속한 시퀀스들의 y 시퀀스를 저장

for fd, df_fd in df_all.groupby('dataset'):
    for tag, df_tag in df_fd.groupby('tag'):
        samples: list[dict] = []

        meta_base = dict(common_meta_base)
        meta_base["base_name"] = fd
        meta_base["tag"] = tag

        for unit_id, df_eng in df_tag.groupby('unit'):
            df_eng = df_eng.sort_values('cycle').reset_index(drop=True)
            n = len(df_eng)

            if n < WINDOW_SIZE:
                print(f"  - {fd}, tag={tag}, unit {int(unit_id)}: length {n} < {WINDOW_SIZE}, 건너뜀")
                continue

            unit_int = int(unit_id)

            for start in range(0, n - WINDOW_SIZE + 1):
                in_slice  = df_eng.iloc[start : start + FORWARD]
                tgt_slice = df_eng.iloc[start + FORWARD : start + WINDOW_SIZE]

                X_fw = in_slice[continuous_cols].to_numpy(dtype=np.float32)   # (FORWARD, F)
                y_bw = tgt_slice["state"].to_numpy(dtype=np.int64)            # (BACKWARD,)

                sample = {
                    "sample_id": f"{fd}_tag{tag}_u{unit_int:03d}_s{len(samples):05d}",
                    "input": {
                        "X": X_fw.tolist()
                    },
                    "target": {
                        "y": y_bw.tolist()
                    },
                    "metadata": {
                        **meta_base,
                        "dataset": fd,
                        "unit": unit_int,
                        "tag": tag,
                        "cycles_input":  in_slice["cycle"].astype(int).tolist(),
                        "cycles_target": tgt_slice["cycle"].astype(int).tolist(),
                        "states_target": y_bw.astype(int).tolist(),
                    },
                }
                samples.append(sample)

        n_samples = len(samples)
        print(f"\n[{fd}][tag={tag}] total samples: {n_samples}")

        if n_samples == 0:
            continue

        indices = np.arange(n_samples)
        rng.shuffle(indices)

        # 9:1 (train+valid : test)
        n_train_valid = int(n_samples * TRAINVAL_RATIO)
        train_valid_idx = indices[:n_train_valid]
        test_idx = indices[n_train_valid:]

        # train_valid 안에서 8:2 (train : valid)
        n_train = int(len(train_valid_idx) * TRAIN_RATIO_IN_TV)
        train_idx = train_valid_idx[:n_train]
        valid_idx = train_valid_idx[n_train:]

        # 통계 + 시퀀스 업데이트
        for phase, idx_arr in [("train", train_idx),
                               ("valid", valid_idx),
                               ("test",  test_idx)]:
            split_counts[phase] += len(idx_arr)
            label_dict = label_counts[phase]
            seq_list   = split_sequences[phase]
            for i in idx_arr:
                ys = samples[int(i)]["target"]["y"]  # list[int]
                seq_list.append(ys)
                for y in ys:
                    label_dict[y] = label_dict.get(y, 0) + 1

        def write_split(root_dir: Path, idx_arr: np.ndarray, phase_name: str):
            for i in idx_arr:
                s = samples[int(i)]
                meta = dict(s["metadata"])
                meta["data_phase"] = phase_name

                out_sample = {
                    "sample_id": s["sample_id"],
                    "input": s["input"],
                    "target": s["target"],
                    "metadata": meta,
                }

                # s["sample_id"] 예: "FD001_tagd_u001_s00027"
                sample_id = s["sample_id"]
                prefix = f"{fd}_"
                if sample_id.startswith(prefix):
                    # "tagd_u001_s00027" 부분만 떼어냄
                    suffix = sample_id[len(prefix):]
                else:
                    suffix = sample_id

                # 최종 파일명:
                #   FD001_d_fw30_bw10_tagd_u001_s00027.jsonl
                filename = f"{fd}_{tag}_fw{FORWARD}_bw{BACKWARD}_{suffix}.jsonl"
                sample_file = root_dir / filename

                with open(sample_file, "w", encoding="utf-8") as f:
                    line = json.dumps(out_sample, ensure_ascii=False)
                    f.write(line + "\n")


        # 각 split 에 대해 디렉터리 + 개별 파일 생성
        write_split(TRAIN_OUTROOT, train_idx, "train")
        write_split(VALID_OUTROOT, valid_idx, "valid")
        write_split(TEST_OUTROOT,  test_idx,  "test")

        print(f"  ↳ train: {len(train_idx)} 샘플 → {TRAIN_OUTROOT / f'{fd}_{tag}_fw{FORWARD}_bw{BACKWARD}'}")
        print(f"  ↳ valid: {len(valid_idx)} 샘플 → {VALID_OUTROOT / f'{fd}_{tag}_fw{FORWARD}_bw{BACKWARD}'}")
        print(f"  ↳ test : {len(test_idx)} 샘플 → {TEST_OUTROOT  / f'{fd}_{tag}_fw{FORWARD}_bw{BACKWARD}'}")


# ── 3단계: split/라벨 분포 막대그래프 ──────────────────────────────

# 3-1) split 별 샘플 수 막대그래프
splits = ["train", "valid", "test"]
counts = [split_counts[s] for s in splits]

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(splits, counts)
ax.set_xlabel("Split")
ax.set_ylabel("Number of samples")
ax.set_title("Number of samples per split")
for i, c in enumerate(counts):
    ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(JSON_OUTROOT / "split_sample_counts.png", dpi=150)
plt.close(fig)
print("✔ split별 샘플 수 그래프 저장 →", JSON_OUTROOT / "split_sample_counts.png")

# 3-2) 라벨 전체 분포 그래프 (train/valid/test 비교)
all_labels_set = set()
for phase_dict in label_counts.values():
    for lbl in phase_dict.keys():
        all_labels_set.add(lbl)
all_labels = sorted(all_labels_set)

if len(all_labels) > 0:
    x = np.arange(len(all_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, phase in enumerate(splits):
        phase_counts = [label_counts[phase].get(lbl, 0) for lbl in all_labels]
        ax.bar(x + (i - 1) * width, phase_counts, width, label=phase)

    ax.set_xticks(x)
    ax.set_xticklabels([str(lbl) for lbl in all_labels])
    ax.set_xlabel("State label")
    ax.set_ylabel("Count")
    ax.set_title("Label distribution by split (overall)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(JSON_OUTROOT / "split_label_distribution_overall.png", dpi=150)
    plt.close(fig)
    print("✔ 라벨 전체 분포 그래프 저장 →",
          JSON_OUTROOT / "split_label_distribution_overall.png")

# 3-3) Step-wise 라벨 분포 그래프 (요청하신 형태)
if len(all_labels) > 0:
    classes = all_labels
    steps = np.arange(1, BACKWARD + 1)   # 1~10 step

    for phase in splits:
        seq_list = split_sequences[phase]
        n_seq = len(seq_list)
        if n_seq == 0:
            print(f"{phase} split에 시퀀스가 없습니다.")
            continue

        # step별, class별 count 누적
        # step_label_counts[step_idx][label] = count
        step_label_counts = [
            {lbl: 0 for lbl in classes} for _ in range(BACKWARD)
        ]

        for ys in seq_list:           # ys: 길이 BACKWARD 인 label 시퀀스
            ys_arr = np.asarray(ys, dtype=int)
            for s_idx in range(BACKWARD):
                lbl = int(ys_arr[s_idx])
                if lbl in step_label_counts[s_idx]:
                    step_label_counts[s_idx][lbl] += 1

        # 플롯: step마다 class별 막대 (그룹드 바)
        fig, ax = plt.subplots(figsize=(10, 5))

        width = 0.8 / len(classes)    # step 안에서 클래스 그룹 폭
        for j, lbl in enumerate(classes):
            counts_per_step = [
                step_label_counts[s_idx][lbl] for s_idx in range(BACKWARD)
            ]
            ax.bar(
                steps + (j - (len(classes)-1)/2) * width,
                counts_per_step,
                width=width,
                label=f"class {lbl}",
            )

        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps])
        ax.set_xlabel("Step (t+1 ~ t+10)")
        ax.set_ylabel("Count")
        ax.set_title(f"Step-wise Label Distribution - {phase}")
        ax.legend()

        fig.tight_layout()
        out_path = JSON_OUTROOT / f"step_label_distribution_{phase}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✔ step-wise 라벨 분포 그래프 저장 → {out_path}")

print("\n✔ 모든 FD 데이터에 대한 슬라이딩 윈도우 JSONL 생성 및 train/valid/test 분할 완료")
