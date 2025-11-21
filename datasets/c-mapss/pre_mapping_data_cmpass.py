import pandas as pd, json, warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── 경로 / 데이터셋 목록 ──────────────────────────────
base_dir = Path('archive/CMaps')
sets     = ['FD001', 'FD002', 'FD003', 'FD004']
map_file = Path('sensor_udc_map.json')
plot_dir = Path('sensor_plots'); plot_dir.mkdir(exist_ok=True)

# ── 매핑 로드(없으면 빈 dict) ─────────────────────────
mapping = json.loads(map_file.read_text()) if map_file.exists() else {}

# ── 공통 설정 ────────────────────────────────────────
drop = ['s1','s5','s6','s10','s16','s18','s19']
cols = ['unit','cycle','set1','set2','set3'] + [f's{i}' for i in range(1,22)]

# ── 데이터셋별 순회 ──────────────────────────────────
for fd in sets:
    data_path = base_dir / f'train_{fd}.txt'
    print(f'\n=== {fd} ===')

    df_raw = pd.read_csv(data_path, sep=r'\s+', header=None, names=cols)

    sensors = [c for c in df_raw.columns if c.startswith('s')
               and c[1:].isdigit() and c not in drop]

    # 전역 min-max
    df = df_raw.copy()
    for s in sensors:
        df[s] = MinMaxScaler().fit_transform(df[[s]])

    # 해당 FD의 매핑 dict 확보
    fd_map = mapping.get(fd, {})

    for s in [x for x in sensors if x not in fd_map]:
        curve = df.groupby('cycle')[s].mean().values
        plt.plot(curve)
        plt.title(f'{fd} – {s}')
        plt.xlabel('cycle'); plt.ylabel('value')
        plt.tight_layout()
        plt.savefig(plot_dir / f'{fd}_{s}.png', dpi=120)
        plt.show()

        ans = input('u(상승) d(하강) c(혼합) o(무관): ').strip().lower()
        fd_map[s] = ans if ans in ['u','d','c','o'] else 'o'
        plt.close()

    mapping[fd] = fd_map       # 업데이트

# ── 저장 ─────────────────────────────────────────────
map_file.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
print('\n✔ 데이터셋별 sensor_udc_map.json 저장 완료')
