# scripts/agg_seeds_total.py

import sys
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# 기존 agg_seeds.py 에 정의된 함수들을 재사용
from scripts.agg_seeds import _set_ylim_from_values


@dataclass
class ModelSeedAgg:
    """
    한 모델(= outputs_seeds/.../test_k 디렉터리 하나)에 대해:
      - dir: metrics_by_ratio_*.csv 가 들어있는 디렉터리
      - ratios: missing ratio 리스트 (float, 정렬된 상태)
      - metrics: 메트릭별 곡선 정보
        metrics[metric_name] = {
          "means": np.ndarray(shape=[num_ratios]),
          "mins":  np.ndarray(shape=[num_ratios]),
          "maxs":  np.ndarray(shape=[num_ratios]),
        }
    """
    label: str
    dir: Path
    ratios: List[float]
    metrics: Dict[str, Dict[str, np.ndarray]]


def _get_family_name(label: str) -> str:
    """
    모델 라벨에서 '베이스 모델 이름'만 뽑아냅니다.
    예:
      "XGBoost NoAug/Z"      -> "XGBoost"
      "MLP RMAug/Z"          -> "MLP"
      "FT-Transformer NoAug" -> "FT-Transformer"

    기준: 첫 번째 공백 앞까지를 family 로 사용.
    """
    parts = label.split()
    if not parts:
        return label
    return parts[0]




def _parse_total_config(cfg: Dict) -> List[Dict]:
    """
    통합 JSON 포맷:

    {
      "models": [
        {
          "label": "XGBoost",
          "dir": "outputs_seeds/.../test_1"
        },
        {
          "label": "LightGBM",
          "dir": "outputs_seeds/.../test_0"
        }
      ]
    }

    반환 형식:
      [
        {"label": "XGBoost", "dir": "..." },
        {"label": "LightGBM", "dir": "..." },
        ...
      ]
    """
    if "models" not in cfg:
        raise ValueError("total json 에 'models' 항목이 없습니다.")

    models: List[Dict] = []
    raw_models = cfg["models"]

    for idx, m in enumerate(raw_models):
        dir_str = m.get("dir")
        if dir_str is None:
            raise ValueError(f"models[{idx}] 에 dir 항목이 없습니다.")

        label = m.get("label") or m.get("name") or f"model_{idx}"

        models.append(
            {
                "label": str(label),
                "dir": dir_str,
            }
        )

    if len(models) == 0:
        raise ValueError("통합 JSON 안에 모델 정보가 없습니다.")

    return models


def _load_model_from_dir(label: str, dir_path: Path) -> ModelSeedAgg:
    """
    dir_path:
      outputs_seeds/.../test_k 디렉터리
      내부에 metrics_by_ratio_seeds.csv 가 있다고 가정.

    CSV 형식 (metrics_by_ratio_seeds.csv):
      메트릭,시드,0.0,0.1,0.2,...,0.9

    여기서 metric별, ratio별로 모든 seed 값들을 모아서
    mean / min / max 를 계산.
    """
    seeds_csv = dir_path / "metrics_by_ratio_seeds.csv"
    if not seeds_csv.exists():
        raise FileNotFoundError(f"{seeds_csv} 파일이 없습니다.")

    with open(seeds_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{seeds_csv} 가 비어 있습니다.")

        if len(header) < 3:
            raise ValueError(f"{seeds_csv} 헤더 형식이 잘못되었습니다: {header}")

        # header: ["메트릭", "시드", "0.0", "0.1", ...]
        ratio_labels = header[2:]
        ratios: List[float] = [float(r) for r in ratio_labels]
        num_ratios = len(ratios)

        # metrics_values[metric][ratio_idx] = [seed1_value, seed2_value, ...]
        metrics_values: Dict[str, List[List[float]]] = {}

        for row in reader:
            if not row or len(row) < 2:
                continue

            metric_name = row[0]
            if metric_name == "":
                continue

            if metric_name not in metrics_values:
                metrics_values[metric_name] = [[] for _ in range(num_ratios)]

            values = row[2:]
            # 부족한 칼럼이 있으면 빈 문자열로 채워진 상태일 수 있음
            if len(values) < num_ratios:
                values = values + [""] * (num_ratios - len(values))

            for i in range(num_ratios):
                cell = values[i]
                if cell == "":
                    continue
                v = float(cell)
                metrics_values[metric_name][i].append(v)

    # metric별 mean/min/max 계산
    metrics: Dict[str, Dict[str, np.ndarray]] = {}

    for metric_name, per_ratio_lists in metrics_values.items():
        means: List[float] = []
        mins: List[float] = []
        maxs: List[float] = []

        for vals in per_ratio_lists:
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                # 해당 ratio에서 값이 하나도 없으면 NaN 넣어둠
                means.append(float("nan"))
                mins.append(float("nan"))
                maxs.append(float("nan"))
            else:
                means.append(float(arr.mean()))
                mins.append(float(arr.min()))
                maxs.append(float(arr.max()))

        metrics[metric_name] = {
            "means": np.asarray(means, dtype=float),
            "mins":  np.asarray(mins, dtype=float),
            "maxs":  np.asarray(maxs, dtype=float),
        }

    return ModelSeedAgg(
        label=label,
        dir=dir_path,
        ratios=ratios,
        metrics=metrics,
    )




def _make_total_output_dir(total_json_path: Path) -> Path:
    """
    outputs_seeds_total/ 아래에 통합 결과 디렉토리를 생성.

    예: total_json_path = total_config.json
        → outputs_seeds_total/total_config/
    """
    base = Path("outputs_seeds_total")
    base.mkdir(parents=True, exist_ok=True)

    name = total_json_path.stem
    out_dir = base / name

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_total_json(
    models_agg: List[ModelSeedAgg],
    out_dir: Path,
    total_json_path: Path,
):
    """
    통합 결과를 JSON 하나로 저장.
    """
    payload: Dict = {
        "total_config": str(total_json_path),
        "models": [],
    }

    for m in models_agg:
        metrics_serialized: Dict[str, Dict[str, List[float]]] = {}
        for metric_name, stats in m.metrics.items():
            metrics_serialized[metric_name] = {
                "means": stats["means"].tolist(),
                "mins":  stats["mins"].tolist(),
                "maxs":  stats["maxs"].tolist(),
            }

        payload["models"].append(
            {
                "label": m.label,
                "dir": str(m.dir),
                "ratios": m.ratios,
                "metrics": metrics_serialized,
            }
        )

    out_path = out_dir / "agg_seeds_total.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)





def _check_compatibility(models_agg: List[ModelSeedAgg]):
    """
    여러 모델이 모두 동일한 ratio / metric 세트를 가지고 있는지 확인.
    다르면 그래프 비교가 의미가 없으므로 에러를 냅니다.
    """
    if len(models_agg) == 0:
        return

    base = models_agg[0]
    base_ratios = base.ratios
    base_metrics = sorted(base.metrics.keys())

    for m in models_agg[1:]:
        # ratio 비교 (길이 + 값)
        if len(m.ratios) != len(base_ratios):
            raise ValueError(
                f"모델 '{base.label}' 과(와) '{m.label}' 의 ratio 개수가 다릅니다. "
                f"{len(base_ratios)} vs {len(m.ratios)}"
            )
        for a, b in zip(base_ratios, m.ratios):
            if abs(a - b) > 1e-9:
                raise ValueError(
                    f"모델 '{base.label}' 과(와) '{m.label}' 의 ratio 값이 다릅니다. "
                    f"{base_ratios} vs {m.ratios}"
                )

        # metric key 세트 비교
        metrics = sorted(m.metrics.keys())
        if metrics != base_metrics:
            raise ValueError(
                f"모델 '{base.label}' 과(와) '{m.label}' 의 metric 목록이 다릅니다. "
                f"{base_metrics} vs {metrics}"
            )

def plot_ratio_curves_multi(
    models_agg: List[ModelSeedAgg],
    out_dir: Path,
    only_model_legends: bool = False,
):
    """
    여러 모델에 대해, ratio 별 scalar metric 을 한 그래프에 그립니다.

    - x축: missing ratio
    - 각 모델: mean (seed 평균)을 선으로 그림
    - 각 모델: seed 최소~최대 범위를 동일 색상으로 fill_between 음영 처리
    - 색상:
        같은 family(예: "XGBoost") 에 속한 모델들은 같은 색
    - 선 스타일:
        같은 family 내에서 NoAug / RMAug 등은
        solid / dashed / dashdot / dotted 등으로 구분
    - 두 버전 저장:
        1) y축 0~1 고정
        2) min/max 기반 zoom (극단값 포함, multi-model)
    """
    if len(models_agg) == 0:
        return

    _check_compatibility(models_agg)

    base = models_agg[0]
    ratios_arr = np.asarray(base.ratios, dtype=float)
    metric_keys = sorted(base.metrics.keys())

    # family / color / linestyle 셋업 부분은 그대로
    families: List[str] = []
    for m in models_agg:
        fam = _get_family_name(m.label)
        if fam not in families:
            families.append(fam)

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if color_cycle is not None:
        colors = color_cycle.by_key().get("color", [])
    else:
        colors = []
    if not colors:
        colors = [f"C{i}" for i in range(10)]

    family_to_color: Dict[str, str] = {}
    for idx, fam in enumerate(families):
        family_to_color[fam] = colors[idx % len(colors)]

    linestyle_cycle = ["solid", "dashed", "dashdot", "dotted"]
    family_style_counter: Dict[str, int] = {}
    model_style: Dict[str, Dict[str, str]] = {}

    for m in models_agg:
        fam = _get_family_name(m.label)
        color = family_to_color[fam]

        idx = family_style_counter.get(fam, 0)
        linestyle = linestyle_cycle[idx % len(linestyle_cycle)]
        family_style_counter[fam] = idx + 1

        model_style[m.label] = {
            "color": color,
            "linestyle": linestyle,
        }

        viz_dir = out_dir / "by_ratio"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for metric_key in metric_keys:
        # zoom용으로 모든 mean/min/max 값을 모아둠
        all_values_for_zoom: List[float] = []

        # ---------------------------
        # 1) y축 0~1 고정 버전
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 4))

        used_families_fixed = set()

        for m in models_agg:
            stats = m.metrics[metric_key]
            means_arr = stats["means"]
            mins_arr = stats["mins"]
            maxs_arr = stats["maxs"]

            style = model_style[m.label]

            # --- 범례에 들어갈 라벨 결정 ---
            if only_model_legends:
                fam = _get_family_name(m.label)
                if fam in used_families_fixed:
                    legend_label = f"_{fam}"  # legend에서 무시되는 라벨
                else:
                    legend_label = fam
                    used_families_fixed.add(fam)
            else:
                legend_label = m.label
            # -----------------------------

            ax.plot(
                ratios_arr,
                means_arr,
                marker="o",
                linewidth=2.0,
                markersize=4,
                label=legend_label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=0.85,  # 선도 약간 투명하게
            )
            ax.fill_between(
                ratios_arr,
                mins_arr,
                maxs_arr,
                alpha=0.15,
                color=style["color"],
            )

            # zoom 계산용으로 mean/min/max 모두 저장
            all_values_for_zoom.extend(means_arr.tolist())
            all_values_for_zoom.extend(mins_arr.tolist())
            all_values_for_zoom.extend(maxs_arr.tolist())

        ax.set_xlabel("missing ratio")
        ax.set_ylabel(metric_key)
        ax.set_title(f"{metric_key} vs missing ratio (fixed 0-1, multi-model)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0.0, 1.0)

        leg = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

        fig.tight_layout()
        fname_fixed = f"{metric_key}_ratio_fixed01_multi.png"
        fig.savefig(
            viz_dir / fname_fixed,
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=(leg,),
        )
        plt.close(fig)

        # ---------------------------
        # 2) zoom 버전 (extrema 포함)
        # ---------------------------
        fig, ax = plt.subplots(figsize=(7, 4))

        used_families_zoom = set()

        for m in models_agg:
            stats = m.metrics[metric_key]
            means_arr = stats["means"]
            mins_arr = stats["mins"]
            maxs_arr = stats["maxs"]

            style = model_style[m.label]

            # --- 범례에 들어갈 라벨 결정 ---
            if only_model_legends:
                fam = _get_family_name(m.label)
                if fam in used_families_zoom:
                    legend_label = f"_{fam}"
                else:
                    legend_label = fam
                    used_families_zoom.add(fam)
            else:
                legend_label = m.label
            # -----------------------------

            ax.plot(
                ratios_arr,
                means_arr,
                marker="o",
                linewidth=2.0,
                markersize=4,
                label=legend_label,
                color=style["color"],
                linestyle=style["linestyle"],
                alpha=0.85,  # 여기도 동일하게 약간 투명
            )
            ax.fill_between(
                ratios_arr,
                mins_arr,
                maxs_arr,
                alpha=0.15,
                color=style["color"],
            )

        ax.set_xlabel("missing ratio")
        ax.set_ylabel(metric_key)
        ax.set_title(f"{metric_key} vs missing ratio (zoom, multi-model)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # ==== 여기부터 y축 범위 계산을 직접 수행 ====
        valid_values = [v for v in all_values_for_zoom if not np.isnan(v)]
        if len(valid_values) > 0:
            vmin = min(valid_values)
            vmax = max(valid_values)

            if vmin == vmax:
                margin = 0.02 if vmax == 0.0 else abs(vmax) * 0.02
            else:
                margin = (vmax - vmin) * 0.05

            ymin = vmin - margin
            ymax = vmax + margin

            lower_name = metric_key.lower()
            if any(k in lower_name for k in ["acc", "f1", "prec", "recall", "auc"]):
                ymin = max(0.0, ymin)
                ymax = min(1.0, ymax)

            ax.set_ylim(ymin, ymax)
        # ==== 여기까지 ====

        leg = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

        fig.tight_layout()
        fname_zoom = f"{metric_key}_ratio_zoom_multi.png"
        fig.savefig(
            viz_dir / fname_zoom,
            dpi=150,
            bbox_inches="tight",
            bbox_extra_artists=(leg,),
        )
        plt.close(fig)



def main():
    """
    사용법:
      python -m scripts.agg_seeds_total TOTAL_JSON_PATH

    예:
      python -m scripts.agg_seeds_total total_config.json
    """
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage:\n"
            "  python -m scripts.agg_seeds_total TOTAL_JSON_PATH\n"
            "예: python -m scripts.agg_seeds_total total_config.json"
        )

    total_json_path = Path(sys.argv[1])

    with open(total_json_path, "r", encoding="utf-8") as f:
        total_cfg = json.load(f)

    model_cfgs = _parse_total_config(total_cfg)

    models_agg: List[ModelSeedAgg] = []

    print("[agg_seeds_total] total json mode:")
    for mc in model_cfgs:
        label = mc["label"]
        dir_path = Path(mc["dir"])

        print(f"  label={label}, dir={dir_path}")

        if not dir_path.exists():
            raise FileNotFoundError(f"{dir_path} 디렉터리가 존재하지 않습니다.")

        model_agg = _load_model_from_dir(label, dir_path)
        models_agg.append(model_agg)

    out_dir = _make_total_output_dir(total_json_path)
    print(f"[agg_seeds_total] output dir: {out_dir}")

    # 통합 JSON 저장
    _save_total_json(models_agg, out_dir, total_json_path)

    # multi-model ratio 곡선 플롯
    plot_ratio_curves_multi(models_agg, out_dir, only_model_legends=True)



if __name__ == "__main__":
    main()


