# scripts/agg_seeds_total.py

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# 기존 agg_seeds.py 에 정의된 함수들을 재사용
from scripts.agg_seeds import (
    RunMetrics,
    load_results,
    aggregate_all,
    _set_ylim_from_values,
    _parse_signature,
)


@dataclass
class ModelSeedAgg:
    """
    한 모델(AUTO JSON 하나)에 대해:
      - run_dirs: seed 별 run 디렉토리 리스트
      - agg_overall: aggregate_all()['agg_overall']
      - agg_by_ratio: aggregate_all()['agg_by_ratio']
    를 묶어서 보관.
    """
    label: str
    auto_json: Path
    test_index: int        # ← 이 줄 추가
    run_dirs: List[Path]
    agg_overall: Dict
    agg_by_ratio: Dict


def _parse_total_config(cfg: Dict) -> List[Dict]:
    """
    통합 JSON 포맷:

    {
      "models": [
        {
          "label": "XGBoost",
          "auto_json": "...xgboost..._auto.json",
          "test": 0
        },
        {
          "label": "LightGBM",
          "auto_json": "...lightgbm..._auto.json",
          "test": 1
        }
      ]
    }

    또는 (옛 방식 호환):

    {
      "0": "...xgboost..._auto.json",
      "1": "...lightgbm..._auto.json"
    }

    반환 형식:
      [
        {"label": "XGBoost", "auto_json": "...", "test_index": 0},
        {"label": "LightGBM", "auto_json": "...", "test_index": 1},
        ...
      ]
    """
    models: List[Dict] = []

    if "models" in cfg:
        raw_models = cfg["models"]
        for idx, m in enumerate(raw_models):
            auto_json = m.get("auto_json") or m.get("path")
            if auto_json is None:
                raise ValueError(f"models[{idx}] 에 auto_json/path 항목이 없습니다.")

            label = m.get("label") or m.get("name") or str(idx)

            # 각 모델별 test 인덱스 (없으면 0으로 기본값)
            if "test" in m:
                test_index = int(m["test"])
            else:
                test_index = 0

            models.append(
                {
                    "label": str(label),
                    "auto_json": auto_json,
                    "test_index": test_index,
                }
            )
    else:
        # 옛날 단순 매핑 방식: test_index 는 0으로 통일
        keys = sorted(cfg.keys(), key=str)
        for k in keys:
            auto_json = cfg[k]
            models.append(
                {
                    "label": str(k),
                    "auto_json": auto_json,
                    "test_index": 0,
                }
            )

    if len(models) == 0:
        raise ValueError("통합 JSON 안에 모델 정보가 없습니다.")

    return models

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
    통합 seed-aggregation 결과 전체를 JSON 하나로 저장.
    """
    payload: Dict = {
        "total_config": str(total_json_path),
        "models": [],
    }

    for m in models_agg:
        payload["models"].append(
            {
                "label": m.label,
                "auto_json": str(m.auto_json),
                "test_index": int(m.test_index),   # ← test_index 추가
                "run_dirs": [str(p) for p in m.run_dirs],
                "agg_overall": m.agg_overall,
                "agg_by_ratio": m.agg_by_ratio,
            }
        )

    out_path = out_dir / "agg_seeds_total.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def _check_compatibility(models_agg: List[ModelSeedAgg]):
    """
    여러 모델이 모두 동일한 pattern / ratio / metric 세트를 가지고 있는지 확인.
    다르면 그래프 비교가 의미가 없으므로 에러를 냅니다.
    """
    if len(models_agg) == 0:
        return

    base = models_agg[0]

    # pattern 세트 비교
    base_patterns = sorted(base.agg_by_ratio.keys())
    for m in models_agg[1:]:
        patterns = sorted(m.agg_by_ratio.keys())
        if patterns != base_patterns:
            raise ValueError(
                f"모델 '{base.label}' 과(와) '{m.label}' 의 pattern 목록이 다릅니다. "
                f"{base_patterns} vs {patterns}"
            )

    # 각 pattern별 ratio / metric 세트 비교
    for pattern in base_patterns:
        base_ratios = sorted(base.agg_by_ratio[pattern].keys(), key=float)

        for m in models_agg[1:]:
            ratios = sorted(m.agg_by_ratio[pattern].keys(), key=float)
            if ratios != base_ratios:
                raise ValueError(
                    f"pattern '{pattern}' 에서 모델 '{base.label}' 과(와) '{m.label}' 의 "
                    f"ratio 목록이 다릅니다. {base_ratios} vs {ratios}"
                )

        # metric key 세트 비교
        first_ratio = base_ratios[0]
        base_metrics = sorted(
            base.agg_by_ratio[pattern][first_ratio]["scalars"].keys()
        )

        for m in models_agg[1:]:
            metrics = sorted(
                m.agg_by_ratio[pattern][first_ratio]["scalars"].keys()
            )
            if metrics != base_metrics:
                raise ValueError(
                    f"pattern '{pattern}', ratio='{first_ratio}' 에서 "
                    f"모델 '{base.label}' 과(와) '{m.label}' 의 metric 목록이 다릅니다. "
                    f"{base_metrics} vs {metrics}"
                )


def plot_ratio_curves_multi(
    models_agg: List[ModelSeedAgg],
    out_dir: Path,
):
    """
    여러 모델에 대해, pattern/ratio 별 scalar metric 을 한 그래프에 그립니다.

    - x축: missing ratio
    - 각 모델: mean (seed 평균)을 선으로 그림
    - 각 모델: seed 최소~최대 범위를 동일 색상으로 fill_between 음영 처리
    - 두 버전 저장:
        1) y축 0~1 고정
        2) 분위수 기반 zoom (_set_ylim_from_values 재사용)
    """
    if len(models_agg) == 0:
        return

    # 호환성 검사 (pattern/ratio/metric 동일 여부)
    _check_compatibility(models_agg)

    # 기준 모델
    base = models_agg[0]
    patterns = sorted(base.agg_by_ratio.keys())

    ratio_dir = out_dir / "by_ratio"
    ratio_dir.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
        pattern_dir = ratio_dir / f"pattern_{pattern}"
        pattern_dir.mkdir(parents=True, exist_ok=True)

        ratio_keys = sorted(base.agg_by_ratio[pattern].keys(), key=float)
        first_ratio = ratio_keys[0]

        metric_keys = sorted(
            base.agg_by_ratio[pattern][first_ratio]["scalars"].keys()
        )

        xs_arr = np.asarray([float(rk) for rk in ratio_keys], dtype=float)

        for metric_key in metric_keys:
            # 각 모델별로 ratio→(mean/min/max, 전체 값) 수집
            per_model_stats = []
            all_values_for_zoom: List[float] = []

            for m in models_agg:
                means: List[float] = []
                mins: List[float] = []
                maxs: List[float] = []
                vals_all: List[float] = []

                for rk in ratio_keys:
                    scalars = m.agg_by_ratio[pattern][rk]["scalars"]
                    if metric_key not in scalars:
                        raise ValueError(
                            f"모델 '{m.label}' 의 pattern='{pattern}', ratio='{rk}' 에 "
                            f"metric '{metric_key}' 가 없습니다."
                        )

                    s = scalars[metric_key]
                    vals = np.asarray(s["values"], dtype=float)

                    if vals.size == 0:
                        raise ValueError(
                            f"모델 '{m.label}' 의 pattern='{pattern}', ratio='{rk}', "
                            f"metric='{metric_key}' 의 values 가 비어 있습니다."
                        )

                    means.append(float(vals.mean()))
                    mins.append(float(vals.min()))
                    maxs.append(float(vals.max()))
                    vals_all.extend(vals.tolist())

                means_arr = np.asarray(means, dtype=float)
                mins_arr = np.asarray(mins, dtype=float)
                maxs_arr = np.asarray(maxs, dtype=float)

                per_model_stats.append(
                    {
                        "label": m.label,
                        "means": means_arr,
                        "mins": mins_arr,
                        "maxs": maxs_arr,
                        "vals_all": vals_all,
                    }
                )

                all_values_for_zoom.extend(vals_all)

            # ---------------------------
            # 1) y축 0~1 고정 버전
            # ---------------------------
            plt.figure(figsize=(6, 4))

            for stats in per_model_stats:
                label = stats["label"]
                means_arr = stats["means"]
                mins_arr = stats["mins"]
                maxs_arr = stats["maxs"]

                line, = plt.plot(xs_arr, means_arr, marker="o", label=label)
                color = line.get_color()
                plt.fill_between(xs_arr, mins_arr, maxs_arr, alpha=0.15, color=color)

            plt.xlabel("missing ratio")
            plt.ylabel(metric_key)
            plt.title(f"{pattern} - {metric_key} vs missing ratio (fixed 0-1, multi-model)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.ylim(0.0, 1.0)
            plt.legend()
            plt.tight_layout()

            fname_fixed = f"{pattern}_ratio_{metric_key}_fixed01_multi.png"
            plt.savefig(pattern_dir / fname_fixed, dpi=150)
            plt.close()

            # ---------------------------
            # 2) zoom 버전
            # ---------------------------
            plt.figure(figsize=(6, 4))

            for stats in per_model_stats:
                label = stats["label"]
                means_arr = stats["means"]
                mins_arr = stats["mins"]
                maxs_arr = stats["maxs"]

                line, = plt.plot(xs_arr, means_arr, marker="o", label=label)
                color = line.get_color()
                plt.fill_between(xs_arr, mins_arr, maxs_arr, alpha=0.15, color=color)

            plt.xlabel("missing ratio")
            plt.ylabel(metric_key)
            plt.title(f"{pattern} - {metric_key} vs missing ratio (zoom, multi-model)")
            plt.grid(True, linestyle="--", alpha=0.5)

            _set_ylim_from_values(all_values_for_zoom)

            plt.legend()
            plt.tight_layout()

            fname_zoom = f"{pattern}_ratio_{metric_key}_zoom_multi.png"
            plt.savefig(pattern_dir / fname_zoom, dpi=150)
            plt.close()

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
        auto_json_path = Path(mc["auto_json"])
        test_index = mc["test_index"]

        print(f"  label={label}, auto_json={auto_json_path}, test_index={test_index}")

        with open(auto_json_path, "r", encoding="utf-8") as f:
            auto_cfg = json.load(f)

        runs = auto_cfg.get("runs", [])
        if len(runs) == 0:
            raise ValueError(f"{auto_json_path} 안에 runs 항목이 비어 있습니다.")

        run_dirs: List[Path] = []
        for r in runs:
            d = r.get("dir")
            if d is None:
                raise ValueError(f"{auto_json_path} 의 runs 항목에 dir 이 없습니다: {r}")
            run_dirs.append(Path(d))

        # seed별 metrics 로딩 (각 모델의 test_index 사용)
        run_metrics: List[RunMetrics] = []
        for rd in run_dirs:
            rm = load_results(rd, test_index=test_index)
            run_metrics.append(rm)

        # seed aggregation (기존 aggregate_all 재사용)
        agg = aggregate_all(run_metrics)

        models_agg.append(
            ModelSeedAgg(
                label=label,
                auto_json=auto_json_path,
                test_index=test_index,         # ← 여기 추가
                run_dirs=run_dirs,
                agg_overall=agg["agg_overall"],
                agg_by_ratio=agg["agg_by_ratio"],
            )
        )

    out_dir = _make_total_output_dir(total_json_path)

    print(f"[agg_seeds_total] output dir: {out_dir}")

    # 통합 JSON 저장
    _save_total_json(models_agg, out_dir, total_json_path)

    # multi-model ratio 곡선 플롯
    plot_ratio_curves_multi(models_agg, out_dir)



if __name__ == "__main__":
    main()
