# scripts/agg_seeds.py

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RunMetrics:
    run_dir: Path
    name: str
    overall: Dict
    by_ratio: Dict[str, Dict[float, Dict]]


def _collect_run_dirs_from_range(start_dir: Path, end_dir: Path) -> list[Path]:
    """
    start_dir, end_dir 가 주어졌을 때:
    - 둘의 parent 디렉토리(보통 outputs/) 안을 훑어서
    - 이름이 start~end 범위 사이이고
    - signature(모델/세팅)가 start_dir 와 같은 것들을 모두 모아 반환.
    """
    if start_dir.parent != end_dir.parent:
        raise ValueError(f"start_dir 와 end_dir 의 부모 디렉토리가 다릅니다: {start_dir.parent}, {end_dir.parent}")

    base_dir = start_dir.parent

    name_lo = min(start_dir.name, end_dir.name)
    name_hi = max(start_dir.name, end_dir.name)

    sig_ref = _parse_signature(start_dir.name)

    selected: list[Path] = []
    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        # 시간 범위 (이름에 날짜+시간이 prefix라서 문자열 비교 가능)
        if n < name_lo or n > name_hi:
            continue

        # 세팅(signature) 동일한 것만 선택
        if _parse_signature(n) != sig_ref:
            continue

        selected.append(p)

    selected.sort(key=lambda x: x.name)
    return selected


def _parse_signature(run_name: str) -> str:
    """
    run_name 예:
      2025-12-03_09-04-06_xgboost_seed42_0.0_to_0.5_0.1_step_multi_mcar_zero

    - 앞의 날짜/시간 2토큰 제거
    - seed** 토큰 제거
    나머지를 이어 붙여서 '같은 모델+세팅' 을 식별하는 signature 로 사용.
    """
    tokens = run_name.split("_")
    if len(tokens) < 3:
        return run_name  # 이상한 이름이면 그대로 반환

    # 날짜, 시간 제거
    tokens = tokens[2:]

    # seed 토큰 제거
    tokens = [t for t in tokens if not t.startswith("seed")]

    return "_".join(tokens)



def _set_ylim_from_values(
    values,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    padding_ratio: float = 0.1,
):
    """
    여러 값들(values)을 받아서:
    - lower_q ~ upper_q 분위수 구간에 맞춰 확대
    - 위아래 padding_ratio 만큼 여유를 둔 ylim 설정
    """
    arr = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(arr)
    if not np.any(finite):
        return

    arr = arr[finite]

    y_low = np.quantile(arr, lower_q)
    y_high = np.quantile(arr, upper_q)

    if y_low == y_high:
        eps = abs(y_low) * 0.1 if y_low != 0 else 1.0
        y_low -= eps
        y_high += eps

    span = y_high - y_low
    pad = span * padding_ratio
    y_min = y_low - pad
    y_max = y_high + pad

    plt.ylim(y_min, y_max)


def load_results(run_dir: Path) -> RunMetrics:
    """
    run_dir:
        outputs/.... (개별 seed run 디렉토리)
    내부 구조 (Trainer 기준):
        run_dir/test/test_k/results_raw.json
    """
    test_root = run_dir / "test"
    test_dirs = []
    for p in test_root.iterdir():
        if p.is_dir() and p.name.startswith("test_"):
            parts = p.name.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                test_dirs.append((int(parts[1]), p))

    if len(test_dirs) == 0:
        raise ValueError(f"'{test_root}' 안에 'test_*' 디렉토리가 없습니다.")

    # 가장 index가 큰 test 디렉토리 사용
    test_dirs.sort(key=lambda x: x[0])
    _, latest_test_dir = test_dirs[-1]

    results_path = latest_test_dir / "results_raw.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    overall = results["metrics_overall"]
    by_ratio = results["metrics_by_ratio"]

    return RunMetrics(
        run_dir=run_dir,
        name=run_dir.name,
        overall=overall,
        by_ratio=by_ratio,
    )


def aggregate_metrics_list(metrics_list: List[Dict]) -> Dict:
    """
    단일 run 에서의 classification metrics(dict)를 여러 seed에 대해 평균/표준편차 산출.

    metrics 예시:
    {
      "accuracy": ...,
      "precision_micro": ...,
      "recall_micro": ...,
      "f1_micro": ...,
      "precision_macro": ...,
      "recall_macro": ...,
      "f1_macro": ...,
      "per_class": { "0": {...}, "1": {...}, ... },
      "num_samples": 74364
    }

    반환:
    {
      "num_runs": K,
      "num_samples": ...,
      "scalars": {
        "accuracy": {"mean": .., "std": .., "values": [...]},
        ...
      },
      "per_class": {
        "0": {
          "precision": {"mean":..,"std":..,"values":[...]},
          "recall":   {...},
          "f1":       {...},
          "support": int,
        },
        ...
      }
    }
    """
    if len(metrics_list) == 0:
        raise ValueError("aggregate_metrics_list(): metrics_list 가 비었습니다.")

    base = metrics_list[0]

    scalar_keys = [
        k for k in base.keys()
        if k not in ("per_class", "num_samples")
    ]

    result: Dict = {
        "num_runs": len(metrics_list),
        "num_samples": int(base["num_samples"]),
        "scalars": {},
        "per_class": {},
    }

    # 스칼라 metric (accuracy 등)
    for key in scalar_keys:
        vals = np.array([float(m[key]) for m in metrics_list], dtype=float)
        result["scalars"][key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "values": [float(v) for v in vals],
        }

    # 클래스별 metric
    per_class_base = base["per_class"]
    class_keys = sorted(per_class_base.keys(), key=lambda s: int(s))

    for cls in class_keys:
        result["per_class"][cls] = {}

        for metric_name in ("precision", "recall", "f1"):
            vals = np.array(
                [float(m["per_class"][cls][metric_name]) for m in metrics_list],
                dtype=float,
            )
            result["per_class"][cls][metric_name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "values": [float(v) for v in vals],
            }

        # support 는 seed 간 동일하다고 가정하고 첫 번째 값 사용
        result["per_class"][cls]["support"] = int(per_class_base[cls]["support"])

    return result


def aggregate_all(run_metrics: List[RunMetrics]) -> Dict:
    """
    여러 seed run 의 metric 을 받아 전체/ratio 별로 평균/표준편차 계산.
    """
    if len(run_metrics) == 0:
        raise ValueError("aggregate_all(): run_metrics 가 비었습니다.")

    # overall
    overall_list = [rm.overall for rm in run_metrics]
    agg_overall = aggregate_metrics_list(overall_list)

    # by_ratio
    first_by_ratio = run_metrics[0].by_ratio
    agg_by_ratio: Dict[str, Dict[float, Dict]] = {}

    for pattern, ratio_dict in first_by_ratio.items():
        agg_by_ratio[pattern] = {}
        # ratio key 는 float 로 캐스팅 가능하다고 가정
        ratios = sorted(ratio_dict.keys(), key=float)
        for ratio in ratios:
            metrics_list = [
                rm.by_ratio[pattern][ratio] for rm in run_metrics
            ]
            agg_by_ratio[pattern][ratio] = aggregate_metrics_list(metrics_list)

    return {
        "agg_overall": agg_overall,
        "agg_by_ratio": agg_by_ratio,
    }


def make_output_dir(run_dirs: List[Path]) -> Path:
    """
    outputs_seeds/ 하위에, 첫 번째 run dir 이름에서 'seed***' 토큰만 제거한
    폴더명을 만들어서 반환.
    예) 2025-12-03_09-04-40_xgboost_seed2025_0.0_to_0.5_...
        → 2025-12-03_09-04-40_xgboost_0.0_to_0.5_...
    """
    base = Path("outputs_seeds")
    base.mkdir(parents=True, exist_ok=True)

    first_name = run_dirs[0].name
    tokens = first_name.split("_")
    filtered = [tok for tok in tokens if not tok.startswith("seed")]
    out_name = "_".join(filtered)

    out_dir = base / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_agg_json_txt(
    agg: Dict,
    run_dirs: List[Path],
    out_dir: Path,
):
    """
    전체 aggregation 결과를 JSON + TXT로 저장.
    """
    payload = {
        "run_dirs": [str(p) for p in run_dirs],
        "agg_overall": agg["agg_overall"],
        "agg_by_ratio": agg["agg_by_ratio"],
    }

    with open(out_dir / "agg_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # summary.txt
    overall = agg["agg_overall"]
    scalars = overall["scalars"]

    lines: List[str] = []
    lines.append(f"num_runs: {overall['num_runs']}")
    lines.append(f"num_samples (per run): {overall['num_samples']}")
    lines.append("")
    lines.append("[overall scalar metrics]")
    for key in sorted(scalars.keys()):
        m = scalars[key]
        lines.append(
            f"{key}: mean={m['mean']:.6f}, std={m['std']:.6f}"
        )

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_overall_bar(agg_overall: Dict, out_dir: Path):
    """
    overall 스칼라 metric 들의 mean/std 를 막대 그래프로 저장.
    """
    scalars = agg_overall["scalars"]
    keys = sorted(scalars.keys())

    means = [scalars[k]["mean"] for k in keys]
    stds = [scalars[k]["std"] for k in keys]

    x = np.arange(len(keys))

    plt.figure(figsize=(8, 4))
    plt.bar(x, means, yerr=stds, alpha=0.7, capsize=4)
    plt.xticks(x, keys, rotation=45, ha="right")
    plt.ylabel("score")
    plt.title("Overall metrics (mean ± std over seeds)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "overall_metrics_bar.png", dpi=150)
    plt.close()

def plot_ratio_curves(agg_by_ratio: Dict[str, Dict[float, Dict]], out_dir: Path):
    """
    pattern, ratio 별로 모든 scalar metric 에 대해

    - x축: missing ratio
    - y축: metric
    - 선: seed 평균
    - 음영: seed 최소~최대 범위

    를 그린다.
    각 metric마다:
      - y축 0~1 고정 그래프
      - zoom(분위수 기반 자동 확대) 그래프
    두 장을 저장한다.
    """
    ratio_dir = out_dir / "by_ratio"
    ratio_dir.mkdir(parents=True, exist_ok=True)

    for pattern, ratios_dict in agg_by_ratio.items():
        pattern_dir = ratio_dir / f"pattern_{pattern}"
        pattern_dir.mkdir(parents=True, exist_ok=True)

        # ratio 값들을 float 기준으로 정렬
        ratio_keys = sorted(ratios_dict.keys(), key=float)
        if len(ratio_keys) == 0:
            continue

        # 어떤 scalar metric 이 있는지: 첫 번째 ratio 기준
        first_ratio = ratio_keys[0]
        metric_keys = sorted(ratios_dict[first_ratio]["scalars"].keys())

        for metric_key in metric_keys:
            xs: list[float] = []
            means: list[float] = []
            mins: list[float] = []
            maxs: list[float] = []
            all_values_for_zoom: list[float] = []

            for rk in ratio_keys:
                agg_metrics = ratios_dict[rk]
                scalars = agg_metrics["scalars"]
                if metric_key not in scalars:
                    continue

                s = scalars[metric_key]
                vals = np.asarray(s["values"], dtype=float)
                m = float(vals.mean())
                lo = float(vals.min())
                hi = float(vals.max())

                xs.append(float(rk))
                means.append(m)
                mins.append(lo)
                maxs.append(hi)
                all_values_for_zoom.append(lo)
                all_values_for_zoom.append(hi)
                all_values_for_zoom.extend(vals.tolist())

            if len(xs) == 0:
                continue

            xs_arr = np.asarray(xs, dtype=float)
            means_arr = np.asarray(means, dtype=float)
            mins_arr = np.asarray(mins, dtype=float)
            maxs_arr = np.asarray(maxs, dtype=float)

            # -------------------------------
            # 1) y축 0~1 고정 그래프
            # -------------------------------
            plt.figure(figsize=(5, 4))
            plt.plot(xs_arr, means_arr, marker="o")
            plt.fill_between(xs_arr, mins_arr, maxs_arr, alpha=0.2)
            plt.xlabel("missing ratio")
            plt.ylabel(metric_key)
            plt.title(f"{pattern} - {metric_key} vs missing ratio (fixed 0-1)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.ylim(0.0, 1.0)
            plt.tight_layout()
            fname_fixed = f"{pattern}_ratio_{metric_key}_fixed01.png"
            plt.savefig(pattern_dir / fname_fixed, dpi=150)
            plt.close()

            # -------------------------------
            # 2) zoom 버전 (분위수 기반)
            # -------------------------------
            plt.figure(figsize=(5, 4))
            plt.plot(xs_arr, means_arr, marker="o")
            plt.fill_between(xs_arr, mins_arr, maxs_arr, alpha=0.2)
            plt.xlabel("missing ratio")
            plt.ylabel(metric_key)
            plt.title(f"{pattern} - {metric_key} vs missing ratio (zoom)")
            plt.grid(True, linestyle="--", alpha=0.5)

            _set_ylim_from_values(all_values_for_zoom)

            plt.tight_layout()
            fname_zoom = f"{pattern}_ratio_{metric_key}_zoom.png"
            plt.savefig(pattern_dir / fname_zoom, dpi=150)
            plt.close()



def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m scripts.agg_seeds RUN_DIR1 [RUN_DIR2 ...]")

    # case 1: 시작/끝 두 개만 준 경우 → 범위 안 run 자동 수집
    if len(sys.argv) == 3:
        start_dir = Path(sys.argv[1])
        end_dir   = Path(sys.argv[2])

        run_dirs = _collect_run_dirs_from_range(start_dir, end_dir)

        if len(run_dirs) == 0:
            raise ValueError(f"범위 내에 해당 세팅의 run 디렉토리가 없습니다.\n  start={start_dir}\n  end={end_dir}")

        print("[agg_seeds] collected run dirs (range mode):")
        for p in run_dirs:
            print(" ", p)

    # case 2: 여러 개를 직접 나열한 경우 → 그대로 사용
    else:
        run_dirs = [Path(p) for p in sys.argv[1:]]
        print("[agg_seeds] explicit run dirs:")
        for p in run_dirs:
            print(" ", p)

    run_metrics: list[RunMetrics] = []
    for rd in run_dirs:
        rm = load_results(rd)
        run_metrics.append(rm)

    agg = aggregate_all(run_metrics)

    out_dir = make_output_dir(run_dirs)
    print(f"[agg_seeds] output dir: {out_dir}")

    save_agg_json_txt(agg, run_dirs, out_dir)
    # overall bar 는 안 쓰신다고 하셔서 호출 안 함
    # plot_overall_bar(agg["agg_overall"], out_dir)
    plot_ratio_curves(agg["agg_by_ratio"], out_dir)





if __name__ == "__main__":
    main()
