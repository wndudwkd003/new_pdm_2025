# scripts/agg_seeds.py

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import csv

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class RunMetrics:
    run_dir: Path
    name: str
    overall: Dict
    by_ratio: Dict[str, Dict[float, Dict]]


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


def load_results(run_dir: Path, test_index: int | None = None) -> RunMetrics:
    """
    run_dir:
        outputs/.... (개별 seed run 디렉토리)
    내부 구조 (Trainer 기준):
        run_dir/test/test_k/results_raw.json

    test_index:
        - None: 가장 index가 큰 test_k 사용
        - 정수 k: test_k 디렉토리를 사용
    """
    test_root = run_dir / "test"
    test_dirs: list[tuple[int, Path]] = []
    for p in test_root.iterdir():
        if p.is_dir() and p.name.startswith("test_"):
            parts = p.name.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                test_dirs.append((idx, p))

    if len(test_dirs) == 0:
        raise ValueError(f"'{test_root}' 안에 'test_*' 디렉토리가 없습니다.")

    if test_index is None:
        # 가장 index가 큰 test 디렉토리 사용
        test_dirs.sort(key=lambda x: x[0])
        _, target_test_dir = test_dirs[-1]
    else:
        # 지정한 index 와 일치하는 test_k 찾기
        found: List[tuple[int, Path]] = []
        for idx, p in test_dirs:
            if idx == test_index:
                found.append((idx, p))

        if len(found) == 0:
            available = [idx for idx, _ in test_dirs]
            raise ValueError(
                f"{run_dir} 에 test_{test_index} 디렉토리가 없습니다. "
                f"사용 가능한 index: {available}"
            )

        target_test_dir = found[0][1]

    results_path = target_test_dir / "results_raw.json"
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
            metrics_list: List[Dict] = []
            for rm in run_metrics:
                metrics_list.append(rm.by_ratio[pattern][ratio])
            agg_by_ratio[pattern][ratio] = aggregate_metrics_list(metrics_list)

    return {
        "agg_overall": agg_overall,
        "agg_by_ratio": agg_by_ratio,
    }


# ─────────────────────────────────────────────
# 0.0 ratio 를 제외한 run 단위 overall 계산용
# ─────────────────────────────────────────────

def _combine_metrics_over_nonzero_ratios(metrics_list: List[Dict]) -> Dict:
    """
    하나의 run 에 대해,
    pattern/ratio 조합 중 ratio > 0.0 인 metrics 들만 모아서
    '0.0 제외 overall' metric 을 하나로 합친다.

    - 스칼라는 단순 평균
    - per_class 스칼라도 단순 평균
    - num_samples / support 는 합으로 둔다
    """
    if len(metrics_list) == 0:
        raise ValueError("_combine_metrics_over_nonzero_ratios(): metrics_list 가 비었습니다.")

    base = metrics_list[0]

    scalar_keys = [
        k for k in base.keys()
        if k not in ("per_class", "num_samples")
    ]

    combined: Dict = {}

    # num_samples 는 모든 ratio>0.0 케이스의 합
    total_samples = 0
    for m in metrics_list:
        total_samples += int(m["num_samples"])
    combined["num_samples"] = int(total_samples)

    # 스칼라 metric 들 평균
    for key in scalar_keys:
        vals = np.array([float(m[key]) for m in metrics_list], dtype=float)
        combined[key] = float(vals.mean())

    # per_class 평균 + support 합
    per_class_base = base["per_class"]
    class_keys = sorted(per_class_base.keys(), key=lambda s: int(s))

    combined_per_class: Dict[str, Dict] = {}

    for cls in class_keys:
        cls_dict: Dict = {}

        for metric_name in ("precision", "recall", "f1"):
            vals: List[float] = []
            for m in metrics_list:
                vals.append(float(m["per_class"][cls][metric_name]))
            arr = np.array(vals, dtype=float)
            cls_dict[metric_name] = float(arr.mean())

        # support 는 합으로 둔다
        total_support = 0
        for m in metrics_list:
            total_support += int(m["per_class"][cls]["support"])
        cls_dict["support"] = int(total_support)

        combined_per_class[cls] = cls_dict

    combined["per_class"] = combined_per_class
    return combined


def aggregate_overall_excl_zero(run_metrics: List[RunMetrics]) -> Dict | None:
    """
    run_metrics 각 run 에 대해:
      - by_ratio 에서 ratio > 0.0 인 것들만 모아서
        그 run 의 '0.0 제외 overall metric' 을 만든 뒤
      - 그것을 여러 seed 에 대해 aggregate_metrics_list 로 평균/표준편차 계산.

    ratio>0.0 이 전혀 없으면 None 반환.
    """
    per_run_metrics: List[Dict] = []

    for rm in run_metrics:
        metrics_nonzero: List[Dict] = []

        for pattern, ratio_dict in rm.by_ratio.items():
            for ratio_key, metrics in ratio_dict.items():
                ratio_val = float(ratio_key)
                if ratio_val > 0.0:
                    metrics_nonzero.append(metrics)

        if len(metrics_nonzero) == 0:
            # 이 run 에 nonzero ratio 가 없으면 건너뜀
            continue

        combined = _combine_metrics_over_nonzero_ratios(metrics_nonzero)
        per_run_metrics.append(combined)

    if len(per_run_metrics) == 0:
        return None

    return aggregate_metrics_list(per_run_metrics)


# ─────────────────────────────────────────────
# 출력 디렉토리/파일 저장
# ─────────────────────────────────────────────

def make_output_dir(run_dirs: List[Path], test_index: int | None = None) -> Path:
    """
    outputs_seeds/ 하위에, 첫 번째 run dir 이름에서 'seed***' 토큰만 제거한
    폴더명을 만들어서 반환.
    예) 2025-12-03_09-04-40_xgboost_seed2025_0.0_to_0.5_...
        → 2025-12-03_09-04-40_xgboost_0.0_to_0.5_...

    test_index 가 주어지면 그 아래에 test_{k} 폴더를 한 번 더 판다.
    예) .../2025-12-04_08-24-29_xgboost_0.0_to_0.0_0.0_step_single_mcar_zero/test_0
    """
    base = Path("outputs_seeds")
    base.mkdir(parents=True, exist_ok=True)

    first_name = run_dirs[0].name
    tokens = first_name.split("_")
    filtered = [tok for tok in tokens if not tok.startswith("seed")]
    out_name = "_".join(filtered)

    out_dir = base / out_name
    if test_index is not None:
        out_dir = out_dir / f"test_{test_index}"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_agg_json_txt(
    agg: Dict,
    run_dirs: List[Path],
    out_dir: Path,
):
    """
    전체 aggregation 결과를 JSON + TXT로 저장.
    (기존: 0.0 포함 전체)
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


def save_agg_excl_zero_json_txt(
    agg_excl_zero: Dict,
    run_dirs: List[Path],
    out_dir: Path,
):
    """
    ratio == 0.0 을 제외한 전체 aggregation 결과를
    별도의 JSON + TXT 로 저장.
    """
    payload = {
        "run_dirs": [str(p) for p in run_dirs],
        "agg_overall_excl_zero": agg_excl_zero,
    }

    with open(out_dir / "agg_results_excl_zero.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    scalars = agg_excl_zero["scalars"]

    lines: List[str] = []
    lines.append(f"num_runs: {agg_excl_zero['num_runs']}  # seed 수")
    lines.append(f"num_samples (per run, base): {agg_excl_zero['num_samples']}")
    lines.append("")
    lines.append("[overall scalar metrics (excluding ratio=0.0)]")
    for key in sorted(scalars.keys()):
        m = scalars[key]
        lines.append(
            f"{key}: mean={m['mean']:.6f}, std={m['std']:.6f}"
        )

    with open(out_dir / "summary_excl_zero.txt", "w", encoding="utf-8") as f:
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


# ─────────────────────────────────────────────
#  missing ratio별 CSV 저장 (소수점 5자리 반올림)
# ─────────────────────────────────────────────

def _parse_seed_from_run_dir_name(name: str) -> str:
    """
    run dir 이름에서 seed 값을 문자열로 파싱.
    예: ..._seed42_... → "42"
    seed 토큰이 없으면 전체 이름을 그대로 반환.
    """
    tokens = name.split("_")
    for tok in tokens:
        if tok.startswith("seed"):
            return tok[4:] if len(tok) > 4 else tok
    return name


def _fmt5(x):
    """
    CSV 저장용: 숫자는 소수점 5자리까지 반올림해서 문자열로,
    그 외는 그대로 반환.
    """
    if isinstance(x, (int, float)):
        return f"{x:.5f}"
    return x


def save_ratio_csvs(
    agg_by_ratio: Dict[str, Dict[float, Dict]],
    run_dirs: List[Path],
    out_dir: Path,
):
    """
    - metrics_by_ratio_seeds.csv
      컬럼: 메트릭, 시드, 0.0, 0.1, ...
      각 행: (metric, seed)에 대해 ratio별 값

    - metrics_by_ratio_mean_std.csv
      컬럼: 구분, 메트릭, 0.0, 0.1, ...
      각 metric마다 2행:
        - 구분=평균: mean
        - 구분=편차: std

    값들은 모두 소수점 5자리에서 반올림한 문자열로 저장.
    """
    if len(agg_by_ratio) == 0:
        return

    # 여러 pattern 이 있을 수 있으나, 여기서는 첫 번째 pattern 기준으로 CSV 생성
    pattern_names = sorted(agg_by_ratio.keys())
    pattern = pattern_names[0]
    ratios_dict = agg_by_ratio[pattern]

    if len(ratios_dict) == 0:
        return

    # ratio 컬럼들 (문자열 그대로 사용)
    ratio_keys_sorted = sorted(ratios_dict.keys(), key=float)

    # metric 목록 (첫 ratio 기준)
    first_ratio = ratio_keys_sorted[0]
    metric_keys = sorted(ratios_dict[first_ratio]["scalars"].keys())

    # seed 리스트
    seed_names: List[str] = []
    for rd in run_dirs:
        seed_names.append(_parse_seed_from_run_dir_name(rd.name))

    num_runs = len(seed_names)

    # 1) 시드별 CSV
    seeds_csv_path = out_dir / "metrics_by_ratio_seeds.csv"
    with open(seeds_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["메트릭", "시드"] + [str(r) for r in ratio_keys_sorted]
        writer.writerow(header)

        for metric in metric_keys:
            for i in range(num_runs):
                row: List[object] = [metric, seed_names[i]]
                for rk in ratio_keys_sorted:
                    scalar = agg_by_ratio[pattern][rk]["scalars"][metric]
                    vals = scalar["values"]
                    # i번째 seed에 해당하는 값
                    if i < len(vals):
                        v = vals[i]
                        v = _fmt5(v)
                    else:
                        v = ""
                    row.append(v)
                writer.writerow(row)

    # 2) mean/std CSV
    agg_csv_path = out_dir / "metrics_by_ratio_mean_std.csv"
    with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["구분", "메트릭"] + [str(r) for r in ratio_keys_sorted]
        writer.writerow(header)

        for metric in metric_keys:
            # 평균 row
            row_mean: List[object] = ["평균", metric]
            for rk in ratio_keys_sorted:
                scalar = agg_by_ratio[pattern][rk]["scalars"][metric]
                row_mean.append(_fmt5(scalar["mean"]))
            writer.writerow(row_mean)

            # 편차 row
            row_std: List[object] = ["편차", metric]
            for rk in ratio_keys_sorted:
                scalar = agg_by_ratio[pattern][rk]["scalars"][metric]
                row_std.append(_fmt5(scalar["std"]))
            writer.writerow(row_std)


def main():
    # 이제는 무조건 AUTO JSON 모드만 사용
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage:\n"
            "  python -m scripts.agg_seeds AUTO_JSON_PATH TEST_INDEX\n"
            "예: python -m scripts.agg_seeds "
            "outputs_auto_run/2025-12-04_08-24-29_xgboost_0.0_to_0.0_0.0_step_single_mcar_zero_auto.json 0"
        )

    auto_json_path = Path(sys.argv[1])
    test_index = int(sys.argv[2])

    with open(auto_json_path, "r", encoding="utf-8") as f:
        auto_cfg = json.load(f)

    runs = auto_cfg.get("runs", [])
    if len(runs) == 0:
        raise ValueError(f"{auto_json_path} 안에 runs 항목이 비어 있습니다.")

    run_dirs: List[Path] = []

    print("[agg_seeds] auto json mode:")
    for r in runs:
        seed = r.get("seed")
        d = r.get("dir")
        print(f"  seed={seed}, dir={d}")
        run_dirs.append(Path(d))

    run_metrics: List[RunMetrics] = []
    for rd in run_dirs:
        rm = load_results(rd, test_index=test_index)
        run_metrics.append(rm)

    # 전체 + ratio별 aggregation
    agg = aggregate_all(run_metrics)

    out_dir = make_output_dir(run_dirs, test_index=test_index)

    print(f"[agg_seeds] output dir: {out_dir}")

    save_agg_json_txt(agg, run_dirs, out_dir)

    # missing ratio별 CSV 두 개 저장 (소수점 5자리 반올림)
    save_ratio_csvs(agg["agg_by_ratio"], run_dirs, out_dir)

    # 0.0 ratio 제외한 overall aggregation
    agg_excl_zero = aggregate_overall_excl_zero(run_metrics)
    if agg_excl_zero is not None:
        print("[agg_seeds] also saving metrics excluding ratio=0.0")
        save_agg_excl_zero_json_txt(agg_excl_zero, run_dirs, out_dir)

    # overall bar 는 안 쓰신다고 하셔서 호출 안 함
    # plot_overall_bar(agg["agg_overall"], out_dir)
    plot_ratio_curves(agg["agg_by_ratio"], out_dir)


if __name__ == "__main__":
    main()
