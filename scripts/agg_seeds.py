# scripts/agg_seeds.py

import sys
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import csv
import re
from decimal import Decimal, ROUND_HALF_UP

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
        test_dirs.sort(key=lambda x: x[0])
        _, target_test_dir = test_dirs[-1]
    else:
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
    """
    if len(metrics_list) == 0:
        raise ValueError("aggregate_metrics_list(): metrics_list 가 비었습니다.")

    base = metrics_list[0]

    scalar_keys = [k for k in base.keys() if k not in ("per_class", "num_samples")]

    result: Dict = {
        "num_runs": len(metrics_list),
        "num_samples": int(base["num_samples"]),
        "scalars": {},
        "per_class": {},
    }

    for key in scalar_keys:
        vals = np.array([float(m[key]) for m in metrics_list], dtype=float)
        result["scalars"][key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "values": [float(v) for v in vals],
        }

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

        result["per_class"][cls]["support"] = int(per_class_base[cls]["support"])

    return result


def aggregate_all(run_metrics: List[RunMetrics]) -> Dict:
    """
    여러 seed run 의 metric 을 받아 전체/ratio 별로 평균/표준편차 계산.
    """
    if len(run_metrics) == 0:
        raise ValueError("aggregate_all(): run_metrics 가 비었습니다.")

    overall_list = [rm.overall for rm in run_metrics]
    agg_overall = aggregate_metrics_list(overall_list)

    first_by_ratio = run_metrics[0].by_ratio
    agg_by_ratio: Dict[str, Dict[float, Dict]] = {}

    for pattern, ratio_dict in first_by_ratio.items():
        agg_by_ratio[pattern] = {}
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


def _combine_metrics_over_nonzero_ratios(metrics_list: List[Dict]) -> Dict:
    if len(metrics_list) == 0:
        raise ValueError(
            "_combine_metrics_over_nonzero_ratios(): metrics_list 가 비었습니다."
        )

    base = metrics_list[0]

    scalar_keys = [k for k in base.keys() if k not in ("per_class", "num_samples")]

    combined: Dict = {}

    total_samples = 0
    for m in metrics_list:
        total_samples += int(m["num_samples"])
    combined["num_samples"] = int(total_samples)

    for key in scalar_keys:
        vals = np.array([float(m[key]) for m in metrics_list], dtype=float)
        combined[key] = float(vals.mean())

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

        total_support = 0
        for m in metrics_list:
            total_support += int(m["per_class"][cls]["support"])
        cls_dict["support"] = int(total_support)

        combined_per_class[cls] = cls_dict

    combined["per_class"] = combined_per_class
    return combined


def aggregate_overall_excl_zero(run_metrics: List[RunMetrics]) -> Dict | None:
    per_run_metrics: List[Dict] = []

    for rm in run_metrics:
        metrics_nonzero: List[Dict] = []

        for pattern, ratio_dict in rm.by_ratio.items():
            for ratio_key, metrics in ratio_dict.items():
                ratio_val = float(ratio_key)
                if ratio_val > 0.0:
                    metrics_nonzero.append(metrics)

        if len(metrics_nonzero) == 0:
            continue

        combined = _combine_metrics_over_nonzero_ratios(metrics_nonzero)
        per_run_metrics.append(combined)

    if len(per_run_metrics) == 0:
        return None

    return aggregate_metrics_list(per_run_metrics)


def make_output_dir(run_dirs: List[Path], test_index: int | None = None) -> Path:
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
    payload = {
        "run_dirs": [str(p) for p in run_dirs],
        "agg_overall": agg["agg_overall"],
        "agg_by_ratio": agg["agg_by_ratio"],
    }

    with open(out_dir / "agg_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    overall = agg["agg_overall"]
    scalars = overall["scalars"]

    lines: List[str] = []
    lines.append(f"num_runs: {overall['num_runs']}")
    lines.append(f"num_samples (per run): {overall['num_samples']}")
    lines.append("")
    lines.append("[overall scalar metrics]")
    for key in sorted(scalars.keys()):
        m = scalars[key]
        lines.append(f"{key}: mean={m['mean']:.6f}, std={m['std']:.6f}")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_agg_excl_zero_json_txt(
    agg_excl_zero: Dict,
    run_dirs: List[Path],
    out_dir: Path,
):
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
        lines.append(f"{key}: mean={m['mean']:.6f}, std={m['std']:.6f}")

    with open(out_dir / "summary_excl_zero.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_ratio_curves(agg_by_ratio: Dict[str, Dict[float, Dict]], out_dir: Path):
    ratio_dir = out_dir / "by_ratio"
    ratio_dir.mkdir(parents=True, exist_ok=True)

    for pattern, ratios_dict in agg_by_ratio.items():
        pattern_dir = ratio_dir / f"pattern_{pattern}"
        pattern_dir.mkdir(parents=True, exist_ok=True)

        ratio_keys = sorted(ratios_dict.keys(), key=float)
        if len(ratio_keys) == 0:
            continue

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


def _parse_seed_from_run_dir_name(name: str) -> str:
    tokens = name.split("_")
    for tok in tokens:
        if tok.startswith("seed"):
            return tok[4:] if len(tok) > 4 else tok
    return name


def _fmt_ndigits_half_up(x, ndigits: int) -> str:
    if isinstance(x, (int, float)):
        q = Decimal(f"1e-{ndigits}")
        return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))
    return str(x)


def save_ratio_csvs(
    agg_by_ratio: Dict[str, Dict[float, Dict]],
    run_dirs: List[Path],
    out_dir: Path,
):
    if len(agg_by_ratio) == 0:
        return

    pattern_names = sorted(agg_by_ratio.keys())
    pattern = pattern_names[0]
    ratios_dict = agg_by_ratio[pattern]

    if len(ratios_dict) == 0:
        return

    ratio_keys_sorted = sorted(ratios_dict.keys(), key=float)

    first_ratio = ratio_keys_sorted[0]
    metric_keys = sorted(ratios_dict[first_ratio]["scalars"].keys())

    seed_names: List[str] = []
    for rd in run_dirs:
        seed_names.append(_parse_seed_from_run_dir_name(rd.name))

    num_runs = len(seed_names)

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
                    if i < len(vals):
                        v = _fmt_ndigits_half_up(vals[i], 5)
                    else:
                        v = ""
                    row.append(v)
                writer.writerow(row)

    agg_csv_path = out_dir / "metrics_by_ratio_mean_std.csv"
    with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["구분", "메트릭"] + [str(r) for r in ratio_keys_sorted]
        writer.writerow(header)

        for metric in metric_keys:
            row_mean: List[object] = ["평균", metric]
            for rk in ratio_keys_sorted:
                scalar = agg_by_ratio[pattern][rk]["scalars"][metric]
                row_mean.append(_fmt_ndigits_half_up(scalar["mean"], 5))
            writer.writerow(row_mean)

            row_std: List[object] = ["편차", metric]
            for rk in ratio_keys_sorted:
                scalar = agg_by_ratio[pattern][rk]["scalars"][metric]
                row_std.append(_fmt_ndigits_half_up(scalar["std"], 5))
            writer.writerow(row_std)


# ─────────────────────────────────────────────
# LaTeX model block 자동 생성 (MODEL 환경변수)
# ─────────────────────────────────────────────


def _read_ratio_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 헤더를 읽지 못했습니다.")
        rows = list(reader)
        return reader.fieldnames, rows


def _read_overall_means_from_summary(summary_path: Path) -> dict[str, float]:
    text = summary_path.read_text(encoding="utf-8", errors="strict").splitlines()

    in_block = False
    means: dict[str, float] = {}

    for line in text:
        s = line.strip()
        if not s:
            continue

        if s == "[overall scalar metrics]":
            in_block = True
            continue

        if not in_block:
            continue

        m = re.match(r"^([A-Za-z0-9_]+):\s*mean=([+-]?\d+(\.\d+)?),\s*std=", s)
        if m is None:
            continue

        key = m.group(1)
        mean_val = float(m.group(2))
        means[key] = mean_val

    if not means:
        raise ValueError(
            "summary.txt 에서 [overall scalar metrics] 블록을 파싱하지 못했습니다."
        )

    return means


def _format_float_half_up(x: float, ndigits: int) -> str:
    q = Decimal(f"1e-{ndigits}")
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def _format_str_number_half_up(x_str: str, ndigits: int) -> str:
    q = Decimal(f"1e-{ndigits}")
    return str(Decimal(x_str).quantize(q, rounding=ROUND_HALF_UP))


def _get_ratio_cols(fieldnames: list[str]) -> list[str]:
    base_exclude = {"구분", "메트릭", "Avg", "avg", "AVG"}
    ratio_cols: list[str] = []
    for c in fieldnames:
        if c in base_exclude:
            continue
        if re.match(r"^\d+(\.\d+)?$", c) is None:
            continue
        ratio_cols.append(c)

    if not ratio_cols:
        raise ValueError("CSV 에 ratio 컬럼(예: 0.0, 0.1, ...)이 없습니다.")

    ratio_cols.sort(key=lambda s: float(s))
    return ratio_cols


def _build_mean_row_map(
    fieldnames: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], dict[str, list[str]]]:
    ratio_cols = _get_ratio_cols(fieldnames)
    mean_rows = [r for r in rows if r["구분"] == "평균"]
    if not mean_rows:
        raise ValueError("CSV 에서 '구분=평균' 행을 찾지 못했습니다.")

    m: dict[str, list[str]] = {}
    for r in mean_rows:
        metric = r["메트릭"]
        m[metric] = [r[c] for c in ratio_cols]

    return ratio_cols, m


def _render_latex_block_model_only(
    model_name: str,
    ratio_cols: list[str],
    mean_map: dict[str, list[str]],
    overall_mean_map: dict[str, float],
    ndigits: int = 5,
) -> str:
    order: list[tuple[str, str]] = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
    ]

    n = len(ratio_cols)
    for k, _ in order:
        if k not in mean_map:
            raise ValueError(f"CSV 평균 행에 '{k}' 가 없습니다.")
        if len(mean_map[k]) != n:
            raise ValueError("메트릭별 ratio 개수가 일치하지 않습니다.")
        if k not in overall_mean_map:
            raise ValueError(f"summary.txt overall scalar metrics에 '{k}' 가 없습니다.")

    def fmt(metric_key: str) -> tuple[str, str]:
        vals_strs = mean_map[metric_key]
        ratio_strs = [_format_str_number_half_up(vs, ndigits) for vs in vals_strs]
        avg = _format_float_half_up(overall_mean_map[metric_key], ndigits)
        return " & ".join(ratio_strs), avg

    # 컬럼 수에 맞춰 cline 자동 계산:
    # (첫 칸 비움) + (model) + (metric name) + (ratios n개) + (avg) = n + 4
    cline_end = n + 4
    sep = rf"\cline{{2-{cline_end}}}"

    lines: list[str] = []

    lines.append(rf"& \multirow{{4}}{{*}}{{{model_name}}}")

    ratios, avg = fmt("accuracy")
    lines.append(rf"& Accuracy  & {ratios} & {avg} \\")
    ratios, avg = fmt("f1_macro")
    lines.append(rf"& & F1        & {ratios} & {avg} \\")
    ratios, avg = fmt("precision_macro")
    lines.append(rf"& & Precision & {ratios} & {avg} \\")
    ratios, avg = fmt("recall_macro")
    lines.append(rf"& & Recall    & {ratios} & {avg} \\")

    lines.append(sep)
    return "\n".join(lines) + "\n"


def _avg_from_ratio_strs(vals_strs: list[str], ndigits: int) -> str:
    n = len(vals_strs)
    if n == 0:
        raise ValueError("ratio 값이 비어 있어 Avg를 계산할 수 없습니다.")

    total = Decimal("0")
    for s in vals_strs:
        total += Decimal(s)

    avg = total / Decimal(str(n))
    q = Decimal(f"1e-{ndigits}")
    return str(avg.quantize(q, rounding=ROUND_HALF_UP))


def _render_latex_block_model_only(
    model_name: str,
    ratio_cols: list[str],
    mean_map: dict[str, list[str]],
    ndigits: int = 5,
    row_sep: str = r"\hline",
) -> str:
    order: list[tuple[str, str]] = [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
    ]

    n = len(ratio_cols)
    for k, _ in order:
        if k not in mean_map:
            raise ValueError(f"CSV 평균 행에 '{k}' 가 없습니다.")
        if len(mean_map[k]) != n:
            raise ValueError("메트릭별 ratio 개수가 일치하지 않습니다.")

    def fmt(metric_key: str) -> tuple[str, str]:
        vals_strs = mean_map[metric_key]
        ratio_strs = [_format_str_number_half_up(vs, ndigits) for vs in vals_strs]
        avg = _avg_from_ratio_strs(vals_strs, ndigits)
        return " & ".join(ratio_strs), avg

    lines: list[str] = []

    # 사용자가 원하신 형태: 첫 줄에 모델 multirow만 두고,
    # 다음 줄부터 "& Metric ..." 로 이어서 같은 행으로 만들기
    lines.append(rf"\multirow{{4}}{{*}}{{{model_name}}}")

    ratios, avg = fmt("accuracy")
    lines.append(rf"& Accuracy  & {ratios} & {avg} \\")
    ratios, avg = fmt("f1_macro")
    lines.append(rf"& F1        & {ratios} & {avg} \\")
    ratios, avg = fmt("precision_macro")
    lines.append(rf"& Precision & {ratios} & {avg} \\")
    ratios, avg = fmt("recall_macro")
    lines.append(rf"& Recall    & {ratios} & {avg} \\")

    lines.append(row_sep)
    return "\n".join(lines) + "\n"


def _maybe_write_model_block(out_dir: Path) -> None:
    if "MODEL" not in os.environ:
        return

    model_name = os.environ["MODEL"].strip()
    if not model_name:
        return

    csv_path = out_dir / "metrics_by_ratio_mean_std.csv"
    if not csv_path.exists():
        raise ValueError(f"model_block 생성 실패: {csv_path} 가 없습니다.")

    fieldnames, rows = _read_ratio_csv(csv_path)
    ratio_cols, mean_map = _build_mean_row_map(fieldnames, rows)

    latex = _render_latex_block_model_only(
        model_name=model_name,
        ratio_cols=ratio_cols,
        mean_map=mean_map,
        ndigits=5,
        row_sep=r"\hline",
    )

    out_path = out_dir / "model_block.txt"
    out_path.write_text(latex, encoding="utf-8", errors="strict")


def main():
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

    if "runs" not in auto_cfg:
        raise ValueError(f"{auto_json_path} 안에 runs 항목이 없습니다.")
    runs = auto_cfg["runs"]
    if len(runs) == 0:
        raise ValueError(f"{auto_json_path} 안에 runs 항목이 비어 있습니다.")

    run_dirs: List[Path] = []

    print("[agg_seeds] auto json mode:")
    for r in runs:
        if "seed" not in r:
            raise ValueError("auto json의 runs 원소에 seed가 없습니다.")
        if "dir" not in r:
            raise ValueError("auto json의 runs 원소에 dir이 없습니다.")
        seed = r["seed"]
        d = r["dir"]
        print(f"  seed={seed}, dir={d}")
        run_dirs.append(Path(d))

    run_metrics: List[RunMetrics] = []
    for rd in run_dirs:
        rm = load_results(rd, test_index=test_index)
        run_metrics.append(rm)

    agg = aggregate_all(run_metrics)

    out_dir = make_output_dir(run_dirs, test_index=test_index)
    print(f"[agg_seeds] output dir: {out_dir}")

    save_agg_json_txt(agg, run_dirs, out_dir)
    save_ratio_csvs(agg["agg_by_ratio"], run_dirs, out_dir)

    agg_excl_zero = aggregate_overall_excl_zero(run_metrics)
    if agg_excl_zero is not None:
        print("[agg_seeds] also saving metrics excluding ratio=0.0")
        save_agg_excl_zero_json_txt(agg_excl_zero, run_dirs, out_dir)

    plot_ratio_curves(agg["agg_by_ratio"], out_dir)

    # 추가: MODEL 환경변수가 있으면 LaTeX 블록 자동 생성
    _maybe_write_model_block(out_dir)


if __name__ == "__main__":
    main()
