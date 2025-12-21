#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def read_ratio_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 헤더를 읽지 못했습니다.")
        rows = list(reader)
        return reader.fieldnames, rows


def read_overall_means_from_summary(summary_path: Path) -> dict[str, float]:
    """
    summary.txt 예:
      [overall scalar metrics]
      accuracy: mean=0.812797, std=0.006773
      f1_macro: mean=0.789300, std=0.007914
      ...

    -> {"accuracy": 0.812797, "f1_macro": 0.789300, ...}
    """
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


def format_float_half_up(x: float, ndigits: int) -> str:
    """
    f-string(은행가 반올림) 대신, 일반적으로 기대하는 ROUND_HALF_UP로 고정.
    """
    q = Decimal(f"1e-{ndigits}")
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def format_str_number_half_up(x_str: str, ndigits: int) -> str:
    """
    CSV에서 읽은 숫자 문자열을 Decimal로 안전하게 ndigits 반올림(half-up)해 출력.
    """
    q = Decimal(f"1e-{ndigits}")
    return str(Decimal(x_str).quantize(q, rounding=ROUND_HALF_UP))


def get_ratio_cols(fieldnames: list[str]) -> list[str]:
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


def build_mean_row_map(
    fieldnames: list[str], rows: list[dict[str, str]]
) -> tuple[list[str], dict[str, list[str]]]:
    ratio_cols = get_ratio_cols(fieldnames)
    mean_rows = [r for r in rows if r["구분"] == "평균"]
    if not mean_rows:
        raise ValueError("CSV 에서 '구분=평균' 행을 찾지 못했습니다.")

    m: dict[str, list[str]] = {}
    for r in mean_rows:
        metric = r["메트릭"]
        m[metric] = [r[c] for c in ratio_cols]

    return ratio_cols, m


def render_latex_block_model_only(
    model_name: str,
    ratio_cols: list[str],
    mean_map: dict[str, list[str]],
    overall_mean_map: dict[str, float],
    ndigits: int = 5,
    sep: str = r"\cline{2-10}",
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
        ratio_strs = [format_str_number_half_up(vs, ndigits) for vs in vals_strs]
        avg = format_float_half_up(overall_mean_map[metric_key], ndigits)
        return " & ".join(ratio_strs), avg

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="test_0 같은 실행 폴더. 내부에서 metrics_by_ratio_mean_std.csv 와 summary.txt 를 자동 탐색",
    )
    p.add_argument(
        "--csv_path", type=str, default=None, help="metrics_by_ratio_mean_std.csv 경로"
    )
    p.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="summary.txt 경로 (기본: run_dir 또는 csv_path의 부모 폴더)",
    )
    p.add_argument("--model", type=str, required=True, help="예: XGBoost")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="출력 txt 경로 (기본: run_dir/model_block.txt 또는 현재 폴더 model_block.txt)",
    )
    args = p.parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        csv_path = run_dir / "metrics_by_ratio_mean_std.csv"
        summary_path = (
            run_dir / "summary.txt"
            if args.summary_path is None
            else Path(args.summary_path)
        )
    else:
        if args.csv_path is None:
            raise ValueError("--run_dir 또는 --csv_path 중 하나는 지정해야 합니다.")
        csv_path = Path(args.csv_path)
        if args.summary_path is not None:
            summary_path = Path(args.summary_path)
        else:
            summary_path = csv_path.parent / "summary.txt"

    if args.out is not None:
        out_path = Path(args.out)
    else:
        if args.run_dir is not None:
            out_path = Path(args.run_dir) / "model_block.txt"
        else:
            out_path = Path("model_block.txt")

    fieldnames, rows = read_ratio_csv(csv_path)
    ratio_cols, mean_map = build_mean_row_map(fieldnames, rows)
    overall_mean_map = read_overall_means_from_summary(summary_path)

    latex = render_latex_block_model_only(
        args.model,
        ratio_cols,
        mean_map,
        overall_mean_map,
        ndigits=5,
    )

    out_path.write_text(latex, encoding="utf-8", errors="strict")


if __name__ == "__main__":
    main()
