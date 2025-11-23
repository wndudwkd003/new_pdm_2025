from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt



CLASS_COLOR_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    "tab:olive", "tab:cyan",
]

def _plot_per_class_step_curves(
    metrics: dict,
    save_dir: Path,
    metric_name: str,                # "accuracy" 또는 "f1" 등
    class_colors: dict[str, str] | None = None,
):
    """
    metrics["per_step"][t]["per_class"][cls][metric_name] 를 이용해서

    1) 통합 그래프(여러 클래스 한 장, 동적 y축 / 고정 y축 0~1)
    2) 클래스별 단독 그래프 (동적 y축 / 고정 y축 0~1)

    를 모두 저장한다.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # per_step 존재 여부 확인
    if "per_step" not in metrics:
        return
    per_step = metrics["per_step"]
    if not isinstance(per_step, list) or len(per_step) == 0:
        return

    # per_class 존재 여부 확인
    if "per_class" not in per_step[0]:
        return

    # 클래스 리스트 (문자열 key 기준, 정수형으로 정렬)
    first_per_class = per_step[0]["per_class"]
    cls_keys = sorted(first_per_class.keys(), key=lambda s: int(s))

    # metric_name 이 실제로 존재하는지 확인
    first_cls_key = cls_keys[0]
    if metric_name not in first_per_class[first_cls_key]:
        # 예: per_class 에 accuracy 가 없으면 아무 것도 안 그림
        return

    # 클래스별로 step 시퀀스를 모음
    series_by_cls: dict[str, list[float]] = {cls: [] for cls in cls_keys}
    for step in per_step:
        step_pc = step["per_class"]
        for cls in cls_keys:
            value = float(step_pc[cls][metric_name])
            series_by_cls[cls].append(value)

    # =========================
    # 1) 통합 그래프 (여러 클래스를 한 장에)
    # =========================
    # 1-1) 동적 y축(분위수 기반 zoom)
    plt.figure()
    all_series = []
    for i, cls in enumerate(cls_keys):
        seq = np.asarray(series_by_cls[cls], dtype=float)
        all_series.append(seq)
        if class_colors is not None and cls in class_colors:
            color = class_colors[cls]
        else:
            color = CLASS_COLOR_PALETTE[i % len(CLASS_COLOR_PALETTE)]
        plt.plot(seq, label=f"class {cls}", color=color)

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"Per-class {metric_name} over step (zoom)")
    plt.grid(True)
    plt.legend()

    _set_ylim_from_series(all_series)

    plt.tight_layout()
    outpath_all_zoom = save_dir / f"per_class_step_{metric_name}_all_zoom.png"
    plt.savefig(outpath_all_zoom, dpi=150)
    plt.close()

    # 1-2) 고정 y축 [0, 1]
    plt.figure()
    for i, cls in enumerate(cls_keys):
        seq = np.asarray(series_by_cls[cls], dtype=float)
        if class_colors is not None and cls in class_colors:
            color = class_colors[cls]
        else:
            color = CLASS_COLOR_PALETTE[i % len(CLASS_COLOR_PALETTE)]
        plt.plot(seq, label=f"class {cls}", color=color)

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"Per-class {metric_name} over step (fixed 0-1)")
    plt.grid(True)
    plt.legend()
    plt.ylim(0.0, 1.0)

    plt.tight_layout()
    outpath_all_fixed = save_dir / f"per_class_step_{metric_name}_all_fixed01.png"
    plt.savefig(outpath_all_fixed, dpi=150)
    plt.close()

    # =========================
    # 2) 클래스별 단독 그래프
    # =========================
    for i, cls in enumerate(cls_keys):
        seq = np.asarray(series_by_cls[cls], dtype=float)

        if class_colors is not None and cls in class_colors:
            color = class_colors[cls]
        else:
            color = CLASS_COLOR_PALETTE[i % len(CLASS_COLOR_PALETTE)]

        # 2-1) 동적 y축(zoom)
        plt.figure()
        plt.plot(seq, label=f"class {cls}", color=color)
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.title(f"class {cls} - {metric_name} over step (zoom)")
        plt.grid(True)
        plt.legend()

        _set_ylim_from_series([seq])

        plt.tight_layout()
        outpath_cls_zoom = save_dir / f"per_class_step_{metric_name}_class{cls}_zoom.png"
        plt.savefig(outpath_cls_zoom, dpi=150)
        plt.close()

        # 2-2) 고정 y축 [0, 1]
        plt.figure()
        plt.plot(seq, label=f"class {cls}", color=color)
        plt.xlabel("Step")
        plt.ylabel(metric_name)
        plt.title(f"class {cls} - {metric_name} over step (fixed 0-1)")
        plt.grid(True)
        plt.legend()
        plt.ylim(0.0, 1.0)

        plt.tight_layout()
        outpath_cls_fixed = save_dir / f"per_class_step_{metric_name}_class{cls}_fixed01.png"
        plt.savefig(outpath_cls_fixed, dpi=150)
        plt.close()




def _set_ylim_from_series(
    series_list,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    padding_ratio: float = 0.1,
):
    """
    여러 시계열 값들을 받아서:
    - lower_q ~ upper_q 분위수 구간에 맞춰 확대
    - 위아래 padding_ratio 만큼 여유를 둔 ylim 설정

    series_list: iterable 들의 리스트 (예: [train_series, valid_series])
    """
    # 1) 합쳐서 하나의 1D array로
    data = []
    for s in series_list:
        arr = np.asarray(s, dtype=float).ravel()
        if arr.size > 0:
            data.append(arr)
    if len(data) == 0:
        return

    data = np.concatenate(data, axis=0)

    # 유한값만 사용
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return
    data = data[finite_mask]

    # 2) 분위수 기반 범위
    y_low = np.quantile(data, lower_q)
    y_high = np.quantile(data, upper_q)

    # 값이 전부 동일한 경우 대비
    if y_low == y_high:
        eps = abs(y_low) * 0.1 if y_low != 0 else 1.0
        y_low -= eps
        y_high += eps

    # 3) 패딩
    span = y_high - y_low
    pad = span * padding_ratio
    y_min = y_low - pad
    y_max = y_high + pad

    # 실제 ylim 적용
    plt.ylim(y_min, y_max)


def _plot_line(series, xlab: str, ylab: str, title: str, outpath: Path):
    """
    기존 동작 그대로: 스케일 조정 없이 전체 범위 그래프 1장 저장
    """
    plt.figure()
    plt.plot(series)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_line_zoom(series, xlab: str, ylab: str, title: str, outpath: Path):
    """
    분위수 기반 확대 + 여백 적용한 버전
    """
    plt.figure()
    plt.plot(series)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title + " (zoom)")
    plt.grid(True)

    _set_ylim_from_series([series])

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_line_fixed_ylim(
    series,
    xlab: str,
    ylab: str,
    title: str,
    outpath: Path,
    y_min: float = 0.0,
    y_max: float = 1.0,
):
    """
    y축을 [y_min, y_max]로 고정해서 그리는 버전.
    (예: 분류 metric의 경우 0.0 ~ 1.0)
    """
    plt.figure()
    plt.plot(series)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title + f" (fixed {y_min:.1f}-{y_max:.1f})")
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def _plot_grouped_bars(classes: list[str], metrics: dict[str, list[float]], title: str, outpath: Path):
    x = np.arange(len(classes))
    width = 0.25
    plt.figure()
    plt.bar(x - width, metrics["precision"], width, label="precision")
    plt.bar(x,          metrics["recall"],   width, label="recall")
    plt.bar(x + width,  metrics["f1"],       width, label="f1")
    plt.xticks(x, classes)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_metric_over_ratio(
    metrics_by_ratio: dict[float, dict],
    metric_key: str,
    save_dir: Path,
    prefix: str,
):
    """
    metrics_by_ratio[ratio] = compute_multitask_classification_metrics(...) 결과 dict
    metric_key: "accuracy", "f1_macro" 등 overall 밑의 키
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ratios = sorted(metrics_by_ratio.keys())
    ys: list[float] = []

    for r in ratios:
        m = metrics_by_ratio[r]
        ys.append(float(m["overall"][metric_key]))

    # 1) 기존: 데이터 범위에 맞게 자동 스케일
    plt.figure()
    plt.plot(ratios, ys, marker="o")
    plt.xlabel("missing ratio")
    plt.ylabel(metric_key)
    plt.title(f"{prefix} - {metric_key} vs missing ratio")
    plt.grid(True)
    plt.tight_layout()
    outpath = save_dir / f"{prefix}_ratio_{metric_key}.png"
    plt.savefig(outpath, dpi=150)
    plt.close()

    # 2) 추가: y축 0.0~1.0 고정 버전
    plt.figure()
    plt.plot(ratios, ys, marker="o")
    plt.xlabel("missing ratio")
    plt.ylabel(metric_key)
    plt.title(f"{prefix} - {metric_key} vs missing ratio (fixed 0-1)")
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    outpath_fixed = save_dir / f"{prefix}_ratio_{metric_key}_fixed01.png"
    plt.savefig(outpath_fixed, dpi=150)
    plt.close()


def save_metrics_artifacts(
    metrics: dict,
    save_dir: Path,
):
    """
    compute_multitask_classification_metrics 결과(dict)를 저장 + 시각화.

    - metrics.json
    - per-step Accuracy / Macro F1 꺾은선 (+ zoom 버전, + 0~1 고정 버전)
    - 클래스별 overall 막대
    - summary.txt
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) JSON 저장
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 2) 스텝별 Accuracy (+ Macro F1)
    if "per_step" in metrics and isinstance(metrics["per_step"], list) and len(metrics["per_step"]) > 0:
        acc = [step["accuracy"] for step in metrics["per_step"]]

        # (1) 원본
        _plot_line(acc, "Step", "Accuracy", "Per-step Accuracy", save_dir / "per_step_accuracy.png")

        # (2) 확대본 (분위수 기반 zoom)
        _plot_line_zoom(acc, "Step", "Accuracy", "Per-step Accuracy", save_dir / "per_step_accuracy_zoom.png")

        # (3) y축 0~1 고정 버전
        _plot_line_fixed_ylim(
            acc,
            "Step",
            "Accuracy",
            "Per-step Accuracy",
            save_dir / "per_step_accuracy_fixed01.png",
            y_min=0.0,
            y_max=1.0,
        )

        if "f1_macro" in metrics["per_step"][0]:
            f1m = [step["f1_macro"] for step in metrics["per_step"]]

            # (1) 원본
            _plot_line(f1m, "Step", "F1 (macro)", "Per-step Macro F1", save_dir / "per_step_f1_macro.png")

            # (2) 확대본 (분위수 기반 zoom)
            _plot_line_zoom(f1m, "Step", "F1 (macro)", "Per-step Macro F1", save_dir / "per_step_f1_macro_zoom.png")

            # (3) y축 0~1 고정 버전
            _plot_line_fixed_ylim(
                f1m,
                "Step",
                "F1 (macro)",
                "Per-step Macro F1",
                save_dir / "per_step_f1_macro_fixed01.png",
                y_min=0.0,
                y_max=1.0,
            )

    # 3) 클래스별 overall 막대
    if "per_class_overall" in metrics and isinstance(metrics["per_class_overall"], dict):
        cls_keys = sorted(metrics["per_class_overall"].keys(), key=lambda s: int(s))
        prec = [metrics["per_class_overall"][k]["precision"] for k in cls_keys]
        rec  = [metrics["per_class_overall"][k]["recall"]    for k in cls_keys]
        f1   = [metrics["per_class_overall"][k]["f1"]        for k in cls_keys]
        _plot_grouped_bars(
            cls_keys,
            {"precision": prec, "recall": rec, "f1": f1},
            "Per-class Overall (precision/recall/f1)",
            save_dir / "per_class_overall.png",
        )

    # 4) 전체 요약 텍스트
    if "overall" in metrics and isinstance(metrics["overall"], dict):
        o = metrics["overall"]
        with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
            if "accuracy" in o:
                f.write(f"overall accuracy: {o['accuracy']:.6f}\n")
            if "precision_micro" in o:
                f.write(
                    f"precision_micro: {o['precision_micro']:.6f}, "
                    f"recall_micro: {o['recall_micro']:.6f}, "
                    f"f1_micro: {o['f1_micro']:.6f}\n"
                )
            if "precision_macro" in o:
                f.write(
                    f"precision_macro: {o['precision_macro']:.6f}, "
                    f"recall_macro: {o['recall_macro']:.6f}, "
                    f"f1_macro: {o['f1_macro']:.6f}\n"
                )

    # 5) 클래스별 step-curve (accuracy / f1)
    if "per_step" in metrics and isinstance(metrics["per_step"], list) and len(metrics["per_step"]) > 0:
        first_step = metrics["per_step"][0]
        if "per_class" in first_step:
            # per_class 에 해당 metric 이 실제로 있는 경우에만 그림
            first_pc = first_step["per_class"]
            cls_keys = sorted(first_pc.keys(), key=lambda s: int(s))

            # accuracy
            if len(cls_keys) > 0 and "accuracy" in first_pc[cls_keys[0]]:
                _plot_per_class_step_curves(
                    metrics,
                    save_dir,
                    metric_name="accuracy",
                    class_colors=None,  # 필요하면 dict 넘기면 됨 {"0": "red", ...}
                )

            # f1 (per_class 에 "f1" 키가 있다고 가정)
            if len(cls_keys) > 0 and "f1" in first_pc[cls_keys[0]]:
                _plot_per_class_step_curves(
                    metrics,
                    save_dir,
                    metric_name="f1",
                    class_colors=None,
                )

def save_history_artifacts(
    history: dict,
    save_dir: Path,
):
    """
    history:
    {
        "metric_name": str,
        "tasks": [
            {"train": [...], "valid": [...]},
            ...
        ]
    }

    각 task(timestep)별로 train/valid curve를 그림.
    - 기존: task{t}_{metric_name}.png  (원본 스케일)
    - 추가: task{t}_{metric_name}_zoom.png  (확대/여백)
    - 추가: 모든 task를 평균낸 overall train/valid curve도 1장 + zoom 1장
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_name: str = history["metric_name"]
    tasks: list[dict] = history["tasks"]

    # -----------------------------
    # 1) 각 task별 개별 그래프
    # -----------------------------
    for t, h in enumerate(tasks):
        train_series = h["train"]
        valid_series = h["valid"]

        # 1) 원본 스케일 그래프 (기존 동작)
        plt.figure()
        plt.plot(train_series, label="train")
        plt.plot(valid_series, label="valid")
        plt.xlabel("Iteration")
        plt.ylabel(metric_name)
        plt.title(f"Task {t} - {metric_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        outpath = save_dir / f"task{t}_{metric_name}.png"
        plt.savefig(outpath, dpi=150)
        plt.close()

        # 2) 확대/여백 적용 그래프
        plt.figure()
        plt.plot(train_series, label="train")
        plt.plot(valid_series, label="valid")
        plt.xlabel("Iteration")
        plt.ylabel(metric_name)
        plt.title(f"Task {t} - {metric_name} (zoom)")
        plt.grid(True)
        plt.legend()

        _set_ylim_from_series([train_series, valid_series])

        plt.tight_layout()
        outpath_zoom = save_dir / f"task{t}_{metric_name}_zoom.png"
        plt.savefig(outpath_zoom, dpi=150)
        plt.close()

    # -----------------------------
    # 2) 통합(전체 task 평균) loss 그래프
    # -----------------------------
    if len(tasks) == 0:
        return

    # 각 task별 길이가 다를 수 있으므로 공통 구간(min length)까지만 사용
    min_len_train = min(len(h["train"]) for h in tasks)
    min_len_valid = min(len(h["valid"]) for h in tasks)
    L = min(min_len_train, min_len_valid)

    if L <= 0:
        return

    # (num_tasks, L) 로 쌓아서 평균
    train_stack = np.stack(
        [np.asarray(h["train"][:L], dtype=float) for h in tasks],
        axis=0
    )
    valid_stack = np.stack(
        [np.asarray(h["valid"][:L], dtype=float) for h in tasks],
        axis=0
    )

    train_mean = train_stack.mean(axis=0)
    valid_mean = valid_stack.mean(axis=0)

    iters = np.arange(L)

    # 2-1) 통합 원본 스케일
    plt.figure()
    plt.plot(iters, train_mean, label="train (mean)")
    plt.plot(iters, valid_mean, label="valid (mean)")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title(f"Overall - {metric_name} (mean over tasks)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    outpath_overall = save_dir / f"overall_{metric_name}.png"
    plt.savefig(outpath_overall, dpi=150)
    plt.close()

    # 2-2) 통합 확대/여백 적용
    plt.figure()
    plt.plot(iters, train_mean, label="train (mean)")
    plt.plot(iters, valid_mean, label="valid (mean)")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title(f"Overall - {metric_name} (mean over tasks, zoom)")
    plt.grid(True)
    plt.legend()

    _set_ylim_from_series([train_mean, valid_mean])

    plt.tight_layout()
    outpath_overall_zoom = save_dir / f"overall_{metric_name}_zoom.png"
    plt.savefig(outpath_overall_zoom, dpi=150)
    plt.close()
