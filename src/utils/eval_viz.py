# src/utils/eval_viz.py

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


CLASS_COLOR_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    "tab:olive", "tab:cyan",
]


def _plot_grouped_bars(
    classes: list[str],
    metrics: dict[str, list[float]],
    title: str,
    outpath: Path,
):
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
    metrics_by_ratio[ratio] = compute_classification_metrics(...) 결과 dict
    metric_key: "accuracy", "f1_macro" 등 최상위 키
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ratios = sorted(metrics_by_ratio.keys())
    ys: list[float] = []

    for r in ratios:
        m = metrics_by_ratio[r]      # m: classification metrics dict
        ys.append(float(m[metric_key]))

    # 1) 자동 스케일 그래프
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

    # 2) y축 [0,1] 고정 그래프
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
    compute_classification_metrics 결과(dict)를 저장 + 시각화.

    - metrics.json
    - 클래스별 overall 막대 (precision/recall/f1)
    - summary.txt
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) JSON 저장
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 2) 클래스별 overall 막대
    if "per_class" in metrics and isinstance(metrics["per_class"], dict):
        cls_keys = sorted(metrics["per_class"].keys(), key=lambda s: int(s))
        prec = [metrics["per_class"][k]["precision"] for k in cls_keys]
        rec  = [metrics["per_class"][k]["recall"]    for k in cls_keys]
        f1   = [metrics["per_class"][k]["f1"]        for k in cls_keys]

        _plot_grouped_bars(
            cls_keys,
            {"precision": prec, "recall": rec, "f1": f1},
            "Per-class metrics (precision/recall/f1)",
            save_dir / "per_class_overall.png",
        )

    # 3) 전체 요약 텍스트
    with open(save_dir / "summary.txt", "w", encoding="utf-8") as f:
        if "accuracy" in metrics:
            f.write(f"accuracy: {metrics['accuracy']:.6f}\n")

        if "precision_micro" in metrics:
            f.write(
                "precision_micro: "
                f"{metrics['precision_micro']:.6f}, "
                "recall_micro: "
                f"{metrics['recall_micro']:.6f}, "
                "f1_micro: "
                f"{metrics['f1_micro']:.6f}\n"
            )

        if "precision_macro" in metrics:
            f.write(
                "precision_macro: "
                f"{metrics['precision_macro']:.6f}, "
                "recall_macro: "
                f"{metrics['recall_macro']:.6f}, "
                "f1_macro: "
                f"{metrics['f1_macro']:.6f}\n"
            )

        if "num_samples" in metrics:
            f.write(f"num_samples: {metrics['num_samples']}\n")


# -------------------- loss history 시각화 --------------------


def _set_ylim_from_series(
    series_list,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    padding_ratio: float = 0.1,
):
    data = []
    for s in series_list:
        arr = np.asarray(s, dtype=float).ravel()
        if arr.size > 0:
            data.append(arr)

    if len(data) == 0:
        return

    data = np.concatenate(data, axis=0)
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return
    data = data[finite_mask]

    y_low = np.quantile(data, lower_q)
    y_high = np.quantile(data, upper_q)

    if y_low == y_high:
        eps = abs(y_low) * 0.1 if y_low != 0 else 1.0
        y_low -= eps
        y_high += eps

    span = y_high - y_low
    pad = span * padding_ratio
    y_min = y_low - pad
    y_max = y_high + pad

    plt.ylim(y_min, y_max)


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

    각 task별 train/valid curve + 전체 평균 curve 저장.
    (여기서의 task는 더 이상 horizon 개념이 아니라, 단순히 여러 loss 시퀀스라는 의미)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_name: str = history["metric_name"]
    tasks: list[dict] = history["tasks"]

    # 1) 각 task별 개별 그래프
    for t, h in enumerate(tasks):
        train_series = h["train"]
        valid_series = h["valid"]

        # (1) 원본 스케일
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

        # (2) 분위수 기반 확대
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

    # 2) 전체 task 평균 그래프
    if len(tasks) == 0:
        return

    min_len_train = min(len(h["train"]) for h in tasks)
    min_len_valid = min(len(h["valid"]) for h in tasks)
    L = min(min_len_train, min_len_valid)

    if L <= 0:
        return

    train_stack = np.stack(
        [np.asarray(h["train"][:L], dtype=float) for h in tasks],
        axis=0,
    )
    valid_stack = np.stack(
        [np.asarray(h["valid"][:L], dtype=float) for h in tasks],
        axis=0,
    )

    train_mean = train_stack.mean(axis=0)
    valid_mean = valid_stack.mean(axis=0)
    iters = np.arange(L)

    # (1) 원본 스케일
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

    # (2) 확대 버전
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
