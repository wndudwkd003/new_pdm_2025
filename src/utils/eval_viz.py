# src/utils/eval_viz.py

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def _plot_line(series, xlab: str, ylab: str, title: str, outpath: Path):
    plt.figure()
    plt.plot(series)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
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


def save_metrics_artifacts(
    metrics: dict,
    save_dir: Path,
):
    """
    compute_multitask_classification_metrics 결과(dict)를 저장 + 시각화.

    - metrics.json
    - per-step Accuracy / Macro F1 꺾은선
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
        _plot_line(acc, "Step", "Accuracy", "Per-step Accuracy", save_dir / "per_step_accuracy.png")

        if "f1_macro" in metrics["per_step"][0]:
            f1m = [step["f1_macro"] for step in metrics["per_step"]]
            _plot_line(f1m, "Step", "F1 (macro)", "Per-step Macro F1", save_dir / "per_step_f1_macro.png")

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
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metric_name: str = history["metric_name"]
    tasks: list[dict] = history["tasks"]

    for t, h in enumerate(tasks):
        train_series = h["train"]
        valid_series = h["valid"]

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
