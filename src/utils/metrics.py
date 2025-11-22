# src/utils/metrics.py
from typing import Any
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def compute_multitask_classification_metrics(
    y_true: np.ndarray,   # (N, T)
    y_pred: np.ndarray,   # (N, T)
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """
    다중시점 다중분류용 메트릭 집계.
    반환 포맷:
    {
      "overall": {accuracy, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro},
      "per_step": [
        {"accuracy", "precision_micro", "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro",
         "per_class": {"0": {"precision","recall","f1","support"}, ...}},
        ...
      ],
      "per_class_overall": {"0": {...}, "1": {...}, ...},
      "num_samples": N,
      "num_tasks": T
    }
    """
    N, T = y_true.shape
    if labels is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        labels = list(range(num_classes))

    # ---- overall (모든 시점/샘플 평탄화) ----
    yt_flat = y_true.reshape(-1)
    yp_flat = y_pred.reshape(-1)

    overall_acc = float((yt_flat == yp_flat).mean())

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        yt_flat, yp_flat, labels=labels, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        yt_flat, yp_flat, labels=labels, average="macro", zero_division=0
    )
    pc_o, rc_o, fc_o, sc_o = precision_recall_fscore_support(
        yt_flat, yp_flat, labels=labels, average=None, zero_division=0
    )
    per_class_overall = {
        str(c): {"precision": float(pc_o[i]), "recall": float(rc_o[i]), "f1": float(fc_o[i]), "support": int(sc_o[i])}
        for i, c in enumerate(labels)
    }

    # ---- per-step ----
    per_step: list[dict[str, Any]] = []
    for j in range(T):
        yj, pj = y_true[:, j], y_pred[:, j]
        acc_j = float((yj == pj).mean())

        pmi, rmi, fmi, _ = precision_recall_fscore_support(
            yj, pj, labels=labels, average="micro", zero_division=0
        )
        pma, rma, fma, _ = precision_recall_fscore_support(
            yj, pj, labels=labels, average="macro", zero_division=0
        )
        pc, rc, fc, sc = precision_recall_fscore_support(
            yj, pj, labels=labels, average=None, zero_division=0
        )
        per_class = {
            str(c): {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sc[i])}
            for i, c in enumerate(labels)
        }

        per_step.append({
            "accuracy": acc_j,
            "precision_micro": float(pmi), "recall_micro": float(rmi), "f1_micro": float(fmi),
            "precision_macro": float(pma), "recall_macro": float(rma), "f1_macro": float(fma),
            "per_class": per_class,
        })

    return {
        "overall": {
            "accuracy": overall_acc,
            "precision_micro": float(p_micro), "recall_micro": float(r_micro), "f1_micro": float(f_micro),
            "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        },
        "per_step": per_step,
        "per_class_overall": per_class_overall,
        "num_samples": int(N),
        "num_tasks": int(T),
    }
