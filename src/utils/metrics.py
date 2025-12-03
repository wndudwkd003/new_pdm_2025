# src/utils/metrics.py

from typing import Any
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_classification_metrics(
    y_true: np.ndarray,   # (N,)
    y_pred: np.ndarray,   # (N,) 또는 (N, C)
    labels: list[int] | None = None,
) -> dict[str, Any]:
    """
    단일 분류용 메트릭 집계.

    반환:
    {
      "accuracy": float,
      "precision_micro": float,
      "recall_micro": float,
      "f1_micro": float,
      "precision_macro": float,
      "recall_macro": float,
      "f1_macro": float,
      "per_class": {
         "0": {"precision":..., "recall":..., "f1":..., "support":...},
         ...
      },
      "num_samples": int
    }
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # y_true: (N,) 강제
    if y_true.ndim != 1:
        raise ValueError(f"[metrics] y_true는 1차원이어야 합니다. 현재 shape: {y_true.shape}")

    N = y_true.shape[0]

    # y_pred:
    # - (N, C) → 마지막 축이 클래스 점수/로짓 → argmax
    # - (N,)   → 이미 클래스
    if y_pred.ndim == 2:
        if y_pred.shape[0] != N:
            raise ValueError(
                f"[metrics] y_pred 첫 번째 차원(N)이 y_true와 다릅니다: "
                f"y_true {y_true.shape}, y_pred {y_pred.shape}"
            )
        y_pred = y_pred.argmax(axis=1)
    elif y_pred.ndim == 1:
        if y_pred.shape[0] != N:
            raise ValueError(
                f"[metrics] y_pred 길이가 y_true와 다릅니다: "
                f"y_true {y_true.shape}, y_pred {y_pred.shape}"
            )
    else:
        raise ValueError(f"[metrics] y_pred는 1차원 또는 2차원이어야 합니다. 현재 shape: {y_pred.shape}")

    # 클래스 레이블 자동 추론
    if labels is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        labels = list(range(num_classes))

    # accuracy
    acc = float((y_true == y_pred).mean())

    # micro / macro
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # per-class
    pc, rc, fc, sc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class = {
        str(c): {
            "precision": float(pc[i]),
            "recall": float(rc[i]),
            "f1": float(fc[i]),
            "support": int(sc[i]),
        }
        for i, c in enumerate(labels)
    }

    return {
        "accuracy": acc,
        "precision_micro": float(p_micro),
        "recall_micro": float(r_micro),
        "f1_micro": float(f_micro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "per_class": per_class,
        "num_samples": int(N),
    }
