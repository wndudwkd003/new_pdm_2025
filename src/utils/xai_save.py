from __future__ import annotations

from pathlib import Path
from typing import Dict

import csv
import numpy as np
import matplotlib.pyplot as plt


def _save_heatmap(
    mat: np.ndarray,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
):
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _save_topk_features_csv(
    attn_cls_feat: np.ndarray,
    save_path: Path,
    topk: int,
):
    n, _ = attn_cls_feat.shape
    rows = []
    for i in range(n):
        scores = attn_cls_feat[i]
        idx = np.argsort(-scores)[:topk]
        rows.append([i] + idx.tolist() + scores[idx].tolist())

    header = (
        ["row"]
        + [f"feat_idx_{j}" for j in range(topk)]
        + [f"feat_score_{j}" for j in range(topk)]
    )
    with open(save_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)


def save_xai_artifacts(
    xai: Dict[str, np.ndarray],
    save_dir: Path,
    tag: str,
    top_rows: int = 64,
    topk_feat: int = 10,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    for k, v in xai.items():
        np.save(save_dir / f"{tag}_{k}.npy", v)

    if "attn_cls_feat" in xai:
        attn = xai["attn_cls_feat"]
        n = min(top_rows, attn.shape[0])
        _save_heatmap(
            attn[:n],
            save_dir / f"{tag}_attn_cls_feat_top{n}.png",
            title=f"{tag} | CLS->Feature Attention (top {n} rows)",
            xlabel="feature index",
            ylabel="sample index",
        )
        _save_topk_features_csv(
            attn[:n],
            save_dir / f"{tag}_attn_cls_feat_top{n}_top{topk_feat}.csv",
            topk=topk_feat,
        )

    if "gate_g" in xai:
        g = xai["gate_g"]
        n = min(top_rows, g.shape[0])
        _save_heatmap(
            g[:n],
            save_dir / f"{tag}_gate_g_top{n}.png",
            title=f"{tag} | FeatGate g (top {n} rows)",
            xlabel="latent dim",
            ylabel="sample index",
        )

    if "retr_w" in xai:
        w = xai["retr_w"]
        n = min(top_rows, w.shape[0])
        _save_heatmap(
            w[:n],
            save_dir / f"{tag}_retr_w_top{n}.png",
            title=f"{tag} | Retrieval weights (top {n} rows)",
            xlabel="neighbor rank",
            ylabel="sample index",
        )

    if "retr_sim" in xai:
        sim = xai["retr_sim"]
        n = min(top_rows, sim.shape[0])
        _save_heatmap(
            sim[:n],
            save_dir / f"{tag}_retr_sim_top{n}.png",
            title=f"{tag} | Retrieval cosine sim (top {n} rows)",
            xlabel="neighbor rank",
            ylabel="sample index",
        )
