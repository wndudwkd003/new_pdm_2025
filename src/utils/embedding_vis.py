# src/utils/embedding_vis.py

from __future__ import annotations

from pathlib import Path
import colorsys

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@torch.no_grad()
def visualize_missing_mu_tsne(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    save_dir: Path,
    tag: str,
    vis_max_points: int,
    vis_pca_dim: int,
    vis_perplexity: float,
    vis_seed: int,
) -> Path:
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    ratios = list(getattr(loader.dataset, "ratios", []))
    if len(ratios) == 0:
        raise ValueError("dataset.ratios is empty")

    mu_all = []
    y_all = []
    ridx_all = []

    for batch in tqdm(loader, desc=f"collect_mu[{tag}]"):
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        ratio_idx = batch["ratio_idx"].to(device)

        out = model(x_cont=x, x_cat=None)
        mu_all.append(out["z_mu"].detach().cpu().float())
        y_all.append(y.detach().cpu().long())
        ridx_all.append(ratio_idx.detach().cpu().long())

    Z = torch.cat(mu_all, dim=0).numpy()
    Y = torch.cat(y_all, dim=0).numpy()
    R = torch.cat(ridx_all, dim=0).numpy()

    n = Z.shape[0]
    if n > vis_max_points:
        rng = np.random.RandomState(vis_seed)
        idx = rng.choice(n, size=vis_max_points, replace=False)
        Z = Z[idx]
        Y = Y[idx]
        R = R[idx]

    if Z.shape[1] > vis_pca_dim:
        Zp = PCA(n_components=vis_pca_dim, random_state=vis_seed).fit_transform(Z)
    else:
        Zp = Z

    perp = min(vis_perplexity, max(5.0, (len(Zp) - 1) / 3.0))
    X2 = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=vis_seed,
    ).fit_transform(Zp)

    base_hues = {
        0: 0.00,  # red
        1: 0.60,  # blue
        2: 0.33,  # green
        3: 0.80,  # purple
    }
    uniq_labels = np.unique(Y)
    extra = [int(v) for v in uniq_labels if int(v) not in base_hues]
    for i, lab in enumerate(extra):
        base_hues[lab] = ((i / max(1, len(extra))) + 0.10) % 1.0

    r_vals = np.array([float(v) for v in ratios], dtype=np.float32)
    r_min = float(r_vals.min())
    r_max = float(r_vals.max())

    sat_by = {}
    if abs(r_max - r_min) < 1e-8:
        for i in np.unique(R):
            sat_by[int(i)] = 0.9
    else:
        for i in np.unique(R):
            rv = float(r_vals[int(i)])
            t = (rv - r_min) / (r_max - r_min)
            sat_by[int(i)] = float(0.15 + 0.85 * t)

    colors = []
    for yv, rv in zip(Y, R):
        hue = base_hues[int(yv)]
        sat = sat_by[int(rv)]
        rgb = colorsys.hsv_to_rgb(hue, sat, 0.95)
        colors.append((*rgb, 0.85))
    colors = np.asarray(colors, dtype=np.float32)

    plt.figure(figsize=(10, 8))
    plt.scatter(X2[:, 0], X2[:, 1], c=colors, s=8, linewidths=0)
    plt.title(f"ReGVAE z_mu TSNE (missing views) - {tag}")
    plt.xticks([])
    plt.yticks([])

    out_path = save_dir / f"tsne_missing_mu_{tag}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path
