# src/utils/embed_viz_sklearn.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("샘플 수 n이 0 이하입니다.")
    if max_points <= 0:
        raise ValueError("max_points는 0보다 커야 합니다.")

    if n <= max_points:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False).astype(np.int64)


def _ratio_sizes_from_idx(
    ratio_idx: np.ndarray,
    ratios: list[float],
    s_min: float,
    s_max: float,
    base_size: float = 30.0,
) -> np.ndarray:
    if ratio_idx.ndim != 1:
        raise ValueError("ratio_idx는 1차원이어야 합니다.")
    if len(ratios) == 0:
        raise ValueError("ratios가 비어 있습니다.")
    if s_min <= 0.0 or s_max <= 0.0:
        raise ValueError("s_min/s_max는 0보다 커야 합니다.")
    if s_max < s_min:
        raise ValueError("s_max는 s_min 이상이어야 합니다.")

    ratio_vals = np.array([ratios[int(i)] for i in ratio_idx], dtype=np.float32)
    r_min = float(np.min(ratio_vals))
    r_max = float(np.max(ratio_vals))

    if r_max == r_min:
        scales = np.full_like(ratio_vals, fill_value=s_max, dtype=np.float32)
    else:
        t = (ratio_vals - r_min) / (r_max - r_min)
        scales = s_min + t * (s_max - s_min)

    return (base_size * scales).astype(np.float32)


def _save_scatter_two_domain(
    coords_clean: np.ndarray,
    coords_missing: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    sizes: np.ndarray | None,
) -> None:
    if coords_clean.shape[1] != 2 or coords_missing.shape[1] != 2:
        raise ValueError("coords는 2차원(2D)이어야 합니다.")

    plt.figure(figsize=(8, 6))
    sc1 = plt.scatter(
        coords_clean[:, 0],
        coords_clean[:, 1],
        c=labels,
        s=sizes,
        alpha=0.65,
        marker="o",
        label="clean",
    )
    plt.scatter(
        coords_missing[:, 0],
        coords_missing[:, 1],
        c=labels,
        s=sizes,
        alpha=0.65,
        marker="x",
        label="missing",
    )
    plt.title(title)
    plt.legend()
    plt.colorbar(sc1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _save_scatter_one_domain(
    coords: np.ndarray,
    color: np.ndarray,
    title: str,
    save_path: Path,
    sizes: np.ndarray | None,
) -> None:
    if coords.shape[1] != 2:
        raise ValueError("coords는 2차원(2D)이어야 합니다.")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color,
        s=sizes,
        alpha=0.75,
    )
    plt.title(title)
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def visualize_embeddings_pca_tsne(
    *,
    model,
    loader,
    device,
    ratios: list[float],
    out_dir: Path,
    split_name: str,
    seed: int,
    max_points_pca: int,
    max_points_tsne: int,
    perplexity: float,
    s_min: float,
    s_max: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    z_missing_list: list[torch.Tensor] = []
    z_clean_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    ratio_idx_list: list[torch.Tensor] = []
    pattern_idx_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            x_missing = batch["x"].to(device)
            y = batch["y"].to(device)
            x_clean = batch["x_originals"].to(device)
            ratio_idx = batch["ratio_idx"]
            pattern_idx = batch["pattern_idx"]

            out_m = model(x_cont=x_missing, x_cat=None)
            out_c = model(x_cont=x_clean, x_cat=None)

            z_m = out_m.get("embedding")
            z_c = out_c.get("embedding")

            if z_m is None or z_c is None:
                raise ValueError("model output에 'embedding'이 없습니다.")

            z_missing_list.append(z_m.detach().cpu())
            z_clean_list.append(z_c.detach().cpu())
            y_list.append(y.detach().cpu())

            ratio_idx_list.append(ratio_idx.detach().cpu())
            pattern_idx_list.append(pattern_idx.detach().cpu())

    z_missing = torch.cat(z_missing_list, dim=0).numpy().astype(np.float32)
    z_clean = torch.cat(z_clean_list, dim=0).numpy().astype(np.float32)
    labels = torch.cat(y_list, dim=0).numpy().astype(np.int64)

    ratio_idx_all = torch.cat(ratio_idx_list, dim=0).numpy().astype(np.int64)
    pattern_idx_all = torch.cat(pattern_idx_list, dim=0).numpy().astype(np.int64)

    n = z_missing.shape[0]
    if n <= 1:
        raise ValueError("시각화할 샘플이 너무 적습니다.")

    # -------------------------
    # PCA (clean/missing 함께)
    # -------------------------
    idx_pca = _subsample_indices(n, max_points_pca, seed=seed)
    zc_p = z_clean[idx_pca]
    zm_p = z_missing[idx_pca]
    y_p = labels[idx_pca]
    ratio_idx_p = ratio_idx_all[idx_pca]
    pattern_idx_p = pattern_idx_all[idx_pca]

    sizes_p = _ratio_sizes_from_idx(ratio_idx_p, ratios, s_min=s_min, s_max=s_max)

    X_pca_fit = np.concatenate([zc_p, zm_p], axis=0)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X_pca_fit)

    zc_2 = X2[: len(idx_pca)]
    zm_2 = X2[len(idx_pca) :]

    _save_scatter_two_domain(
        coords_clean=zc_2,
        coords_missing=zm_2,
        labels=y_p,
        title=f"[{split_name}] PCA (color=class) clean vs missing",
        save_path=out_dir / f"{split_name}_pca_clean_vs_missing_by_class.png",
        sizes=sizes_p,
    )

    ratio_vals_p = np.array([ratios[int(i)] for i in ratio_idx_p], dtype=np.float32)
    _save_scatter_two_domain(
        coords_clean=zc_2,
        coords_missing=zm_2,
        labels=ratio_vals_p,
        title=f"[{split_name}] PCA (color=ratio) clean vs missing",
        save_path=out_dir / f"{split_name}_pca_clean_vs_missing_by_ratio.png",
        sizes=sizes_p,
    )

    np.save(out_dir / f"{split_name}_pca_idx.npy", idx_pca)
    np.save(out_dir / f"{split_name}_pca_clean_2d.npy", zc_2)
    np.save(out_dir / f"{split_name}_pca_missing_2d.npy", zm_2)
    np.save(out_dir / f"{split_name}_pca_labels.npy", y_p)
    np.save(out_dir / f"{split_name}_pca_ratio_idx.npy", ratio_idx_p)
    np.save(out_dir / f"{split_name}_pca_pattern_idx.npy", pattern_idx_p)

    # -------------------------
    # t-SNE (clean/missing 함께)
    # -------------------------
    idx_tsne = _subsample_indices(n, max_points_tsne, seed=seed + 1)
    zc_t = z_clean[idx_tsne]
    zm_t = z_missing[idx_tsne]
    y_t = labels[idx_tsne]
    ratio_idx_t = ratio_idx_all[idx_tsne]
    pattern_idx_t = pattern_idx_all[idx_tsne]

    sizes_t = _ratio_sizes_from_idx(ratio_idx_t, ratios, s_min=s_min, s_max=s_max)

    X_tsne_fit = np.concatenate([zc_t, zm_t], axis=0)
    n_tsne = X_tsne_fit.shape[0]
    if n_tsne <= 2:
        raise ValueError("t-SNE를 수행하기에 샘플이 너무 적습니다.")

    p = float(perplexity)
    if p >= n_tsne:
        p = float(n_tsne - 1)

    tsne = TSNE(
        n_components=2,
        perplexity=p,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    T2 = tsne.fit_transform(X_tsne_fit)

    zc_t2 = T2[: len(idx_tsne)]
    zm_t2 = T2[len(idx_tsne) :]

    _save_scatter_two_domain(
        coords_clean=zc_t2,
        coords_missing=zm_t2,
        labels=y_t,
        title=f"[{split_name}] t-SNE (color=class) clean vs missing",
        save_path=out_dir / f"{split_name}_tsne_clean_vs_missing_by_class.png",
        sizes=sizes_t,
    )

    ratio_vals_t = np.array([ratios[int(i)] for i in ratio_idx_t], dtype=np.float32)
    _save_scatter_two_domain(
        coords_clean=zc_t2,
        coords_missing=zm_t2,
        labels=ratio_vals_t,
        title=f"[{split_name}] t-SNE (color=ratio) clean vs missing",
        save_path=out_dir / f"{split_name}_tsne_clean_vs_missing_by_ratio.png",
        sizes=sizes_t,
    )

    # (옵션) missing만 따로 저장하고 싶으면 아래 2개도 같이 저장해 둡니다.
    _save_scatter_one_domain(
        coords=zm_t2,
        color=y_t,
        title=f"[{split_name}] t-SNE (missing only, color=class)",
        save_path=out_dir / f"{split_name}_tsne_missing_only_by_class.png",
        sizes=sizes_t,
    )
    _save_scatter_one_domain(
        coords=zm_t2,
        color=ratio_vals_t,
        title=f"[{split_name}] t-SNE (missing only, color=ratio)",
        save_path=out_dir / f"{split_name}_tsne_missing_only_by_ratio.png",
        sizes=sizes_t,
    )

    np.save(out_dir / f"{split_name}_tsne_idx.npy", idx_tsne)
    np.save(out_dir / f"{split_name}_tsne_clean_2d.npy", zc_t2)
    np.save(out_dir / f"{split_name}_tsne_missing_2d.npy", zm_t2)
    np.save(out_dir / f"{split_name}_tsne_labels.npy", y_t)
    np.save(out_dir / f"{split_name}_tsne_ratio_idx.npy", ratio_idx_t)
    np.save(out_dir / f"{split_name}_tsne_pattern_idx.npy", pattern_idx_t)
