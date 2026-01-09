# src/utils/hybrid_xgvae_utils.py
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


# ============================================================
# 1) Label / Objective inference utilities
# ============================================================


def infer_num_class_from_y(y_tr: np.ndarray, y_val: np.ndarray) -> int:
    """
    y가 0..C-1 연속 라벨이라는 가정 하에 class 개수를 추론합니다.
    """
    y_cat = np.concatenate([y_tr, y_val], axis=0)
    uniq = np.unique(y_cat)
    if uniq.ndim != 1 or uniq.size < 2:
        raise ValueError(f"num_class must be >=2, got uniq={uniq}")
    mn = int(uniq.min())
    mx = int(uniq.max())
    if mn != 0:
        raise ValueError(f"class labels must start at 0, got min={mn}, uniq={uniq}")
    if mx != int(uniq.size - 1):
        raise ValueError(
            f"class labels must be contiguous 0..C-1, got max={mx}, size={uniq.size}, uniq={uniq}"
        )
    return int(uniq.size)


def resolve_xgb_params_auto(
    is_reg: bool, y_tr: np.ndarray, y_val: np.ndarray
) -> Tuple[str, str, Optional[int]]:
    if is_reg:
        return "reg:squarederror", "rmse", None
    nc = infer_num_class_from_y(y_tr, y_val)
    if nc == 2:
        return "binary:logistic", "logloss", None
    return "multi:softprob", "mlogloss", nc


def resolve_xgb_from_config_or_auto(
    config_model: Any, is_reg: bool, y_tr: np.ndarray, y_val: np.ndarray
) -> Tuple[str, str, Optional[int]]:
    """
    config_model.objective / config_model.eval_metric 이 있으면 그것을 우선,
    "auto"면 y로부터 자동 결정합니다.
    """
    auto_obj, auto_metric, auto_num_class = resolve_xgb_params_auto(is_reg, y_tr, y_val)

    objective_cfg = str(getattr(config_model, "objective", "auto") or "auto")
    eval_metric_cfg = str(getattr(config_model, "eval_metric", "auto") or "auto")

    objective = auto_obj if objective_cfg == "auto" else objective_cfg
    eval_metric = auto_metric if eval_metric_cfg == "auto" else eval_metric_cfg

    if objective in ("multi:softprob", "multi:softmax"):
        if auto_num_class is None or auto_num_class <= 2:
            raise ValueError(
                f"objective={objective} requires num_class>=3, got {auto_num_class}"
            )
        return objective, eval_metric, auto_num_class

    if objective == "binary:logistic":
        if auto_num_class is not None and auto_num_class > 2:
            raise ValueError(
                f"objective=binary:logistic but detected num_class={auto_num_class}"
            )
        return objective, eval_metric, None

    if objective.startswith("reg:"):
        return objective, eval_metric, None

    raise ValueError(f"Unsupported XGB objective: {objective}")


def resolve_lgbm_params_auto(
    is_reg: bool, y_tr: np.ndarray, y_val: np.ndarray
) -> Tuple[str, str, Optional[int]]:
    if is_reg:
        return "regression", "rmse", None
    nc = infer_num_class_from_y(y_tr, y_val)
    if nc == 2:
        return "binary", "binary_logloss", None
    return "multiclass", "multi_logloss", nc


def resolve_lgbm_from_config_or_auto(
    config_model: Any, is_reg: bool, y_tr: np.ndarray, y_val: np.ndarray
) -> Tuple[str, str, Optional[int]]:
    """
    config_model.lgbm_objective / config_model.lgbm_metric 이 있으면 우선,
    "auto"면 y로부터 자동 결정합니다.
    """
    auto_obj, auto_metric, auto_num_class = resolve_lgbm_params_auto(
        is_reg, y_tr, y_val
    )

    obj_cfg = str(getattr(config_model, "lgbm_objective", "auto")).lower()
    metric_cfg = str(getattr(config_model, "lgbm_metric", "auto")).lower()

    objective = (
        auto_obj if obj_cfg == "auto" else getattr(config_model, "lgbm_objective")
    )
    eval_metric = (
        auto_metric if metric_cfg == "auto" else getattr(config_model, "lgbm_metric")
    )

    objective_str = str(objective).lower()

    if is_reg:
        if not (
            objective_str.startswith("reg")
            or objective_str in ("huber", "fair", "poisson", "gamma", "tweedie")
        ):
            raise ValueError(
                f"Regression task requires regression-like objective, got objective={objective}"
            )
        return str(objective), str(eval_metric), None

    if objective_str in ("binary",):
        if auto_num_class is not None and auto_num_class > 2:
            raise ValueError(
                f"objective=binary but detected num_class={auto_num_class}"
            )
        return str(objective), str(eval_metric), None

    if objective_str in ("multiclass", "multiclassova"):
        if auto_num_class is None or auto_num_class <= 2:
            raise ValueError(
                f"objective={objective} requires num_class>=3, got {auto_num_class}"
            )
        return str(objective), str(eval_metric), auto_num_class

    raise ValueError(f"Unsupported LGBM objective: {objective}")


# ============================================================
# 2) Memory bank utilities
# ============================================================


@dataclass
class MemoryBank:
    """
    - mu:        (N, D)  (device or cpu)
    - mu_norm:   (N, D)  normalized float32 on device (retrieval에 사용)
    - y:         (N,)    (device or cpu)
    - idx:       (N,)    (device or cpu)
    - x_raw_cpu: (N, F)  반드시 CPU 텐서로 유지 (메모리 큼)
    """

    mu: torch.Tensor
    mu_norm: torch.Tensor
    y: torch.Tensor
    idx: torch.Tensor
    x_raw_cpu: torch.Tensor

    @staticmethod
    def load(path: Path, device: str, to_device: bool = True) -> "MemoryBank":
        d = torch.load(path, map_location="cpu")

        mu = d["mu"]
        y = d["y"]
        idx = d["idx"]
        x_raw = d.get("x_raw", None)

        if x_raw is None:
            raise ValueError(
                f"Loaded memory bank has no 'x_raw': {path}. "
                "Re-export memory bank with export_clean_memory_bank(...)."
            )

        if to_device:
            mu = mu.to(device)
            y = y.to(device)
            idx = idx.to(device)

        mu_norm = F.normalize(mu.float(), dim=1)  # retrieval은 float32 정규화
        x_raw_cpu = x_raw.cpu()  # 반드시 CPU로 고정

        return MemoryBank(mu=mu, mu_norm=mu_norm, y=y, idx=idx, x_raw_cpu=x_raw_cpu)

    def save(self, path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
        out = {
            "mu": self.mu.detach().cpu(),
            "x_raw": self.x_raw_cpu.detach().cpu(),
            "y": self.y.detach().cpu(),
            "idx": self.idx.detach().cpu(),
            "meta": meta or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)


@torch.no_grad()
def export_clean_memory_bank(
    *,
    model: Any,
    dataset: Any,
    device: str,
    is_regression: bool,
    output_dim: int,
    save_dir: Path,
    tag: str = "train",
    bs: int = 4096,
) -> Path:
    """
    dataset.imputed_dict["original"]["X"/"y"] 기반 clean memory bank 생성.
    - mu, x_raw(=X_clean), y, idx 저장
    """
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    X_clean = dataset.imputed_dict["original"]["X"]
    y_clean = dataset.imputed_dict["original"]["y"]

    X_clean_np = np.asarray(X_clean, dtype=np.float32)
    x_raw_t = torch.from_numpy(X_clean_np).half().cpu()  # x_raw는 보통 크므로 fp16 CPU

    N = int(X_clean_np.shape[0])

    mu_list = []
    for s in tqdm(range(0, N, bs), desc=f"export_clean_mu[{tag}]"):
        e = min(N, s + bs)
        xb = torch.from_numpy(X_clean_np[s:e]).to(device).float()
        enc = model.encode_only(x_cont=xb, x_cat=None, return_attn=False)
        mu_list.append(enc["z_mu"].half().cpu())

    mu_all = torch.cat(mu_list, dim=0)

    if is_regression:
        y_t = torch.from_numpy(np.asarray(y_clean, dtype=np.float32)).float()
    else:
        y_t = torch.from_numpy(np.asarray(y_clean, dtype=np.int64)).long()

    out = {
        "mu": mu_all,
        "x_raw": x_raw_t,
        "y": y_t,
        "idx": torch.arange(N, dtype=torch.long),
        "meta": {
            "task": "regression" if is_regression else "classification",
            "input_dim": int(dataset.meta.input_dim),
            "output_dim": int(output_dim),
            "ratios": getattr(dataset, "ratios", None),
            "patterns": [p.value for p in dataset.config.data.missing_patterns],
        },
    }

    path = save_dir / f"memory_clean_{tag}.pt"
    torch.save(out, path)

    if tag == "train":
        alias = save_dir / "memory_clean_train.pt"
        try:
            shutil.copyfile(path, alias)
        except Exception:
            pass

    return path


# ============================================================
# 3) Retrieval + retrieved raw aggregation utilities
# ============================================================


@torch.no_grad()
def retrieve_agg_mu_xai(
    *,
    bank: MemoryBank,
    mu_q: torch.Tensor,
    device: str,
    retrieval_k: int,
    retrieval_tau: float,
    retrieval_chunk: int,
    exclude_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    bank.mu_norm 에 대해 chunked top-k retrieval 수행.
    return:
      mu_r     (B, D)
      best_idx (B, k)
      best_sim (B, k)
      w        (B, k) softmax(sim/tau)
    """
    qn = F.normalize(mu_q.float(), dim=1)
    B = int(qn.shape[0])
    N = int(bank.mu_norm.shape[0])
    k = int(min(retrieval_k, N))

    best_sim = torch.full((B, k), -1e9, device=device)
    best_idx = torch.full((B, k), -1, device=device, dtype=torch.long)

    chunk = int(retrieval_chunk)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        bn = bank.mu_norm[s:e]  # (chunk, D) on device
        sim = qn @ bn.t()  # (B, chunk)

        if exclude_idx is not None:
            ex = exclude_idx
            m = (ex >= s) & (ex < e)
            if m.any():
                rows = torch.nonzero(m, as_tuple=False).squeeze(1)
                cols = (ex[m] - s).long()
                sim[rows, cols] = -1e9

        top_sim, top_local = torch.topk(sim, k, dim=1)  # (B,k)
        top_global = top_local + s

        comb_sim = torch.cat([best_sim, top_sim], dim=1)
        comb_idx = torch.cat([best_idx, top_global], dim=1)

        new_sim, pos = torch.topk(comb_sim, k, dim=1)
        new_idx = torch.gather(comb_idx, 1, pos)

        best_sim = new_sim
        best_idx = new_idx

    mu_knn = bank.mu[best_idx].float()  # (B, k, D)
    w = F.softmax(best_sim / float(retrieval_tau), dim=1)
    mu_r = (mu_knn * w.unsqueeze(-1)).sum(dim=1)
    return mu_r, best_idx, best_sim, w


@torch.no_grad()
def aggregate_retrieved_x_raw(
    *,
    bank: MemoryBank,
    best_idx: torch.Tensor,
    w: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    bank.x_raw_cpu에서 (B,k,F) 가져와 가중합 -> (B,F) 를 device로 반환
    """
    idx_cpu = best_idx.detach().cpu()  # (B, k)
    w_cpu = w.detach().cpu().float()  # (B, k)

    x_knn = bank.x_raw_cpu[idx_cpu].float()  # (B, k, F) on CPU
    x_r = (x_knn * w_cpu.unsqueeze(-1)).sum(dim=1)  # (B, F) on CPU

    return x_r.to(device)
