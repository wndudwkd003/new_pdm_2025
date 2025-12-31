# src/models/agata_adapter.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split
from src.core.utils.losses import info_nce_loss


# -----------------------------
# AGATa (continuous-only)
# - attention-guided feature selection (final-layer attention)
# - random augmentation per batch: masking / shuffling / CutMix
# - Stage1: InfoNCE pretrain
# - Stage2: CE fine-tuning
# -----------------------------


class _ContFeatEmbedding(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,F) -> (B,F,d)
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (B,F). got={tuple(x.shape)}")
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _TransformerBlockReturnAttn(nn.Module):
    def __init__(self, d_token: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_token)

        self.ff = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token),
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_attn: bool):
        y, w = self.attn(
            x,
            x,
            x,
            need_weights=return_attn,
            average_attn_weights=False,  # (B,H,seq,seq)
        )
        x = self.norm1(x + self.drop(y))
        f = self.ff(x)
        x = self.norm2(x + self.drop(f))
        if return_attn:
            return x, w
        return x, None


class AGATaModel(nn.Module):
    def __init__(
        self,
        n_cont_features: int,
        d_token: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        proj_dim: int,
        num_class: int,
    ):
        super().__init__()
        self.n_cont_features = int(n_cont_features)
        self.d_token = int(d_token)

        self.cont_embed = _ContFeatEmbedding(self.n_cont_features, self.d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _TransformerBlockReturnAttn(
                    self.d_token, int(n_heads), int(d_ff), float(dropout)
                )
                for _ in range(int(n_layers))
            ]
        )

        # projection head: 1 hidden layer MLP
        self.proj = nn.Sequential(
            nn.Linear(self.d_token, self.d_token),
            nn.ReLU(),
            nn.Linear(self.d_token, int(proj_dim)),
        )

        # downstream head: 1 hidden layer MLP
        self.cls_head = nn.Sequential(
            nn.Linear(self.d_token, self.d_token),
            nn.ReLU(),
            nn.Linear(self.d_token, int(num_class)),
        )

    def encode(self, x_cont: Tensor, return_final_attn: bool):
        # x_cont: (B,F) -> tokens: (B,1+F,d)
        if x_cont.dim() != 2:
            raise ValueError(f"x_cont must be 2D (B,F). got={tuple(x_cont.shape)}")

        b = x_cont.shape[0]
        feat = self.cont_embed(x_cont)  # (B,F,d)
        cls = self.cls_token.expand(b, 1, -1)  # (B,1,d)
        x = torch.cat([cls, feat], dim=1)  # (B,1+F,d)

        final_attn = None
        for li, blk in enumerate(self.blocks):
            need = (li == (len(self.blocks) - 1)) and return_final_attn
            x, w = blk(x, return_attn=need)
            if need:
                final_attn = w  # (B,H,seq,seq)

        return x, final_attn

    def proj_from_enc(self, enc: Tensor) -> Tensor:
        cls = enc[:, 0, :]
        return self.proj(cls)

    def classify_from_enc(self, enc: Tensor) -> Tensor:
        cls = enc[:, 0, :]
        return self.cls_head(cls)


class AGATaAdapter(BaseModelAdapter):
    def __init__(self, config: Config):
        super().__init__(config)

        self.model: AGATaModel | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.num_class: int | None = None

        m = self.config.model
        self.lambda_cls = float(m.lambda_cls)
        self.lambda_view = float(m.lambda_view)
        self.tau = float(m.tau)

        # low-attention feature selection ratio k (default 0.4)
        self.k = 0.4
        if hasattr(m, "agata_k"):
            self.k = float(m.agata_k)

        # stage1/2 split
        self.pretrain_ratio = 0.5
        if hasattr(m, "pretrain_ratio"):
            self.pretrain_ratio = float(m.pretrain_ratio)

        # feature-wise mean for masking
        self.feature_means: Tensor | None = None

    def _get_model(self, input_dim: int, num_class: int) -> AGATaModel:
        d_token = 192
        n_heads = 8
        n_layers = 4
        d_ff = 768
        dropout = 0.1
        proj_dim = 128

        if hasattr(self.config.model, "agata_d_token"):
            d_token = int(self.config.model.agata_d_token)
        if hasattr(self.config.model, "agata_n_heads"):
            n_heads = int(self.config.model.agata_n_heads)
        if hasattr(self.config.model, "agata_n_layers"):
            n_layers = int(self.config.model.agata_n_layers)
        if hasattr(self.config.model, "agata_d_ff"):
            d_ff = int(self.config.model.agata_d_ff)
        if hasattr(self.config.model, "agata_dropout"):
            dropout = float(self.config.model.agata_dropout)
        if hasattr(self.config.model, "agata_proj_dim"):
            proj_dim = int(self.config.model.agata_proj_dim)

        return AGATaModel(
            n_cont_features=int(input_dim),
            d_token=int(d_token),
            n_heads=int(n_heads),
            n_layers=int(n_layers),
            d_ff=int(d_ff),
            dropout=float(dropout),
            proj_dim=int(proj_dim),
            num_class=int(num_class),
        ).to(self.device)

    def _feature_importance_from_attn(self, attn: Tensor) -> Tensor:
        # attn: (B,H,seq,seq) where seq=1+F (includes CLS)
        if attn.dim() != 4:
            raise ValueError(f"attn must be 4D (B,H,seq,seq). got={tuple(attn.shape)}")

        # drop CLS -> (B,H,F,F)
        a = attn[:, :, 1:, 1:]
        # head mean -> (B,F,F)
        a_mean = a.mean(dim=1)
        # query mean -> (B,F): feature importance per "key feature"
        imp = a_mean.mean(dim=1)
        return imp

    def _select_low_attention_indices(self, imp: Tensor, k_ratio: float) -> Tensor:
        # imp: (B,F). 낮을수록 덜 중요한 feature.
        if imp.dim() != 2:
            raise ValueError(f"imp must be 2D (B,F). got={tuple(imp.shape)}")

        b, f = imp.shape
        k = int(round(float(k_ratio) * f))
        if k < 1:
            k = 1
        _, idx = torch.topk(imp, k=k, dim=1, largest=False)
        return idx  # (B,k)

    def _apply_agata_aug(self, x: Tensor, idx: Tensor) -> Tensor:
        # x: (B,F), idx: (B,k)
        if self.feature_means is None:
            raise ValueError("feature_means is None")
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (B,F). got={tuple(x.shape)}")
        if idx.dim() != 2:
            raise ValueError(f"idx must be 2D (B,k). got={tuple(idx.shape)}")

        b, _ = x.shape
        k = idx.shape[1]
        row = torch.arange(b, device=x.device).unsqueeze(1).expand(b, k)

        # per-batch random choice among {masking, shuffling, cutmix}
        mode = int(torch.randint(low=0, high=3, size=(1,), device=x.device).item())

        if mode == 0:
            # masking: replace selected features with feature mean
            x_aug = x.clone()
            x_aug[row, idx] = self.feature_means[idx]
            return x_aug

        if mode == 1:
            # shuffling: replace selected features with values from another sample in batch
            perm = torch.randperm(b, device=x.device)
            partner = x[perm]
            x_aug = x.clone()
            x_aug[row, idx] = partner[row, idx]
            return x_aug

        # cutmix: mix selected feature subset using Bernoulli mask
        perm = torch.randperm(b, device=x.device)
        partner = x[perm]

        m = torch.bernoulli(torch.full((b, k), 0.5, device=x.device))
        xa = x[row, idx]
        xb = partner[row, idx]
        mixed = m * xa + (1.0 - m) * xb

        x_aug = x.clone()
        x_aug[row, idx] = mixed
        return x_aug

    def fit(self, train_data: Datasets, valid_data: Datasets):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.input_dim = int(train_data.meta.input_dim)
        self.num_class = train_data.meta.num_class
        if self.num_class is None:
            raise ValueError("num_class is None")

        self.model = self._get_model(self.input_dim, int(self.num_class))

        # feature means (train original 기준)
        x0 = train_data.imputed_dict["original"]["X"]  # numpy (N,F)
        mu = x0.mean(axis=0).astype(np.float32)  # (F,)
        self.feature_means = torch.from_numpy(mu).to(self.device)

        total_epochs = int(self.config.train.epochs)
        stage1_epochs = int(round(total_epochs * float(self.pretrain_ratio)))
        if stage1_epochs < 1:
            stage1_epochs = 1
        stage2_epochs = total_epochs - stage1_epochs
        if stage2_epochs < 1:
            stage2_epochs = 1

        opt1 = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.train.lr_stage_1),
            weight_decay=0.01,
        )
        sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt1,
            T_max=stage1_epochs,
            eta_min=float(self.config.train.lr_min_stage_1),
        )

        opt2 = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.train.lr_stage_2),
            weight_decay=0.01,
        )
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt2,
            T_max=stage2_epochs,
            eta_min=float(self.config.train.lr_min_stage_2),
        )

        st1_tr_total: list[float] = []
        st1_vl_total: list[float] = []
        st1_tr_info: list[float] = []
        st1_vl_info: list[float] = []

        st2_tr_total: list[float] = []
        st2_vl_total: list[float] = []
        st2_tr_ce: list[float] = []
        st2_vl_ce: list[float] = []

        # -------------------
        # Stage 1: pretrain (InfoNCE)
        # -------------------
        for epoch in range(stage1_epochs):
            tr = self._run_epoch_stage1(tr_loader, opt1, Split.TRAIN)
            vl = self._run_epoch_stage1(vl_loader, None, Split.VALID)

            lr = float(opt1.param_groups[0]["lr"])
            print(
                f"[AGATa Stage1 Epoch {epoch + 1}/{stage1_epochs}] "
                f"Train: total={tr['total']:.4f}, info={tr['info']:.4f} | "
                f"Valid: total={vl['total']:.4f}, info={vl['info']:.4f} | "
                f"LR: {lr:.6f}"
            )

            st1_tr_total.append(tr["total"])
            st1_vl_total.append(vl["total"])
            st1_tr_info.append(tr["info"])
            st1_vl_info.append(vl["info"])

            sch1.step()

        # -------------------
        # Stage 2: finetune (CE)
        # -------------------
        best_valid = None
        best_state = None
        patience = 0
        max_patience = int(self.config.train.early_stopping_rounds)

        for epoch in range(stage2_epochs):
            tr = self._run_epoch_stage2(tr_loader, opt2, Split.TRAIN)
            vl = self._run_epoch_stage2(vl_loader, None, Split.VALID)

            lr = float(opt2.param_groups[0]["lr"])
            print(
                f"[AGATa Stage2 Epoch {epoch + 1}/{stage2_epochs}] "
                f"Train: total={tr['total']:.4f}, ce={tr['ce']:.4f} | "
                f"Valid: total={vl['total']:.4f}, ce={vl['ce']:.4f} | "
                f"LR: {lr:.6f}"
            )

            st2_tr_total.append(tr["total"])
            st2_vl_total.append(vl["total"])
            st2_tr_ce.append(tr["ce"])
            st2_vl_ce.append(vl["ce"])

            sch2.step()

            if best_valid is None or vl["total"] < best_valid:
                best_valid = vl["total"]
                patience = 0
                best_state = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience += 1

            if patience >= max_patience:
                print(f"[AGATa Stage2] Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # metrics (stage2 기준)
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader, Split.TRAIN)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader, Split.VALID)

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        return {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": "total_loss",
                "tasks": [
                    {Split.TRAIN.value: st1_tr_total, Split.VALID.value: st1_vl_total},
                    {Split.TRAIN.value: st2_tr_total, Split.VALID.value: st2_vl_total},
                ],
                "components": {
                    "stage1_pretrain": {
                        "train": {"info": st1_tr_info},
                        "valid": {"info": st1_vl_info},
                    },
                    "stage2_finetune": {
                        "train": {"ce": st2_tr_ce},
                        "valid": {"ce": st2_vl_ce},
                    },
                },
            },
        }

    def _run_epoch_stage1(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
    ):
        if self.model is None:
            raise ValueError("model is None")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        total_sum = 0.0
        info_sum = 0.0
        n = 0

        desc = self.get_desc("AGATa-Stage1", split)

        for batch in tqdm(loader, desc=desc):
            # Stage1 positive pair: original vs attention-guided augmented
            if "x_originals" in batch:
                x_orig = batch["x_originals"].to(self.device)
            else:
                x_orig = batch["x"].to(self.device)

            # 만약 loader가 (base sample * views) 형태면, stage1은 clean(base)만 사용
            if "views_per_base" in batch and "base_batch_size" in batch:
                v = int(batch["views_per_base"].item())
                b0 = int(batch["base_batch_size"].item())
                if v > 1 and b0 > 0:
                    x_orig = x_orig[:b0]

            with torch.set_grad_enabled(is_train):
                enc, attn = self.model.encode(x_orig, return_final_attn=True)
                if attn is None:
                    raise ValueError("final attn is None")

                z1 = self.model.proj_from_enc(enc)

                imp = self._feature_importance_from_attn(attn)
                idx = self._select_low_attention_indices(imp, self.k)

                x_aug = self._apply_agata_aug(x_orig, idx)

                enc2, _ = self.model.encode(x_aug, return_final_attn=False)
                z2 = self.model.proj_from_enc(enc2)

                loss_info = info_nce_loss(z1, z2, self.tau)
                loss_total = self.lambda_view * loss_info

                if is_train:
                    if optimizer is None:
                        raise ValueError("TRAIN split인데 optimizer가 None 입니다.")
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

            total_sum += float(loss_total.item())
            info_sum += float(loss_info.item())
            n += 1

        denom = max(1, n)
        return {"total": total_sum / denom, "info": info_sum / denom}

    def _run_epoch_stage2(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
    ):
        if self.model is None:
            raise ValueError("model is None")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        total_sum = 0.0
        ce_sum = 0.0
        n = 0

        desc = self.get_desc("AGATa-Stage2", split)

        for batch in tqdm(loader, desc=desc):
            x = batch["x"].to(self.device)  # (B,F)
            y = batch["y"].to(self.device)

            with torch.set_grad_enabled(is_train):
                enc, _ = self.model.encode(x, return_final_attn=False)
                logits = self.model.classify_from_enc(enc)

                loss_ce = F.cross_entropy(logits, y)
                loss_total = self.lambda_cls * loss_ce

                if is_train:
                    if optimizer is None:
                        raise ValueError("TRAIN split인데 optimizer가 None 입니다.")
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()

            total_sum += float(loss_total.item())
            ce_sum += float(loss_ce.item())
            n += 1

        denom = max(1, n)
        return {"total": total_sum / denom, "ce": ce_sum / denom}

    def predict(self, loader: DataLoader, split: Split = Split.TEST):
        if self.model is None:
            raise ValueError("model is None")

        self.model.eval()

        total_loss = 0.0
        n = 0

        all_preds: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_pattern_idx: list[Tensor] = []
        all_ratio_idx: list[Tensor] = []

        desc = self.get_desc("AGATa", split)

        for batch in tqdm(loader, desc=desc):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            pattern_idx = batch["pattern_idx"]
            ratio_idx = batch["ratio_idx"]

            with torch.no_grad():
                enc, _ = self.model.encode(x, return_final_attn=False)
                logits = self.model.classify_from_enc(enc)
                loss = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=1)

            total_loss += float(loss.item())
            n += 1

            all_preds.append(preds.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_pattern_idx.append(pattern_idx.detach().cpu())
            all_ratio_idx.append(ratio_idx.detach().cpu())

        avg_loss = total_loss / max(1, n)
        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    def test(self, test_data: Datasets):
        if self.model is None:
            raise ValueError("model is None")

        te_loader = test_data.get_loader_for_deep(shuffle=False)
        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader, Split.TEST
        )

        metrics_overall = compute_classification_metrics(labels_all, preds_all)

        patterns = test_data.config.data.missing_patterns
        ratios = test_data.ratios

        metrics_by_ratio: dict[str, dict[float, dict]] = {}

        for p_i, pattern in enumerate(patterns):
            p_val = pattern.value
            metrics_by_ratio[p_val] = {}
            for r_i, ratio in enumerate(ratios):
                mask = (pattern_idx_all == p_i) & (ratio_idx_all == r_i)
                if np.any(mask):
                    y_sub = labels_all[mask]
                    y_hat_sub = preds_all[mask]
                    metrics_by_ratio[p_val][ratio] = compute_classification_metrics(
                        y_sub, y_hat_sub
                    )

        return {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

    def save(self, path: Path):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "input_dim": int(self.input_dim),
            "num_class": int(self.num_class),
            "model_path": str(model_path),
        }
        self.save_meta(save_dir, meta)
        return model_path

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])
        model_path = Path(meta["model_path"])

        self.model = self._get_model(self.input_dim, self.num_class)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        return True
