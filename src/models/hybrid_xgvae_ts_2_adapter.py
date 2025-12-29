from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from src.core.models.ReGVAE import ReGVAE
from src.core.utils.losses import ReGVAEFinalStage1Loss

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split

from src.utils.embedding_vis import visualize_missing_mu_tsne


class HybridXGVAE_TS_2_Adapter(BaseModelAdapter):
    def __init__(self, config: Config):
        super().__init__(config)

        self.model: ReGVAE | None = None
        self.teacher: XGBClassifier | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.num_class: int | None = None

        m = self.config.model

        self.lambda_contrast = m.lambda_contrast
        self.lambda_mu = m.lambda_mu
        self.lambda_prior = m.lambda_prior
        self.lambda_recon = m.lambda_recon

        self.stage1_loss = ReGVAEFinalStage1Loss(
            tau=m.tau,
            lambda_cls=m.lambda_cls,
            w_contrast=self.lambda_contrast,
            w_mu=self.lambda_mu,
            w_prior=self.lambda_prior,
            w_recon=self.lambda_recon,
            mu_align=m.mu_align,
        )

        self.distill_temp = m.tau
        self.lambda_kd = m.lambda_kd

        self.retrieval_k = m.retrieval_k
        self.retrieval_tau = m.retrieval_tau
        self.retrieval_chunk = m.retrieval_chunk

        self._bank_mu: torch.Tensor | None = None
        self._bank_mu_norm: torch.Tensor | None = None
        self._bank_y: torch.Tensor | None = None
        self._bank_idx: torch.Tensor | None = None

        self.vis_max_points = m.vis_max_points
        self.vis_pca_dim = m.vis_pca_dim
        self.vis_perplexity = m.vis_perplexity
        self.vis_seed = m.vis_seed

    def fit(self, train_data: Datasets, valid_data: Datasets):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.input_dim = train_data.meta.input_dim
        self.num_class = train_data.meta.num_class

        self.model = self._get_model(self.input_dim, self.num_class)

        # -------------------------
        # Stage 1
        # -------------------------
        stage1_epochs = self.config.train.epochs
        opt1, sch1 = self._make_optimizer_scheduler(stage=1, num_epochs=stage1_epochs)

        best_valid = None
        best_state = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        train_total_1: list[float] = []
        valid_total_1: list[float] = []
        train_ce_1: list[float] = []
        valid_ce_1: list[float] = []
        train_view_1: list[float] = []
        valid_view_1: list[float] = []
        train_kl_1: list[float] = []
        valid_kl_1: list[float] = []
        train_recon_1: list[float] = []
        valid_recon_1: list[float] = []

        for epoch in range(stage1_epochs):
            tr = self.run_epoch(tr_loader, opt1, Split.TRAIN, stage=1)
            vl = self.run_epoch(vl_loader, None, Split.VALID, stage=1)

            lr = opt1.param_groups[0]["lr"]
            print(
                f"[{self.config.model.model.name} Stage1 Epoch {epoch + 1}] "
                f"Train: total={tr['total']:.4f}, ce={tr['ce']:.4f}, view={tr['view']:.4f}, kl={tr['kl']:.4f}, recon={tr['recon']:.4f} | "
                f"Valid: total={vl['total']:.4f}, ce={vl['ce']:.4f}, view={vl['view']:.4f}, kl={vl['kl']:.4f}, recon={vl['recon']:.4f} | "
                f"LR={lr:.6f}"
            )

            train_total_1.append(tr["total"])
            valid_total_1.append(vl["total"])
            train_ce_1.append(tr["ce"])
            valid_ce_1.append(vl["ce"])
            train_view_1.append(tr["view"])
            valid_view_1.append(vl["view"])
            train_kl_1.append(tr["kl"])
            valid_kl_1.append(vl["kl"])
            train_recon_1.append(tr["recon"])
            valid_recon_1.append(vl["recon"])

            sch1.step()

            if best_valid is None or vl["total"] < best_valid:
                best_valid = vl["total"]
                patience = 0
                best_state = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience += 1
                if patience >= max_patience:
                    print(
                        f"[{self.config.model.model.name}] Stage1 Early stopping at epoch {epoch + 1}"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            self.model.eval()

        # -------------------------
        # memory bank export + embedding vis
        # -------------------------
        root = (
            self.work_dir
            if self.work_dir is not None
            else Path(self.config.train.output_dir)
        )
        mem_dir = root / "memory_bank" / self.config.model.model.name
        mem_path = self.export_clean_memory_bank(
            train_data, save_dir=mem_dir, tag="train"
        )
        print("memory saved:", mem_path)

        vis_dir = root / "embedding_vis" / self.config.model.model.name
        vis_path = visualize_missing_mu_tsne(
            model=self.model,
            loader=vl_loader,
            device=self.device,
            save_dir=vis_dir,
            tag="valid_stage1",
            vis_max_points=self.vis_max_points,
            vis_pca_dim=self.vis_pca_dim,
            vis_perplexity=self.vis_perplexity,
            vis_seed=self.vis_seed,
        )

        # -------------------------
        # teacher training (XGB)
        # -------------------------
        X_tr, y_tr = self._collect_xy_from_loader(tr_loader, split=Split.TRAIN)
        X_vl, y_vl = self._collect_xy_from_loader(vl_loader, split=Split.VALID)

        eval_metric = self.config.model.eval_metric
        num_class = train_data.get_num_class()

        teacher = XGBClassifier(
            n_estimators=self.config.train.tree_est,
            objective=self.config.model.objective,
            num_class=num_class,
            random_state=self.config.train.seed,
            eval_metric=eval_metric,
            early_stopping_rounds=self.config.train.early_stopping_rounds,
            device=self.config.train.device,
            tree_method=self.config.train.tree_method,
        )
        teacher.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_vl, y_vl)])
        self.teacher = teacher

        # -------------------------
        # Stage 2 (retrieval + feat gate + KD + CE)
        # -------------------------
        self._load_memory_bank(mem_path, to_device=True)
        self._set_stage2_trainable()

        stage2_epochs = self.config.train.epochs
        opt2, sch2 = self._make_optimizer_scheduler(stage=2, num_epochs=stage2_epochs)

        best_valid2 = None
        best_state2 = None
        patience2 = 0
        max_patience2 = self.config.train.early_stopping_rounds

        train_total_2: list[float] = []
        valid_total_2: list[float] = []
        train_ce_2: list[float] = []
        valid_ce_2: list[float] = []
        train_kd_2: list[float] = []
        valid_kd_2: list[float] = []

        # stage2에서도 components 구조를 안정적으로 유지하기 위해 기록합니다(값은 0으로 들어옵니다).
        train_view_2: list[float] = []
        valid_view_2: list[float] = []
        train_kl_2: list[float] = []
        valid_kl_2: list[float] = []
        train_recon_2: list[float] = []
        valid_recon_2: list[float] = []

        for epoch in range(stage2_epochs):
            tr2 = self.run_epoch(tr_loader, opt2, Split.TRAIN, stage=2)
            vl2 = self.run_epoch(vl_loader, None, Split.VALID, stage=2)

            lr2 = opt2.param_groups[0]["lr"]
            print(
                f"[{self.config.model.model.name} Stage2 Epoch {epoch + 1}] "
                f"Train: total={tr2['total']:.4f}, ce={tr2['ce']:.4f}, kd={tr2['kd']:.4f} | "
                f"Valid: total={vl2['total']:.4f}, ce={vl2['ce']:.4f}, kd={vl2['kd']:.4f} | "
                f"LR={lr2:.6f}"
            )

            train_total_2.append(tr2["total"])
            valid_total_2.append(vl2["total"])
            train_ce_2.append(tr2["ce"])
            valid_ce_2.append(vl2["ce"])
            train_kd_2.append(tr2["kd"])
            valid_kd_2.append(vl2["kd"])

            train_view_2.append(tr2["view"])
            valid_view_2.append(vl2["view"])
            train_kl_2.append(tr2["kl"])
            valid_kl_2.append(vl2["kl"])
            train_recon_2.append(tr2["recon"])
            valid_recon_2.append(vl2["recon"])

            sch2.step()

            if best_valid2 is None or vl2["total"] < best_valid2:
                best_valid2 = vl2["total"]
                patience2 = 0
                best_state2 = {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                }
            else:
                patience2 += 1
                if patience2 >= max_patience2:
                    print(
                        f"[{self.config.model.model.name}] Stage2 Early stopping at epoch {epoch + 1}"
                    )
                    break

        if best_state2 is not None:
            self.model.load_state_dict(best_state2)
            self.model.to(self.device)
            self.model.eval()

        # -------------------------
        # final eval (stage2 student)
        # -------------------------
        tr_loss, tr_preds, tr_labels, _, _ = self.predict(
            tr_loader, split=Split.TRAIN, memory_path=mem_path
        )
        vl_loss, vl_preds, vl_labels, _, _ = self.predict(
            vl_loader, split=Split.VALID, memory_path=mem_path
        )

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        # -------------------------
        # loss history format (ReGVAEAdapter 호환)
        # -------------------------
        metric_name = "total_loss"
        tasks = [
            {Split.TRAIN.value: train_total_1, Split.VALID.value: valid_total_1},
            {Split.TRAIN.value: train_total_2, Split.VALID.value: valid_total_2},
        ]

        components = {
            "stage1": {
                "train": {
                    "ce": train_ce_1,
                    "view": train_view_1,
                    "kl": train_kl_1,
                    "recon": train_recon_1,
                },
                "valid": {
                    "ce": valid_ce_1,
                    "view": valid_view_1,
                    "kl": valid_kl_1,
                    "recon": valid_recon_1,
                },
            },
            "stage2": {
                "train": {
                    "ce": train_ce_2,
                    "kd": train_kd_2,
                    "view": train_view_2,
                    "kl": train_kl_2,
                    "recon": train_recon_2,
                },
                "valid": {
                    "ce": valid_ce_2,
                    "kd": valid_kd_2,
                    "view": valid_view_2,
                    "kl": valid_kl_2,
                    "recon": valid_recon_2,
                },
            },
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": metric_name,
                "tasks": tasks,
                "components": components,
                "student_eval_ce": {
                    Split.TRAIN.value: float(tr_loss),
                    Split.VALID.value: float(vl_loss),
                },
            },
            "artifacts": {"memory_bank": str(mem_path), "embedding_vis": str(vis_path)},
        }
        return results

    def test(self, test_data: Datasets):
        if self.model is None:
            raise ValueError("model is None")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        root = (
            self.work_dir
            if self.work_dir is not None
            else Path(self.config.train.output_dir)
        )
        mem_path = (
            root
            / "memory_bank"
            / self.config.model.model.name
            / "memory_clean_train.pt"
        )
        if not mem_path.exists():
            raise ValueError(f"train memory bank not found: {mem_path}")

        self._load_memory_bank(mem_path, to_device=True)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader, split=Split.TEST, memory_path=mem_path
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
                    m = compute_classification_metrics(y_sub, y_hat_sub)
                    metrics_by_ratio[p_val][ratio] = m

        return {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

    def run_epoch(self, loader: DataLoader, optimizer, split: Split, stage: int = 1):
        if self.model is None:
            raise ValueError("model is None")

        is_train = split == Split.TRAIN
        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(f"{self.config.model.model.name}_stage{stage}", split)

        total_sum = 0.0
        ce_sum = 0.0
        view_sum = 0.0
        kl_sum = 0.0
        recon_sum = 0.0
        kd_sum = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, x_ori, _, _, _, base_idx, _, V = self._prepare_batch(batch)

            if stage == 1:
                x_clean = x_ori[::V] if V > 1 else x_ori
                y_base = y[::V] if V > 1 else y

                with torch.set_grad_enabled(is_train):
                    out_clean = self.model(x_cont=x_clean, x_cat=None)
                    out_missing = self.model(x_cont=x, x_cat=None)

                    loss_dict = self.stage1_loss(
                        mu_clean=out_clean["z_mu"],
                        logvar_clean=out_clean["z_logvar"],
                        mu_missing=out_missing["z_mu"],
                        logvar_missing=out_missing["z_logvar"],
                        x_clean=x_clean,
                        recon_clean=out_clean["recon"],
                        recon_missing=out_missing["recon"],
                        y_base=y_base,
                        views_per_base=V,
                    )

                    loss_total = loss_dict["total"]
                    if is_train:
                        optimizer.zero_grad()
                        loss_total.backward()
                        optimizer.step()

                loss_ce = loss_dict["contrast"]
                loss_view = loss_dict["mu_align"]
                loss_kl = loss_dict["prior"]
                loss_recon = loss_dict["recon"]
                loss_kd = torch.zeros((), device=self.device)

            else:
                if self.teacher is None:
                    raise ValueError("teacher is None")
                if self.num_class is None:
                    raise ValueError("num_class is None")

                with torch.set_grad_enabled(is_train):
                    out = self.model(x_cont=x, x_cat=None)
                    mu_q = out["z_mu"]

                    mu_r = self._retrieve_agg_mu(
                        mu_q, exclude_idx=base_idx if split == Split.TRAIN else None
                    )
                    fused = self.model.feat_gate(mu_q, mu_r)
                    logits = self.model.cls_head(fused)

                    loss_ce = F.cross_entropy(logits, y)

                    x_np = x.detach().cpu().numpy().astype(np.float32)
                    t_margin = self.teacher.predict(x_np, output_margin=True)

                    if self.num_class == 2:
                        t_margin = t_margin.reshape(-1, 1)
                        t_logits = np.concatenate([-t_margin, t_margin], axis=1)
                    else:
                        t_logits = t_margin

                    t_logits_t = torch.from_numpy(t_logits).to(self.device).float()

                    T = float(self.distill_temp)
                    t_prob = F.softmax(t_logits_t / T, dim=1)
                    s_logprob = F.log_softmax(logits / T, dim=1)
                    loss_kd = F.kl_div(s_logprob, t_prob, reduction="batchmean") * (
                        T * T
                    )

                    loss_total = loss_ce + (self.lambda_kd * loss_kd)

                    if is_train:
                        optimizer.zero_grad()
                        loss_total.backward()
                        optimizer.step()

                loss_view = torch.zeros((), device=self.device)
                loss_kl = torch.zeros((), device=self.device)
                loss_recon = torch.zeros((), device=self.device)

            num_batches += 1
            total_sum += loss_total.item()
            ce_sum += loss_ce.item()
            view_sum += loss_view.item()
            kl_sum += loss_kl.item()
            recon_sum += loss_recon.item()
            kd_sum += loss_kd.item()

        denom = max(1, num_batches)
        return {
            "total": total_sum / denom,
            "ce": ce_sum / denom,
            "view": view_sum / denom,
            "kl": kl_sum / denom,
            "recon": recon_sum / denom,
            "kd": kd_sum / denom,
        }

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        split: Split = Split.TEST,
        memory_path: Path | None = None,
    ):
        if self.model is None:
            raise ValueError("model is None")

        self.model.eval()

        if memory_path is None:
            raise ValueError("predict requires memory_path")
        if self._bank_mu is None:
            self._load_memory_bank(memory_path, to_device=True)

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        desc = self.get_desc(self.config.model.model.name, split)

        for batch in tqdm(loader, desc=desc):
            x, y, _, _, pattern_idx, ratio_idx, base_idx, _, _ = self._prepare_batch(
                batch
            )

            out = self.model(x_cont=x, x_cat=None)
            mu_q = out["z_mu"]

            mu_r = self._retrieve_agg_mu(
                mu_q, exclude_idx=base_idx if split == Split.TRAIN else None
            )
            fused = self.model.feat_gate(mu_q, mu_r)
            logits = self.model.cls_head(fused)

            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(preds.detach().cpu())
            all_labels.append(y.detach().cpu())
            all_pattern_idx.append(pattern_idx.detach().cpu())
            all_ratio_idx.append(ratio_idx.detach().cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    def _collect_xy_from_loader(self, loader: DataLoader, split: Split):
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        desc = self.get_desc(f"{self.config.model.model.name}_teacher_xy", split)
        for batch in tqdm(loader, desc=desc):
            x = batch["x"]
            y = batch["y"]
            X_list.append(x.detach().cpu().numpy().astype(np.float32))
            y_list.append(y.detach().cpu().numpy().astype(np.int64))

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return X_all, y_all

    @torch.no_grad()
    def export_clean_memory_bank(
        self, dataset: Datasets, save_dir: Path, tag: str = "train"
    ):
        if self.model is None:
            raise ValueError("model is None")

        self.model.eval()
        save_dir.mkdir(parents=True, exist_ok=True)

        X_clean = dataset.imputed_dict["original"]["X"]
        y_clean = dataset.imputed_dict["original"]["y"]
        N = X_clean.shape[0]

        mu_list = []
        bs = 4096

        for s in tqdm(range(0, N, bs), desc=f"export_clean_mu[{tag}]"):
            e = min(N, s + bs)
            xb = torch.from_numpy(X_clean[s:e]).to(self.device).float()
            enc = self.model.encode_only(x_cont=xb, x_cat=None)
            mu_list.append(enc["z_mu"].half().cpu())

        mu_all = torch.cat(mu_list, dim=0)

        out = {
            "mu": mu_all,
            "y": torch.from_numpy(y_clean).long(),
            "idx": torch.arange(N, dtype=torch.long),
            "meta": {
                "input_dim": dataset.meta.input_dim,
                "num_class": dataset.meta.num_class,
                "ratios": dataset.ratios,
                "patterns": [p.value for p in dataset.config.data.missing_patterns],
            },
        }

        path = save_dir / f"memory_clean_{tag}.pt"
        torch.save(out, path)
        return path

    @torch.no_grad()
    def _load_memory_bank(self, path: Path, to_device: bool = True):
        d = torch.load(path, map_location="cpu")
        mu = d["mu"]
        y = d["y"]
        idx = d["idx"]

        if to_device:
            mu = mu.to(self.device)
            y = y.to(self.device)
            idx = idx.to(self.device)

        self._bank_mu = mu
        self._bank_y = y
        self._bank_idx = idx
        self._bank_mu_norm = F.normalize(mu.float(), dim=1)

    @torch.no_grad()
    def _retrieve_agg_mu(self, mu_q: torch.Tensor, exclude_idx: torch.Tensor | None):
        if self._bank_mu is None or self._bank_mu_norm is None:
            raise ValueError("memory bank not loaded")

        qn = F.normalize(mu_q.float(), dim=1)
        B = qn.shape[0]
        N = self._bank_mu_norm.shape[0]
        k = min(self.retrieval_k, N)

        best_sim = torch.full((B, k), -1e9, device=self.device)
        best_idx = torch.full((B, k), -1, device=self.device, dtype=torch.long)

        chunk = self.retrieval_chunk

        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            bn = self._bank_mu_norm[s:e]
            sim = qn @ bn.t()

            if exclude_idx is not None:
                ex = exclude_idx
                m = (ex >= s) & (ex < e)
                if m.any():
                    rows = torch.nonzero(m, as_tuple=False).squeeze(1)
                    cols = (ex[m] - s).long()
                    sim[rows, cols] = -1e9

            top_sim, top_local = torch.topk(sim, k, dim=1)
            top_global = top_local + s

            comb_sim = torch.cat([best_sim, top_sim], dim=1)
            comb_idx = torch.cat([best_idx, top_global], dim=1)

            new_sim, pos = torch.topk(comb_sim, k, dim=1)
            new_idx = torch.gather(comb_idx, 1, pos)

            best_sim = new_sim
            best_idx = new_idx

        mu_knn = self._bank_mu[best_idx].float()
        w = F.softmax(best_sim / self.retrieval_tau, dim=1).unsqueeze(-1)
        mu_r = (mu_knn * w).sum(dim=1)
        return mu_r

    def _set_stage2_trainable(self):
        if self.model is None:
            raise ValueError("model is None")

        for p in self.model.parameters():
            p.requires_grad_(False)

        for p in self.model.feat_gate.parameters():
            p.requires_grad_(True)

        for p in self.model.cls_head.parameters():
            p.requires_grad_(True)

    def _make_optimizer_scheduler(self, stage: int, num_epochs: int):
        if self.model is None:
            raise ValueError("model is None")

        if stage == 1:
            lr = self.config.train.lr_stage_1
            lr_min = self.config.train.lr_min_stage_1
        else:
            lr = self.config.train.lr_stage_2
            lr_min = self.config.train.lr_min_stage_2

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, num_epochs), eta_min=lr_min
        )
        return opt, sch

    def _prepare_batch(self, batch: dict):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        x_ori = batch["x_originals"].to(self.device)
        bemv = batch["bemv"].to(self.device)
        pattern_idx = batch["pattern_idx"].to(self.device)
        ratio_idx = batch["ratio_idx"].to(self.device)
        base_idx = batch["base_idx"].to(self.device)
        V = batch["views_per_base"].item()
        B0 = batch["base_batch_size"].item()
        return x, y, x_ori, bemv, pattern_idx, ratio_idx, base_idx, B0, V

    def _get_model(self, input_dim: int, num_class: int | None) -> ReGVAE:
        if num_class is None:
            raise ValueError("num_class is None")

        ft_kwargs = ReGVAE.get_default_kwargs()
        model = ReGVAE(
            n_cont_features=input_dim,
            cat_cardinalities=[],
            d_out=num_class,
            latent_dim=None,
            logits_from="mu",
            **ft_kwargs,
        ).to(self.device)
        return model

    def save(self, path: Path):
        if self.model is None:
            raise ValueError("model is None")
        if self.teacher is None:
            raise ValueError("teacher is None")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        teacher_path = save_dir / f"{self.config.model.model.name}_teacher_xgb.json"
        self.teacher.save_model(teacher_path)

        meta = {
            "input_dim": self.input_dim,
            "num_class": self.num_class,
            "model_path": str(model_path),
            "teacher_path": str(teacher_path),
        }
        self.save_meta(save_dir, meta)
        return {"model": model_path, "teacher": teacher_path}

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        self.input_dim = meta["input_dim"]
        self.num_class = meta["num_class"]

        model_path = Path(meta["model_path"])
        self.model = self._get_model(self.input_dim, self.num_class)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        teacher_path = meta["teacher_path"]
        teacher = XGBClassifier()
        teacher.load_model(teacher_path)
        self.teacher = teacher

        return True
