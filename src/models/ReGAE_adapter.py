# src/models/ReGAE_adapter.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets
from src.utils.metrics import compute_classification_metrics
from src.params.data_model import Split
from src.core.models.ReGAE import RetrievalGatedAutoEncoder as ReGAE
from src.core.utils.losses import info_nce_loss



class ReGAEAdapter(BaseModelAdapter):

    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: ReGAE | None = None
        self.device = self.config.train.device

        self.input_dim: int | None = None
        self.num_class: int | None = None

        # 손실 가중치
        model_cfg = self.config.model
        self.lambda_class = model_cfg.lambda_class
        self.lambda_recon = model_cfg.lambda_recon
        self.lambda_view = model_cfg.lambda_view
        self.lambda_rscore = model_cfg.lambda_rscore

        # Missing dictionary (train split 기반)
        self.memory_embeddings: torch.Tensor | None = None  # (N, D_embed)
        self.memory_labels: torch.Tensor | None = None      # (N,)
        self.memory_r: torch.Tensor | None = None           # (N,)

    # ------------------------------------------------------------------
    # 학습 루프: Stage1 → Dictionary 구축 → Stage2
    # ------------------------------------------------------------------
    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.input_dim = train_data.meta.input_dim
        self.num_class = train_data.meta.num_class

        self.model = self._get_model(self.input_dim, self.num_class)

        num_epochs_stage1 = self.config.train.epochs
        num_epochs_stage2 = self.config.train.epochs

        print(f"[{self.config.model.model.name}] Stage 1: Pretrain 시작")

        optimizer1, scheduler1 = self.get_deeplearning_utils()

        best_valid_loss_stage1 = None
        best_state_stage1 = None
        patience = 0
        max_patience = self.config.train.early_stopping_rounds

        lrs: list[float] = []
        train_losses: list[float] = []
        valid_losses: list[float] = []

        # --- 추가: stage1 / stage2, train / valid별 loss component 기록용 ---
        stage1_train_components = {"info": [], "recon": [], "r": []}
        stage1_valid_components = {"info": [], "recon": [], "r": []}
        stage2_train_components = {"ce": [], "recon": [], "r": []}
        stage2_valid_components = {"ce": [], "recon": [], "r": []}

        # ---------------- Stage 1 ----------------
        for epoch in range(num_epochs_stage1):
            train_stats = self.run_epoch(
                loader=tr_loader,
                optimizer=optimizer1,
                split=Split.TRAIN,
                stage=1,
            )
            valid_stats = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID,
                stage=1,
            )

            train_loss = train_stats["total"]
            valid_loss = valid_stats["total"]
            lr = float(optimizer1.param_groups[0]["lr"])

            print(
                f"[{self.config.model.model.name} Stage1 Epoch {epoch + 1}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"LR: {lr:.6f}"
            )

            lrs.append(lr)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # stage1 component 저장
            stage1_train_components["info"].append(train_stats["info"])
            stage1_train_components["recon"].append(train_stats["recon"])
            stage1_train_components["r"].append(train_stats["r"])

            stage1_valid_components["info"].append(valid_stats["info"])
            stage1_valid_components["recon"].append(valid_stats["recon"])
            stage1_valid_components["r"].append(valid_stats["r"])

            scheduler1.step()

            if best_valid_loss_stage1 is None or valid_loss < best_valid_loss_stage1:
                best_valid_loss_stage1 = valid_loss
                patience = 0
                if self.model is not None:
                    best_state_stage1 = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if patience >= max_patience:
                print(f"[{self.config.model.model.name} Stage1 Early stopping at epoch {epoch + 1}")
                break

        if best_state_stage1 is not None and self.model is not None:
            self.model.load_state_dict(best_state_stage1)
            self.model.to(self.device)

        print(f"[{self.config.model.model.name}] Missing Dictionary 구축 시작")
        self.build_missing_dictionary(train_data)

        # ---------------- Stage 2 ----------------
        print(f"[{self.config.model.model.name}] Stage 2: Finetune 시작")

        optimizer2, scheduler2 = self.get_deeplearning_utils()

        best_valid_loss_stage2 = None
        best_state_stage2 = None
        patience = 0

        for epoch in range(num_epochs_stage2):
            train_stats = self.run_epoch(
                loader=tr_loader,
                optimizer=optimizer2,
                split=Split.TRAIN,
                stage=2,
            )
            valid_stats = self.run_epoch(
                loader=vl_loader,
                optimizer=None,
                split=Split.VALID,
                stage=2,
            )

            train_loss = train_stats["total"]
            valid_loss = valid_stats["total"]
            lr = float(optimizer2.param_groups[0]["lr"])

            print(
                f"[{self.config.model.model.name} Stage2 Epoch {epoch + 1}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"LR: {lr:.6f}"
            )

            lrs.append(lr)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # stage2 component 저장
            stage2_train_components["ce"].append(train_stats["ce"])
            stage2_train_components["recon"].append(train_stats["recon"])
            stage2_train_components["r"].append(train_stats["r"])

            stage2_valid_components["ce"].append(valid_stats["ce"])
            stage2_valid_components["recon"].append(valid_stats["recon"])
            stage2_valid_components["r"].append(valid_stats["r"])

            scheduler2.step()

            if best_valid_loss_stage2 is None or valid_loss < best_valid_loss_stage2:
                best_valid_loss_stage2 = valid_loss
                patience = 0
                if self.model is not None:
                    best_state_stage2 = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience += 1

            if patience >= max_patience:
                print(f"[{self.config.model.model.name} Stage2 Early stopping at epoch {epoch + 1}")
                break

        if best_state_stage2 is not None and self.model is not None:
            self.model.load_state_dict(best_state_stage2)
            self.model.to(self.device)

        # 이후 predict / metrics 부분은 그대로
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader, split=Split.TRAIN)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader, split=Split.VALID)

        train_metrics = compute_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_classification_metrics(vl_labels, vl_preds)

        metric_name = "total_loss"

        tasks = [
            {
                Split.TRAIN.value: train_losses,
                Split.VALID.value: valid_losses,
            }
        ]

        # --- 여기서 stage별 loss component까지 results에 넣어서 저장 ---
        loss_components = {
            "stage1": {
                "train": stage1_train_components,
                "valid": stage1_valid_components,
            },
            "stage2": {
                "train": stage2_train_components,
                "valid": stage2_valid_components,
            },
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": metric_name,
                "tasks": tasks,
                "components": loss_components,
            },
        }

        return results

    # ------------------------------------------------------------------
    # 한 epoch 실행
    #   stage=1: Pretrain (Recon + InfoNCE + R-score)
    #   stage=2: Finetune (Retrieval + Gate + CE + Recon + R-score)
    # ------------------------------------------------------------------
    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
        stage: int,
    ):
        is_train = (split == Split.TRAIN)

        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")

        self.model.train() if is_train else self.model.eval()

        desc = self.get_desc(f"{self.config.model.model.name}-stage{stage}", split)

        total_loss = 0.0
        num_batches = 0

        # step별(배치별) loss를 epoch 평균으로 만들기 위한 합계들
        total_info_loss = 0.0   # stage 1 전용
        total_ce_loss = 0.0     # stage 2 전용
        total_recon_loss = 0.0
        total_r_loss = 0.0

        for batch in tqdm(loader, desc=desc):
            x, y, x_ori, bemv, _, _ = self._prepare_batch(batch)

            clean_bemv = torch.ones_like(bemv)

            with torch.set_grad_enabled(is_train):
                if stage == 1:
                    # --------------------------------------------------
                    # Stage1: Pretrain
                    # --------------------------------------------------
                    out_masked = self.model(
                        x_missing=x,
                        missing_mask=bemv,
                        retrieved_embedding=None,
                        retrieved_r_score=None,
                        retrieved_label=None,
                        use_gate=False,
                    )
                    out_clean = self.model(
                        x_missing=x_ori,
                        missing_mask=clean_bemv,
                        retrieved_embedding=None,
                        retrieved_r_score=None,
                        retrieved_label=None,
                        use_gate=False,
                    )

                    z_masked = out_masked["embedding_curr"]
                    z_clean = out_clean["embedding_curr"]

                    info_loss = info_nce_loss(
                        z_clean,
                        z_masked,
                        self.config.train.temperature,
                    )

                    recon_masked = out_masked["recon_from_curr"]
                    recon_clean = out_clean["recon_from_curr"]

                    recon_loss = (
                        F.mse_loss(recon_masked, x_ori)
                        + F.mse_loss(recon_clean, x_ori)
                    )

                    with torch.no_grad():
                        sim_masked = F.cosine_similarity(recon_masked, x_ori, dim=-1).clamp(0.0, 1.0)
                        sim_clean = F.cosine_similarity(recon_clean, x_ori, dim=-1).clamp(0.0, 1.0)

                    r_masked = out_masked["r_from_curr"]
                    r_clean = out_clean["r_from_curr"]

                    r_loss = (
                        F.mse_loss(r_masked, sim_masked)
                        + F.mse_loss(r_clean, sim_clean)
                    )

                    loss = (
                        self.lambda_view * info_loss
                        + self.lambda_recon * recon_loss
                        + self.lambda_rscore * r_loss
                    )

                    # 합계 업데이트
                    total_info_loss += float(info_loss.item())
                    total_recon_loss += float(recon_loss.item())
                    total_r_loss += float(r_loss.item())

                elif stage == 2:
                    # --------------------------------------------------
                    # Stage2: Finetune with retrieval + gate + classifier
                    # --------------------------------------------------
                    if self.memory_embeddings is None:
                        raise ValueError("Missing dictionary 가 구축되지 않았습니다.")

                    with torch.no_grad():
                        z_query = self.model.encode(x, bemv)  # (B, D_embed)
                        z_ret, y_ret, r_ret = self._retrieve_top1(z_query)

                    out = self.model(
                        x_missing=x,
                        missing_mask=bemv,
                        retrieved_embedding=z_ret,
                        retrieved_r_score=r_ret,
                        retrieved_label=y_ret,
                        use_gate=True,
                    )

                    logits = out["logits"]

                    x_hat = out["recon_from_curr"]
                    r_pred = out["r_from_curr"]

                    loss_ce = F.cross_entropy(logits, y)
                    loss_recon = F.mse_loss(x_hat, x_ori)

                    with torch.no_grad():
                        sim = F.cosine_similarity(x_hat, x_ori, dim=-1).clamp(0.0, 1.0)
                    loss_r = F.mse_loss(r_pred, sim)

                    loss = (
                        self.lambda_class * loss_ce
                        + self.lambda_recon * loss_recon
                        + self.lambda_rscore * loss_r
                    )

                    # 합계 업데이트
                    total_ce_loss += float(loss_ce.item())
                    total_recon_loss += float(loss_recon.item())
                    total_r_loss += float(loss_r.item())

                else:
                    raise ValueError(f"stage 는 1 또는 2 여야 합니다. (현재: {stage})")

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            num_batches += 1
            total_loss += float(loss.item())

        denom = max(1, num_batches)

        # stage에 따라 사용되지 않는 항목은 None 처리
        epoch_losses = {
            "total": total_loss / denom,
            "info": total_info_loss / denom if stage == 1 else None,
            "ce":   total_ce_loss / denom if stage == 2 else None,
            "recon": total_recon_loss / denom,
            "r":     total_r_loss / denom,
        }

        return epoch_losses

    # ------------------------------------------------------------------
    # Missing Dictionary 구축
    #   - train split 전체에 대해 encoder output / R-score / label 저장
    # ------------------------------------------------------------------
    def build_missing_dictionary(
        self,
        train_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다.")

        loader = train_data.get_loader_for_deep(shuffle=False)

        self.model.eval()

        all_emb = []
        all_label = []
        all_r = []

        for batch in tqdm(loader, desc="BuildMissingDict"):
            x, y, x_ori, bemv, _, _ = self._prepare_batch(batch)

            with torch.no_grad():
                out = self.model(
                    x_missing=x,
                    missing_mask=bemv,
                    retrieved_embedding=None,
                    retrieved_r_score=None,
                    retrieved_label=None,
                    use_gate=False,
                )
                z = out["embedding_curr"]          # (B, D_embed)
                r = out["r_from_curr"]             # (B,)

            all_emb.append(z.cpu())
            all_label.append(y.cpu())
            all_r.append(r.cpu())

        self.memory_embeddings = torch.cat(all_emb, dim=0)  # (N, D_embed)
        self.memory_labels = torch.cat(all_label, dim=0)    # (N,)
        self.memory_r = torch.cat(all_r, dim=0)             # (N,)

        print(
            f"Missing dictionary 크기: {self.memory_embeddings.shape[0]} samples, "
            f"embed_dim={self.memory_embeddings.shape[1]}"
        )

    # ------------------------------------------------------------------
    # 단순 Top-1 retrieval (cosine similarity 기반)
    # ------------------------------------------------------------------
    def _retrieve_top1(
        self,
        z_query: torch.Tensor,  # (B, D_embed), GPU
    ):
        if self.memory_embeddings is None:
            raise ValueError("Missing dictionary 가 없습니다.")

        # 1) query 는 CPU 로 내려서
        z_q = F.normalize(z_query.detach().cpu(), dim=-1)      # (B, D)

        # 2) memory embeddings 도 CPU 상에서 normalize
        mem_emb = F.normalize(self.memory_embeddings, dim=-1)  # (N, D)  CPU

        # 3) CPU matmul
        sim = torch.matmul(z_q, mem_emb.t())                   # (B, N)  CPU
        idx = sim.argmax(dim=-1)                               # (B,)

        z_ret = self.memory_embeddings[idx]                    # (B, D)
        y_ret = self.memory_labels[idx]
        r_ret = self.memory_r[idx]

        # 4) 최종적으로만 GPU 로 올리기
        device = z_query.device
        z_ret = z_ret.to(device)
        y_ret = y_ret.to(device)
        r_ret = r_ret.to(device)

        return z_ret, y_ret, r_ret


    # ------------------------------------------------------------------
    # 테스트 / 예측 (Stage2 기준, dictionary가 있으면 retrieval 사용)
    # ------------------------------------------------------------------
    def test(
        self,
        test_data: Datasets,
    ):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(te_loader)

        metrics_overall = compute_classification_metrics(labels_all, preds_all)

        patterns = test_data.config.data.missing_patterns
        ratios = test_data.ratios

        metrics_by_ratio: dict[str, dict[float, dict]] = {}

        for p_i, pattern in enumerate(patterns):
            p_val = pattern.value
            metrics_by_ratio[p_val] = {}

            for r_i, ratio in enumerate(ratios):
                mask = (pattern_idx_all == p_i) & (ratio_idx_all == r_i)
                if not np.any(mask):
                    continue

                y_sub = labels_all[mask]
                y_hat_sub = preds_all[mask]

                m = compute_classification_metrics(y_sub, y_hat_sub)
                metrics_by_ratio[p_val][ratio] = m

        results = {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

        return results

    def predict(
        self,
        loader: DataLoader,
        split: Split = Split.TEST,
    ):
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        desc = self.get_desc(f"{self.config.model.model.name}-predict", split)

        for batch in tqdm(loader, desc=desc):
            x, y, x_ori, bemv, pattern_idx, ratio_idx = self._prepare_batch(batch)

            with torch.no_grad():
                # dictionary 가 있으면 retrieval + gate 사용 (Stage2 스타일)
                if self.memory_embeddings is not None:
                    z_query = self.model.encode(x, bemv)
                    z_ret, y_ret, r_ret = self._retrieve_top1(z_query)

                    out = self.model(
                        x_missing=x,
                        missing_mask=bemv,
                        retrieved_embedding=z_ret,
                        retrieved_r_score=r_ret,
                        retrieved_label=y_ret,
                        use_gate=True,
                    )
                else:
                    # 사전 없으면 Stage1 스타일로 fallback
                    out = self.model(
                        x_missing=x,
                        missing_mask=bemv,
                        retrieved_embedding=None,
                        retrieved_r_score=None,
                        retrieved_label=None,
                        use_gate=False,
                    )

                logits = out["logits"]
                loss_ce = F.cross_entropy(logits, y)
                preds = logits.argmax(dim=1)

            total_loss += float(loss_ce.item())
            num_batches += 1

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_pattern_idx.append(pattern_idx.cpu())
            all_ratio_idx.append(ratio_idx.cpu())

        avg_loss = total_loss / max(1, num_batches)

        preds_all = torch.cat(all_preds, dim=0).numpy()
        labels_all = torch.cat(all_labels, dim=0).numpy()
        pattern_idx_all = torch.cat(all_pattern_idx, dim=0).numpy()
        ratio_idx_all = torch.cat(all_ratio_idx, dim=0).numpy()

        return avg_loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all

    # ------------------------------------------------------------------
    # 배치 준비
    # ------------------------------------------------------------------
    def _prepare_batch(
        self,
        batch: dict,
    ):
        x = batch["x"].to(self.device)               # 결측 포함 입력
        y = batch["y"].to(self.device)
        x_ori = batch["x_originals"].to(self.device) # 원본 intact 입력
        bemv = batch["bemv"].to(self.device)         # missing mask (1/0)
        pattern_idx = batch["pattern_idx"]
        ratio_idx = batch["ratio_idx"]

        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    # ------------------------------------------------------------------
    # 모델 생성
    # ------------------------------------------------------------------
    def _get_model(
        self,
        input_dim: int,
        num_class: int | None,
    ) -> ReGAE:

        model_cfg = self.config.model.model

        # latent 차원
        embed_dim = getattr(model_cfg, "embed_dim", 32)

        # Encoder / Decoder 피라미드 구조
        enc_hidden = getattr(
            model_cfg,
            "enc_hidden",
            (128, 128, 64, 64, 32),
        )
        dec_hidden = getattr(
            model_cfg,
            "dec_hidden",
            (32, 64, 64, 128, 128),
        )

        r_hidden    = getattr(model_cfg, "r_hidden",    (64,))       # RScoreHead
        gate_hidden = getattr(model_cfg, "gate_hidden", (128, 128))  # FeatGate
        clf_hidden  = getattr(model_cfg, "clf_hidden",  (128, 128))  # classifier

        model = ReGAE(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_classes=num_class,
            enc_hidden=tuple(enc_hidden),
            dec_hidden=tuple(dec_hidden),
            r_hidden=tuple(r_hidden),
            gate_hidden=tuple(gate_hidden),
            clf_hidden=tuple(clf_hidden),
        ).to(self.device)

        return model



    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------
    def save(
        self,
        path: Path,
    ):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다. fit() 이후에 save()를 호출하십시오.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self.config.model.model.name}_model.pt"
        torch.save(self.model.state_dict(), model_path)

        meta = {
            "input_dim": self.input_dim,
            "num_class": self.num_class,
            "model_path": str(model_path),
        }

        self.save_meta(save_dir, meta)

        return model_path

    def load(
        self,
        path: Path,
    ):
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

        # load() 할 때는 dictionary 를 새로 만들어야 하므로
        # adapter 밖에서 build_missing_dictionary 를 다시 호출해 주는 편이 안전합니다.
        # (여기서는 자동으로 만들지 않고 None 으로 둡니다.)
        self.memory_embeddings = None
        self.memory_labels = None
        self.memory_r = None

        return True
