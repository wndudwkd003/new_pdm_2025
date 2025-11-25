# src/models/tab_transformer_adapter.py

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
from src.utils.metrics import compute_multitask_classification_metrics
from src.params.data_model import Split, StageType
from tab_transformer_pytorch import TabTransformer


class TabTransformerAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.model: TabTransformer | None = None
        self.models: list[TabTransformer] | None = None

        self.device = self.config.train.device
        self.train_mode = self.config.model.stage

        self.feature_dim: int | None = None   # F (센서 수)
        self.seq_len: int | None = None       # S (윈도우 길이)
        self.input_dim: int | None = None     # S * F (TabTransformer 입력 차원)
        self.num_class: int | None = None
        self.horizon: int | None = None

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        """
        각 horizon step마다 TabTransformer 하나씩 학습.
        입력은 항상 (B, S, F) 라고 가정하고, (B, S*F) 로 flatten 해서 사용.
        """
        # 1) shape 파악 (항상 B, S, F 라고 가정)
        probe_loader = train_data.get_loader_for_deep(shuffle=False)
        first_batch = next(iter(probe_loader))

        x_probe = first_batch["x"]  # (B, S, F)
        _, S, F_feat = x_probe.shape
        self.seq_len = int(S)
        self.feature_dim = int(F_feat)
        self.input_dim = int(S * F_feat)

        print(
            f"[TabTransformerAdapter] seq_len={self.seq_len}, "
            f"feature_dim={self.feature_dim}, input_dim={self.input_dim}"
        )

        # 2) 실제 학습용 loader
        tr_loader = train_data.get_loader_for_deep(shuffle=True)
        vl_loader = valid_data.get_loader_for_deep(shuffle=False)

        self.num_class = int(train_data.meta.num_class)
        self.horizon = int(train_data.get_horizon())

        H = self.horizon
        C = self.num_class

        print(f"[TabTransformerAdapter] num_class={C}, horizon={H}")

        # 3) horizon 개수만큼 TabTransformer 모델 생성 및 학습
        self.models = []
        loss_tasks: list[dict] = []

        for t in range(H):
            print(f"[TabTransformerAdapter] === Train step {t} / {H-1} ===")

            model_t = self._get_model(self.input_dim, C)
            model_t.to(self.device)

            self.models.append(model_t)
            self.model = model_t  # BaseModelAdapter.get_deeplearning_utils에서 사용

            optimizer, scheduler = self.get_deeplearning_utils()

            best_valid_loss = None
            best_state = None
            patience = 0
            max_patience = self.config.train.early_stopping_rounds

            train_losses_t: list[float] = []
            valid_losses_t: list[float] = []

            for epoch in range(self.config.train.epochs):
                train_loss = self.run_epoch(
                    loader=tr_loader,
                    optimizer=optimizer,
                    split=Split.TRAIN,
                    step_idx=t,
                )
                valid_loss = self.run_epoch(
                    loader=vl_loader,
                    optimizer=None,
                    split=Split.VALID,
                    step_idx=t,
                )

                lr = float(optimizer.param_groups[0]["lr"])

                print(
                    f"[{self.config.model.model.name} step {t} Epoch {epoch+1}] "
                    f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | LR: {lr:.6f}"
                )

                train_losses_t.append(train_loss)
                valid_losses_t.append(valid_loss)

                scheduler.step()

                if best_valid_loss is None or valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience = 0
                    best_state = {k: v.cpu() for k, v in model_t.state_dict().items()}
                else:
                    patience += 1

                if patience >= max_patience:
                    print(
                        f"[{self.config.model.model.name} step {t}] "
                        f"Early stopping at epoch {epoch+1}"
                    )
                    break

            if best_state is not None:
                model_t.load_state_dict(best_state)
                model_t.to(self.device)

            loss_tasks.append(
                {
                    Split.TRAIN.value: train_losses_t,
                    Split.VALID.value: valid_losses_t,
                }
            )

        # 4) 전체 horizon 모델로 train/valid 메트릭 계산
        _, tr_preds, tr_labels, _, _ = self.predict(tr_loader)
        _, vl_preds, vl_labels, _, _ = self.predict(vl_loader)

        train_metrics = compute_multitask_classification_metrics(tr_labels, tr_preds)
        valid_metrics = compute_multitask_classification_metrics(vl_labels, vl_preds)

        metric_name = "cross_entropy_loss"

        loss_info = {
            "metric_name": metric_name,
            "tasks": loss_tasks,
        }

        results = {
            "split": Split.TRAIN.value,
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": loss_info,
        }

        return results

    # ------------------------------------------------------------------
    # 단일 step (timestep)용 epoch 러너
    # ------------------------------------------------------------------
    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None,
        split: Split,
        step_idx: int,
    ) -> float:
        assert self.models is not None
        assert self.input_dim is not None

        model = self.models[step_idx]

        is_train = split == Split.TRAIN
        model.train() if is_train else model.eval()

        desc = f"[{self.config.model.model.name} {split.name} step {step_idx}]"

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc=desc):
            x, y, _, bemv, _, _ = self._prepare_batch(batch)

            # x: (B, S, F) 고정
            B, S, F_feat = x.shape
            x_in = x.reshape(B, S * F_feat)


            # TabTransformer는 (x_categ, x_cont) 입력
            x_categ = torch.empty(B, 0, dtype=torch.long, device=self.device)
            x_cont = x_in

            y_step = y[:, step_idx]

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                logits = model(x_categ, x_cont)  # (B, num_class)
                loss = F.cross_entropy(logits, y_step)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / num_batches

    # ------------------------------------------------------------------
    # 테스트
    # ------------------------------------------------------------------
    def test(
        self,
        test_data: Datasets,
    ):
        if self.models is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        te_loader = test_data.get_loader_for_deep(shuffle=False)

        loss, preds_all, labels_all, pattern_idx_all, ratio_idx_all = self.predict(
            te_loader
        )

        metrics_overall = compute_multitask_classification_metrics(
            labels_all, preds_all
        )

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

                m = compute_multitask_classification_metrics(y_sub, y_hat_sub)
                metrics_by_ratio[p_val][ratio] = m

        results = {
            "split": Split.TEST.value,
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": metrics_by_ratio,
            "loss": float(loss),
        }

        return results

    # ------------------------------------------------------------------
    # 예측 (전체 horizon 동시에)
    # ------------------------------------------------------------------
    def predict(
        self,
        loader: DataLoader,
    ):
        if self.models is None:
            raise ValueError("모델이 로드되지 않았습니다.")
        assert self.input_dim is not None

        for m in self.models:
            m.eval()

        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_pattern_idx = []
        all_ratio_idx = []

        H = len(self.models)

        for batch in tqdm(loader, desc=f"[{self.config.model.model.name} PRED]"):
            x, y, _, bemv, pattern_idx, ratio_idx = self._prepare_batch(batch)

            # x: (B, S, F) 고정
            B, S, F_feat = x.shape
            x_in = x.reshape(B, S * F_feat)

            if x_in.shape[1] != self.input_dim:
                raise ValueError(
                    f"입력 차원 불일치: x_in.shape[1]={x_in.shape[1]}, "
                    f"self.input_dim={self.input_dim}"
                )

            x_categ = torch.empty(B, 0, dtype=torch.long, device=self.device)
            x_cont = x_in

            logits_list = []
            preds_list = []

            for t, model in enumerate(self.models):
                logits_t = model(x_categ, x_cont)  # (B, C)
                preds_t = logits_t.argmax(dim=-1)  # (B,)

                logits_list.append(logits_t)
                preds_list.append(preds_t)

            logits = torch.stack(logits_list, dim=1)  # (B, H, C)
            preds = torch.stack(preds_list, dim=1)    # (B, H)

            B_cur, H_cur, C = logits.shape
            if H_cur != H:
                raise ValueError(f"H mismatch: H_cur={H_cur}, H={H}")

            logits_flat = logits.reshape(B_cur * H_cur, C)
            y_flat = y.reshape(B_cur * H_cur)

            loss = F.cross_entropy(logits_flat, y_flat)

            total_loss += float(loss.item())
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
        x = batch["x"].to(self.device)                # (B, S, F)
        y = batch["y"].to(self.device)                # (B, H)
        x_ori = batch["x_originals"].to(self.device)  # 안 써도 일단 유지
        bemv = batch["bemv"].to(self.device)          # 안 써도 일단 유지
        pattern_idx = batch["pattern_idx"]
        ratio_idx = batch["ratio_idx"]

        return x, y, x_ori, bemv, pattern_idx, ratio_idx

    # ------------------------------------------------------------------
    # TabTransformer 생성
    # ------------------------------------------------------------------
    def _get_model(
        self,
        input_dim: int,
        num_class: int,
    ):
        model = TabTransformer(
            categories=(),                # 카테고리 없음
            num_continuous=input_dim,     # flatten된 연속형 feature 수 (S*F)
            dim=32, # 32 # 16
            dim_out=num_class,            # 클래스 수
            depth=6,  # 6 # 4
            heads=8,  # 8 # 4
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU(),
        )
        return model

    # ------------------------------------------------------------------
    # 저장 / 로드
    # ------------------------------------------------------------------
    def save(
        self,
        path: Path,
    ):
        if self.models is None:
            raise ValueError("저장할 모델이 없습니다. fit() 이후에 save()를 호출하십시오.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_paths: list[str] = []

        for t, model in enumerate(self.models):
            model_path = save_dir / f"{self.config.model.model.name}_step_{t}.pt"
            torch.save(model.state_dict(), model_path)
            model_paths.append(str(model_path))

        meta = {
            "feature_dim": self.feature_dim,
            "seq_len": self.seq_len,
            "input_dim": self.input_dim,
            "num_class": self.num_class,
            "horizon": self.horizon,
            "model_paths": model_paths,
        }

        self.save_meta(save_dir, meta)

        return save_dir

    def load(
        self,
        path: Path,
    ):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        self.feature_dim = int(meta["feature_dim"])
        self.seq_len = int(meta["seq_len"])
        self.input_dim = int(meta["input_dim"])
        self.num_class = int(meta["num_class"])
        self.horizon = int(meta["horizon"])
        model_paths: list[str] = list(meta["model_paths"])

        self.device = self.config.train.device

        self.models = []

        for model_path_str in model_paths:
            model_path = Path(model_path_str)
            model = self._get_model(self.input_dim, self.num_class)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        self.model = self.models[0] if len(self.models) > 0 else None

        return len(self.models) == self.horizon
