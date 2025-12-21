# src/core/retrieval/regae_retriever.py

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class ReGAERetriever:

    def __init__(
        self,
        memory_embeddings: torch.Tensor,
        memory_labels: torch.Tensor,
    ):
        self.memory_embeddings = memory_embeddings
        self.memory_labels = memory_labels

    @classmethod
    def build_from_train_data(
        cls,
        model,
        train_data,
        desc: str = "BuildMissingDict",
    ):
        if model is None:
            raise ValueError("모델이 초기화되지 않았습니다. (model is None)")

        model.eval()

        X0 = train_data.imputed_dict["original"][
            "X"
        ]  # (N, F) : 결측 없는 원본(현재는 zscore 적용된 값)
        y0 = train_data.imputed_dict["original"]["y"]  # (N,)

        N = X0.shape[0]
        batch_size = train_data.config.train.batch_size

        device = next(model.parameters()).device

        all_emb = []
        all_label = []

        for start in tqdm(range(0, N, batch_size), desc=desc):
            end = min(start + batch_size, N)

            x = torch.from_numpy(X0[start:end]).to(device).to(torch.float32)
            bemv = torch.ones_like(x)

            with torch.no_grad():
                z = model.encode(x_missing=x, missing_mask=bemv)  # (B, D)

            all_emb.append(z.cpu())
            all_label.append(torch.from_numpy(y0[start:end]).to(torch.long).cpu())

        memory_embeddings = torch.cat(all_emb, dim=0)  # (N, D)
        memory_labels = torch.cat(all_label, dim=0)  # (N,)

        print(
            f"Missing dictionary 크기(원본만): {memory_embeddings.shape[0]} samples, "
            f"embed_dim={memory_embeddings.shape[1]}"
        )

        return cls(
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
        )

    def top1(
        self,
        z_query: torch.Tensor,  # (B, D), GPU
    ):
        if self.memory_embeddings is None:
            raise ValueError(
                "Missing dictionary 가 없습니다. (memory_embeddings is None)"
            )

        # 1) query -> CPU normalize
        z_q = F.normalize(z_query.detach().cpu(), dim=-1)  # (B, D)

        # 2) mem -> CPU normalize
        mem_emb = F.normalize(self.memory_embeddings, dim=-1)  # (N, D)

        # 3) similarity (B, N)
        sim = torch.matmul(z_q, mem_emb.t())
        idx = sim.argmax(dim=-1)  # (B,)

        # 4) gather on CPU
        z_ret = self.memory_embeddings[idx]  # (B, D)
        y_ret = self.memory_labels[idx]  # (B,)

        # 5) move to query device
        device = z_query.device
        z_ret = z_ret.to(device)
        y_ret = y_ret.to(device)

        return z_ret, y_ret

    def state_dict(self) -> dict:
        if self.memory_embeddings is None or self.memory_labels is None:
            raise ValueError("retriever state_dict: memory가 없습니다.")

        return {
            "memory_embeddings": self.memory_embeddings.detach().cpu(),
            "memory_labels": self.memory_labels.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        if "memory_embeddings" not in state or "memory_labels" not in state:
            raise ValueError("from_state_dict: state key가 올바르지 않습니다.")

        return cls(
            memory_embeddings=state["memory_embeddings"].detach().cpu(),
            memory_labels=state["memory_labels"].detach().cpu(),
        )
