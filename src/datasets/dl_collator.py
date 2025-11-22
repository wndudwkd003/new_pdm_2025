# src/datasets/dl_collator.py

import numpy as np
import torch
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.datasets.data_class import Datasets

class DefaultMissingCollator:
    def __init__(self, dataset: "Datasets"):
        self.dataset = dataset

        self.patterns = dataset.config.data.missing_patterns  # [MissingPattern.MCAR, ...]
        self.ratios = dataset.ratios                          # [0.0, 0.1, ...]
        self.imputed_dict = dataset.imputed_dict              # pattern/ratio별 X,y,bemv

        self.original_X = dataset.imputed_dict["original"]["X"]


    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # batch: [{"base_idx": i}, {"base_idx": j}, ...]
        base_indices = [b["base_idx"] for b in batch]

        xs = []
        ys = []
        bemvs = []

        x_originals = []

        pattern_idx = []
        ratio_idx = []

        for base_idx in base_indices:
            x_orig = self.original_X[base_idx]

            # 원본 인덱스 하나에 대해 모든 pattern × ratio 돌면서 쌓기
            for p_i, pattern in enumerate(self.patterns):
                pattern_v = pattern.value
                ratio_dict = self.imputed_dict[pattern_v]

                for r_i, ratio in enumerate(self.ratios):
                    d = ratio_dict[ratio]

                    X_imp = d["X"][base_idx]     # (S, F)
                    y = d["y"][base_idx]         # (T,)
                    bemv = d["bemv"][base_idx]   # (S, F)

                    xs.append(X_imp)
                    ys.append(y)
                    bemvs.append(bemv)

                    x_originals.append(x_orig)

                    pattern_idx.append(p_i)
                    ratio_idx.append(r_i)

        x = torch.from_numpy(np.stack(xs, axis=0)).to(torch.float32)      # (B * P * R, S, F)
        y = torch.from_numpy(np.stack(ys, axis=0)).long()              # (B * P * R, T)
        bemv = torch.from_numpy(np.stack(bemvs, axis=0)).to(torch.float32)

        x_originals = torch.from_numpy(np.stack(x_originals, axis=0)).to(torch.float32)

        pattern_idx = torch.tensor(pattern_idx, dtype=torch.long)      # (B * P * R,)
        ratio_idx = torch.tensor(ratio_idx, dtype=torch.long)          # (B * P * R,)

        return {
            "x": x,
            "y": y,
            "bemv": bemv,
            "x_originals": x_originals,
            "pattern_idx": pattern_idx,
            "ratio_idx": ratio_idx,
        }
