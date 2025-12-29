# src/datasets/dl_collator.py

import numpy as np
import torch
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.datasets.data_class import Datasets


class DefaultMissingCollator:
    def __init__(self, dataset: "Datasets"):
        self.dataset = dataset
        self.patterns = dataset.config.data.missing_patterns
        self.ratios = dataset.ratios
        self.imputed_dict = dataset.imputed_dict

        self.original_X = dataset.imputed_dict["original"]["X"]

        self.use_on_batch = dataset.config.data.use_on_batch
        self.different_mode = dataset.config.data.different_mode

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        base_indices = [int(b["base_idx"]) for b in batch]

        xs, ys, bemvs, x_originals = [], [], [], []
        pattern_idx, ratio_idx = [], []
        base_ids = []

        if self.use_on_batch:
            V = len(self.patterns) * len(self.ratios)

            for base_idx in base_indices:
                x_orig = self.original_X[base_idx]

                for p_i, pattern in enumerate(self.patterns):
                    pattern_v = pattern.value
                    ratio_dict = self.imputed_dict[pattern_v]

                    for r_i, ratio in enumerate(self.ratios):
                        d = ratio_dict[ratio]

                        xs.append(d["X"][base_idx])
                        ys.append(d["y"][base_idx])
                        bemvs.append(d["bemv"][base_idx])
                        x_originals.append(x_orig)

                        pattern_idx.append(p_i)
                        ratio_idx.append(r_i)
                        base_ids.append(base_idx)

        else:
            V = 1
            num_patterns = len(self.patterns)
            num_ratios = len(self.ratios)
            num_scenarios = num_patterns * num_ratios

            if self.different_mode:
                scenario_pairs = [
                    (p_i, r_i)
                    for p_i in range(num_patterns)
                    for r_i in range(num_ratios)
                ]
                perm = np.random.permutation(num_scenarios)

                for i, base_idx in enumerate(base_indices):
                    x_orig = self.original_X[base_idx]
                    p_i, r_i = scenario_pairs[int(perm[i % num_scenarios])]

                    pattern = self.patterns[p_i]
                    pattern_v = pattern.value
                    ratio = self.ratios[r_i]
                    d = self.imputed_dict[pattern_v][ratio]

                    xs.append(d["X"][base_idx])
                    ys.append(d["y"][base_idx])
                    bemvs.append(d["bemv"][base_idx])
                    x_originals.append(x_orig)

                    pattern_idx.append(p_i)
                    ratio_idx.append(r_i)
                    base_ids.append(base_idx)
            else:
                for base_idx in base_indices:
                    x_orig = self.original_X[base_idx]
                    p_i = int(np.random.randint(num_patterns))
                    r_i = int(np.random.randint(num_ratios))

                    pattern = self.patterns[p_i]
                    pattern_v = pattern.value
                    ratio = self.ratios[r_i]
                    d = self.imputed_dict[pattern_v][ratio]

                    xs.append(d["X"][base_idx])
                    ys.append(d["y"][base_idx])
                    bemvs.append(d["bemv"][base_idx])
                    x_originals.append(x_orig)

                    pattern_idx.append(p_i)
                    ratio_idx.append(r_i)
                    base_ids.append(base_idx)

        x = torch.from_numpy(np.stack(xs, axis=0)).to(torch.float32)
        y = torch.from_numpy(np.stack(ys, axis=0)).long()
        bemv = torch.from_numpy(np.stack(bemvs, axis=0)).to(torch.float32)
        x_originals = torch.from_numpy(np.stack(x_originals, axis=0)).to(torch.float32)

        pattern_idx_t = torch.tensor(pattern_idx, dtype=torch.long)
        ratio_idx_t = torch.tensor(ratio_idx, dtype=torch.long)
        base_idx_t = torch.tensor(base_ids, dtype=torch.long)

        return {
            "x": x,
            "y": y,
            "bemv": bemv,
            "x_originals": x_originals,
            "pattern_idx": pattern_idx_t,
            "ratio_idx": ratio_idx_t,
            "base_idx": base_idx_t,
            "views_per_base": torch.tensor(V, dtype=torch.long),
            "base_batch_size": torch.tensor(len(base_indices), dtype=torch.long),
        }
