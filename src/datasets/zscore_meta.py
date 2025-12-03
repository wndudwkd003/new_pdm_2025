# src/datasets/zscore_meta.py

from dataclasses import dataclass

@dataclass
class ZScoreMeta:
    mean: list[float]  # 각 feature별 평균
    std: list[float]   # 각 feature별 표준편차
