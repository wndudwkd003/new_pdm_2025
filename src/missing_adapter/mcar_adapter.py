# src/missing_adapter/mcar_adapter.py

import numpy as np


class MCARAdapter:
    def __init__(self, ratio: float, seed: int):
        self.ratio = float(ratio)
        self.seed = int(seed)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        - X: (N, F)
        - 각 행마다 정확히 round(ratio*F)개를 결측으로 만든다.
        - 각 행은 독립적으로 랜덤하게 선택한다.
        - 어떤 컬럼도 전체 행에서 전부 결측이 되지 않도록 보정한다.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected X.ndim==2, got {X.ndim}")

        N, F = X.shape
        rng = np.random.default_rng(self.seed)

        k_miss = int(round(self.ratio * F))  # 행당 결측 개수
        k_miss = max(0, min(k_miss, F))  # 안전 클램프

        if k_miss >= F:
            k_miss = F - 1  # 최소 1개는 남김

        # 1) 행별로 정확히 k_miss개 결측
        mask = np.zeros((N, F), dtype=bool)  # True=결측
        for i in range(N):
            miss_idx = rng.choice(F, size=k_miss, replace=False)
            mask[i, miss_idx] = True

        # 2) 컬럼이 "전부 결측"인 경우 보정: 해당 컬럼에서 임의의 한 행은 관측으로 되돌림
        col_all_missing = np.where(mask.all(axis=0))[0]  # True=컬럼 전체가 결측
        for j in col_all_missing:
            i = int(rng.integers(0, N))  # 아무 행 하나 선택
            mask[i, j] = False  # 그 위치는 관측으로 변경

            # 행별 결측 개수(k_miss)를 유지하기 위해, 같은 행에서 다른 관측 하나를 결측으로 바꿈
            # (j를 관측으로 만들었으니 결측 개수가 1 줄었음 -> 다시 1개 늘려야 함)
            candidates = np.where(~mask[i])[0]  # 현재 관측인 위치들
            candidates = candidates[candidates != j]  # 방금 살린 j는 제외
            if candidates.size > 0:
                t = int(rng.choice(candidates))
                mask[i, t] = True

        X_missing = X.copy()
        X_missing[mask] = np.nan
        return X_missing
