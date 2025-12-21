from __future__ import annotations

import math
from typing import Iterator, List, Sequence

import numpy as np
import torch


class BalancedClassBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    base 샘플 인덱스(batch)를 "클래스 밸런스"로 생성합니다.
    - batch_size: base 배치 크기 (예: 512)
    - min_per_class: 선택된 클래스마다 최소 몇 개를 넣을지 (SupCon/cls-contrast면 2 이상 권장)
    - classes_per_batch: 한 배치에서 보장할 클래스 개수. None이면 자동으로 floor(batch_size/min_per_class)
    - drop_last: 마지막 미완성 배치 버릴지
    """

    def __init__(
        self,
        *,
        labels: Sequence[int] | np.ndarray,
        batch_size: int,
        min_per_class: int = 2,
        classes_per_batch: int | None = None,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if min_per_class <= 0:
            raise ValueError(f"min_per_class must be positive, got {min_per_class}")
        if batch_size < min_per_class:
            raise ValueError(
                f"batch_size must be >= min_per_class, got {batch_size=} {min_per_class=}"
            )

        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.min_per_class = int(min_per_class)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.num_classes = int(self.labels.max()) + 1
        if self.num_classes <= 1:
            raise ValueError(
                f"num_classes must be >= 2 for class-balanced sampling, got {self.num_classes}"
            )

        # 클래스별 인덱스 풀
        self.indices_by_class: list[np.ndarray] = []
        self.valid_classes: list[int] = []
        for c in range(self.num_classes):
            idx = np.where(self.labels == c)[0]
            if idx.size > 0:
                self.indices_by_class.append(idx)
                self.valid_classes.append(c)
            else:
                self.indices_by_class.append(idx)

        if len(self.valid_classes) < 2:
            raise ValueError("At least two classes must have samples.")

        if classes_per_batch is None:
            classes_per_batch = self.batch_size // self.min_per_class
        if classes_per_batch <= 0:
            raise ValueError(
                f"classes_per_batch must be positive, got {classes_per_batch}"
            )

        self.classes_per_batch = int(min(classes_per_batch, len(self.valid_classes)))
        self._epoch = 0

    def __len__(self) -> int:
        n = int(self.labels.shape[0])
        if self.drop_last:
            return n // self.batch_size
        return int(math.ceil(n / self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        # 각 클래스 인덱스 셔플 + 포인터
        pools: list[np.ndarray] = []
        ptrs = np.zeros((self.num_classes,), dtype=np.int64)

        for c in range(self.num_classes):
            idx = self.indices_by_class[c]
            if idx.size > 0:
                pools.append(rng.permutation(idx))
            else:
                pools.append(idx)

        n_batches = len(self)

        for _ in range(n_batches):
            # 이번 배치에 포함시킬 클래스 선택
            chosen = rng.choice(
                np.asarray(self.valid_classes, dtype=np.int64),
                size=self.classes_per_batch,
                replace=False,
            ).tolist()

            batch: list[int] = []

            # 1) 클래스별 최소 min_per_class 채우기
            for c in chosen:
                need = self.min_per_class
                for _k in range(need):
                    # 해당 클래스 pool이 소진되면 다시 셔플해서 재사용(오버샘플링)
                    if pools[c].size == 0:
                        raise ValueError(f"class {c} has no samples, but was chosen")

                    if ptrs[c] >= pools[c].size:
                        pools[c] = rng.permutation(self.indices_by_class[c])
                        ptrs[c] = 0

                    batch.append(int(pools[c][ptrs[c]]))
                    ptrs[c] += 1

            # 2) 남은 슬롯 채우기 (선택된 클래스들에서 라운드로빈)
            remain = self.batch_size - len(batch)
            if remain < 0:
                raise ValueError(
                    f"min_per_class * classes_per_batch exceeds batch_size: "
                    f"{self.min_per_class=} {self.classes_per_batch=} {self.batch_size=}"
                )

            ci = 0
            while len(batch) < self.batch_size:
                c = chosen[ci % len(chosen)]
                if ptrs[c] >= pools[c].size:
                    pools[c] = rng.permutation(self.indices_by_class[c])
                    ptrs[c] = 0
                batch.append(int(pools[c][ptrs[c]]))
                ptrs[c] += 1
                ci += 1

            yield batch
