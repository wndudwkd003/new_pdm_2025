from abc import ABC, abstractmethod

from src.datasets.data_class import DatasetClass
from src.params.literals import Split


class BaseMissingAdapter(ABC):
    @abstractmethod
    def transform(
        self,
        dataset: DatasetClass,
        split: Split,
    ):
        pass

