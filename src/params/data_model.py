from enum import Enum

from attr import dataclass


class DatasetType(Enum):
    MPTMS = "datasets/MPTMS/processed_data"
    CMAPSS = "datasets/c-mapss/processed_data"






class ModelType(Enum):
    XGBOOST = "xgboost"
