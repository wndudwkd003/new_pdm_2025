from enum import Enum


class MissingScenario(Enum):
    SINGLE = "single"
    MULTI = "multi"


class MissingPattern(Enum):
    MCAR = "mcar"


class StackMode(Enum):
    BY_DATASET = "by_dataset"
    BY_ROW = "by_row"


class ImputeMethod(Enum):
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"
    MICE = "mice"
    GAIN = "gain"
