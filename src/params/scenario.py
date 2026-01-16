from enum import Enum


class MissingScenario(Enum):
    SINGLE = "single"
    MULTI = "multi"


class MissingPattern(Enum):
    MCAR = "mcar"
    MAR = "mar"
    MNAR_MBOV = "mnar_mbov"
    MNAR_MBUV = "mnar_mbuv"
    MNAR_ExS = "mnar_exs"
    MNAR_LCP = "mnar_lcp"
    MNAR_NR = "mnar_nr"


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
