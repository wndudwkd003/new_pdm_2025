# src/utils/imputer_utils.py

from pathlib import Path
import pickle
from typing import Dict

from src.imputer.base_imputer import BaseImputeAdapter


def imputer_save(work_dir: Path, imputer_dict: Dict[str, BaseImputeAdapter]):
    """
    work_dir (예: work_dir / 'train') 아래에
    패턴별 imputer 딕셔너리를 그대로 저장합니다.
    """
    path = work_dir / "imputer_meta.pkl"
    with open(path, "wb") as f:
        pickle.dump(imputer_dict, f)


def load_imputer_data(work_dir: Path) -> Dict[str, BaseImputeAdapter]:
    """
    work_dir (예: work_dir / 'train') 아래에서
    imputer 딕셔너리를 다시 읽어옵니다.
    """
    path = work_dir / "imputer_meta.pkl"
    with open(path, "rb") as f:
        imputer_dict = pickle.load(f)
    return imputer_dict
