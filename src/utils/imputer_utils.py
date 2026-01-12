# src/utils/imputer_utils.py

from pathlib import Path


from typing import Dict

from src.imputer.base_imputer import BaseImputeAdapter


import cloudpickle as pickler


def imputer_save(work_dir: Path, imputer_dict: Dict[str, BaseImputeAdapter]):
    path = work_dir / "imputer_meta.pkl"
    with open(path, "wb") as f:
        pickler.dump(imputer_dict, f)


def load_imputer_data(work_dir: Path) -> Dict[str, BaseImputeAdapter]:
    path = work_dir / "imputer_meta.pkl"
    with open(path, "rb") as f:
        return pickler.load(f)
