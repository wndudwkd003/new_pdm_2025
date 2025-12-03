from dataclasses import asdict
import json
from pathlib import Path
from src.datasets.zscore_meta import ZScoreMeta

def zscore_save(work_dir: Path, z: ZScoreMeta):
    path = work_dir / "zscore_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(z), f, ensure_ascii=False, indent=2)


def load_zscore_data(work_dir: Path) -> ZScoreMeta:
    path = work_dir / "zscore_meta.json"
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return ZScoreMeta(
        mean=d["mean"],
        std=d["std"],
    )
