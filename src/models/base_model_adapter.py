# src/models/base_model_adapter.py

from pathlib import Path
import json

from abc import ABC, abstractmethod

from src.configs.configs import Config

class BaseModelAdapter(ABC):
    def __init__(
        self,
        config: Config,
    ):
        self.config = config

    @abstractmethod
    def fit(
        self,
        train_data,
        valid_data,
    ):
        pass


    @abstractmethod
    def predict(
        self,
        test_data,
    ):
        pass


    @abstractmethod
    def save(
        self,
        path: Path,
    ):
        pass


    @abstractmethod
    def load(
        self,
        path: Path,
    ):
        pass


    def save_meta(
        self,
        save_dir: Path,
        meta: dict,
    ):
        meta_path = save_dir / "meta.json"

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)




    def load_meta(
        self,
        save_dir: Path,
    ):
        meta_path = save_dir / "meta.json"

        if meta_path.exists() is False:
            raise FileNotFoundError(f"모델 메타 파일이 없는데요? {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return meta

