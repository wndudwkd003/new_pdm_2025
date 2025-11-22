# src/trainer/trainer.py

from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json, shutil
import numpy as np

from src.configs.configs import Config
from src.params.data_model import ModelType
from src.params.literals import Split, Workspace
from src.datasets.data_class import Datasets
from src.models.base_model_adapter import BaseModelAdapter



class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config

        self.model_type = config.model.model
        self.stage_type = config.model.stage

        self.adapter = self.model_type.value(config)


    def get_work_dir(self):
        # 현재 run의 ws 만드는 함수임
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mn = self.model_type.name.lower() # model name

        sr = self.config.data.start_missing_ratio       # start ratio
        tr = self.config.data.target_missing_ratio      # target ratio
        step = self.config.data.step_missing_ratio      # step ratio

        stage = self.stage_type.name.lower()

        other_prefix = self.config.model.other_prefix
        other_prefix = f"_{other_prefix}" if other_prefix != "" else ""

        run_name = f"{now}_{mn}-{sr}_to_{tr}_{step}step_{stage}{other_prefix}"

        # ws dir 생성
        work_dir = Path(self.config.train.output_dir) / run_name
        work_dir.mkdir(parents=True, exist_ok=True)

        # config 백업
        config_dir = work_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        with open(config_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

        # configs, scripts, src 폴더 복사
        backup_dir = work_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for dir in list(Workspace):
            source = Path(dir)
            dist = backup_dir / dir
            shutil.copytree(source, dist, dirs_exist_ok=True)

        return work_dir

    def train(self):
        # train 시 작업 디렉토리 생성
        self.work_dir = self.get_work_dir()

        train_ds = Datasets(self.config, "train")
        valid_ds = Datasets(self.config, "valid")

        results = self.adapter.fit(train_ds, valid_ds)
        self.adapter.save(self.work_dir)

        results_dir = self.work_dir / "history"
        results_dir.mkdir(parents=True, exist_ok=True)

        metrics = self.save_results(results, results_dir, "train")

        return metrics


    def test(self):
        # test 시 저장된 디렉토리 불러옴
        self.work_dir = Path(self.config.model.save_work_dir)

        results_dir = self.work_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)


        test_ds = Datasets(self.config, "test")

        if self.adapter.load(self.work_dir):
            results = self.adapter.predict(test_ds)
            metrics = self.save_results(results, results_dir, "test")
            return metrics

        else:
            raise ValueError("모델 로드에 실패했습니다.")


    def save_results(self, results, path, split: Split):
        metrics = self.todo_fun(results)
        metric_dir = self.work_dir / "metrics"
        metric_dir.mkdir(parents=True, exist_ok=True)
        self.save_metrics(metrics, metric_dir)
        return metrics



    def todo_fun(self, results):
        pass


    def save_metrics(self, metrics, path):
        pass






