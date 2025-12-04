# src/trainer/trainer.py

from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json, shutil
from src.models.base_model_adapter import BaseModelAdapter
from src.configs.configs import Config
from src.params.literals import Workspace
from src.params.data_model import Split, StageType
from src.params.model_map import MODEL_MAP
from src.datasets.data_class import Datasets
from src.utils.eval_viz import (
    save_metrics_artifacts,
    save_history_artifacts,
    plot_metric_over_ratio,
)
from src.utils.zscore_utils import zscore_save, load_zscore_data
from src.utils.imputer_utils import imputer_save, load_imputer_data


class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config

        self.model_type = config.model.model
        self.stage_type = config.model.stage

        self.adapter: BaseModelAdapter = MODEL_MAP[self.model_type](config)


    def get_work_dir(self):
        # 현재 run의 ws 만드는 함수임
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mn = self.model_type.value.lower() # model name

        sr = self.config.data.start_missing_ratio       # start ratio
        tr = self.config.data.target_missing_ratio      # target ratio
        step = self.config.data.step_missing_ratio      # step ratio

        stage = self.stage_type.value.lower()

        model_size = self.config.model.model_size.value.lower()

        other_prefix = self.config.model.other_prefix
        other_prefix = f"-{other_prefix}" if other_prefix != "" else ""

        missing_scenario = self.config.data.missing_scenario.value
        missing_patterns = "_".join([p.value for p in self.config.data.missing_patterns])
        impute_method = self.config.data.impute_method.value

        seed_txt = f"seed{self.config.train.seed}"
        run_name = f"{now}_{mn}_{seed_txt}_{sr}_to_{tr}_{step}_step_{stage}{model_size}{missing_scenario}_{missing_patterns}_{impute_method}{other_prefix}"

        # ws dir 생성
        work_dir = Path(self.config.train.output_dir) / run_name
        work_dir.mkdir(parents=True, exist_ok=True)

        # configs, scripts, src 폴더 복사
        backup_dir = work_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for p in list(Workspace.__args__):
            source = Path(p)
            dist = backup_dir / p
            shutil.copytree(source, dist, dirs_exist_ok=True)

        return work_dir


    def run(self, split: Split):
        if split == Split.TRAIN:
            # train 시 작업 디렉토리 생성
            self.work_dir = self.get_work_dir()

            train_ds = Datasets(
                self.config,
                Split.TRAIN,
                None,
                None,
            )

            # train 시 만드는 것들
            train_zscore_meta = train_ds.zscore_meta
            imputer_dict = train_ds.imputer_dict

            valid_ds = Datasets(
                self.config,
                Split.VALID,
                train_zscore_meta,
                imputer_dict
            )

            model_save_dir = self.work_dir / split.value
            model_save_dir.mkdir(parents=True, exist_ok=True)

            zscore_save(model_save_dir, train_zscore_meta)
            imputer_save(model_save_dir, imputer_dict)

            if self.config.model.stage == StageType.FINETUNE:
                pre_trained_dir = Path(self.config.model.save_work_dir)
                if not self.adapter.load(pre_trained_dir):
                    raise ValueError("사전학습된 모델 로드에 실패했습니다.")
                print(f"Pre-trained model loaded from {pre_trained_dir}")

            results = self.adapter.fit(train_ds, valid_ds)
            print(f"Training completed.")
            self.adapter.save(model_save_dir)

            results_dir = self.work_dir / "history"
            results_dir.mkdir(parents=True, exist_ok=True)


        elif split == Split.TEST:
            # test 시 저장된 디렉토리 불러옴
            self.work_dir = Path(self.config.model.save_work_dir)
            results_dir = self.get_next_result_dir("test")

            train_train_dir = self.work_dir / Split.TRAIN.value

            train_zscore_meta = load_zscore_data(train_train_dir)
            imputer_dict = load_imputer_data(train_train_dir)

            test_ds = Datasets(
                self.config,
                Split.TEST,
                train_zscore_meta,
                imputer_dict,
            )

            if not self.adapter.load(self.work_dir):
                raise ValueError("모델 로드에 실패했습니다.")

            results = self.adapter.test(test_ds)


        self.save_results(results, results_dir, split)

        return results_dir


    def save_results(self, results: dict, path: Path, split: Split) -> Path:

        # raw results 백업
        with open(path / "results_raw.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if split == Split.TRAIN:
            # XGBoostAdapter.fit() 결과 포맷 가정:
            # {
            #   "split": "train",
            #   "train_metrics": {...},
            #   "valid_metrics": {...},
            #   "loss": {
            #       "metric_name": str,
            #       "tasks": [ {"train": [...], "valid": [...]}, ... ]
            #   },
            # }
            train_metrics = results[f"{Split.TRAIN.value}_metrics"]
            valid_metrics = results[f"{Split.VALID.value}_metrics"]
            loss = results["loss"]

            train_dir = path / f"{Split.TRAIN.value}_metrics"
            valid_dir = path / f"{Split.VALID.value}_metrics"
            loss_dir = path / "loss"

            train_dir.mkdir(parents=True, exist_ok=True)
            valid_dir.mkdir(parents=True, exist_ok=True)
            loss_dir.mkdir(parents=True, exist_ok=True)

            # train/valid 지표 + 스텝/클래스별 그래프
            save_metrics_artifacts(train_metrics, train_dir)
            save_metrics_artifacts(valid_metrics, valid_dir)

            # task별 metric history (eval metric curve)
            save_history_artifacts(loss, loss_dir)

        elif split == Split.TEST:
            # XGBoostAdapter.test() 결과 포맷 가정:
            # {
            #   "split": "test",
            #   "metrics_overall": {...},
            #   "metrics_by_ratio": {
            #       pattern_value: {
            #           ratio: {...},  # compute_multitask_classification_metrics 결과
            #           ...
            #       },
            #       ...
            #   },
            # }
            overall = results["metrics_overall"]
            by_ratio = results["metrics_by_ratio"]

            overall_dir = path / "overall"
            overall_dir.mkdir(parents=True, exist_ok=True)
            save_metrics_artifacts(overall, overall_dir)

            ratio_base_dir = path / "by_ratio"
            ratio_base_dir.mkdir(parents=True, exist_ok=True)

            # 패턴별 / ratio별 지표 + ratio 곡선
            for pattern, ratio_dict in by_ratio.items():
                pattern_dir = ratio_base_dir / f"pattern_{pattern}"
                pattern_dir.mkdir(parents=True, exist_ok=True)

                # 각 ratio별 metrics 저장
                for ratio_value, m in ratio_dict.items():
                    r_dir = pattern_dir / f"ratio_{ratio_value}"
                    r_dir.mkdir(parents=True, exist_ok=True)
                    save_metrics_artifacts(m, r_dir)

                # ratio에 따른 꺾은선 (accuracy, f1_macro)
                plot_metric_over_ratio(
                    metrics_by_ratio=ratio_dict,
                    metric_key="accuracy",
                    save_dir=pattern_dir,
                    prefix=f"{pattern}",
                )
                # f1_macro가 있는 경우에만
                sample_metrics = next(iter(ratio_dict.values()))
                if "f1_macro" in sample_metrics:
                    plot_metric_over_ratio(
                        metrics_by_ratio=ratio_dict,
                        metric_key="f1_macro",
                        save_dir=pattern_dir,
                        prefix=f"{pattern}",
                    )
        else:
            raise ValueError(f"알 수 없는 split: {split}")

        return path




    def get_next_result_dir(self, prefix: str):
        exist_idx = []

        ws_dir = self.work_dir / Split.TEST.value
        ws_dir.mkdir(parents=True, exist_ok=True)

        for p in ws_dir.iterdir():
            if p.is_dir() and p.name.startswith(prefix + "_"):
                _, idx = p.name.split("_")

                idx = int(idx)
                exist_idx.append(idx)


        target_idx = max(exist_idx) + 1 if len(exist_idx) > 0 else 0

        dir_name = f"{prefix}_{target_idx}"

        result_dir = ws_dir / dir_name
        result_dir.mkdir(parents=True, exist_ok=True)

        return result_dir







