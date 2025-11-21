# src/models/xgboost_adapter.py

from xgboost import XGBClassifier
from pathlib import Path

from src.configs.configs import Config
from src.models.base_model_adapter import BaseModelAdapter
from src.datasets.data_class import Datasets


class XGBoostAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__(config)

        self.models: list[XGBClassifier] | None = None
        self.results = dict()

    def fit(
        self,
        train_data: Datasets,
        valid_data: Datasets,
    ):
        # adapter --> 데이터 세트 반환 (numpy)
        X_tr, y_tr = train_data.as_numpy()
        X_val, y_val = valid_data.as_numpy()


        # XGBoost 설정
        eval_metric = self.config.model.eval_metric
        horizon = train_data.get_horizon()
        self.horizon = horizon
        num_class = train_data.get_num_class()

        eval_results = dict()

        self.models = [] # <-- 초기화

        # 시계열 예측 --> 멀티 태스크 분류로 처리
        for t in range(horizon):
            model = XGBClassifier(
                objective=self.config.model.objective,
                num_class=num_class,
                random_state=self.config.train.seed,
                eval_metric=eval_metric,
                early_stopping_rounds=self.config.train.early_stopping_rounds,
            )

            model.fit(
                X_tr, y_tr[:, t],
                eval_set=[(X_val, y_val[:, t])],
            )

            self.models.append(model) # 학습된 모델 저장

            eval_results[f"horizon_{t}_eval_result"] = model.evals_result() # 평가 결과 저장

        self.results["eval_results"] = eval_results # 전체 평가 결과 저장

        return self.results


    def save(
        self,
        path: Path
    ):
        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)


        meta = {
            "horizon": self.horizon,
        }

        save_model_dirs = []

        for t, model in enumerate(self.models):
            model_dir = save_dir / f"xgb_horizon_{t}.json"
            model.save_model(model_dir)

            save_model_dirs.append(str(model_dir))

        meta["save_model_dirs"] = save_model_dirs

        self.save_meta(save_dir, meta)



    def load(
        self,
        path: Path
    ):
        save_dir = path / "save"
        meta = self.load_meta(save_dir)

        save_model_dirs = meta["save_model_dirs"]

        self.models = []

        for model_dir in save_model_dirs:
            model = XGBClassifier()
            model.load_model(model_dir)
            self.models.append(model)

        return len(self.models) == len(save_model_dirs) == meta["horizon"]









