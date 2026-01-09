# src/models/xgboost_adapter.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier, XGBRegressor

from src.configs.configs import Config
from src.datasets.data_class import Datasets
from src.models.base_model_adapter import BaseModelAdapter
from src.params.data_model import Split
from src.utils.metrics import compute_classification_metrics, compute_regression_metrics


class XGBoostAdapter(BaseModelAdapter):
    """
    - bemv 기반 NaN 복원: bemv == 0 인 위치만 NaN (0=결측, 1=관측)
    - 분류:
        * num_class == 2  -> binary:logistic
        * num_class >  2  -> multi:softprob + num_class
      (config.model.objective / eval_metric 이 "auto"면 자동 결정)
    - 회귀:
        * meta.task 가 regression 계열이면 reg:squarederror (옵션)
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.model: XGBClassifier | XGBRegressor | None = None
        self.is_regression: bool = False

    # -------------------------
    # task / label inference
    # -------------------------
    def _is_regression_from_meta(self, data: Datasets) -> bool:
        meta = data.meta

        # dict meta
        if isinstance(meta, dict):
            if "task" not in meta:
                return False
            task = meta["task"]
            t = str(task).lower()
            if "regress" in t:
                return True
            if "class" in t:
                return False
            raise ValueError(f"Unknown meta.task: {task}")

        # object meta
        if hasattr(meta, "task"):
            task = meta.task
            t = str(task).lower()
            if "regress" in t:
                return True
            if "class" in t:
                return False
            raise ValueError(f"Unknown meta.task: {task}")

        return False

    def _infer_num_class_from_y(self, y_tr: np.ndarray, y_val: np.ndarray) -> int:
        y_cat = np.concatenate([y_tr, y_val], axis=0)
        uniq = np.unique(y_cat)
        assert uniq.ndim == 1
        assert uniq.size >= 2, f"num_class must be >= 2, got uniq={uniq}"

        # 분류 라벨은 0..C-1 (연속) 강제
        mn = int(uniq.min())
        mx = int(uniq.max())
        assert mn == 0, f"class labels must start at 0. got min={mn}, uniq={uniq}"
        assert mx == int(
            uniq.size - 1
        ), f"class labels must be contiguous 0..C-1. got max={mx}, size={uniq.size}, uniq={uniq}"
        return int(uniq.size)

    def _resolve_xgb_params_auto(
        self, data: Datasets, y_tr: np.ndarray, y_val: np.ndarray
    ) -> tuple[str, str, int | None]:
        """
        returns: (objective, eval_metric, num_class_or_none)
        """
        is_reg = self._is_regression_from_meta(data)
        if is_reg:
            return "reg:squarederror", "rmse", None

        # num_class는 (meta가 틀릴 수 있으므로) y 기반을 우선
        num_class_y = self._infer_num_class_from_y(y_tr, y_val)

        if num_class_y == 2:
            return "binary:logistic", "logloss", None
        return "multi:softprob", "mlogloss", num_class_y

    def _resolve_from_config_or_auto(
        self,
        data: Datasets,
        y_tr: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[str, str, int | None]:
        auto_objective, auto_eval_metric, auto_num_class = (
            self._resolve_xgb_params_auto(data, y_tr, y_val)
        )

        objective_cfg = self.config.model.objective
        eval_metric_cfg = self.config.model.eval_metric

        objective = auto_objective if objective_cfg == "auto" else objective_cfg
        eval_metric = auto_eval_metric if eval_metric_cfg == "auto" else eval_metric_cfg

        # 최종 objective 기준으로 num_class 전달 여부 결정 + 정합성 체크
        if objective in ("multi:softprob", "multi:softmax"):
            if auto_num_class is None:
                raise ValueError(
                    f"objective={objective} requires num_class, but auto_num_class is None"
                )
            if auto_num_class <= 2:
                raise ValueError(
                    f"objective={objective} requires num_class >= 3, got {auto_num_class}"
                )
            return objective, eval_metric, auto_num_class

        if objective == "binary:logistic":
            # binary인데 라벨이 3클래스 이상이면 즉시 오류로 원인 드러내기
            if auto_num_class is not None and auto_num_class > 2:
                raise ValueError(
                    f"objective=binary:logistic but detected num_class={auto_num_class}"
                )
            return objective, eval_metric, None

        if objective.startswith("reg:"):
            return objective, eval_metric, None

        # 그 외 objective는 여기서 명확히 막아두는 편이 디버깅에 유리
        raise ValueError(f"Unsupported objective: {objective}")

    # -------------------------
    # bemv -> NaN restore
    # -------------------------
    def _apply_nan_from_bemv(self, X: np.ndarray, bemv: np.ndarray) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        assert isinstance(bemv, np.ndarray)
        assert X.shape == bemv.shape, f"X.shape={X.shape}, bemv.shape={bemv.shape}"

        if X.dtype.kind != "f":
            X = X.astype(np.float32)

        X_nan = X.copy()

        # bemv: 1=값 있음, 0=값 없음(결측) -> 0 위치만 NaN
        miss_mask = bemv == 0
        X_nan[miss_mask] = np.nan
        return X_nan

    def _stack_gbdt_xy_with_nan(self, data: Datasets) -> tuple[np.ndarray, np.ndarray]:
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []

        for pattern in self.config.data.missing_patterns:
            pattern_v = pattern.value
            ratio_dict = data.imputed_dict[pattern_v]

            for ratio in data.ratios:
                d = ratio_dict[ratio]
                X_imp = d["X"]  # (N, F)
                y = d["y"]  # (N,)
                bemv = d["bemv"]  # (N, F)

                X_nan = self._apply_nan_from_bemv(X_imp, bemv)
                X_list.append(X_nan)
                y_list.append(y)

        X_cat = np.concatenate(X_list, axis=0)
        y_cat = np.concatenate(y_list, axis=0)
        return X_cat, y_cat

    def _get_ratio_xy_with_nan(
        self, data: Datasets, pattern_v: str, ratio: float
    ) -> tuple[np.ndarray, np.ndarray]:
        d = data.imputed_dict[pattern_v][ratio]
        X_imp = d["X"]
        y = d["y"]
        bemv = d["bemv"]
        X_nan = self._apply_nan_from_bemv(X_imp, bemv)
        return X_nan, y

    # -------------------------
    # train / eval
    # -------------------------
    def fit(self, train_data: Datasets, valid_data: Datasets):
        X_tr, y_tr = self._stack_gbdt_xy_with_nan(train_data)
        X_val, y_val = self._stack_gbdt_xy_with_nan(valid_data)

        self.is_regression = self._is_regression_from_meta(train_data)

        objective, eval_metric, num_class_param = self._resolve_from_config_or_auto(
            train_data, y_tr, y_val
        )

        if self.is_regression:
            print("[XGBoostAdapter] task: regression")
            print(f"[XGBoostAdapter] objective={objective}, eval_metric={eval_metric}")

            model = XGBRegressor(
                n_estimators=self.config.train.tree_est,
                objective=objective,
                random_state=self.config.train.seed,
                eval_metric=eval_metric,
                early_stopping_rounds=self.config.train.early_stopping_rounds,
                device=self.config.train.device,
                tree_method=self.config.train.tree_method,
                missing=np.nan,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_tr, y_tr), (X_val, y_val)],
            )

            self.model = model

            ev = model.evals_result()
            train_vals = ev["validation_0"][eval_metric]
            valid_vals = ev["validation_1"][eval_metric]

            y_tr_pred = model.predict(X_tr)
            y_val_pred = model.predict(X_val)

            train_metrics = compute_regression_metrics(y_tr, y_tr_pred)
            valid_metrics = compute_regression_metrics(y_val, y_val_pred)

            return {
                "split": Split.TRAIN.value,
                "task": "regression",
                f"{Split.TRAIN.value}_metrics": train_metrics,
                f"{Split.VALID.value}_metrics": valid_metrics,
                "loss": {
                    "metric_name": eval_metric,
                    "tasks": [
                        {Split.TRAIN.value: train_vals, Split.VALID.value: valid_vals}
                    ],
                },
            }

        # -----------------------------
        # classification: binary/multiclass (auto)
        # -----------------------------
        num_class_y = self._infer_num_class_from_y(y_tr, y_val)

        print("[XGBoostAdapter] task: classification")
        print(f"[XGBoostAdapter] num_class_from_y: {num_class_y}")
        print(f"[XGBoostAdapter] objective={objective}, eval_metric={eval_metric}")

        xgb_kwargs = dict(
            n_estimators=self.config.train.tree_est,
            objective=objective,
            random_state=self.config.train.seed,
            eval_metric=eval_metric,
            early_stopping_rounds=self.config.train.early_stopping_rounds,
            device=self.config.train.device,
            tree_method=self.config.train.tree_method,
            missing=np.nan,
        )

        if num_class_param is not None:
            xgb_kwargs["num_class"] = num_class_param

        model = XGBClassifier(**xgb_kwargs)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
        )

        self.model = model

        ev = model.evals_result()
        train_vals = ev["validation_0"][eval_metric]
        valid_vals = ev["validation_1"][eval_metric]

        y_tr_pred = model.predict(X_tr)
        y_val_pred = model.predict(X_val)

        train_metrics = compute_classification_metrics(y_tr, y_tr_pred)
        valid_metrics = compute_classification_metrics(y_val, y_val_pred)

        return {
            "split": Split.TRAIN.value,
            "task": "classification",
            f"{Split.TRAIN.value}_metrics": train_metrics,
            f"{Split.VALID.value}_metrics": valid_metrics,
            "loss": {
                "metric_name": eval_metric,
                "tasks": [
                    {Split.TRAIN.value: train_vals, Split.VALID.value: valid_vals}
                ],
            },
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")
        return self.model.predict(X)

    def test(self, test_data: Datasets):
        if self.model is None:
            raise ValueError("모델이 학습되거나 로드되지 않았습니다.")

        X_all, y_all = self._stack_gbdt_xy_with_nan(test_data)
        y_pred_all = self.predict(X_all)

        if self.is_regression:
            metrics_overall = compute_regression_metrics(y_all, y_pred_all)
        else:
            metrics_overall = compute_classification_metrics(y_all, y_pred_all)

        by_ratio: dict[str, dict[float, dict]] = {}

        for pattern in self.config.data.missing_patterns:
            p_v = pattern.value
            by_ratio[p_v] = {}

            for ratio in test_data.ratios:
                X, y = self._get_ratio_xy_with_nan(test_data, p_v, ratio)
                y_pred = self.predict(X)

                if self.is_regression:
                    m = compute_regression_metrics(y, y_pred)
                else:
                    m = compute_classification_metrics(y, y_pred)

                by_ratio[p_v][ratio] = m

        return {
            "split": Split.TEST.value,
            "task": "regression" if self.is_regression else "classification",
            "metrics_overall": metrics_overall,
            "metrics_by_ratio": by_ratio,
        }

    # -------------------------
    # save / load
    # -------------------------
    def save(self, path: Path):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")

        save_dir = path / "save"
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "xgb_model.json"
        self.model.save_model(model_path)

        meta = {
            "model_path": str(model_path),
            "task": "regression" if self.is_regression else "classification",
        }
        self.save_meta(save_dir, meta)

    def load(self, path: Path):
        save_dir = path / Split.TRAIN.value / "save"
        meta = self.load_meta(save_dir)

        model_path = meta["model_path"]

        if "task" in meta:
            task = meta["task"]
        else:
            task = "classification"

        if task == "regression":
            model = XGBRegressor()
            self.is_regression = True
        elif task == "classification":
            model = XGBClassifier()
            self.is_regression = False
        else:
            raise ValueError(f"Unknown saved task: {task}")

        model.load_model(model_path)
        self.model = model
        return True
