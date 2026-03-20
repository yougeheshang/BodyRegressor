from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


@dataclass
class XGBConfig:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.1
    objective: str = "reg:squarederror"
    n_jobs: int = -1
    random_state: int = 42


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    return StandardScaler().fit(X_train)


def transform_with_scaler(
    scaler: StandardScaler,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    return tuple(scaler.transform(a) for a in arrays)


class XGBSingleTargetRegressor:
    def __init__(self, config: Optional[XGBConfig] = None) -> None:
        self.config = config or XGBConfig()
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBSingleTargetRegressor":
        params: dict[str, Any] = asdict(self.config)
        model = xgb.XGBRegressor(**params)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        return self.model.predict(X).astype(np.float32)


class XGBMultiTargetRegressor:
    def __init__(
        self,
        config: Optional[XGBConfig] = None,
        target_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.config = config or XGBConfig()
        self.target_names = list(target_names) if target_names is not None else []
        self.models: list[xgb.XGBRegressor] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBMultiTargetRegressor":
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        if y_train.ndim == 1:
            y_train = y_train[:, None]
        n_targets = y_train.shape[1]
        if not self.target_names or len(self.target_names) != n_targets:
            self.target_names = [str(i) for i in range(n_targets)]

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            if y_val.ndim == 1:
                y_val = y_val[:, None]
        else:
            X_val, y_val = None, None

        params: dict[str, Any] = asdict(self.config)
        self.models = []
        for i in range(n_targets):
            model = xgb.XGBRegressor(**params)
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val[:, i])]
            model.fit(X_train, y_train[:, i], eval_set=eval_set, verbose=False)
            self.models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("模型尚未训练")
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.astype(np.float32)


def save_checkpoint(
    path: Union[Path, str],
    *,
    models: Any,
    scaler_x: StandardScaler,
    feature_cols: list[str],
    target_cols: list[str],
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "models": models,
        "scaler_x_mean": scaler_x.mean_,
        "scaler_x_scale": scaler_x.scale_,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }
    if extra:
        payload.update(extra)
    joblib.dump(payload, p)
    return p


def load_checkpoint(
    path: Union[Path, str],
) -> tuple[Any, StandardScaler, list[str], list[str], dict[str, Any]]:
    obj: dict[str, Any] = joblib.load(Path(path))
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(obj["scaler_x_mean"])
    scaler.scale_ = np.asarray(obj["scaler_x_scale"])
    feature_cols = list(obj["feature_cols"])
    target_cols = list(obj["target_cols"])
    extra = {k: v for k, v in obj.items() if k not in {"models", "scaler_x_mean", "scaler_x_scale", "feature_cols", "target_cols"}}
    return obj["models"], scaler, feature_cols, target_cols, extra

