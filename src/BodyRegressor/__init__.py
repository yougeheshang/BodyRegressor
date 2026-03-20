"""BodyRegressor: 从人体测量/mesh 特征预测体脂指标。"""

from .features import (
    FAT_TARGET_COLS,
    WeightMode,
)
from .models import (
    XGBConfig,
    XGBMultiTargetRegressor,
    XGBSingleTargetRegressor,
)


__all__ = [
    "WeightMode",
    "FAT_TARGET_COLS",
    "XGBConfig",
    "XGBSingleTargetRegressor",
    "XGBMultiTargetRegressor",
]

