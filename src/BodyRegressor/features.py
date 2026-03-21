from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class WeightMode(str, Enum):
    REAL = "real"   # 使用真实 Weight（训练/可选推理）
    HAT = "hat"     # 使用 Weight_hat（stacking 二阶段）


FAT_TARGET_COLS = [
    "Total_fat_mass",
    "Percentage_body_fat",
    "Android_fat_mass",
    "Gynoid_fat_mass",
    "Subcutaneous_fat_mass",
    "Total_abdominal_fat_mass",
    "Visceral_adipose_tissue_mass",
    "Peripheral_fat_mass",
]

WEIGHT_TARGET_COL = "Weight"
WEIGHT_HAT_COL = "Weight_hat"


def encode_sex(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = df if inplace else df.copy()
    out["Sex"] = out["Sex"].map({1: 0, 2: 1})
    return out


def compute_bmi(weight_kg: pd.Series, height_cm: pd.Series) -> pd.Series:
    h_m = height_cm / 100.0
    return weight_kg / (h_m**2)


def compute_bri(waist_cm: pd.Series, height_cm: pd.Series) -> pd.Series:
    r = waist_cm / (2 * np.pi)
    h_half = 0.5 * height_cm
    return 364.2 - 365.5 * np.sqrt(1 - (r / h_half) ** 2)


def build_feature_frame(
    df: pd.DataFrame,
    *,
    weight_mode: WeightMode = WeightMode.REAL,
    weight_hat_col: str = WEIGHT_HAT_COL,
    encode_sex_col: bool = True,
    ensure_derived: bool = True,
) -> pd.DataFrame:
    work = encode_sex(df.copy(), inplace=True) if encode_sex_col else df.copy()

    if weight_mode == WeightMode.REAL:
        if WEIGHT_TARGET_COL not in work.columns:
            raise ValueError("weight_mode=real 但 df 中没有 Weight 列")
        weight_series = work[WEIGHT_TARGET_COL]
    elif weight_mode == WeightMode.HAT:
        if weight_hat_col not in work.columns:
            raise ValueError(f"weight_mode=hat 但 df 中没有 {weight_hat_col} 列")
        weight_series = work[weight_hat_col]
    else:
        raise ValueError(f"未知 weight_mode: {weight_mode}")

    if ensure_derived:
        if "BMI" not in work.columns or weight_mode == WeightMode.HAT:
            work["BMI"] = compute_bmi(weight_series, work["Height"])
        if "BRI" not in work.columns:
            work["BRI"] = compute_bri(work["Waist"], work["Height"]).round(1)

    if weight_mode == WeightMode.HAT:
        work["Weight_used"] = weight_series
    return work


def fat_feature_cols(weight_mode: WeightMode) -> list[str]:
    if weight_mode == WeightMode.REAL:
        return ["Age", "Sex", "Height", "Weight", "Waist", "Hip", "BMI", "BRI"]
    if weight_mode == WeightMode.HAT:
        return ["Age", "Sex", "Height", "Weight_used", "Waist", "Hip", "BMI", "BRI"]
    raise ValueError(f"未知 weight_mode: {weight_mode}")


def df_to_x_fat(
    df: pd.DataFrame,
    *,
    weight_mode: WeightMode,
    weight_hat_col: str = WEIGHT_HAT_COL,
    feature_cols: Optional[list[str]] = None,
    encode_sex_col: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """仅构建体脂预测特征 X（推理阶段无真实目标列）。"""
    feat_df = build_feature_frame(
        df,
        weight_mode=weight_mode,
        weight_hat_col=weight_hat_col,
        encode_sex_col=encode_sex_col,
    )
    feats = feature_cols or fat_feature_cols(weight_mode)
    X = feat_df[feats].values.astype(np.float32)
    return X, feats


def df_to_xy_fat(
    df: pd.DataFrame,
    *,
    weight_mode: WeightMode = WeightMode.REAL,
    weight_hat_col: str = WEIGHT_HAT_COL,
    target_cols: Optional[list[str]] = None,
    encode_sex_col: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    targets = target_cols or FAT_TARGET_COLS
    feat_df = build_feature_frame(
        df,
        weight_mode=weight_mode,
        weight_hat_col=weight_hat_col,
        encode_sex_col=encode_sex_col,
    )
    feats = fat_feature_cols(weight_mode)
    X = feat_df[feats].values.astype(np.float32)
    y = feat_df[targets].values.astype(np.float32)
    return X, y, feats, targets


def df_to_x_weight(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    encode_sex_col: bool = True,
) -> tuple[np.ndarray, list[str]]:
    cols = feature_cols or ["Age", "Sex", "Height", "Waist", "Hip"]
    work = encode_sex(df.copy(), inplace=True) if encode_sex_col else df.copy()
    X = work[cols].values.astype(np.float32)
    return X, cols


def df_to_xy_weight(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[list[str]] = None,
    encode_sex_col: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = feature_cols or ["Age", "Sex", "Height", "Waist", "Hip"]
    work = encode_sex(df.copy(), inplace=True) if encode_sex_col else df.copy()
    X = work[cols].values.astype(np.float32)
    y = work[WEIGHT_TARGET_COL].values.astype(np.float32)
    return X, y, cols

