from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .data import split_train_val_test
from .features import (
    FAT_TARGET_COLS,
    WEIGHT_HAT_COL,
    WeightMode,
    df_to_xy_fat,
    df_to_xy_weight,
)
from .metrics import fat_units_for_targets, format_summary, summarize
from .models import (
    XGBConfig,
    XGBMultiTargetRegressor,
    XGBSingleTargetRegressor,
    fit_scaler,
    save_checkpoint,
    transform_with_scaler,
)


def train_weight_oof(
    df: pd.DataFrame,
    *,
    k_folds: int = 5,
    config: Optional[XGBConfig] = None,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, object]]:
    X, y, feat_cols = df_to_xy_weight(df)
    cfg = config or XGBConfig()
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    oof = np.zeros(len(df), dtype=np.float32)
    for train_idx, val_idx in kf.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        scaler = fit_scaler(X_tr)
        X_tr_s, X_va_s = transform_with_scaler(scaler, X_tr, X_va)

        m = XGBSingleTargetRegressor(cfg).fit(X_tr_s, y_tr, X_va_s, y_va)
        oof[val_idx] = m.predict(X_va_s)

    meta = {"feature_cols": feat_cols, "k_folds": k_folds, "random_state": random_state, "config": cfg}
    return oof, meta


def train_weight_final(
    df: pd.DataFrame,
    *,
    config: Optional[XGBConfig] = None,
) -> tuple[XGBSingleTargetRegressor, object, list[str]]:
    X, y, feat_cols = df_to_xy_weight(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(X, y, test_ratio=0.1, val_ratio=0.2)
    cfg = config or XGBConfig()

    scaler = fit_scaler(X_tr)
    X_tr_s, X_va_s, X_te_s = transform_with_scaler(scaler, X_tr, X_va, X_te)

    model = XGBSingleTargetRegressor(cfg).fit(X_tr_s, y_tr, X_va_s, y_va)
    preds = model.predict(X_te_s)
    report = summarize(preds, y_te, target_names=["Weight"], units={"Weight": "kg"})
    print(format_summary(report, title="Test set evaluation:"))
    return model, scaler, feat_cols


def train_fat(
    df: pd.DataFrame,
    *,
    weight_mode: WeightMode,
    config: Optional[XGBConfig] = None,
    weight_hat_col: str = WEIGHT_HAT_COL,
) -> tuple[XGBMultiTargetRegressor, object, list[str], list[str]]:
    X, y, feat_cols, tgt_cols = df_to_xy_fat(df, weight_mode=weight_mode, weight_hat_col=weight_hat_col)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(X, y, test_ratio=0.1, val_ratio=0.2)

    cfg = config or XGBConfig()
    scaler = fit_scaler(X_tr)
    X_tr_s, X_va_s, X_te_s = transform_with_scaler(scaler, X_tr, X_va, X_te)

    model = XGBMultiTargetRegressor(cfg, target_names=tgt_cols).fit(X_tr_s, y_tr, X_va_s, y_va)
    preds = model.predict(X_te_s)
    units = fat_units_for_targets(tgt_cols)
    report = summarize(preds, y_te, target_names=tgt_cols, units=units)
    print(format_summary(report, title="Test set evaluation:", target_order=tgt_cols))
    return model, scaler, feat_cols, tgt_cols


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["weight", "fat_full", "fat_stacked"], required=True)
    p.add_argument("--csv", default="data/nhanes/processed/nhanes_merged.csv")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--k-folds", type=int, default=5, help="仅用于 fat_stacked（OOF 生成 Weight_hat）")

    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.05)

    args = p.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = XGBConfig(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.lr)

    if args.task == "weight":
        model, scaler, feat_cols = train_weight_final(df, config=cfg)
        save_checkpoint(
            out_dir / "weight_xgb.joblib",
            models=model.model,
            scaler_x=scaler,
            feature_cols=feat_cols,
            target_cols=["Weight"],
            extra={"config": cfg.__dict__},
        )
        return

    if args.task == "fat_full":
        model, scaler, feat_cols, tgt_cols = train_fat(df, weight_mode=WeightMode.REAL, config=cfg)
        save_checkpoint(
            out_dir / "fat_xgb_full.joblib",
            models=model.models,
            scaler_x=scaler,
            feature_cols=feat_cols,
            target_cols=tgt_cols,
            extra={"config": cfg.__dict__, "weight_mode": WeightMode.REAL.value},
        )
        return

    if args.task == "fat_stacked":
        oof, _ = train_weight_oof(df, k_folds=args.k_folds, config=cfg)
        df_hat = df.copy()
        df_hat[WEIGHT_HAT_COL] = oof
        df_hat.to_csv(out_dir / "train_with_weight_hat.csv", index=False)
        model, scaler, feat_cols, tgt_cols = train_fat(df_hat, weight_mode=WeightMode.HAT, config=cfg)
        save_checkpoint(
            out_dir / "fat_xgb_stacked.joblib",
            models=model.models,
            scaler_x=scaler,
            feature_cols=feat_cols,
            target_cols=tgt_cols,
            extra={"config": cfg.__dict__, "weight_mode": WeightMode.HAT.value, "weight_hat_col": WEIGHT_HAT_COL},
        )
        return

    raise RuntimeError(f"未知 task: {args.task}")


if __name__ == "__main__":
    main()

