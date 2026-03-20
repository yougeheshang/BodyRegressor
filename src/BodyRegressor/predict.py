from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import WeightMode, df_to_xy_fat, df_to_xy_weight, WEIGHT_HAT_COL
from .models import (
    XGBMultiTargetRegressor,
    XGBSingleTargetRegressor,
    load_checkpoint,
    transform_with_scaler,
)


def predict_weight(df: pd.DataFrame, weight_ckpt: Path) -> np.ndarray:
    model_obj, scaler, feat_cols, _, _ = load_checkpoint(weight_ckpt)
    wrapper = XGBSingleTargetRegressor()
    wrapper.model = model_obj
    X, _, _ = df_to_xy_weight(df, feature_cols=feat_cols)
    (X_s,) = transform_with_scaler(scaler, X)
    return wrapper.predict(X_s)


def predict_fat(df: pd.DataFrame, fat_ckpt: Path, *, weight_mode: WeightMode) -> tuple[np.ndarray, list[str]]:
    model_obj, scaler, feat_cols, tgt_cols, _ = load_checkpoint(fat_ckpt)
    wrapper = XGBMultiTargetRegressor(target_names=tgt_cols)
    wrapper.models = model_obj
    X, _, _, _ = df_to_xy_fat(df, weight_mode=weight_mode)
    (X_s,) = transform_with_scaler(scaler, X)
    preds = wrapper.predict(X_s)
    return preds, tgt_cols


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-csv", help="批量预测输入 CSV（需包含所需列）")
    src.add_argument("--one", action="store_true", help="单样本预测（从命令行参数读取）")

    p.add_argument("--out-csv", default="../../output/preds.csv", help="批量模式输出；单样本模式将打印结果")

    p.add_argument("--weight-mode", choices=[m.value for m in WeightMode], default=WeightMode.HAT.value)
    p.add_argument("--weight-ckpt", default="models/weight_xgb.joblib")
    p.add_argument("--fat-ckpt-full", default="models/fat_xgb_full.joblib")
    p.add_argument("--fat-ckpt-stacked", default="models/fat_xgb_stacked.joblib")

    # 单样本输入（与 NHANES/SMPL 输出列名一致）
    p.add_argument("--age", type=float)
    p.add_argument("--sex", type=int, choices=[1, 2], help="1=Male, 2=Female")
    p.add_argument("--height", type=float, help="cm")
    p.add_argument("--waist", type=float, help="cm")
    p.add_argument("--hip", type=float, help="cm")
    p.add_argument("--weight", type=float, help="kg（仅 weight-mode=real 时需要）")
    args = p.parse_args()

    if args.one:
        required = ["age", "sex", "height", "waist", "hip"]
        missing = [k for k in required if getattr(args, k) is None]
        if missing:
            raise SystemExit(f"--one 模式缺少参数: {missing}")
        row = {
            "Age": args.age,
            "Sex": args.sex,
            "Height": args.height,
            "Waist": args.waist,
            "Hip": args.hip,
        }
        if args.weight is not None:
            row["Weight"] = args.weight
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(args.input_csv)
    mode = WeightMode(args.weight_mode)

    if mode == WeightMode.HAT:
        weight_hat = predict_weight(df, Path(args.weight_ckpt))
        df = df.copy()
        df[WEIGHT_HAT_COL] = weight_hat
        fat_ckpt = Path(args.fat_ckpt_stacked)
    else:
        fat_ckpt = Path(args.fat_ckpt_full)

    preds, tgt_cols = predict_fat(df, fat_ckpt, weight_mode=mode)
    out = pd.DataFrame(preds, columns=tgt_cols)
    if args.one:
        print(out.iloc[0].to_dict())
    else:
        out.to_csv(args.out_csv, index=False)
        print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()

