"""
SMPL 标准 verts -> SMPL-Anthropometry 量测 -> 体脂等指标预测（默认 stacking: Weight_hat）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .features import WEIGHT_HAT_COL, WeightMode, compute_bmi, compute_bri
from .metrics import fat_units_for_targets, format_summary, summarize_predictions
from .predict import predict_fat, predict_weight
from .smpl_measure import default_anthro_dir, load_verts_from_file, measure_smpl_verts


def neutral_smpl_verts_for_demo(*, anthro_dir: Path) -> np.ndarray:
    """生成中性 mean shape 的 SMPL 顶点 (6890,3)，用于无 verts 文件时的快速联调。"""
    import os

    anthro_dir = Path(anthro_dir).resolve()
    anthro_str = str(anthro_dir)
    if anthro_str not in sys.path:
        sys.path.insert(0, anthro_str)

    import torch

    prev = os.getcwd()
    os.chdir(anthro_dir)
    try:
        from measure import MeasureBody  # type: ignore[import-untyped]

        m = MeasureBody("smpl")
        betas = torch.zeros((1, 10), dtype=torch.float32)
        m.from_body_model(gender="FEMALE", shape=betas)
        return np.asarray(m.verts, dtype=np.float32)
    finally:
        os.chdir(prev)


def mesh_to_fat_row(
    *,
    measurements: dict[str, float],
    age: float,
    sex: int,
    weight_ckpt: Path,
    fat_ckpt_stacked: Path,
    weight_mode: WeightMode = WeightMode.HAT,
    fat_ckpt_full: Path | None = None,
    body_weight_kg: float | None = None,
    weight_hat_kg: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """单样本：量测结果 + 人口学 -> 体脂预测矩阵 (1, D) 与目标名列表。"""
    row = {
        "Age": age,
        "Sex": sex,
        "Height": measurements["Height"],
        "Waist": measurements["Waist"],
        "Hip": measurements["Hip"],
    }
    mode = weight_mode
    if mode == WeightMode.REAL:
        if body_weight_kg is None:
            raise ValueError("weight_mode=real 需要提供 body_weight_kg（真实体重 kg）")
        row["Weight"] = body_weight_kg
        fat_ckpt = fat_ckpt_full or Path("models/fat_xgb_full.joblib")
        df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])
        df = df.copy()
        if weight_hat_kg is None:
            df[WEIGHT_HAT_COL] = predict_weight(df, weight_ckpt)
        else:
            df[WEIGHT_HAT_COL] = np.float32(weight_hat_kg)
        fat_ckpt = fat_ckpt_stacked

    preds, tgt_cols = predict_fat(df, Path(fat_ckpt), weight_mode=mode)
    return preds, tgt_cols


def main() -> None:
    p = argparse.ArgumentParser(description="SMPL verts -> 量测 -> 体脂预测")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--verts", type=Path, help=".npy / .npz / .pt 路径，形状 (6890,3)")
    src.add_argument("--demo", action="store_true", help="使用中性 mean shape SMPL 顶点（需 data/smpl/*.pkl）")

    p.add_argument("--anthro-dir", type=Path, default=None, help="SMPL-Anthropometry 内层目录（默认 third-party/.../SMPL-Anthropometry）")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--sex", type=int, choices=[1, 2], help="1=Male, 2=Female")

    p.add_argument("--weight-mode", choices=[m.value for m in WeightMode], default=WeightMode.HAT.value)
    p.add_argument("--weight-ckpt", type=Path, default=Path("models/weight_xgb.joblib"))
    p.add_argument("--fat-ckpt-stacked", type=Path, default=Path("models/fat_xgb_stacked.joblib"))
    p.add_argument("--fat-ckpt-full", type=Path, default=Path("models/fat_xgb_full.joblib"))
    p.add_argument("--weight-kg", type=float, default=None, help="仅 weight-mode=real：真实体重（kg）")

    args = p.parse_args()

    anthro = args.anthro_dir if args.anthro_dir is not None else default_anthro_dir()

    if args.demo:
        verts = neutral_smpl_verts_for_demo(anthro_dir=anthro)
    else:
        verts = load_verts_from_file(args.verts)

    meas = measure_smpl_verts(verts, anthro_dir=anthro)
    mode = WeightMode(args.weight_mode)

    sex_str = "male" if args.sex == 1 else "female"

    weight_used_kg: float
    weight_hat_kg: float | None = None
    if mode == WeightMode.REAL:
        if args.weight_kg is None:
            raise SystemExit("--weight-kg 必填：weight-mode=real")
        weight_used_kg = float(args.weight_kg)
    else:
        # 用 weight_xgb 预测 Weight_hat，供 Features 计算 BMI（推理时无需真实 Weight）
        tmp_df = pd.DataFrame(
            [
                {
                    "Age": args.age,
                    "Sex": args.sex,
                    "Height": meas["Height"],
                    "Waist": meas["Waist"],
                    "Hip": meas["Hip"],
                }
            ]
        )
        weight_hat_arr = predict_weight(tmp_df, args.weight_ckpt)
        weight_hat_kg = float(weight_hat_arr[0])
        weight_used_kg = weight_hat_kg

    # BMI/BRI（与 features.py 同公式）
    height_cm = meas["Height"]
    waist_cm = meas["Waist"]
    bmi = float(compute_bmi(np.float32(weight_used_kg), np.float32(height_cm)))
    bri = float(compute_bri(np.float32(waist_cm), np.float32(height_cm)).round(1))

    print("Features:")
    print(f"Age: {args.age}")
    print(f"Sex: {sex_str} ({args.sex})")
    print(f"Height: {height_cm:.2f}cm")
    print(f"Weight: {weight_used_kg:.2f}kg")
    print(f"Waist: {waist_cm:.2f}cm")
    print(f"Hip: {meas['Hip']:.2f}cm")
    print(f"BMI: {bmi:.2f}")
    print(f"BRI: {bri:.2f}")



    preds, tgt_cols = mesh_to_fat_row(
        measurements=meas,
        age=args.age,
        sex=args.sex,
        weight_ckpt=args.weight_ckpt,
        fat_ckpt_stacked=args.fat_ckpt_stacked,
        weight_mode=mode,
        fat_ckpt_full=args.fat_ckpt_full,
        body_weight_kg=args.weight_kg,
        weight_hat_kg=weight_hat_kg,
    )
    units = fat_units_for_targets(tgt_cols)
    report = summarize_predictions(preds, tgt_cols, units=units)
    print(format_summary(report, title="Prediction:", target_order=tgt_cols, style="predict"))


if __name__ == "__main__":
    main()
