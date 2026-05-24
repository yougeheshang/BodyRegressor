from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from BodyRegressor.features import WeightMode, compute_bmi, compute_bri
from BodyRegressor.image_to_smpl import default_spin_dir, infer_smpl_vertices_from_image, resolve_smpl_gender
from BodyRegressor.mesh_predict import mesh_to_fat_row
from BodyRegressor.metrics import fat_units_for_targets, format_summary, summarize_predictions
from BodyRegressor.predict import predict_weight
from BodyRegressor.cli.llm_args import add_llm_arguments
from BodyRegressor.health_report import build_health_report, format_report_markdown
from BodyRegressor.health_report.pipeline import report_input_from_predictions
from BodyRegressor.smpl_measure import default_anthro_dir, measure_smpl_verts, measure_smpl_verts_height_normalized


def main() -> None:
    p = argparse.ArgumentParser(description="RGB 图像 -> SPIN(SMPL) -> 量测 -> 体脂预测")
    p.add_argument("--img", type=Path, required=True, help="输入图像路径")
    p.add_argument("--age", type=float, required=True)
    p.add_argument("--sex", type=int, choices=[1, 2], required=True, help="1=Male, 2=Female")

    p.add_argument("--bbox", type=Path, default=None, help="可选 bbox json，格式同 SPIN demo")
    p.add_argument("--openpose", type=Path, default=None, help="可选 openpose json")
    p.add_argument("--spin-dir", type=Path, default=None, help="SPIN 根目录（默认 third-party/SPIN）")
    p.add_argument("--spin-ckpt", type=Path, default=Path("third-party/SPIN/data/model_checkpoint.pt"))
    p.add_argument("--device", type=str, default=None, help="例如 cuda / cpu，默认自动选择")
    p.add_argument("--save-verts", type=Path, default=None, help="可选：保存 SPIN 输出 verts(.npy)")
    p.add_argument(
        "--smpl-gender",
        choices=["by_sex", "neutral", "male", "female"],
        default="by_sex",
        help="SMPL 模板：by_sex 时与 --sex 一致（1=male, 2=female）；neutral 与原版 SPIN demo 一致",
    )

    p.add_argument("--anthro-dir", type=Path, default=None, help="SMPL-Anthropometry 根目录")
    p.add_argument(
        "--height-cm",
        type=float,
        default=None,
        help="可选：真实身高(cm)，将按 Anthropometry height_normalize 比例缩放 mesh 并用于量测",
    )
    p.add_argument("--weight-mode", choices=[m.value for m in WeightMode], default=WeightMode.HAT.value)
    p.add_argument("--weight-ckpt", type=Path, default=Path("models/weight_xgb.joblib"))
    p.add_argument("--fat-ckpt-stacked", type=Path, default=Path("models/fat_xgb_stacked.joblib"))
    p.add_argument("--fat-ckpt-full", type=Path, default=Path("models/fat_xgb_full.joblib"))
    p.add_argument("--weight-kg", type=float, default=None, help="仅 weight-mode=real 时使用")
    p.add_argument(
        "--health-report",
        action="store_true",
        help="在预测后生成健康报告（规则 + 知识库检索，可选 LLM）",
    )
    p.add_argument("--health-report-out", type=Path, default=None, help="报告 Markdown 输出路径")
    p.add_argument(
        "--health-report-no-llm",
        action="store_true",
        help="健康报告不调用大模型，仅规则结论 + 知识库摘录",
    )
    add_llm_arguments(p)
    args = p.parse_args()

    spin_dir = args.spin_dir if args.spin_dir is not None else default_spin_dir()
    anthro_dir = args.anthro_dir if args.anthro_dir is not None else default_anthro_dir()
    mode = WeightMode(args.weight_mode)

    smpl_gender_used = resolve_smpl_gender(mode=args.smpl_gender, sex=args.sex)
    print(f"SMPL gender: {smpl_gender_used} (--smpl-gender={args.smpl_gender}, --sex={args.sex})")

    verts = infer_smpl_vertices_from_image(
        image_path=args.img,
        checkpoint=args.spin_ckpt,
        spin_dir=spin_dir,
        bbox_path=args.bbox,
        openpose_path=args.openpose,
        device=args.device,
        smpl_gender=args.smpl_gender,
        sex=args.sex,
    )

    if args.height_cm is not None:
        verts, meas, _hn_meta = measure_smpl_verts_height_normalized(
            verts,
            target_height_cm=float(args.height_cm),
            anthro_dir=anthro_dir,
        )
    else:
        meas = measure_smpl_verts(verts, anthro_dir=anthro_dir)

    if args.save_verts is not None:
        args.save_verts.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_verts, verts.astype(np.float32))

    weight_used_kg: float
    weight_hat_kg: float | None = None
    if mode == WeightMode.REAL:
        if args.weight_kg is None:
            raise SystemExit("--weight-mode=real 时必须提供 --weight-kg")
        weight_used_kg = float(args.weight_kg)
    else:
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

    bmi = float(compute_bmi(np.float32(weight_used_kg), np.float32(meas["Height"])))
    bri = float(compute_bri(np.float32(meas["Waist"]), np.float32(meas["Height"])).round(1))

    sex_str = "male" if args.sex == 1 else "female"
    print("Features:")
    print(f"Age: {args.age}")
    print(f"Sex: {sex_str} ({args.sex})")
    print(f"Height: {meas['Height']:.2f}cm")
    print(f"Weight: {weight_used_kg:.2f}kg")
    print(f"Waist: {meas['Waist']:.2f}cm")
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

    if args.health_report:
        inp = report_input_from_predictions(
            age=args.age,
            sex=args.sex,
            height_cm=float(meas["Height"]),
            weight_kg=weight_used_kg,
            waist_cm=float(meas["Waist"]),
            hip_cm=float(meas["Hip"]),
            bmi=bmi,
            bri=bri,
            preds=preds,
            target_names=tgt_cols,
        )
        hr = build_health_report(
            inp,
            use_llm=not args.health_report_no_llm,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
        )
        md = format_report_markdown(hr)
        if args.health_report_out:
            args.health_report_out.parent.mkdir(parents=True, exist_ok=True)
            args.health_report_out.write_text(md, encoding="utf-8")
            print(f"\nHealth report written to {args.health_report_out}")
        else:
            print("\n" + md)


if __name__ == "__main__":
    main()

