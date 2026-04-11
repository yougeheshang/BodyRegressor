from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st 
import numpy as np
from datasets import load_dataset
from PIL import Image

# Force PyOpenGL to use OSMesa backend (must be set before any OpenGL import).
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# Ensure `import BodyRegressor` works when running `streamlit run .../celeb_fbi_viewer.py`
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from BodyRegressor.features import WeightMode, compute_bmi, compute_bri
from BodyRegressor.image_to_smpl import infer_smpl_vertices_and_overlay_from_image
from BodyRegressor.mesh_predict import mesh_to_fat_row
from BodyRegressor.predict import predict_weight
from BodyRegressor.smpl_measure import default_anthro_dir, measure_smpl_verts


@st.cache_data(show_spinner=False)
def _load_ds(parquet_path: str):
    parquet_path = str(Path(parquet_path).resolve())
    ds = load_dataset("parquet", data_files={"test": parquet_path})["test"]
    return ds


def _gender_to_sex(gender_gt: int) -> int:
    # celeb-fbi: 0=male, 1=female ; our pipeline: 1=Male, 2=Female
    return 1 if int(gender_gt) == 0 else 2


def _hf_image_value_to_pil(image: object) -> Image.Image:
    """to_pandas() 后 image 常为 dict(bytes/path)；Dataset 行里常为 PIL。"""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        b = image.get("bytes")
        if b:
            return Image.open(io.BytesIO(b))
        p = image.get("path")
        if p:
            return Image.open(p)
        raise ValueError(f"无法从 image dict 解码: keys={list(image.keys())}")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    raise TypeError(f"不支持的 image 类型: {type(image)}")


def _pil_to_jpg_path(img: Image.Image, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_rgb = img.convert("RGB")
    img_rgb.save(out_path, format="JPEG")
    return out_path


def main() -> None:
    st.set_page_config(page_title="BodyRegressor Viewer", layout="wide")
    st.title("BodyRegressor Viewer (celeb-fbi test)")

    parquet_default = "data/celeb-fbi/0000.parquet"
    parquet_path = st.sidebar.text_input("parquet_path", value=parquet_default)

    device = st.sidebar.selectbox("device", options=["auto", "cpu", "cuda"], index=0)
    device_arg = None if device == "auto" else device

    spin_ckpt = st.sidebar.text_input(
        "spin_ckpt",
        value="third-party/SPIN/data/model_checkpoint.pt",
    )
    spin_dir = st.sidebar.text_input("spin_dir", value="third-party/SPIN")
    anthro_dir = st.sidebar.text_input(
        "anthro_dir",
        value=str(default_anthro_dir()),
    )

    weight_ckpt = st.sidebar.text_input("weight_ckpt", value="models/weight_xgb.joblib")
    fat_ckpt_stacked = st.sidebar.text_input("fat_ckpt_stacked", value="models/fat_xgb_stacked.joblib")
    fat_ckpt_full = st.sidebar.text_input("fat_ckpt_full", value="models/fat_xgb_full.joblib")

    weight_mode = st.sidebar.selectbox("weight_mode", options=[m.value for m in WeightMode], index=1)
    mode = WeightMode(weight_mode)

    use_gt_height_normalize = st.sidebar.checkbox(
        "用真实身高校准 mesh（Anthropometry height-normalize）",
        value=True,
    )

    # We want to match dataset gender: 0/1 -> male/female templates.
    smpl_gender = "neutral"

    ds = _load_ds(parquet_path)
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str)

    id_list = df["id"].tolist()
    selected_id = st.selectbox("Select id", id_list)
    row = df[df["id"] == selected_id].iloc[0]

    col_img, col_metrics = st.columns([1, 1.6])

    # Cache per id during the session to avoid re-running SPIN every click.
    cache_key = f"{selected_id}|{device}|{mode}|{smpl_gender}|{use_gt_height_normalize}"
    if "session_cache" not in st.session_state:
        st.session_state.session_cache = {}

    if cache_key not in st.session_state.session_cache:
        with st.spinner("SPIN 推理 + 量测 + 体脂预测中..."):
            img_pil = _hf_image_value_to_pil(row["image"])
            original_buf = io.BytesIO()
            img_pil.convert("RGB").save(original_buf, format="PNG")
            original_png_bytes = original_buf.getvalue()
            tmp_img_dir = Path("outputs/celeb-fbi-tmp")
            tmp_img_path = tmp_img_dir / f"{selected_id}.jpg"
            tmp_img_path = _pil_to_jpg_path(img_pil, tmp_img_path)

            gender_gt = int(row["gender"])
            sex_for_model = _gender_to_sex(gender_gt)

            age_gt = float(row["age"])
            height_gt = float(row["height"])
            weight_gt = float(row["weight"])

            verts, overlay_img, meas_hn = infer_smpl_vertices_and_overlay_from_image(
                image_path=tmp_img_path,
                checkpoint=Path(spin_ckpt).resolve(),
                spin_dir=Path(spin_dir).resolve(),
                bbox_path=None,
                openpose_path=None,
                device=device_arg,
                smpl_gender=smpl_gender,  # type: ignore[arg-type]
                sex=sex_for_model,
                target_height_cm=height_gt if use_gt_height_normalize else None,
                anthro_dir=Path(anthro_dir).resolve() if use_gt_height_normalize else None,
            )

            if meas_hn is not None:
                meas = meas_hn
            else:
                meas = measure_smpl_verts(verts, anthro_dir=Path(anthro_dir).resolve())

            # Predicted weight (Weight_hat) & fat preds
            tmp_df = pd.DataFrame(
                [
                    {
                        "Age": age_gt,
                        "Sex": sex_for_model,
                        "Height": meas["Height"],
                        "Waist": meas["Waist"],
                        "Hip": meas["Hip"],
                    }
                ]
            )

            weight_hat_kg = None
            if mode == WeightMode.HAT:
                weight_hat_kg = float(predict_weight(tmp_df, Path(weight_ckpt).resolve())[0])
                weight_used_kg = weight_hat_kg
            else:
                # weight_mode=real: use dataset real weight directly
                weight_used_kg = weight_gt

            bmi_pred = float(compute_bmi(np.float32(weight_used_kg), np.float32(meas["Height"])))
            bri_pred = float(compute_bri(np.float32(meas["Waist"]), np.float32(meas["Height"])).round(1))

            preds_fat, tgt_cols = mesh_to_fat_row(
                measurements=meas,
                age=age_gt,
                sex=sex_for_model,
                weight_ckpt=Path(weight_ckpt).resolve(),
                fat_ckpt_stacked=Path(fat_ckpt_stacked).resolve(),
                weight_mode=mode,
                fat_ckpt_full=Path(fat_ckpt_full).resolve(),
                body_weight_kg=(weight_gt if mode == WeightMode.REAL else None),
                weight_hat_kg=weight_hat_kg,
            )

            fat_pred_dict = {
                col: float(preds_fat[0, i]) for i, col in enumerate(list(tgt_cols))
            }

            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            overlay_png_bytes = buf.getvalue()

            st.session_state.session_cache[cache_key] = {
                "overlay_png_bytes": overlay_png_bytes,
                "original_png_bytes": original_png_bytes,
                "gender_gt": gender_gt,
                "age_gt": age_gt,
                "height_gt": height_gt,
                "weight_gt": weight_gt,
                "sex_for_model": sex_for_model,
                "meas": meas,
                "weight_used_kg": weight_used_kg,
                "weight_hat_kg": weight_hat_kg,
                "bmi_pred": bmi_pred,
                "bri_pred": bri_pred,
                "fat_pred_dict": fat_pred_dict,
            }

    result = st.session_state.session_cache[cache_key]

    with col_img:
        overlay_img = Image.open(io.BytesIO(result["overlay_png_bytes"]))
        # 放大到占满当前列宽
        st.image(
            overlay_img,
            caption=f"id={selected_id} (SMPL overlay, cropped)",
            use_container_width=True,
        )

        original_img = Image.open(io.BytesIO(result["original_png_bytes"]))
        st.image(
            original_img,
            caption="Original image (from dataset)",
            use_container_width=True,
        )

    with col_metrics:
        st.markdown("### GT (celeb-fbi)")
        st.write(
            {
                "age": result["age_gt"],
                "gender": result["gender_gt"],  # 0 male, 1 female
                "height_cm": result["height_gt"],
                "weight_kg": result["weight_gt"],
            }
        )

        st.markdown("### Predicted")
        pred_height = float(result["meas"]["Height"])
        pred_waist = float(result["meas"]["Waist"])
        pred_hip = float(result["meas"]["Hip"])

        pred_weight = float(result["weight_hat_kg"]) if result["weight_hat_kg"] is not None else float(result["weight_used_kg"])
        st.write(
            {
                "height_cm": pred_height,
                "weight_kg": pred_weight,
                "waist_cm": pred_waist,
                "hip_cm": pred_hip,
                "BMI": float(result["bmi_pred"]),
                "BRI": float(result["bri_pred"]),
            }
        )

        st.markdown("### Fat targets")
        fat_df = pd.DataFrame(
            [{"target": k, "pred": v} for k, v in result["fat_pred_dict"].items()]
        ).sort_values("target")
        st.dataframe(fat_df, use_container_width=True)


if __name__ == "__main__":
    main()

