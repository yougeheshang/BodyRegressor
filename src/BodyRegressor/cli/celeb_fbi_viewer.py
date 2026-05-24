from __future__ import annotations

import io
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from PIL import Image

# Force PyOpenGL to use OSMesa backend (must be set before any OpenGL import).
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from BodyRegressor.features import WeightMode, compute_bmi, compute_bri
from BodyRegressor.health_report import build_health_report, format_report_markdown
from BodyRegressor.health_report.pipeline import report_input_from_predictions
from BodyRegressor.image_to_smpl import infer_smpl_vertices_and_overlay_from_image
from BodyRegressor.mesh_predict import mesh_to_fat_row
from BodyRegressor.predict import predict_weight
from BodyRegressor.smpl_measure import default_anthro_dir, measure_smpl_verts

UPLOAD_DIR = _REPO_ROOT / "outputs" / "uploads"
REPORT_DIR = _REPO_ROOT / "outputs" / "reports"
CELEB_TMP_DIR = _REPO_ROOT / "outputs" / "celeb-fbi-tmp"
# 报告 Markdown 结构变更时递增，使 session 内旧缓存失效
_REPORT_MD_CACHE_VER = "v3"


@dataclass
class PipelineResult:
    overlay_png_bytes: bytes
    original_png_bytes: bytes
    saved_image_path: Path
    age: float
    sex: int
    height_cm_input: Optional[float]
    weight_used_kg: float
    weight_hat_kg: Optional[float]
    meas: dict[str, float]
    bmi: float
    bri: float
    fat_pred_dict: dict[str, float]
    tgt_cols: list[str]
    preds: np.ndarray


def launch() -> None:
    """pip 安装后的入口：启动 Streamlit。"""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_THIS_FILE),
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.call(cmd))


@st.cache_data(show_spinner=False)
def _load_ds(parquet_path: str):
    parquet_path = str(Path(parquet_path).resolve())
    ds = load_dataset("parquet", data_files={"test": parquet_path})["test"]
    return ds


def _gender_to_sex(gender_gt: int) -> int:
    return 1 if int(gender_gt) == 0 else 2


def _hf_image_value_to_pil(image: object) -> Image.Image:
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
    img_rgb.save(out_path, format="JPEG", quality=95)
    return out_path


def _save_uploaded_image(uploaded_file) -> Path:
    """将 Streamlit 上传文件保存到 outputs/uploads/。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = Path(uploaded_file.name).name
    stem = Path(raw_name).stem[:80] or "upload"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = UPLOAD_DIR / f"{ts}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"
    data = uploaded_file.getvalue()
    if suffix in {".jpg", ".jpeg"}:
        out_path.write_bytes(data)
    else:
        img = Image.open(io.BytesIO(data))
        _pil_to_jpg_path(img, out_path.with_suffix(".jpg"))
        out_path = out_path.with_suffix(".jpg")
    return out_path.resolve()


def _pil_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _run_pipeline(
    *,
    image_path: Path,
    age: float,
    sex: int,
    spin_ckpt: Path,
    spin_dir: Path,
    anthro_dir: Path,
    weight_ckpt: Path,
    fat_ckpt_stacked: Path,
    fat_ckpt_full: Path,
    mode: WeightMode,
    device_arg: Optional[str],
    smpl_gender: str,
    target_height_cm: Optional[float],
    body_weight_kg: Optional[float],
) -> PipelineResult:
    img_pil = Image.open(image_path).convert("RGB")
    original_png_bytes = _pil_png_bytes(img_pil)

    verts, overlay_img, meas_hn = infer_smpl_vertices_and_overlay_from_image(
        image_path=image_path,
        checkpoint=spin_ckpt,
        spin_dir=spin_dir,
        bbox_path=None,
        openpose_path=None,
        device=device_arg,
        smpl_gender=smpl_gender,  # type: ignore[arg-type]
        sex=sex,
        target_height_cm=target_height_cm,
        anthro_dir=anthro_dir if target_height_cm is not None else None,
    )
    meas = meas_hn if meas_hn is not None else measure_smpl_verts(verts, anthro_dir=anthro_dir)

    tmp_df = pd.DataFrame(
        [{"Age": age, "Sex": sex, "Height": meas["Height"], "Waist": meas["Waist"], "Hip": meas["Hip"]}]
    )
    weight_hat_kg: Optional[float] = None
    if mode == WeightMode.HAT:
        weight_hat_kg = float(predict_weight(tmp_df, weight_ckpt)[0])
        weight_used_kg = weight_hat_kg
    else:
        if body_weight_kg is None:
            raise ValueError("weight_mode=real 需要提供体重 (kg)")
        weight_used_kg = float(body_weight_kg)

    bmi = float(compute_bmi(np.float32(weight_used_kg), np.float32(meas["Height"])))
    bri = float(compute_bri(np.float32(meas["Waist"]), np.float32(meas["Height"])).round(1))

    preds, tgt_cols = mesh_to_fat_row(
        measurements=meas,
        age=age,
        sex=sex,
        weight_ckpt=weight_ckpt,
        fat_ckpt_stacked=fat_ckpt_stacked,
        weight_mode=mode,
        fat_ckpt_full=fat_ckpt_full,
        body_weight_kg=body_weight_kg,
        weight_hat_kg=weight_hat_kg,
    )
    fat_pred_dict = {col: float(preds[0, i]) for i, col in enumerate(list(tgt_cols))}

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return PipelineResult(
        overlay_png_bytes=buf.getvalue(),
        original_png_bytes=original_png_bytes,
        saved_image_path=image_path.resolve(),
        age=age,
        sex=sex,
        height_cm_input=target_height_cm,
        weight_used_kg=weight_used_kg,
        weight_hat_kg=weight_hat_kg,
        meas=meas,
        bmi=bmi,
        bri=bri,
        fat_pred_dict=fat_pred_dict,
        tgt_cols=list(tgt_cols),
        preds=preds,
    )


def _build_report_md(result: PipelineResult, *, use_llm: bool) -> str:
    inp = report_input_from_predictions(
        age=result.age,
        sex=result.sex,
        height_cm=float(result.meas["Height"]),
        weight_kg=result.weight_used_kg,
        waist_cm=float(result.meas["Waist"]),
        hip_cm=float(result.meas["Hip"]),
        bmi=result.bmi,
        bri=result.bri,
        preds=result.preds,
        target_names=result.tgt_cols,
    )
    hr = build_health_report(inp, use_llm=use_llm)
    return format_report_markdown(hr)


def _render_metrics(result: PipelineResult, *, gt: Optional[dict[str, Any]] = None) -> None:
    if gt is not None:
        st.markdown("### 数据集真值 (GT)")
        st.json(gt)

    st.markdown("### 预测量测")
    st.json(
        {
            "height_cm": float(result.meas["Height"]),
            "weight_kg": result.weight_used_kg,
            "waist_cm": float(result.meas["Waist"]),
            "hip_cm": float(result.meas["Hip"]),
            "BMI": result.bmi,
            "BRI": result.bri,
            "image_path": str(result.saved_image_path),
        }
    )

    st.markdown("### 体脂指标")
    fat_df = pd.DataFrame([{"target": k, "pred": v} for k, v in result.fat_pred_dict.items()]).sort_values(
        "target"
    )
    st.dataframe(fat_df, use_container_width=True, hide_index=True)


def _display_images_and_metrics(
    result: PipelineResult,
    *,
    gt: Optional[dict[str, Any]] = None,
    original_caption: str = "原图",
    original_path: Optional[Path] = None,
) -> None:
    """上：左图右数据；报告区在调用方于列外单独渲染。"""
    col_img, col_right = st.columns([1, 1.6])
    with col_img:
        st.image(
            Image.open(io.BytesIO(result.overlay_png_bytes)),
            caption="SMPL overlay",
            use_container_width=True,
        )
        if original_path is not None and original_path.is_file():
            st.image(str(original_path), caption=original_caption, use_container_width=True)
        else:
            st.image(
                Image.open(io.BytesIO(result.original_png_bytes)),
                caption=original_caption,
                use_container_width=True,
            )
    with col_right:
        _render_metrics(result, gt=gt)


def _report_md_key(report_key: str) -> str:
    return f"report_md_{_REPORT_MD_CACHE_VER}_{report_key}"


def _maybe_auto_report(result: PipelineResult, *, report_key: str, use_llm: bool, auto: bool) -> None:
    md_key = _report_md_key(report_key)
    if auto and md_key not in st.session_state:
        with st.spinner("生成健康报告…"):
            st.session_state[md_key] = _build_report_md(result, use_llm=use_llm)


def _render_report_section(
    result: PipelineResult,
    *,
    use_llm: bool,
    report_key: str,
) -> None:
    st.markdown("## 健康解读报告")
    if st.button("生成 / 刷新报告", key=f"gen_report_{report_key}"):
        with st.spinner("规则分级 + 知识库检索" + (" + DeepSeek 润色…" if use_llm else "…")):
            md = _build_report_md(result, use_llm=use_llm)
            st.session_state[_report_md_key(report_key)] = md
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            out = REPORT_DIR / f"{report_key}.md"
            out.write_text(md, encoding="utf-8")
            st.session_state[f"report_path_{_REPORT_MD_CACHE_VER}_{report_key}"] = str(out)

    md_key = _report_md_key(report_key)
    if md_key in st.session_state:
        # 避免 [id] 被 Streamlit 解析为链接/占位；用容器 + 纯文本块展示长报告
        with st.container():
            st.markdown(st.session_state[md_key], unsafe_allow_html=False)
        path_key = f"report_path_{_REPORT_MD_CACHE_VER}_{report_key}"
        if path_key in st.session_state:
            st.caption(f"已保存：{st.session_state[path_key]}")
        st.download_button(
            "下载报告 (.md)",
            data=st.session_state[md_key].encode("utf-8"),
            file_name=f"health_report_{report_key}.md",
            mime="text/markdown",
            key=f"dl_{report_key}",
        )


def _sidebar_paths() -> dict[str, Path]:
    device = st.sidebar.selectbox("device", options=["auto", "cpu", "cuda"], index=0)
    return {
        "device_arg": None if device == "auto" else device,
        "spin_ckpt": Path(st.sidebar.text_input("spin_ckpt", value="third-party/SPIN/data/model_checkpoint.pt")).resolve(),
        "spin_dir": Path(st.sidebar.text_input("spin_dir", value="third-party/SPIN")).resolve(),
        "anthro_dir": Path(st.sidebar.text_input("anthro_dir", value=str(default_anthro_dir()))).resolve(),
        "weight_ckpt": Path(st.sidebar.text_input("weight_ckpt", value="models/weight_xgb.joblib")).resolve(),
        "fat_ckpt_stacked": Path(st.sidebar.text_input("fat_ckpt_stacked", value="models/fat_xgb_stacked.joblib")).resolve(),
        "fat_ckpt_full": Path(st.sidebar.text_input("fat_ckpt_full", value="models/fat_xgb_full.joblib")).resolve(),
    }


def _sidebar_report_options() -> dict[str, Any]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**健康报告**")
    use_llm = st.sidebar.checkbox("调用 DeepSeek 润色（需 .env）", value=True)
    auto_report = st.sidebar.checkbox("分析完成后自动生成报告", value=True)
    return {"use_llm": use_llm, "auto_report": auto_report}


def _tab_dataset(cfg: dict[str, Any], report_opts: dict[str, Any]) -> None:
    parquet_default = "data/celeb-fbi/0000.parquet"
    parquet_path = st.text_input("parquet_path", value=parquet_default)
    mode = WeightMode(st.selectbox("weight_mode", options=[m.value for m in WeightMode], index=1))
    use_gt_height = st.checkbox("用真实身高校准 mesh", value=True)
    smpl_gender = "neutral"

    ds = _load_ds(parquet_path)
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str)
    selected_id = st.selectbox("选择样本 id", df["id"].tolist())
    row = df[df["id"] == selected_id].iloc[0]

    cache_key = (
        f"ds|{selected_id}|{cfg['device_arg']}|{mode}|{smpl_gender}|{use_gt_height}|{parquet_path}"
    )
    if "pipeline_cache" not in st.session_state:
        st.session_state.pipeline_cache = {}

    if cache_key not in st.session_state.pipeline_cache:
        with st.spinner("SPIN 推理 + 量测 + 体脂预测中…"):
            img_pil = _hf_image_value_to_pil(row["image"])
            CELEB_TMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = _pil_to_jpg_path(img_pil, CELEB_TMP_DIR / f"{selected_id}.jpg")

            gender_gt = int(row["gender"])
            sex = _gender_to_sex(gender_gt)
            age = float(row["age"])
            height_gt = float(row["height"])
            weight_gt = float(row["weight"])

            result = _run_pipeline(
                image_path=tmp_path,
                age=age,
                sex=sex,
                spin_ckpt=cfg["spin_ckpt"],
                spin_dir=cfg["spin_dir"],
                anthro_dir=cfg["anthro_dir"],
                weight_ckpt=cfg["weight_ckpt"],
                fat_ckpt_stacked=cfg["fat_ckpt_stacked"],
                fat_ckpt_full=cfg["fat_ckpt_full"],
                mode=mode,
                device_arg=cfg["device_arg"],
                smpl_gender=smpl_gender,
                target_height_cm=height_gt if use_gt_height else None,
                body_weight_kg=weight_gt if mode == WeightMode.REAL else None,
            )
            st.session_state.pipeline_cache[cache_key] = result

    result: PipelineResult = st.session_state.pipeline_cache[cache_key]
    report_key = f"ds_{selected_id}"

    _display_images_and_metrics(
        result,
        gt={
            "age": float(row["age"]),
            "gender": int(row["gender"]),
            "height_cm": float(row["height"]),
            "weight_kg": float(row["weight"]),
        },
        original_caption="原图",
    )
    st.divider()
    _maybe_auto_report(
        result,
        report_key=report_key,
        use_llm=report_opts["use_llm"],
        auto=report_opts["auto_report"],
    )
    _render_report_section(result, use_llm=report_opts["use_llm"], report_key=report_key)


def _tab_upload(cfg: dict[str, Any], report_opts: dict[str, Any]) -> None:
    st.markdown("上传全身 RGB 照片，填写基本信息后进行分析。图片会保存到 `outputs/uploads/`。")
    uploaded = st.file_uploader("选择图片", type=["jpg", "jpeg", "png", "webp", "bmp"])

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("年龄", min_value=1.0, max_value=120.0, value=30.0, step=1.0)
    with c2:
        sex_label = st.selectbox("性别", options=["男", "女"], index=0)
        sex = 1 if sex_label == "男" else 2
    with c3:
        mode = WeightMode(st.selectbox("weight_mode", options=[m.value for m in WeightMode], index=1, key="upload_wm"))

    c4, c5 = st.columns(2)
    with c4:
        height_cm = st.number_input("真实身高 (cm，可选，用于 mesh 校准)", min_value=0.0, value=0.0, step=0.1)
    with c5:
        weight_kg = st.number_input(
            "真实体重 (kg，weight_mode=real 时必填)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )

    smpl_gender = st.selectbox(
        "SMPL 模板",
        options=["by_sex", "neutral", "male", "female"],
        index=0,
        help="by_sex：与所选性别一致",
    )
    run = st.button("开始分析", type="primary", disabled=uploaded is None)

    if not run or uploaded is None:
        if UPLOAD_DIR.is_dir():
            saved = sorted(UPLOAD_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
            if saved:
                st.markdown("**最近保存的上传图片**")
                for p in saved:
                    st.caption(str(p))
        return

    saved_path = _save_uploaded_image(uploaded)
    st.success(f"图片已保存：{saved_path}")

    target_h = float(height_cm) if height_cm > 0 else None
    body_w = float(weight_kg) if weight_kg > 0 else None
    if mode == WeightMode.REAL and (body_w is None or body_w <= 0):
        st.error("weight_mode=real 时请填写真实体重 (kg)")
        return

    cache_key = f"up|{saved_path}|{age}|{sex}|{mode}|{smpl_gender}|{target_h}|{body_w}|{cfg['device_arg']}"
    with st.spinner("SPIN 推理 + 量测 + 体脂预测中…"):
        result = _run_pipeline(
            image_path=saved_path,
            age=float(age),
            sex=sex,
            spin_ckpt=cfg["spin_ckpt"],
            spin_dir=cfg["spin_dir"],
            anthro_dir=cfg["anthro_dir"],
            weight_ckpt=cfg["weight_ckpt"],
            fat_ckpt_stacked=cfg["fat_ckpt_stacked"],
            fat_ckpt_full=cfg["fat_ckpt_full"],
            mode=mode,
            device_arg=cfg["device_arg"],
            smpl_gender=smpl_gender,
            target_height_cm=target_h,
            body_weight_kg=body_w,
        )
    st.session_state.pipeline_cache = st.session_state.get("pipeline_cache", {})
    st.session_state.pipeline_cache[cache_key] = result

    report_key = saved_path.stem
    _display_images_and_metrics(
        result,
        original_caption="已保存原图",
        original_path=saved_path,
    )
    st.divider()
    _maybe_auto_report(
        result,
        report_key=report_key,
        use_llm=report_opts["use_llm"],
        auto=report_opts["auto_report"],
    )
    _render_report_section(result, use_llm=report_opts["use_llm"], report_key=report_key)


def main() -> None:
    st.set_page_config(page_title="BodyRegressor", layout="wide", initial_sidebar_state="expanded")
    st.title("BodyRegressor")
    st.caption("人体测量 · 体脂预测 · 健康解读报告")

    cfg = _sidebar_paths()
    report_opts = _sidebar_report_options()

    tab_ds, tab_up = st.tabs(["数据集 (celeb-fbi)", "上传图片"])
    with tab_ds:
        _tab_dataset(cfg, report_opts)
    with tab_up:
        _tab_upload(cfg, report_opts)


if __name__ == "__main__":
    main()
