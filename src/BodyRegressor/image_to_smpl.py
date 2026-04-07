from __future__ import annotations

import json
import os
import sys
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Literal
from PIL import Image

# Force PyOpenGL to use OSMesa backend (must be set before any OpenGL import).
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SPIN_DIR = _REPO_ROOT / "third-party" / "SPIN"
_DEFAULT_ANTHRO_SMPL_DIR = _REPO_ROOT / "third-party" / "SMPL-Anthropometry" / "data" / "smpl"


@contextmanager
def _working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def default_spin_dir() -> Path:
    return _DEFAULT_SPIN_DIR


def _ensure_spin_layout(spin_dir: Path) -> None:
    required = [
        spin_dir / "config.py",
        spin_dir / "constants.py",
        spin_dir / "models" / "hmr.py",
        spin_dir / "models" / "smpl.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"SPIN 目录不完整，缺少文件: {missing}")


def _resolve_smpl_model_dir(spin_dir: Path) -> Path:
    spin_smpl = spin_dir / "data" / "smpl"
    required = ["SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl"]

    def has_required(p: Path) -> bool:
        return p.is_dir() and all((p / f).exists() for f in required)

    if has_required(spin_smpl):
        return spin_smpl
    if has_required(_DEFAULT_ANTHRO_SMPL_DIR):
        return _DEFAULT_ANTHRO_SMPL_DIR
    raise FileNotFoundError(
        "未找到可用的 SMPL 模型目录。需要包含 "
        f"{required}。已检查: {spin_smpl} 和 {_DEFAULT_ANTHRO_SMPL_DIR}"
    )


def _bbox_from_openpose(openpose_file: Path, rescale: float = 1.2, detection_thresh: float = 0.2) -> tuple[np.ndarray, float]:
    with open(openpose_file, "r", encoding="utf-8") as f:
        keypoints = json.load(f)["people"][0]["pose_keypoints_2d"]
    keypoints = np.reshape(np.array(keypoints), (-1, 3))
    valid = keypoints[:, -1] > detection_thresh
    valid_keypoints = keypoints[valid][:, :-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    scale = (bbox_size / 200.0) * rescale
    return center, float(scale)


def _bbox_from_json(bbox_file: Path) -> tuple[np.ndarray, float]:
    with open(bbox_file, "r", encoding="utf-8") as f:
        bbox = np.array(json.load(f)["bbox"]).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    return center, float(scale)


def _process_image(
    img_file: Path,
    *,
    bbox_file: Path | None,
    openpose_file: Path | None,
    input_res: int,
) -> tuple[torch.Tensor, np.ndarray]:
    normalize_img = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = cv2.imread(str(img_file))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_file}")
    img = img[:, :, ::-1].copy()

    if bbox_file is None and openpose_file is None:
        h, w = img.shape[:2]
        center = np.array([w // 2, h // 2])
        scale = max(h, w) / 200.0
    elif bbox_file is not None:
        center, scale = _bbox_from_json(bbox_file)
    else:
        center, scale = _bbox_from_openpose(openpose_file)  # type: ignore[arg-type]

    from utils.imutils import crop  # type: ignore[import-untyped]

    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1)
    norm_img = normalize_img(img_t.clone())[None]
    # img: (H, W, 3) float in [0,1]
    return norm_img, img


SmplGenderMode = Literal["by_sex", "neutral", "male", "female"]


def resolve_smpl_gender(*, mode: SmplGenderMode, sex: int | None) -> str:
    """解析 smplx.SMPL 的 gender 字符串：neutral / male / female。"""
    if mode == "neutral":
        return "neutral"
    if mode == "male":
        return "male"
    if mode == "female":
        return "female"
    if mode == "by_sex":
        if sex is None:
            raise ValueError("mode=by_sex 时必须提供 sex（1=Male, 2=Female）")
        if sex == 1:
            return "male"
        if sex == 2:
            return "female"
        raise ValueError(f"无效的 sex: {sex}，应为 1 或 2")
    raise ValueError(f"未知的 mode: {mode}")


def infer_smpl_vertices_from_image(
    *,
    image_path: Path,
    checkpoint: Path,
    spin_dir: Path | None = None,
    bbox_path: Path | None = None,
    openpose_path: Path | None = None,
    device: str | None = None,
    smpl_gender: SmplGenderMode = "neutral",
    sex: int | None = None,
) -> np.ndarray:
    spin_dir = Path(spin_dir) if spin_dir is not None else default_spin_dir()
    spin_dir = spin_dir.resolve()
    _ensure_spin_layout(spin_dir)

    spin_str = str(spin_dir)
    if spin_str not in sys.path:
        sys.path.insert(0, spin_str)

    image_path = Path(image_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    bbox_path = Path(bbox_path).resolve() if bbox_path is not None else None
    openpose_path = Path(openpose_path).resolve() if openpose_path is not None else None

    if bbox_path is not None and openpose_path is not None:
        raise ValueError("bbox_path 与 openpose_path 只能二选一")

    dev = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    with _working_directory(spin_dir):
        import config  # type: ignore[import-untyped]
        import constants  # type: ignore[import-untyped]
        from models import SMPL, hmr  # type: ignore[import-untyped]

        model = hmr(config.SMPL_MEAN_PARAMS).to(dev)
        load_kw: dict[str, object] = {"map_location": dev}
        if "weights_only" in inspect.signature(torch.load).parameters:
            # SPIN checkpoint stores a full pickle dict, not weights-only format.
            load_kw["weights_only"] = False
        ckpt = torch.load(checkpoint, **load_kw)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        smpl_model_dir = _resolve_smpl_model_dir(spin_dir)
        gender = resolve_smpl_gender(mode=smpl_gender, sex=sex)
        smpl = SMPL(
            str(smpl_model_dir),
            batch_size=1,
            create_transl=False,
            gender=gender,
        ).to(dev)

        norm_img, _img_crop = _process_image(
            image_path,
            bbox_file=bbox_path,
            openpose_file=openpose_path,
            input_res=constants.IMG_RES,
        ).to(dev)

        with torch.no_grad():
            pred_rotmat, pred_betas, _pred_camera = model(norm_img)
            pred_output = smpl(
                betas=pred_betas,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            pred_vertices = pred_output.vertices[0].detach().cpu().numpy().astype(np.float32)

    if pred_vertices.shape != (6890, 3):
        raise RuntimeError(f"SPIN 输出 verts 形状异常: {pred_vertices.shape}")
    return pred_vertices


def infer_smpl_vertices_and_overlay_from_image(
    *,
    image_path: Path,
    checkpoint: Path,
    spin_dir: Path | None = None,
    bbox_path: Path | None = None,
    openpose_path: Path | None = None,
    device: str | None = None,
    smpl_gender: SmplGenderMode = "neutral",
    sex: int | None = None,
) -> tuple[np.ndarray, Image.Image]:
    """
    推理得到 verts(6890,3) 并渲染得到拟合可视化图片（裁剪到 224x224）。
    """
    spin_dir = Path(spin_dir) if spin_dir is not None else default_spin_dir()
    spin_dir = spin_dir.resolve()
    _ensure_spin_layout(spin_dir)

    spin_str = str(spin_dir)
    if spin_str not in sys.path:
        sys.path.insert(0, spin_str)

    image_path = Path(image_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    bbox_path = Path(bbox_path).resolve() if bbox_path is not None else None
    openpose_path = Path(openpose_path).resolve() if openpose_path is not None else None

    if bbox_path is not None and openpose_path is not None:
        raise ValueError("bbox_path 与 openpose_path 只能二选一")

    dev = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    with _working_directory(spin_dir):
        import config  # type: ignore[import-untyped]
        import constants  # type: ignore[import-untyped]
        from models import SMPL, hmr  # type: ignore[import-untyped]
        from utils.renderer import Renderer  # type: ignore[import-untyped]

        model = hmr(config.SMPL_MEAN_PARAMS).to(dev)
        load_kw: dict[str, object] = {"map_location": dev}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        ckpt = torch.load(checkpoint, **load_kw)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        smpl_model_dir = _resolve_smpl_model_dir(spin_dir)
        gender = resolve_smpl_gender(mode=smpl_gender, sex=sex)
        smpl = SMPL(
            str(smpl_model_dir),
            batch_size=1,
            create_transl=False,
            gender=gender,
        ).to(dev)

        renderer = Renderer(
            focal_length=constants.FOCAL_LENGTH,
            img_res=constants.IMG_RES,
            faces=smpl.faces,
        )

        norm_img, img_crop = _process_image(
            image_path,
            bbox_file=bbox_path,
            openpose_file=openpose_path,
            input_res=constants.IMG_RES,
        )
        norm_img = norm_img.to(dev)

        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(norm_img)
            pred_output = smpl(
                betas=pred_betas,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, 0].unsqueeze(1),
                pose2rot=False,
            )
            pred_vertices = pred_output.vertices[0].detach().cpu().numpy().astype(np.float32)

            # Same camera conversion as SPIN demo.py
            camera_translation = torch.stack(
                [
                    pred_camera[:, 1],
                    pred_camera[:, 2],
                    2 * constants.FOCAL_LENGTH / (constants.IMG_RES * pred_camera[:, 0] + 1e-9),
                ],
                dim=-1,
            )[0].cpu().numpy()

            overlay_float = renderer(pred_vertices, camera_translation, img_crop)
            overlay_u8 = (np.clip(overlay_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            overlay_img = Image.fromarray(overlay_u8)

    if pred_vertices.shape != (6890, 3):
        raise RuntimeError(f"SPIN 输出 verts 形状异常: {pred_vertices.shape}")
    return pred_vertices, overlay_img

