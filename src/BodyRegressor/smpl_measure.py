"""
从标准 SMPL 顶点 (6890, 3) 量测身高/腰围/臀围（cm），依赖 third-party SMPL-Anthropometry。

SMPL-Anthropometry 在内部假定 verts 单位为米，量测结果以 cm 输出。
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

# 仓库根：src/BodyRegressor/smpl_measure.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_ANTHRO_DIR = _REPO_ROOT / "third-party" / "SMPL-Anthropometry" / "SMPL-Anthropometry"

# label_measurements(STANDARD_LABELS) 后的键 -> 与 NHANES/特征列对应
_LABEL_HEIGHT = "P"  # height
_LABEL_WAIST = "E"   # waist circumference
_LABEL_HIP = "F"     # hip circumference


@contextmanager
def _working_directory(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def default_anthro_dir() -> Path:
    return _DEFAULT_ANTHRO_DIR


def _ensure_anthro_layout(anthro_dir: Path) -> None:
    smpl_data = anthro_dir / "data" / "smpl"
    if not smpl_data.is_dir():
        raise FileNotFoundError(
            f"未找到 SMPL 数据目录: {smpl_data}（需含 *.pkl 与 smpl_body_parts_2_faces.json）"
        )


def load_verts_from_file(path: Path) -> np.ndarray:
    """支持 .npy（或 .npz 单数组）与 .pt/.pth（torch 保存的 tensor）。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        z = np.load(path)
        keys = list(z.keys())
        if not keys:
            raise ValueError("npz 中无数组")
        arr = z[keys[0]]
    elif suffix in (".pt", ".pth"):
        import inspect
        import torch

        load_kw: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kw["weights_only"] = False
        t = torch.load(path, **load_kw)
        if isinstance(t, dict) and "verts" in t:
            t = t["verts"]
        if not hasattr(t, "numpy"):
            raise TypeError(f"无法从 {path} 得到 tensor")
        arr = t.detach().cpu().numpy()
    else:
        raise ValueError(f"不支持的 verts 文件后缀: {suffix}（可用 .npy / .npz / .pt）")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (6890, 3):
        raise ValueError(f"SMPL verts 形状应为 (6890, 3)，当前 {arr.shape}")
    return arr


def measure_smpl_verts(
    verts: np.ndarray,
    *,
    anthro_dir: Path | None = None,
) -> dict[str, float]:
    """
    对标准 SMPL 顶点做量测，返回 cm 下的 Height / Waist / Hip。

    Parameters
    ----------
    verts : (6890, 3) ndarray，单位建议为米（与官方 SMPL 一致）。
    anthro_dir : SMPL-Anthropometry 内层目录（内含 measure.py 与 data/smpl）。
    """
    anthro_dir = Path(anthro_dir) if anthro_dir is not None else default_anthro_dir()
    anthro_dir = anthro_dir.resolve()
    _ensure_anthro_layout(anthro_dir)

    verts = np.asarray(verts, dtype=np.float32)
    if verts.shape != (6890, 3):
        raise ValueError(f"SMPL verts 形状应为 (6890, 3)，当前 {verts.shape}")

    import torch

    anthro_str = str(anthro_dir)
    if anthro_str not in sys.path:
        sys.path.insert(0, anthro_str)

    with _working_directory(anthro_dir):
        from measure import MeasureBody  # type: ignore[import-untyped]
        from measurement_definitions import STANDARD_LABELS  # type: ignore[import-untyped]

        measurer: Any = MeasureBody("smpl")
        measurer.from_verts(torch.as_tensor(verts, dtype=torch.float32))
        measurer.measure(measurer.all_possible_measurements)
        measurer.label_measurements(STANDARD_LABELS)
        labeled = measurer.labeled_measurements

    missing = [k for k in (_LABEL_HEIGHT, _LABEL_WAIST, _LABEL_HIP) if k not in labeled]
    if missing:
        raise RuntimeError(f"量测结果缺少标签: {missing}，当前 keys={list(labeled.keys())}")

    return {
        "Height": float(labeled[_LABEL_HEIGHT]),
        "Waist": float(labeled[_LABEL_WAIST]),
        "Hip": float(labeled[_LABEL_HIP]),
    }
