from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
from sklearn.metrics import r2_score


def mae(preds: np.ndarray, targets: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return np.mean(np.abs(preds - targets), axis=axis)


def mape(
    preds: np.ndarray,
    targets: np.ndarray,
    axis: Optional[int] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    denom = np.where(np.abs(targets) < eps, eps, targets)
    return np.mean(np.abs((preds - targets) / denom), axis=axis) * 100.0


def r2_per_target(preds: np.ndarray, targets: np.ndarray) -> list[float]:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape != targets.shape:
        raise ValueError(f"preds/targets 形状不一致: {preds.shape} vs {targets.shape}")
    if preds.ndim == 1:
        return [float(r2_score(targets, preds))]
    return [float(r2_score(targets[:, i], preds[:, i])) for i in range(preds.shape[1])]


def summarize(
    preds: np.ndarray,
    targets: np.ndarray,
    target_names: Optional[Iterable[str]] = None,
    *,
    units: Optional[Mapping[str, str]] = None,
) -> dict[str, dict[str, float]]:
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    if preds.shape != targets.shape:
        raise ValueError(f"preds/targets 形状不一致: {preds.shape} vs {targets.shape}")

    if preds.ndim == 1:
        preds = preds[:, None]
        targets = targets[:, None]

    n_targets = preds.shape[1]
    if target_names is None:
        names = [str(i) for i in range(n_targets)]
    else:
        names = list(target_names)
        if len(names) != n_targets:
            raise ValueError(f"target_names 长度不匹配: {len(names)} vs {n_targets}")

    units = units or {}
    mae_vals = mae(preds, targets, axis=0)
    mape_vals = mape(preds, targets, axis=0)
    r2_vals = r2_per_target(preds, targets)

    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        out[name] = {"mae": float(mae_vals[i]), "mape": float(mape_vals[i]), "r2": float(r2_vals[i])}
        if name in units:
            out[name]["unit"] = units[name]
    return out


def format_summary(
    summary: Mapping[str, Mapping[str, float]],
    *,
    title: Optional[str] = None,
    target_order: Optional[Iterable[str]] = None,
    mae_decimals: int = 2,
    mape_decimals: int = 2,
    r2_decimals: int = 3,
    r2_label: str = "R^2",
) -> str:
    """
    把 summarize() 的结果格式成“每行一个 target”的可读文本，便于控制台对比。
    """

    if target_order is None:
        targets = list(summary.keys())
    else:
        targets = list(target_order)

    lines: list[str] = []
    if title:
        lines.append(title)

    for name in targets:
        if name not in summary:
            # 允许 target_order 比 summary 多一些，跳过缺失项
            continue
        row = summary[name]
        mae_val = row.get("mae", float("nan"))
        mape_val = row.get("mape", float("nan"))
        r2_val = row.get("r2", float("nan"))
        unit = row.get("unit", "")

        mae_str = f"MAE {mae_val:.{mae_decimals}f}{unit}"
        mape_str = f"MAPE {mape_val:.{mape_decimals}f}%"
        r2_str = f"{r2_label} {r2_val:.{r2_decimals}f}"
        lines.append(f"{name}: {mae_str}, {mape_str}, {r2_str}")

    return "\n".join(lines)

