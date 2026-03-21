from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Optional

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


def fat_units_for_targets(target_names: Iterable[str]) -> dict[str, str]:
    """体脂目标默认单位（与 training 中 summarize 一致）。"""
    return {c: ("%" if c == "Percentage_body_fat" else "g") for c in target_names}


def summarize_predictions(
    preds: np.ndarray,
    target_names: Iterable[str],
    *,
    units: Optional[Mapping[str, str]] = None,
) -> dict[str, dict[str, Any]]:
    """
    推理用：仅预测值，无真实标签。输出结构与 format_summary(style=\"predict\") 兼容。
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim == 1:
        preds = preds.reshape(1, -1)
    if preds.shape[0] != 1:
        raise ValueError(f"summarize_predictions 当前仅支持单样本，得到 batch={preds.shape[0]}")
    names = list(target_names)
    if len(names) != preds.shape[1]:
        raise ValueError(f"target_names 长度 {len(names)} 与 preds 列数 {preds.shape[1]} 不一致")
    units = units or {}
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(names):
        row: dict[str, Any] = {"pred": float(preds[0, i])}
        if name in units:
            row["unit"] = units[name]
        out[name] = row
    return out


def format_summary(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    title: Optional[str] = None,
    target_order: Optional[Iterable[str]] = None,
    style: Literal["eval", "predict"] = "eval",
    mae_decimals: int = 2,
    mape_decimals: int = 2,
    r2_decimals: int = 3,
    r2_label: str = "R^2",
    pred_decimals: int = 2,
) -> str:
    """
    把 summarize() 或 summarize_predictions() 的结果格式成“每行一个 target”的可读文本。

    - style=\"eval\"：MAE / MAPE / R^2（与 training 一致）
    - style=\"predict\"：仅 Pred + 单位（推理）
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
            continue
        row = summary[name]
        unit = row.get("unit", "")

        if style == "predict":
            pred_val = float(row.get("pred", float("nan")))
            lines.append(f"{name}: Pred {pred_val:.{pred_decimals}f}{unit}")
            continue

        mae_val = row.get("mae", float("nan"))
        mape_val = row.get("mape", float("nan"))
        r2_val = row.get("r2", float("nan"))
        mae_str = f"MAE {mae_val:.{mae_decimals}f}{unit}"
        mape_str = f"MAPE {mape_val:.{mape_decimals}f}%"
        r2_str = f"{r2_label} {r2_val:.{r2_decimals}f}"
        lines.append(f"{name}: {mae_str}, {mape_str}, {r2_str}")

    return "\n".join(lines)

