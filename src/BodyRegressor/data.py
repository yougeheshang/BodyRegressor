from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


def calculate_bri(waist_cm: pd.Series, height_cm: pd.Series) -> pd.Series:
    r = waist_cm / (2 * np.pi)
    h_half = 0.5 * height_cm
    return 364.2 - 365.5 * np.sqrt(1 - (r / h_half) ** 2)


def _find_xpt(raw_dir: Path, base_name: str) -> Path:
    raw_dir = Path(raw_dir)
    for c in (base_name, base_name.upper(), base_name.lower()):
        for ext in ("", ".xpt", ".XPT"):
            p = raw_dir / f"{c}{ext}" if ext else raw_dir / c
            if p.exists():
                return p
    for pattern in (f"{base_name}_*.xpt", f"{base_name}_*.XPT"):
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"找不到 XPT: {base_name} in {raw_dir}")


def merge_nhanes_raw(
    raw_dir: Path,
    *,
    include_hip: bool = True,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    demo = pd.read_sas(_find_xpt(raw_dir, "DEMO"), format="xport", encoding="utf-8").set_index("SEQN")
    bmx = pd.read_sas(_find_xpt(raw_dir, "BMX"), format="xport", encoding="utf-8").set_index("SEQN")
    dxx = pd.read_sas(_find_xpt(raw_dir, "DXX"), format="xport", encoding="utf-8").set_index("SEQN")
    dxxag = pd.read_sas(_find_xpt(raw_dir, "DXXAG"), format="xport", encoding="utf-8").set_index("SEQN")

    demo_filt = demo[demo["RIDSTATR"] == 2]
    bmx_filt = bmx[bmx["BMDSTATS"] == 1]
    dxx_filt = dxx[dxx["DXAEXSTS"] == 1]
    dxxag_filt = dxxag[(dxxag["DXXAGST"] == 1) & (np.abs(dxxag["DXXVATV"]) < 1e-6)]

    data = (
        bmx_filt.join(demo_filt, how="inner", rsuffix="_demo")
        .join(dxx_filt, how="inner", rsuffix="_dxx")
        .join(dxxag_filt, how="inner", rsuffix="_dxxag")
    )

    out = pd.DataFrame(index=data.index)
    out["ID"] = data.index
    out["Age"] = data["RIDAGEYR"]
    out["Sex"] = data["RIAGENDR"]
    out["Height"] = data["BMXHT"]
    out["Weight"] = data["BMXWT"]
    out["Waist"] = data["BMXWAIST"]
    if include_hip and "BMXHIP" in data.columns:
        out["Hip"] = data["BMXHIP"]
    out["BMI"] = data["BMXBMI"]
    out["BRI"] = calculate_bri(out["Waist"], out["Height"]).round(1)

    out["Total_fat_mass"] = data["DXDTOFAT"]
    out["Percentage_body_fat"] = data["DXDTOPF"]
    out["Android_fat_mass"] = data["DXXANFM"]
    out["Gynoid_fat_mass"] = data["DXXGYFM"]
    out["Subcutaneous_fat_mass"] = data["DXXSATM"]
    out["Total_abdominal_fat_mass"] = data["DXXTATM"]
    out["Visceral_adipose_tissue_mass"] = data["DXXVFATM"]
    out["Peripheral_fat_mass"] = (
        data["DXXLAFAT"] + data["DXXRAFAT"] + data["DXXLLFAT"] + data["DXXRLFAT"]
    ).round(1)

    return out.dropna()


def write_merged_csv(raw_dir: Path, output_csv: Path, *, include_hip: bool = True) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = merge_nhanes_raw(raw_dir, include_hip=include_hip)
    df.to_csv(output_csv, index=False)
    return output_csv


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_ratio: float = 0.1,
    val_ratio: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError("X 与 y 行数不一致")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, shuffle=shuffle
    )
    val_rel = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, random_state=random_state, shuffle=shuffle
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/nhanes/raw", help="包含 DEMO/BMX/DXX/DXXAG 的目录")
    p.add_argument(
        "--out-csv",
        default="data/nhanes/processed/nhanes_merged.csv",
        help="输出合并后的 CSV 路径",
    )
    p.add_argument("--include-hip", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    out = write_merged_csv(Path(args.raw_dir), Path(args.out_csv), include_hip=args.include_hip)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

