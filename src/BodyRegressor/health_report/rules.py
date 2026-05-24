from __future__ import annotations

from typing import Iterable, Optional

from BodyRegressor.features import FAT_TARGET_COLS

from .types import GRADE_LABEL_ZH, Grade, MetricAssessment, ReportInput

# ---------------------------------------------------------------------------
# 参考依据（科普分级，非临床诊断）：
# - NHANES DXA 体成分参考：Kelly et al., Am J Clin Nutr 2009 / PMC2737140（FMI、分区脂肪）
# - NHANES 变量 DXXAGRAT（Android/Gynoid 脂肪质量比）人群范围约 0.5–1.9
# - 体脂率：ACE 等成人常见分段（与 Percentage_body_fat 一致）
# - 内脏脂肪：按 VAT 质量占体重比例（与代谢风险文献中“相对内脏负荷”一致）
# 模型预测值为 DXA 同源指标（见 data.py 中 DXDTOFAT、DXXANFM 等），阈值按成人尺度设定。
# ---------------------------------------------------------------------------

_BMI_LOW = 18.5
_BMI_HIGH_BORDER = 24.0
_BMI_HIGH = 28.0

_PBF_MALE = (10.0, 20.0, 25.0)
_PBF_FEMALE = (18.0, 28.0, 32.0)

_WAIST_MALE = (94.0, 102.0)
_WAIST_FEMALE = (80.0, 88.0)

_VAT_RATIO_BORDER = 0.10
_VAT_RATIO_HIGH = 0.15

# 全身脂肪量指数 FMI = 脂肪(kg) / 身高(m)²  （NHANES 成人 FMI 分层思路）
_FMI_TOTAL_MALE = (4.0, 7.5, 9.5)
_FMI_TOTAL_FEMALE = (5.0, 10.0, 12.5)

_FMI_ANDROID_MALE = (0.7, 1.6, 2.2)
_FMI_ANDROID_FEMALE = (1.0, 2.0, 2.8)

_FMI_GYNOID_MALE = (1.0, 2.5, 3.5)
_FMI_GYNOID_FEMALE = (2.0, 4.0, 5.5)

_FMI_SUBCUT_MALE = (3.0, 8.0, 10.0)
_FMI_SUBCUT_FEMALE = (5.0, 12.0, 15.0)

_FMI_ABDOM_MALE = (0.8, 2.0, 2.8)
_FMI_ABDOM_FEMALE = (1.2, 2.8, 3.8)

_FMI_PERIPH_MALE = (1.5, 4.0, 5.5)
_FMI_PERIPH_FEMALE = (2.5, 6.0, 8.0)

# Android/Gynoid 脂肪质量比（DXXAGRAT，中心性分布）
_AG_RATIO_MALE = (0.75, 1.15, 1.35)
_AG_RATIO_FEMALE = (0.70, 0.95, 1.08)

_METRIC_DISPLAY = {
    "BMI": "体质指数 (BMI)",
    "BRI": "身体圆润指数 (BRI)",
    "Waist": "腰围",
    "Percentage_body_fat": "体脂率",
    "Visceral_adipose_tissue_mass": "内脏脂肪 (VAT)",
    "Total_abdominal_fat_mass": "腹部脂肪总量",
    "Android_fat_mass": "腹部（Android）脂肪",
    "Gynoid_fat_mass": "臀腿（Gynoid）脂肪",
    "Total_fat_mass": "全身脂肪总量",
    "Subcutaneous_fat_mass": "皮下脂肪",
    "Peripheral_fat_mass": "外周脂肪",
}

_SUMMARY: dict[tuple[str, Grade], str] = {
    ("BMI", Grade.LOW): "BMI 低于常见成人参考下限，可能存在偏瘦风险，建议结合饮食与力量训练情况综合看待。",
    ("BMI", Grade.NORMAL): "BMI 处于常见成人参考范围。",
    ("BMI", Grade.BORDERLINE): "BMI 处于超重边缘区间，建议关注体重管理与活动量。",
    ("BMI", Grade.HIGH): "BMI 达到常见成人肥胖参考区间，建议重视体重与代谢风险。",
    ("Waist", Grade.NORMAL): "腰围未达常见中心性肥胖风险阈值。",
    ("Waist", Grade.BORDERLINE): "腰围处于中心性肥胖风险边缘，建议关注腹部脂肪。",
    ("Waist", Grade.HIGH): "腰围达到常见中心性肥胖风险阈值，建议重视代谢与心血管风险因素。",
    ("Percentage_body_fat", Grade.LOW): "预测体脂率偏低，若伴随乏力或月经异常等，建议咨询专业人士。",
    ("Percentage_body_fat", Grade.NORMAL): "预测体脂率处于常见成人参考范围。",
    ("Percentage_body_fat", Grade.BORDERLINE): "预测体脂率略高，可关注饮食结构与有氧运动。",
    ("Percentage_body_fat", Grade.HIGH): "预测体脂率偏高，建议重视体脂管理与生活方式。",
    ("Total_fat_mass", Grade.LOW): "预测全身脂肪量（FMI）偏低，若与过轻或营养摄入不足相关，建议综合评估。",
    ("Total_fat_mass", Grade.NORMAL): "预测全身脂肪量（FMI）处于常见成人参考范围。",
    ("Total_fat_mass", Grade.BORDERLINE): "预测全身脂肪量（FMI）略高，建议关注总能量平衡与活动量。",
    ("Total_fat_mass", Grade.HIGH): "预测全身脂肪量（FMI）偏高，与整体肥胖负荷增加相关，建议系统管理体成分。",
    ("Android_fat_mass", Grade.LOW): "预测腹部（Android）脂肪偏低，通常与总体脂偏低或测量误差有关。",
    ("Android_fat_mass", Grade.NORMAL): "预测腹部（Android）脂肪处于常见范围，中心性脂肪负荷未见明显偏高。",
    ("Android_fat_mass", Grade.BORDERLINE): "预测腹部（Android）脂肪略高，建议关注腰围与内脏脂肪是否同步升高。",
    ("Android_fat_mass", Grade.HIGH): "预测腹部（Android）脂肪偏高，提示中心性脂肪堆积倾向，宜优先改善饮食与活动。",
    ("Gynoid_fat_mass", Grade.LOW): "预测臀腿（Gynoid）脂肪偏低，多与总体脂偏低相关。",
    ("Gynoid_fat_mass", Grade.NORMAL): "预测臀腿（Gynoid）脂肪处于常见范围。",
    ("Gynoid_fat_mass", Grade.BORDERLINE): "预测臀腿（Gynoid）脂肪略高，常与总体脂偏高并存，可结合全身指标综合看待。",
    ("Gynoid_fat_mass", Grade.HIGH): "预测臀腿（Gynoid）脂肪偏高，多与全身脂肪量增加相关；相对 Android 单纯偏高，代谢风险通常较低。",
    ("Subcutaneous_fat_mass", Grade.LOW): "预测皮下脂肪偏低，可见于偏瘦或肌肉量相对较高者。",
    ("Subcutaneous_fat_mass", Grade.NORMAL): "预测皮下脂肪处于常见范围。",
    ("Subcutaneous_fat_mass", Grade.BORDERLINE): "预测皮下脂肪略高，提示体表脂肪储存增加，可配合体脂率与腰围观察。",
    ("Subcutaneous_fat_mass", Grade.HIGH): "预测皮下脂肪偏高，反映总体皮下脂肪负荷较大，建议通过生活方式干预改善体成分。",
    ("Total_abdominal_fat_mass", Grade.LOW): "预测腹部脂肪总量偏低。",
    ("Total_abdominal_fat_mass", Grade.NORMAL): "预测腹部脂肪总量处于常见范围。",
    ("Total_abdominal_fat_mass", Grade.BORDERLINE): "预测腹部脂肪总量略高，建议与 VAT、腰围一并评估中心性肥胖风险。",
    ("Total_abdominal_fat_mass", Grade.HIGH): "预测腹部脂肪总量偏高，提示腹腔周围脂肪负荷增加，宜重视代谢健康。",
    ("Visceral_adipose_tissue_mass", Grade.NORMAL): "预测内脏脂肪占体重比例未达常见关注阈值。",
    ("Visceral_adipose_tissue_mass", Grade.BORDERLINE): "预测内脏脂肪占体重比例偏高，建议减少精制碳水与久坐。",
    ("Visceral_adipose_tissue_mass", Grade.HIGH): "预测内脏脂肪占体重比例较高，建议优先改善饮食与活动习惯并定期体检。",
    ("Peripheral_fat_mass", Grade.LOW): "预测四肢（外周）脂肪偏低。",
    ("Peripheral_fat_mass", Grade.NORMAL): "预测外周脂肪处于常见范围。",
    ("Peripheral_fat_mass", Grade.BORDERLINE): "预测外周脂肪略高，多与全身脂肪量增加相关。",
    ("Peripheral_fat_mass", Grade.HIGH): "预测外周脂肪偏高，反映肢体脂肪储存较多，建议结合全身体脂与腰围综合评估。",
}


def _height_m(height_cm: float) -> float:
    return height_cm / 100.0


def _fat_kg(grams: float) -> float:
    return grams / 1000.0


def _fmi(grams: float, height_cm: float) -> float:
    h = _height_m(height_cm)
    if h <= 0:
        return float("nan")
    return _fat_kg(grams) / (h**2)


def _grade_from_thresholds(value: float, low_hi: float, normal_hi: float, border_hi: float) -> Grade:
    if value < low_hi:
        return Grade.LOW
    if value <= normal_hi:
        return Grade.NORMAL
    if value <= border_hi:
        return Grade.BORDERLINE
    return Grade.HIGH


def _grade_from_fmi(grams: float, height_cm: float, *, sex: int, bands: tuple[float, float, float]) -> Grade:
    fmi = _fmi(grams, height_cm)
    if fmi != fmi:  # nan
        return Grade.UNKNOWN
    low_hi, normal_hi, border_hi = bands
    if sex == 1:
        return _grade_from_thresholds(fmi, low_hi, normal_hi, border_hi)
    return _grade_from_thresholds(fmi, low_hi, normal_hi, border_hi)


def _sex_bands(sex: int, male: tuple[float, float, float], female: tuple[float, float, float]) -> tuple[float, float, float]:
    return male if sex == 1 else female


def grade_bmi(bmi: float) -> Grade:
    if bmi < _BMI_LOW:
        return Grade.LOW
    if bmi < _BMI_HIGH_BORDER:
        return Grade.NORMAL
    if bmi < _BMI_HIGH:
        return Grade.BORDERLINE
    return Grade.HIGH


def grade_body_fat_percent(pbf: float, *, sex: int) -> Grade:
    bands = _sex_bands(sex, _PBF_MALE, _PBF_FEMALE)
    return _grade_from_thresholds(pbf, *bands)


def grade_waist_cm(waist_cm: float, *, sex: int) -> Grade:
    border, high = (_WAIST_MALE if sex == 1 else _WAIST_FEMALE)
    if waist_cm < border:
        return Grade.NORMAL
    if waist_cm < high:
        return Grade.BORDERLINE
    return Grade.HIGH


def grade_vat_ratio(vat_g: float, weight_kg: float) -> Grade:
    if weight_kg <= 0:
        return Grade.UNKNOWN
    ratio = _fat_kg(vat_g) / weight_kg
    if ratio < _VAT_RATIO_BORDER:
        return Grade.NORMAL
    if ratio < _VAT_RATIO_HIGH:
        return Grade.BORDERLINE
    return Grade.HIGH


def grade_total_fat_mass(grams: float, *, height_cm: float, sex: int) -> Grade:
    bands = _sex_bands(sex, _FMI_TOTAL_MALE, _FMI_TOTAL_FEMALE)
    return _grade_from_fmi(grams, height_cm, sex=sex, bands=bands)


def grade_android_fat_mass(
    android_g: float,
    *,
    height_cm: float,
    sex: int,
    gynoid_g: Optional[float] = None,
) -> Grade:
    bands = _sex_bands(sex, _FMI_ANDROID_MALE, _FMI_ANDROID_FEMALE)
    by_fmi = _grade_from_fmi(android_g, height_cm, sex=sex, bands=bands)
    if gynoid_g is None or gynoid_g <= 0:
        return by_fmi
    by_ag = grade_android_gynoid_ratio(android_g, gynoid_g, sex=sex)
    return _worst_grade(by_fmi, by_ag)


def grade_gynoid_fat_mass(grams: float, *, height_cm: float, sex: int) -> Grade:
    bands = _sex_bands(sex, _FMI_GYNOID_MALE, _FMI_GYNOID_FEMALE)
    return _grade_from_fmi(grams, height_cm, sex=sex, bands=bands)


def grade_subcutaneous_fat_mass(grams: float, *, height_cm: float, sex: int) -> Grade:
    bands = _sex_bands(sex, _FMI_SUBCUT_MALE, _FMI_SUBCUT_FEMALE)
    return _grade_from_fmi(grams, height_cm, sex=sex, bands=bands)


def grade_total_abdominal_fat_mass(grams: float, *, height_cm: float, sex: int) -> Grade:
    bands = _sex_bands(sex, _FMI_ABDOM_MALE, _FMI_ABDOM_FEMALE)
    return _grade_from_fmi(grams, height_cm, sex=sex, bands=bands)


def grade_peripheral_fat_mass(grams: float, *, height_cm: float, sex: int) -> Grade:
    bands = _sex_bands(sex, _FMI_PERIPH_MALE, _FMI_PERIPH_FEMALE)
    return _grade_from_fmi(grams, height_cm, sex=sex, bands=bands)


def grade_android_gynoid_ratio(android_g: float, gynoid_g: float, *, sex: int) -> Grade:
    if gynoid_g <= 0:
        return Grade.UNKNOWN
    r = android_g / gynoid_g
    low_hi, normal_hi, border_hi = _sex_bands(sex, _AG_RATIO_MALE, _AG_RATIO_FEMALE)
    return _grade_from_thresholds(r, low_hi, normal_hi, border_hi)


_GRADE_ORDER = {Grade.LOW: 0, Grade.NORMAL: 1, Grade.BORDERLINE: 2, Grade.HIGH: 3, Grade.UNKNOWN: -1}


def _worst_grade(a: Grade, b: Grade) -> Grade:
    if _GRADE_ORDER.get(a, -1) >= _GRADE_ORDER.get(b, -1):
        return a
    return b


def _grade_fat_metric(metric_id: str, value_g: float, inp: ReportInput) -> Grade:
    sex = inp.sex
    h = inp.height_cm
    w = inp.weight_kg
    gynoid = inp.fat_predictions.get("Gynoid_fat_mass")

    if metric_id == "Percentage_body_fat":
        return grade_body_fat_percent(value_g, sex=sex)
    if metric_id == "Total_fat_mass":
        return grade_total_fat_mass(value_g, height_cm=h, sex=sex)
    if metric_id == "Android_fat_mass":
        return grade_android_fat_mass(value_g, height_cm=h, sex=sex, gynoid_g=gynoid)
    if metric_id == "Gynoid_fat_mass":
        return grade_gynoid_fat_mass(value_g, height_cm=h, sex=sex)
    if metric_id == "Subcutaneous_fat_mass":
        return grade_subcutaneous_fat_mass(value_g, height_cm=h, sex=sex)
    if metric_id == "Total_abdominal_fat_mass":
        return grade_total_abdominal_fat_mass(value_g, height_cm=h, sex=sex)
    if metric_id == "Visceral_adipose_tissue_mass":
        return grade_vat_ratio(value_g, w)
    if metric_id == "Peripheral_fat_mass":
        return grade_peripheral_fat_mass(value_g, height_cm=h, sex=sex)
    return Grade.UNKNOWN


def _tags(metric_id: str, grade: Grade, sex: int) -> tuple[str, ...]:
    sex_tag = "male" if sex == 1 else "female"
    return (f"metric:{metric_id}", f"grade:{grade.value}", f"sex:{sex_tag}")


def _assess(
    metric_id: str,
    value: float,
    unit: str,
    grade: Grade,
    *,
    sex: int,
    extra_note: str = "",
) -> MetricAssessment:
    display = _METRIC_DISPLAY.get(metric_id, metric_id)
    summary = _SUMMARY.get((metric_id, grade), f"{display}：{GRADE_LABEL_ZH[grade]}（规则默认说明）。")
    if grade != Grade.UNKNOWN:
        summary = f"【{GRADE_LABEL_ZH[grade]}】{summary}"
    if extra_note:
        summary = f"{summary} {extra_note}"
    return MetricAssessment(
        metric_id=metric_id,
        display_name=display,
        value=value,
        unit=unit,
        grade=grade,
        summary=summary,
        tags=_tags(metric_id, grade, sex) if grade != Grade.UNKNOWN else (),
    )


def _android_extra_note(inp: ReportInput) -> str:
    android = inp.fat_predictions.get("Android_fat_mass")
    gynoid = inp.fat_predictions.get("Gynoid_fat_mass")
    if android is None or gynoid is None or gynoid <= 0:
        return ""
    r = android / gynoid
    return f"（Android/Gynoid 脂肪质量比≈{r:.2f}，NHANES 常见约 0.5–1.9）"


def assess_all(inp: ReportInput) -> list[MetricAssessment]:
    units = inp.fat_units or {}
    sex = inp.sex
    out: list[MetricAssessment] = []

    out.append(_assess("BMI", inp.bmi, "", grade_bmi(inp.bmi), sex=sex))
    out.append(_assess("Waist", inp.waist_cm, "cm", grade_waist_cm(inp.waist_cm, sex=sex), sex=sex))

    for metric_id in FAT_TARGET_COLS:
        value = inp.fat_predictions.get(metric_id)
        if value is None:
            continue
        unit = units.get(metric_id, "g" if metric_id != "Percentage_body_fat" else "%")
        grade = _grade_fat_metric(metric_id, float(value), inp)
        extra = _android_extra_note(inp) if metric_id == "Android_fat_mass" else ""
        out.append(
            _assess(
                metric_id,
                float(value),
                unit,
                grade,
                sex=sex,
                extra_note=extra,
            )
        )

    return out


def executive_summary_lines(assessments: Iterable[MetricAssessment]) -> list[str]:
    highs = [a for a in assessments if a.grade == Grade.HIGH]
    borders = [a for a in assessments if a.grade == Grade.BORDERLINE]
    if not highs and not borders:
        return ["规则引擎摘要：主要指标未触发偏高/边缘阈值，请结合个人史与专业检查综合判断。"]
    parts: list[str] = []
    if highs:
        names = "、".join(a.display_name for a in highs)
        parts.append(f"需优先关注（规则判定偏高）：{names}。")
    if borders:
        names = "、".join(a.display_name for a in borders)
        parts.append(f"建议留意（规则判定边缘）：{names}。")
    parts.append("以下为各指标数值与固定说明；科普段落来自知识库检索，不代表诊断。")
    return parts
