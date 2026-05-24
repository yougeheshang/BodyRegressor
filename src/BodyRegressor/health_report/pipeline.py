from __future__ import annotations

from pathlib import Path
from typing import Optional

from BodyRegressor.features import FAT_TARGET_COLS
from BodyRegressor.metrics import fat_units_for_targets, summarize_predictions

from .llm import generate_narrative, sanitize_snippet_citations
from .retrieve import retrieve_snippets
from .rules import assess_all, executive_summary_lines
from .types import HealthReport, ReportInput

DISCLAIMER = (
    "【免责声明】本报告由算法预测值 + 规则引擎 + 知识库检索自动生成，仅供健康科普与生活方式参考，"
    "不构成医疗诊断、处方或治疗建议。如有不适或慢性病，请咨询执业医师。"
)


def report_input_from_predictions(
    *,
    age: float,
    sex: int,
    height_cm: float,
    weight_kg: float,
    waist_cm: float,
    hip_cm: float,
    bmi: float,
    bri: float,
    preds,
    target_names: Optional[list[str]] = None,
) -> ReportInput:
    names = target_names or list(FAT_TARGET_COLS)
    units = fat_units_for_targets(names)
    summary = summarize_predictions(preds, names, units=units)
    fat_predictions = {name: float(summary[name]["pred"]) for name in names if name in summary}
    return ReportInput(
        age=age,
        sex=sex,
        height_cm=height_cm,
        weight_kg=weight_kg,
        waist_cm=waist_cm,
        hip_cm=hip_cm,
        bmi=bmi,
        bri=bri,
        fat_predictions=fat_predictions,
        fat_units=units,
    )


def build_health_report(
    inp: ReportInput,
    *,
    kb_path: Path | None = None,
    top_k: int = 5,
    use_llm: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> HealthReport:
    assessments = assess_all(inp)
    retrieved = retrieve_snippets(assessments, kb_path=kb_path, top_k=top_k)
    sex_str = "男" if inp.sex == 1 else "女"
    profile = [
        f"年龄 {inp.age:.0f} 岁，性别 {sex_str}",
        f"身高 {inp.height_cm:.1f} cm，体重 {inp.weight_kg:.1f} kg",
        f"腰围 {inp.waist_cm:.1f} cm，臀围 {inp.hip_cm:.1f} cm",
        f"BMI {inp.bmi:.2f}，BRI {inp.bri:.1f}",
    ]
    narrative, mode, llm_meta = generate_narrative(
        assessments=assessments,
        snippets=retrieved,
        profile_lines=profile,
        use_llm=use_llm,
        provider=llm_provider,
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    meta = {"narrative_mode": mode, **llm_meta}
    return HealthReport(
        disclaimer=DISCLAIMER,
        executive_summary=executive_summary_lines(assessments),
        assessments=assessments,
        retrieved=retrieved,
        narrative=narrative,
        meta=meta,
    )


def format_report_markdown(report: HealthReport) -> str:
    lines: list[str] = [
        "# 健康解读报告",
        "",
        report.disclaimer,
        "",
        "## 摘要",
        "",
    ]
    for s in report.executive_summary:
        lines.append(f"- {s}")
    lines.extend(["", "## 指标解读（规则引擎，确定性）", ""])
    for a in report.assessments:
        val = f"{a.value:.2f}{a.unit}" if a.unit else f"{a.value:.2f}"
        lines.append(f"### {a.display_name}")
        lines.append(f"- **测量/预测值**：{val}")
        lines.append(f"- **规则分级**：{a.grade.value}")
        lines.append(f"- **说明**：{a.summary}")
        lines.append("")

    if report.retrieved:
        lines.extend(["## 参考知识库摘录", ""])
        for s in report.retrieved:
            lines.append(f"### {s.title}")
            lines.append(f"来源：{s.source}")
            lines.append("")
            lines.append(s.body)
            lines.append("")

    lines.extend(["## 科普与生活方式", "", report.narrative, ""])
    mode = report.meta.get("narrative_mode", "unknown")
    model = report.meta.get("model", "")
    provider = report.meta.get("provider", "")
    footer = f"报告生成模式：{mode}"
    if provider or model:
        footer += f"（{provider}/{model}）".replace("(/)", "")
    lines.append(f"---\n*{footer}*")
    md = "\n".join(lines)
    return sanitize_snippet_citations(md, report.retrieved)
