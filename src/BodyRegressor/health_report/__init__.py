"""规则分级 + 知识库检索 + 可选 LLM 生成健康报告。"""

from .pipeline import build_health_report, format_report_markdown
from .types import HealthReport, ReportInput

__all__ = [
    "ReportInput",
    "HealthReport",
    "build_health_report",
    "format_report_markdown",
]
