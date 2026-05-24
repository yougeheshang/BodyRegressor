from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Grade(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    BORDERLINE = "borderline"
    HIGH = "high"
    UNKNOWN = "unknown"


GRADE_LABEL_ZH = {
    Grade.LOW: "偏低",
    Grade.NORMAL: "正常",
    Grade.BORDERLINE: "边缘/需关注",
    Grade.HIGH: "偏高",
    Grade.UNKNOWN: "无法分级",
}


@dataclass(frozen=True)
class MetricAssessment:
    """单指标规则分级结果（确定性，可审计）。"""

    metric_id: str
    display_name: str
    value: float
    unit: str
    grade: Grade
    summary: str
    tags: tuple[str, ...] = ()


@dataclass
class ReportInput:
    """生成报告所需的测量与预测上下文。"""

    age: float
    sex: int  # 1=Male, 2=Female（与项目其它模块一致）
    height_cm: float
    weight_kg: float
    waist_cm: float
    hip_cm: float
    bmi: float
    bri: float
    fat_predictions: dict[str, float]
    fat_units: Optional[dict[str, str]] = None


@dataclass
class RetrievedSnippet:
    snippet_id: str
    title: str
    body: str
    source: str
    matched_tags: tuple[str, ...]
    score: float


@dataclass
class HealthReport:
    disclaimer: str
    executive_summary: list[str]
    assessments: list[MetricAssessment]
    retrieved: list[RetrievedSnippet]
    narrative: str
    meta: dict[str, Any] = field(default_factory=dict)
