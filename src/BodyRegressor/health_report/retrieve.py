from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .types import MetricAssessment, RetrievedSnippet

_DEFAULT_KB = Path(__file__).resolve().parent / "knowledge" / "snippets.json"


def load_snippets(kb_path: Path | None = None) -> list[dict]:
    path = kb_path or _DEFAULT_KB
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"知识库格式错误，应为 JSON 数组: {path}")
    return data


def _collect_query_tags(assessments: Iterable[MetricAssessment]) -> set[str]:
    tags: set[str] = {"general"}
    for a in assessments:
        tags.update(a.tags)
    return tags


def retrieve_snippets(
    assessments: Sequence[MetricAssessment],
    *,
    kb_path: Path | None = None,
    top_k: int = 5,
) -> list[RetrievedSnippet]:
    """按标签重叠打分检索（轻量 RAG，无需向量库）。"""
    query_tags = _collect_query_tags(assessments)
    raw = load_snippets(kb_path)
    scored: list[RetrievedSnippet] = []

    for item in raw:
        item_tags = set(item.get("tags") or [])
        overlap = query_tags & item_tags
        if not overlap:
            continue
        score = float(len(overlap))
        scored.append(
            RetrievedSnippet(
                snippet_id=str(item["id"]),
                title=str(item.get("title", item["id"])),
                body=str(item["body"]),
                source=str(item.get("source", "unknown")),
                matched_tags=tuple(sorted(overlap)),
                score=score,
            )
        )

    scored.sort(key=lambda s: (-s.score, s.snippet_id))
    seen: set[str] = set()
    out: list[RetrievedSnippet] = []
    for s in scored:
        if s.snippet_id in seen:
            continue
        seen.add(s.snippet_id)
        out.append(s)
        if len(out) >= top_k:
            break
    return out
