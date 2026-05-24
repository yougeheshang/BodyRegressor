from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional, Sequence

from .llm_config import resolve_llm_settings
from .types import MetricAssessment, RetrievedSnippet

# LLM 常输出的错误引用格式 → 替换为可读出处或删除
_SNIPPET_COLON_RE = re.compile(r"\[snippet:\s*([a-zA-Z0-9_-]+)\s*\]", re.IGNORECASE)
_SNIPPET_ID_LITERAL_RE = re.compile(r"\[snippet_id\]", re.IGNORECASE)
_BACKTICK_ID_RE = re.compile(r"`\[([a-zA-Z][a-zA-Z0-9_-]*)\]`")
_BRACKET_ID_RE = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_-]*)\]")


def sanitize_snippet_citations(text: str, snippets: Sequence[RetrievedSnippet]) -> str:
    """移除/替换 [snippet:xxx]、`[id]`、[id] 等 Streamlit 易误解析的引用占位。"""
    id_to_title = {s.snippet_id: s.title for s in snippets}
    valid = set(id_to_title)

    def _to_title(sid: str) -> str:
        if sid in valid:
            return f"（参考：{id_to_title[sid]}）"
        return ""

    text = _SNIPPET_COLON_RE.sub(lambda m: _to_title(m.group(1)), text)
    text = _SNIPPET_ID_LITERAL_RE.sub("", text)
    text = _BACKTICK_ID_RE.sub(lambda m: _to_title(m.group(1)), text)

    def _bracket(m: re.Match[str]) -> str:
        return _to_title(m.group(1))

    text = _BRACKET_ID_RE.sub(_bracket, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([，。；：])", r"\1", text)
    text = re.sub(r"（参考：）", "", text)
    return text.strip()


def _openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float = 60.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"]).strip()


def generate_narrative(
    *,
    assessments: Sequence[MetricAssessment],
    snippets: Sequence[RetrievedSnippet],
    profile_lines: Sequence[str],
    use_llm: bool = True,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str, dict]:
    """
    返回 (narrative, mode, llm_meta)。
    mode 为 ``llm`` 或 ``template``。
    """
    llm_meta: dict = {}
    if not use_llm:
        return _template_narrative(snippets), "template", llm_meta

    settings = resolve_llm_settings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    if settings is None:
        return _template_narrative(snippets), "template", llm_meta

    llm_meta = {
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url,
    }

    assess_text = "\n".join(
        f"- {a.display_name}: {a.value}{a.unit} → {a.grade.value}；{a.summary}" for a in assessments
    )
    id_list = "、".join(s.snippet_id for s in snippets) or "（无）"
    kb_text = "\n\n".join(
        f"（出处 id={s.snippet_id}）{s.title}（来源 {s.source}）\n{s.body}" for s in snippets
    )
    system = (
        "你是健康科普写作助手。只能根据用户提供的「规则结论」与「知识库片段」写作，"
        "不得编造医学诊断、药物剂量或检查结果。不得违背规则分级结论。"
        "输出简体中文 Markdown，分节：## 生活方式建议、## 指标理解、## 注意事项。"
        "若需标注依据，在句末写知识条目的中文标题，例如（参考：BMI 是什么），"
        "禁止使用 [snippet:...]、[snippet_id]、方括号 id、或未提供的占位符。"
        "不要用bullet list分点论述，请分段输出。"
    )
    user = (
        "【用户概况】\n"
        + "\n".join(profile_lines)
        + "\n\n【规则引擎结论（必须遵守）】\n"
        + assess_text
        + f"\n\n【知识库片段 id 列表（仅可引用以下 id 对应内容）】\n{id_list}"
        + "\n\n【知识库片段正文】\n"
        + kb_text
    )
    debug = os.environ.get("BODYREG_LLM_DEBUG", "").lower() in ("1", "true", "yes")
    try:
        text = _openai_compatible_chat(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            system=system,
            user=user,
        )
        text = sanitize_snippet_citations(text, snippets)
        return text, "llm", llm_meta
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError, IndexError) as exc:
        if debug:
            detail = exc.read().decode("utf-8", errors="replace") if isinstance(exc, urllib.error.HTTPError) else str(exc)
            print(f"[BODYREG_LLM_DEBUG] LLM 调用失败: {detail}", flush=True)
        return _template_narrative(snippets), "template", llm_meta


def _template_narrative(snippets: Sequence[RetrievedSnippet]) -> str:
    if not snippets:
        return "（未检索到知识库片段；请扩充 knowledge/snippets.json 或检查标签。）"
    lines = ["## 科普摘录（知识库检索，非诊断）", ""]
    for s in snippets:
        lines.append(f"### {s.title}")
        lines.append(f"来源：{s.source}")
        lines.append("")
        lines.append(s.body)
        lines.append("")
    return "\n".join(lines)
