from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# DeepSeek OpenAI 兼容接口（见 https://api-docs.deepseek.com/）
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    api_key: str
    base_url: str
    model: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dotenv(path: Optional[Path] = None) -> None:
    """加载项目根目录 .env（无第三方依赖）。已存在的环境变量不会被覆盖。"""
    env_path = path or (_repo_root() / ".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_base_url(base_url: str) -> str:
    """OpenAI SDK 风格可能带 /v1；本项目的请求路径为 {base}/chat/completions。"""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def resolve_llm_settings(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    load_env: bool = True,
) -> Optional[LlmSettings]:
    if load_env:
        load_dotenv()

    prov = (provider or os.environ.get("BODYREG_LLM_PROVIDER", "deepseek")).strip().lower()

    if prov == "deepseek":
        key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("BODYREG_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        base = base_url or os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        mdl = model or os.environ.get("BODYREG_LLM_MODEL") or os.environ.get("DEEPSEEK_MODEL")
        default_base = DEEPSEEK_DEFAULT_BASE_URL
        default_model = DEEPSEEK_DEFAULT_MODEL
    elif prov == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("BODYREG_LLM_API_KEY")
        base = base_url or os.environ.get("OPENAI_BASE_URL")
        mdl = model or os.environ.get("BODYREG_LLM_MODEL")
        default_base = "https://api.openai.com/v1"
        default_model = "gpt-4o-mini"
    else:
        raise ValueError(f"未知 LLM provider: {prov}，支持 deepseek / openai")

    if not key:
        return None

    return LlmSettings(
        provider=prov,
        api_key=key.strip(),
        base_url=normalize_base_url(base or default_base),
        model=(mdl or default_model).strip(),
    )
