"""LLM utilities: Ollama availability, model selection, query translation."""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)


def ollama_available() -> bool:
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return resp.status == 200
    except Exception:
        return False


def anthropic_key_available() -> bool:
    """Check if ANTHROPIC_API_KEY or LLM_API_KEY env var is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY"))


def llm_available() -> bool:
    """Check if any LLM provider is available (Ollama or Anthropic)."""
    return ollama_available() or anthropic_key_available()


def get_available_models() -> list[str]:
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def get_qwen_model() -> str | None:
    """Find available qwen3 model, preferring larger ones."""
    models = get_available_models()
    for preferred in ["qwen3:8b", "qwen3:4b", "qwen3:1.7b", "qwen3:0.6b"]:
        if preferred in models:
            return preferred
    for m in models:
        if "qwen3" in m:
            return m
    return None


def get_all_available_models() -> list[str]:
    """Return all available model IDs for UI selector (Claude + Ollama)."""
    models = []
    if anthropic_key_available():
        models += ["claude-sonnet-4-6", "claude-haiku-4", "claude-opus-4-7"]
    for m in get_available_models():
        models.append(f"ollama:{m}")
    return models


def is_chinese(text: str) -> bool:
    """Return True if text contains CJK Unified Ideograph characters."""
    return bool(re.search(r'[一-鿿]', text))


async def translate_query(query: str, provider=None) -> str | None:
    """Translate a Chinese 3GPP query to English using the LLM.

    Returns the English translation, or None if not needed/possible.
    """
    if not is_chinese(query):
        return None
    if provider is None:
        return None
    try:
        result = await provider.chat(
            messages=[{"role": "user", "content": (
                "将以下 3GPP 通信技术相关的中文查询翻译为英文，"
                "保留原有的英文缩写不变，只输出翻译结果：\n\n" + query
            )}],
            system="You are a concise 3GPP technical term translator. Output only the English translation.",
            temperature=0,
            max_tokens=256,
        )
        return result.strip() if result else None
    except Exception as e:
        logger.warning(f"Query translation failed: {e}")
        return None
