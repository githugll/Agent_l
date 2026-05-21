"""LLM utilities: Ollama availability, model selection."""

import json
import logging
import os

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


def get_available_models() -> list[str]:
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
