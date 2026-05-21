"""LLM factory and configuration."""

import logging
import os
from typing import Literal

from .base import LLMProvider
from .claude_provider import ClaudeProvider
from .ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for LLM provider selection."""

    def __init__(
        self,
        provider: Literal["ollama", "claude"] = "ollama",
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Args:
            provider: "ollama" or "claude"
            model: Model name (auto-selected if None)
            api_key: For cloud providers (Claude)
            base_url: Custom API base URL (for proxy services)
            ollama_url: Base URL for local Ollama
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.ollama_url = ollama_url

    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        provider = os.environ.get("LLM_PROVIDER", "ollama")
        model = os.environ.get("LLM_MODEL")
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

        # Auto-detect: if API key is set without explicit provider, use Claude
        if not os.environ.get("LLM_PROVIDER") and api_key:
            provider = "claude"

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            ollama_url=ollama_url,
        )


def create_llm_provider(config: LLMConfig = None) -> LLMProvider:
    """Factory function to create LLM provider."""
    if config is None:
        config = LLMConfig.from_env()

    if config.provider == "claude":
        model = config.model or "claude-sonnet-4-6"
        logger.info(f"Creating Claude provider: {model} (base_url={config.base_url})")
        return ClaudeProvider(model=model, api_key=config.api_key, base_url=config.base_url)

    elif config.provider == "ollama":
        model = config.model or _find_ollama_model()
        logger.info(f"Creating Ollama provider: {model}")
        return OllamaProvider(model=model, base_url=config.ollama_url)

    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _find_ollama_model() -> str:
    """Auto-detect available Ollama model."""
    try:
        import urllib.request

        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        import json

        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]

        # Prefer qwen models
        for preferred in ["qwen3:8b", "qwen2.5:7b", "qwen3:4b"]:
            if preferred in models:
                return preferred

        # Fallback to first available
        if models:
            return models[0]

        logger.warning("No Ollama models found, defaulting to qwen3:8b")
        return "qwen3:8b"

    except Exception as e:
        logger.warning(f"Failed to detect Ollama models: {e}, using qwen3:8b")
        return "qwen3:8b"
