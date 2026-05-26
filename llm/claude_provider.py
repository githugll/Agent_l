"""Claude / Anthropic API provider."""

import logging
import os
from typing import AsyncIterator

import httpx
from anthropic import Anthropic, AsyncAnthropic

from .base import LLMProvider

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """Use Claude from Anthropic API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str = None, base_url: str = None):
        """
        Args:
            model: Model ID (e.g. claude-sonnet-4-6)
            api_key: Anthropic API key (or read from ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL (for proxy services)
        """
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set via env var or pass api_key param."
            )

        # Create httpx clients with trust_env=False so they don't auto-detect
        # macOS system proxies that may be unavailable in the terminal environment
        sync_client = httpx.Client(trust_env=False, timeout=60.0)
        async_client = httpx.AsyncClient(trust_env=False, timeout=60.0)

        client_kwargs = {"api_key": self.api_key, "http_client": sync_client}
        async_kwargs = {"api_key": self.api_key, "http_client": async_client}
        if base_url:
            client_kwargs["base_url"] = base_url
            async_kwargs["base_url"] = base_url

        self.client = Anthropic(**client_kwargs)
        self.async_client = AsyncAnthropic(**async_kwargs)
        logger.info(f"Initialized ClaudeProvider: {model}")

    async def stream_chat(
        self,
        messages: list[dict],
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream from Claude API."""
        # Convert messages format if needed
        # Claude uses same format as OpenAI for messages

        stream = self.async_client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=messages,
            temperature=temperature,
        )
        async with stream as s:
            async for text in s.text_stream:
                yield text

    async def chat(
        self,
        messages: list[dict],
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Get complete response from Claude."""
        full_response = ""
        async for token in self.stream_chat(messages, system, temperature, max_tokens):
            full_response += token
        return full_response
