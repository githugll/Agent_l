"""Claude / Anthropic API provider."""

import logging
from typing import AsyncIterator

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
        import os

        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Set via env var or pass api_key param."
            )

        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = Anthropic(**client_kwargs)
        self.async_client = AsyncAnthropic(**client_kwargs)
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

        with self.async_client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "",
            messages=messages,
            temperature=temperature,
        ) as stream:
            async for text in stream.text_stream:
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
