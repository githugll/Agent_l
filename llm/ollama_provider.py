"""Ollama local LLM provider."""

import logging
from typing import AsyncIterator

import ollama

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Use local Ollama instance."""

    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        logger.info(f"Initialized OllamaProvider: {model}")

    async def stream_chat(
        self,
        messages: list[dict],
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream from local Ollama."""
        if system:
            messages = [{"role": "system", "content": system}] + messages

        stream = self.client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )

        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    async def chat(
        self,
        messages: list[dict],
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Get complete response from Ollama."""
        full_response = ""
        async for token in self.stream_chat(messages, system, temperature, max_tokens):
            full_response += token
        return full_response
