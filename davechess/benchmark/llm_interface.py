"""Generic LLM API client (OpenAI-compatible)."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("davechess.benchmark")


class LLMClient:
    """Client for OpenAI-compatible chat completion APIs."""

    def __init__(self, base_url: str = "https://api.openai.com/v1",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 max_tokens: int = 256):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            )
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request and return the response text.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.

        Returns:
            The assistant's response text.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            raise

    def get_move_response(self, system_prompt: str, conversation: list[dict]) -> str:
        """Get a move response from the LLM.

        Args:
            system_prompt: System message with game rules and examples.
            conversation: Ongoing conversation messages.

        Returns:
            The raw text response from the LLM.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        return self.chat(messages)
