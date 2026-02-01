"""LLM API clients for benchmark evaluation.

Includes the original OpenAI-compatible client and a new ToolUseLLMClient
that supports tool-use protocols for both OpenAI and Anthropic.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from davechess.benchmark.token_tracker import TokenUsage

logger = logging.getLogger("davechess.benchmark")


@dataclass
class ToolCallResult:
    """A single tool call extracted from an LLM response."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response from an LLM API call with tool-use support."""
    text: Optional[str] = None
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", "max_tokens"


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


class ToolUseLLMClient:
    """LLM client with tool-use support and token tracking.

    Supports both OpenAI and Anthropic providers with a unified interface.
    """

    def __init__(self, provider: str = "openai",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 4096):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

        if provider == "openai":
            self._init_openai(base_url, api_key)
        elif provider == "anthropic":
            self._init_anthropic(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    def _init_openai(self, base_url: Optional[str], api_key: Optional[str]):
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=base_url or "https://api.openai.com/v1",
                api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            )
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    def _init_anthropic(self, api_key: Optional[str]):
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            )
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    def chat_with_tools(self, system: str, messages: list[dict],
                        tools: list[dict]) -> LLMResponse:
        """Send a chat request with tool definitions.

        Args:
            system: System message content.
            messages: Conversation messages.
            tools: Tool definitions (in provider-native format).

        Returns:
            LLMResponse with tool_calls, text, and token usage.
        """
        if self.provider == "openai":
            return self._openai_chat(system, messages, tools)
        else:
            return self._anthropic_chat(system, messages, tools)

    def _openai_chat(self, system: str, messages: list[dict],
                     tools: list[dict]) -> LLMResponse:
        """OpenAI tool-use implementation."""
        full_messages = [{"role": "system", "content": system}] + messages

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

        msg = response.choices[0].message
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append(ToolCallResult(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        stop = response.choices[0].finish_reason
        stop_reason = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }.get(stop, stop or "end_turn")

        return LLMResponse(
            text=msg.content,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
        )

    def _anthropic_chat(self, system: str, messages: list[dict],
                        tools: list[dict]) -> LLMResponse:
        """Anthropic tool-use implementation."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallResult(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        stop_reason = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
        }.get(response.stop_reason, response.stop_reason or "end_turn")

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=stop_reason,
        )
