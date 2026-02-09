"""LLM provider abstraction layer."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call with tool support."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def call(self, prompt: str, system_prompt: str | None = None) -> str:
        """Make a simple text completion call."""
        pass

    @abstractmethod
    async def call_structured(
        self, prompt: str, response_format: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Make a call expecting structured JSON output."""
        pass

    @abstractmethod
    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Make a call with tool/function calling support."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url

        try:
            from openai import AsyncOpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = AsyncOpenAI(**kwargs)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    async def call(self, prompt: str, system_prompt: str | None = None) -> str:
        """Make a simple text completion call."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content or ""

    async def call_structured(
        self, prompt: str, response_format: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Make a call expecting structured JSON output."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Make a call with tool/function calling support."""
        full_messages: list[dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls if present
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        tool_responses: dict[str, list[ToolCall]] | None = None,
        default_response: str = "Mock response",
    ):
        self.responses = responses or {}
        self.tool_responses = tool_responses or {}
        self.default_response = default_response
        self.call_history: list[dict[str, Any]] = []

    async def call(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a mock response."""
        self.call_history.append({"type": "call", "prompt": prompt, "system_prompt": system_prompt})

        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        return self.default_response

    async def call_structured(
        self, prompt: str, response_format: dict[str, Any], system_prompt: str | None = None
    ) -> dict[str, Any]:
        """Return a mock structured response."""
        self.call_history.append(
            {"type": "call_structured", "prompt": prompt, "system_prompt": system_prompt}
        )

        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                if isinstance(response, str):
                    return json.loads(response)
                return response

        return {"result": self.default_response}

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Return a mock response with optional tool calls."""
        self.call_history.append(
            {"type": "call_with_tools", "messages": messages, "system_prompt": system_prompt}
        )

        # Check if there are tool responses for any keyword in messages
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        for key, tool_calls in self.tool_responses.items():
            if key.lower() in last_user_msg.lower():
                return LLMResponse(content=None, tool_calls=tool_calls, finish_reason="tool_calls")

        # Default: return text response
        response_text = self.default_response
        for key, response in self.responses.items():
            if key.lower() in last_user_msg.lower():
                response_text = response
                break

        return LLMResponse(content=response_text, tool_calls=[], finish_reason="stop")
