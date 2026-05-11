"""
LLM client wrapper supporting OpenAI and Anthropic.

Provides a unified interface so the rest of the codebase doesn't need to know
which provider is being used.  Set LLM_EVAL_MODE=mock in .env to run without
real API calls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "anthropic", "mock"]


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: Provider
    usage: dict = field(default_factory=dict)


class MockLLMClient:
    """A deterministic mock client that returns canned responses — no API key needed."""

    model = "mock-gpt"
    provider: Provider = "mock"

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        response = f"[MOCK] This is a mock response to: '{last_user[:60]}...'"
        return LLMResponse(content=response, model=self.model, provider=self.provider)

    def complete(self, prompt: str, **kwargs) -> str:
        return f"[MOCK] Completion for: '{prompt[:60]}...'"


class OpenAIClient:
    """Thin wrapper around the OpenAI chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai>=1.0") from exc

        self.model = model
        self.provider: Provider = "openai"
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            usage=dict(response.usage) if response.usage else {},
        )

    def complete(self, prompt: str, **kwargs) -> str:
        resp = self.chat([Message(role="user", content=prompt)], **kwargs)
        return resp.content


class AnthropicClient:
    """Thin wrapper around the Anthropic messages API."""

    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        self.model = model
        self.provider: Provider = "anthropic"
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def chat(self, messages: list[Message], **kwargs) -> LLMResponse:
        system_msgs = [m.content for m in messages if m.role == "system"]
        user_msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        system_text = "\n".join(system_msgs) if system_msgs else None

        create_kwargs = dict(model=self.model, max_tokens=1024, messages=user_msgs, **kwargs)
        if system_text:
            create_kwargs["system"] = system_text

        response = self._client.messages.create(**create_kwargs)
        content = response.content[0].text if response.content else ""
        return LLMResponse(content=content, model=self.model, provider=self.provider)

    def complete(self, prompt: str, **kwargs) -> str:
        resp = self.chat([Message(role="user", content=prompt)], **kwargs)
        return resp.content


def get_client(
    provider: Provider | None = None,
    model: str | None = None,
) -> MockLLMClient | OpenAIClient | AnthropicClient:
    """
    Factory function.  Priority:
    1. LLM_EVAL_MODE=mock → MockLLMClient (no API key needed)
    2. provider kwarg
    3. Auto-detect from available API keys in environment
    """
    mode = os.getenv("LLM_EVAL_MODE", "").lower()
    if mode == "mock" or provider == "mock":
        return MockLLMClient()

    resolved_provider = provider or _auto_detect_provider()

    if resolved_provider == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    if resolved_provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-haiku-20240307")

    # Fallback
    return MockLLMClient()


def _auto_detect_provider() -> Provider:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"
