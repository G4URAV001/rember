"""Unit tests for LLMRegistry."""

from __future__ import annotations

import pytest

from rember.llm.registry import LLMRegistry
from rember.models import ExtractedInfo


class AnotherMockLLM:
    """Minimal mock for registry tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    def generate(self, prompt, system_instruction=None):
        return f"Response from {self._name}"

    def extract_info(self, content, document_id, content_type="text", extra_metadata=None):
        return ExtractedInfo(
            document_id=document_id,
            summary=f"Summary from {self._name}",
            key_facts=[],
        )


class TestLLMRegistry:
    def test_default_provider(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        assert registry.get_provider("default") is mock_llm
        assert registry.get_provider("any_task") is mock_llm

    def test_register_additional_provider(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        other = AnotherMockLLM("openai")
        registry.register(other)
        assert "openai" in registry.registered_providers

    def test_task_routing(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        other = AnotherMockLLM("openai")
        registry.register(other)
        registry.set_task_route("extraction", "openai")

        provider = registry.get_provider("extraction")
        assert provider.provider_name == "openai"

    def test_unrouted_task_uses_default(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        provider = registry.get_provider("summarization")
        assert provider is mock_llm

    def test_set_task_route_unknown_provider_raises(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        with pytest.raises(ValueError, match="not registered"):
            registry.set_task_route("extraction", "unknown_provider")

    def test_set_default_changes_fallback(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        other = AnotherMockLLM("other")
        registry.register(other)
        registry.set_default("other")
        assert registry.default_provider_name == "other"

    def test_set_default_unknown_raises(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        with pytest.raises(ValueError, match="not registered"):
            registry.set_default("nonexistent")

    def test_load_routing_from_config(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        other = AnotherMockLLM("gemini_pro")
        registry.register(other)

        registry.load_routing_from_config({
            "extraction": "gemini_pro",
            "missing": "nonexistent",   # should be skipped, not raise
        })

        assert registry.get_provider("extraction").provider_name == "gemini_pro"
        assert registry.get_provider("missing") is mock_llm  # falls back

    def test_routing_table(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        other = AnotherMockLLM("x")
        registry.register(other)
        registry.set_task_route("q", "x")

        table = registry.routing_table()
        assert table == {"q": "x"}

    def test_registered_providers_list(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        assert "mock" in registry.registered_providers
