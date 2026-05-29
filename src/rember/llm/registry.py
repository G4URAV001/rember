"""
LLM Registry — per-task provider routing.

Usage:
    registry = LLMRegistry(default_provider=gemini_provider)
    registry.register("openai", openai_provider)
    registry.set_task_route("extraction", "gemini")
    registry.set_task_route("query_answering", "openai")

    provider = registry.get_provider("extraction")  # → gemini
    provider = registry.get_provider("unknown_task") # → default
"""

from __future__ import annotations

import logging

from rember.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LLMRegistry:
    """
    Manages multiple LLM providers and routes tasks to the right one.

    Task routing:
      - If a task has an explicit route → use that provider.
      - Otherwise → use the default provider.
    """

    def __init__(self, default_provider: LLMProvider) -> None:
        self._providers: dict[str, LLMProvider] = {
            default_provider.provider_name: default_provider
        }
        self._default_name: str = default_provider.provider_name
        self._task_routing: dict[str, str] = {}  # task_name → provider_name

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, provider: LLMProvider) -> None:
        """Register an LLM provider under its provider_name."""
        self._providers[provider.provider_name] = provider
        logger.debug("Registered LLM provider: %s", provider.provider_name)

    def set_default(self, provider_name: str) -> None:
        """Change the default provider."""
        if provider_name not in self._providers:
            raise ValueError(
                f"Provider '{provider_name}' is not registered. "
                f"Available: {list(self._providers.keys())}"
            )
        self._default_name = provider_name

    def set_task_route(self, task: str, provider_name: str) -> None:
        """Route a specific task to a specific provider."""
        if provider_name not in self._providers:
            raise ValueError(
                f"Provider '{provider_name}' is not registered. "
                f"Available: {list(self._providers.keys())}"
            )
        self._task_routing[task] = provider_name
        logger.debug("Routed task '%s' → provider '%s'", task, provider_name)

    def load_routing_from_config(self, routing: dict[str, str]) -> None:
        """
        Bulk-load task routing from a config dict.
        Unknown providers are logged and skipped (not an error).
        """
        for task, provider_name in routing.items():
            if provider_name in self._providers:
                self._task_routing[task] = provider_name
            else:
                logger.warning(
                    "Task routing for '%s' references unknown provider '%s'. Skipping.",
                    task,
                    provider_name,
                )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_provider(self, task: str = "default") -> LLMProvider:
        """Return the provider for a given task, falling back to the default."""
        provider_name = self._task_routing.get(task, self._default_name)
        provider = self._providers.get(provider_name)
        if provider is None:
            logger.warning(
                "Provider '%s' for task '%s' not found. Using default '%s'.",
                provider_name,
                task,
                self._default_name,
            )
            provider = self._providers[self._default_name]
        return provider

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_providers(self) -> list[str]:
        return list(self._providers.keys())

    @property
    def default_provider_name(self) -> str:
        return self._default_name

    def routing_table(self) -> dict[str, str]:
        """Return current task → provider routing table."""
        return dict(self._task_routing)
