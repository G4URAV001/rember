"""
Abstract base class for LLM providers.

All LLM implementations must subclass LLMProvider and implement:
  - generate()               : free-form text generation
  - extract_info()           : structured info extraction from text (Phase 1)
  - extract_info_multimodal(): structured info extraction from image/video (Phase 2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rember.models import ExtractedInfo


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of this provider."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The user-facing prompt / question.
            system_instruction: Optional system-level instruction to prepend.

        Returns:
            Generated text string.
        """

    @abstractmethod
    def extract_info(
        self,
        content: str,
        document_id: str,
        content_type: str = "text",
        extra_metadata: dict | None = None,
    ) -> ExtractedInfo:
        """
        Use the LLM to extract structured information from raw text content.

        Args:
            content: The raw text (or description) to process.
            document_id: Parent document ID for the returned ExtractedInfo.
            content_type: Hint about content type ("text", "file", etc.).
            extra_metadata: Additional metadata to carry forward.

        Returns:
            ExtractedInfo with summary, key_facts, and topics.
        """

    def extract_info_multimodal(
        self,
        content_parts: list[Any],
        document_id: str,
        content_type: str = "image",
        extra_metadata: dict | None = None,
    ) -> ExtractedInfo:
        """
        Extract structured information from multimodal content (image/video + text).

        Default implementation concatenates any text parts and calls extract_info()
        as a text-only fallback. Subclasses should override this to use native
        multimodal capabilities.

        Args:
            content_parts: Mixed list of content — may include text strings,
                          image bytes wrapped in provider-specific Part objects,
                          or uploaded file references.
            document_id: Parent document ID.
            content_type: "image" or "video".
            extra_metadata: Metadata to carry forward.

        Returns:
            ExtractedInfo with summary, key_facts, and topics.
        """
        # Default fallback: extract any text parts and process as text
        text_parts = [p for p in content_parts if isinstance(p, str)]
        combined_text = "\n".join(text_parts) or f"[{content_type} content — no text available]"

        return self.extract_info(
            content=combined_text,
            document_id=document_id,
            content_type=content_type,
            extra_metadata=extra_metadata,
        )
