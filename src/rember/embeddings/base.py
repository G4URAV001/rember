"""
Abstract base class for embedding providers.

All embedding implementations must subclass EmbeddingProvider and implement:
  - embed()       : embed a batch of document texts (RETRIEVAL_DOCUMENT task type)
  - embed_query() : embed a single query text (RETRIEVAL_QUERY task type)
  - dimension     : the dimensionality of returned vectors
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of this provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of embedding vectors produced by this provider."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of document texts.

        Args:
            texts: List of strings to embed.

        Returns:
            Float32 numpy array of shape (len(texts), dimension).
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Uses RETRIEVAL_QUERY task type where supported (Gemini), which is
        optimised for query-side asymmetric retrieval.

        Args:
            query: The question or search string.

        Returns:
            Float32 numpy array of shape (1, dimension).
        """
