"""
Gemini embedding provider.

Uses gemini-embedding-001 (3072 dimensions) via the google-genai SDK.

Key design points:
  - Uses RETRIEVAL_DOCUMENT task type for document embedding
  - Uses RETRIEVAL_QUERY task type for query embedding (asymmetric retrieval)
  - Batches requests (up to 250 texts / 20K tokens per call)
  - Returns float32 numpy arrays ready for FAISS
  - Implements exponential backoff on rate-limit errors
"""

from __future__ import annotations

import logging
import time

import numpy as np

from rember.config import EmbeddingProviderConfig
from rember.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Gemini API limits per request
_MAX_TEXTS_PER_BATCH = 250
_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0

# Default dimension for gemini-embedding-001
_DEFAULT_DIMENSION = 3072


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Google's gemini-embedding-001 model."""

    def __init__(self, api_key: str, config: EmbeddingProviderConfig) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from e

        self._genai = genai
        self._types = genai_types
        self._client = genai.Client(api_key=api_key)
        self._config = config
        self._dimension: int = config.dimension or _DEFAULT_DIMENSION

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of document texts using RETRIEVAL_DOCUMENT task type.

        Automatically batches large lists. Returns float32 array (N, dim).
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), _MAX_TEXTS_PER_BATCH):
            batch = texts[batch_start : batch_start + _MAX_TEXTS_PER_BATCH]
            batch_embeddings = self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string using RETRIEVAL_QUERY task type.

        Returns float32 array of shape (1, dim).
        """
        embeddings = self._embed_batch([query], task_type="RETRIEVAL_QUERY")
        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        """Call the Gemini embedding API for a batch of texts with backoff."""
        config_kwargs: dict = {"task_type": task_type}
        if self._config.dimension:
            config_kwargs["output_dimensionality"] = self._config.dimension

        embed_config = self._types.EmbedContentConfig(**config_kwargs)

        backoff = _INITIAL_BACKOFF
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.embed_content(
                    model=self._config.model,
                    contents=texts,
                    config=embed_config,
                )
                return [e.values for e in response.embeddings]

            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = (
                    "429" in exc_str or "quota" in exc_str or "rate" in exc_str
                )
                if not is_rate_limit:
                    raise

                last_exc = exc
                wait = backoff * (2 ** attempt)
                logger.warning(
                    "Gemini embedding rate limit (attempt %d/%d). Retrying in %.1fs…",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini embedding API failed after {_MAX_RETRIES} retries."
        ) from last_exc
