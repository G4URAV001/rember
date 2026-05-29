"""Embedding abstraction layer."""
from rember.embeddings.base import EmbeddingProvider
from rember.embeddings.gemini import GeminiEmbeddingProvider

__all__ = ["EmbeddingProvider", "GeminiEmbeddingProvider"]
