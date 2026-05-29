"""
Shared pytest fixtures for the Rember test suite.

Key fixtures:
  - tmp_settings     : Settings with temp data_dir, no real API keys needed
  - mock_llm         : A mock LLMProvider that returns canned responses
  - mock_embedder    : A mock EmbeddingProvider returning random float32 vectors
  - vector_store     : A temporary FAISSVectorStore (dim=8 for speed)
  - metadata_store   : A temporary MetadataStore backed by a tmp SQLite file
  - sample_document  : A Document fixture for pipeline tests
  - sample_jpeg_path : Path to a programmatically-generated 100x100 JPEG (Phase 2)
"""

from __future__ import annotations

import numpy as np
import pytest

from rember.config import (
    ChunkingConfig,
    EmbeddingProviderConfig,
    LLMProviderConfig,
    MediaConfig,
    PipelineConfig,
    QueryConfig,
    Settings,
    StorageConfig,
)
from rember.embeddings.base import EmbeddingProvider
from rember.llm.base import LLMProvider
from rember.models import Document, ExtractedInfo, SourceType
from rember.storage.metadata import MetadataStore
from rember.storage.vector import FAISSVectorStore

# Embedding dimension used in tests (small for speed)
TEST_DIM = 8


# ---------------------------------------------------------------------------
# Mock LLM Provider
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """
    A deterministic LLM provider for testing. No API calls.

    Phase 2 note: extract_info_multimodal() is inherited from LLMProvider base class,
    which provides a text-fallback implementation. Individual tests that need to verify
    multimodal dispatch should set mock_llm.extract_info_multimodal = MagicMock(...).
    """

    def __init__(self, generate_response: str = "Test response.") -> None:
        self._response = generate_response

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return self._response

    def extract_info(
        self,
        content: str,
        document_id: str,
        content_type: str = "text",
        extra_metadata: dict | None = None,
    ) -> ExtractedInfo:
        return ExtractedInfo(
            document_id=document_id,
            summary="Mock summary of the content.",
            key_facts=["Mock fact 1.", "Mock fact 2.", "Mock fact 3."],
            topics=["mock", "testing"],
            metadata=extra_metadata or {},
        )

    # upload_video / delete_uploaded_file added by individual tests via MagicMock

# ---------------------------------------------------------------------------
# Mock Embedding Provider
# ---------------------------------------------------------------------------


class MockEmbeddingProvider(EmbeddingProvider):
    """Returns deterministic (seeded) random vectors. No API calls."""

    def __init__(self, dimension: int = TEST_DIM, seed: int = 42) -> None:
        self._dimension = dimension
        self._rng = np.random.default_rng(seed)

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._rng.random((len(texts), self._dimension)).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._rng.random((1, self._dimension)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=TEST_DIM)


@pytest.fixture
def tmp_settings(tmp_path) -> Settings:
    """Settings pointing to a temp directory, no real API key needed."""
    return Settings(
        pipeline=PipelineConfig(default_llm="gemini", default_embedding="gemini"),
        llm={"gemini": LLMProviderConfig()},
        embeddings={
            "gemini": EmbeddingProviderConfig(
                model="gemini-embedding-001",
                dimension=TEST_DIM,
            )
        },
        storage=StorageConfig(data_dir=str(tmp_path / "rember_data")),
        chunking=ChunkingConfig(
            adaptive_threshold=10,  # low threshold so we can test both paths
            max_chunk_size=20,
            chunk_overlap=5,
        ),
        media=MediaConfig(
            image_max_dimension=256,   # small for test speed
            num_frames_to_extract=2,   # minimal for test speed
            prefer_native_video=False, # disable native in tests (no API)
            enable_transcription=False, # disable Whisper in tests
        ),
        query=QueryConfig(top_k=5, min_score=0.0),
    )


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    return FAISSVectorStore(dimension=TEST_DIM)


@pytest.fixture
def metadata_store(tmp_path) -> MetadataStore:
    return MetadataStore(db_path=tmp_path / "test.db")


@pytest.fixture
def sample_document() -> Document:
    return Document(
        source_type=SourceType.TEXT,
        raw_content="Python was created by Guido van Rossum and first released in 1991.",
        metadata={"tag": "test"},
    )


@pytest.fixture
def sample_extracted(sample_document) -> ExtractedInfo:
    return ExtractedInfo(
        document_id=sample_document.id,
        summary="Python is a programming language created by Guido van Rossum in 1991.",
        key_facts=[
            "Python was created by Guido van Rossum.",
            "Python was first released in 1991.",
            "Python is a programming language.",
        ],
        topics=["python", "programming"],
        metadata={"tag": "test"},
    )


@pytest.fixture
def sample_jpeg_path(tmp_path) -> "Path":
    """
    Generate a real 100x100 red JPEG image for image processing tests.
    Does not require any external files.
    """
    from pathlib import Path
    from PIL import Image

    img_path = tmp_path / "sample.jpg"
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    img.save(img_path, format="JPEG", quality=85)
    return img_path
