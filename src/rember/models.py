"""
Core data models that flow through the Rember pipeline.

Flow:
  raw input
    → Document      (IngestStage)
    → ExtractedInfo (ExtractStage)
    → list[Chunk]   (ChunkStage)
    → StoredChunk   (after embedding + storage)

Query flow:
  question → list[QueryResult] → Answer
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """Type of the original input source."""

    TEXT = "text"
    FILE = "file"
    IMAGE = "image"  # Phase 2
    VIDEO = "video"  # Phase 2


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Raw input unit entering the pipeline."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType
    source_path: str | None = None  # filesystem path, URL, etc.
    mime_type: str | None = None  # e.g. "image/jpeg", "video/mp4" (Phase 2)
    raw_content: str  # text content (or human-readable description for binary files)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(use_enum_values=True)


class ExtractedInfo(BaseModel):
    """
    Structured output of the ExtractStage.
    The LLM produces a summary + a list of discrete key facts.
    """

    document_id: str
    summary: str  # concise summary of the content
    key_facts: list[str]  # individual, self-contained facts
    topics: list[str] = Field(default_factory=list)  # inferred topic tags
    metadata: dict[str, Any] = Field(default_factory=dict)  # inherited from Document


class Chunk(BaseModel):
    """An embeddable unit of information produced by ChunkStage."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str  # the text that will be embedded
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0  # position within the parent document


class StoredChunk(BaseModel):
    """A Chunk after it has been embedded and saved to FAISS + SQLite."""

    chunk_id: str
    document_id: str
    faiss_id: int  # int64 ID used in the FAISS index
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


class QueryResult(BaseModel):
    """A single retrieval result returned by the Retriever."""

    chunk_id: str
    document_id: str
    content: str
    score: float  # cosine similarity (0.0 – 1.0, higher = more relevant)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    """Final answer produced by the Answerer, with source attribution."""

    question: str
    answer: str
    sources: list[QueryResult] = Field(default_factory=list)
    model_used: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
