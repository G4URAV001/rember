"""Unit tests for core data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rember.models import (
    Answer,
    Chunk,
    Document,
    ExtractedInfo,
    QueryResult,
    SourceType,
    StoredChunk,
)


class TestDocument:
    def test_auto_id(self):
        doc = Document(source_type=SourceType.TEXT, raw_content="hello")
        assert len(doc.id) == 36  # UUID format

    def test_two_docs_have_different_ids(self):
        d1 = Document(source_type=SourceType.TEXT, raw_content="a")
        d2 = Document(source_type=SourceType.TEXT, raw_content="b")
        assert d1.id != d2.id

    def test_metadata_defaults_to_empty(self):
        doc = Document(source_type=SourceType.TEXT, raw_content="x")
        assert doc.metadata == {}

    def test_source_type_enum_value(self):
        doc = Document(source_type=SourceType.FILE, raw_content="x", source_path="/tmp/f.txt")
        # With use_enum_values=True, the value should be the string
        assert doc.source_type in ("file", SourceType.FILE)

    def test_requires_raw_content(self):
        with pytest.raises(ValidationError):
            Document(source_type=SourceType.TEXT)  # missing raw_content


class TestExtractedInfo:
    def test_creation(self):
        ei = ExtractedInfo(
            document_id="doc-123",
            summary="A summary.",
            key_facts=["Fact 1", "Fact 2"],
            topics=["ai"],
        )
        assert ei.document_id == "doc-123"
        assert len(ei.key_facts) == 2

    def test_topics_defaults_empty(self):
        ei = ExtractedInfo(
            document_id="doc-abc",
            summary="s",
            key_facts=[],
        )
        assert ei.topics == []


class TestChunk:
    def test_auto_id(self):
        c = Chunk(document_id="doc-1", content="hello world")
        assert len(c.id) == 36

    def test_default_token_count_zero(self):
        c = Chunk(document_id="doc-1", content="text")
        assert c.token_count == 0

    def test_chunk_index_default(self):
        c = Chunk(document_id="doc-1", content="text")
        assert c.chunk_index == 0


class TestStoredChunk:
    def test_creation(self):
        sc = StoredChunk(
            chunk_id="chunk-abc",
            document_id="doc-xyz",
            faiss_id=42,
            content="stored content",
        )
        assert sc.faiss_id == 42
        assert sc.content == "stored content"


class TestQueryResult:
    def test_score_range(self):
        qr = QueryResult(
            chunk_id="c1",
            document_id="d1",
            content="relevant chunk",
            score=0.87,
        )
        assert 0.0 <= qr.score <= 1.0

    def test_metadata_defaults_empty(self):
        qr = QueryResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            score=0.5,
        )
        assert qr.metadata == {}


class TestAnswer:
    def test_creation(self):
        a = Answer(question="What?", answer="Because.")
        assert a.question == "What?"
        assert a.sources == []

    def test_with_sources(self, sample_document):
        qr = QueryResult(
            chunk_id="c1",
            document_id=sample_document.id,
            content="relevant",
            score=0.9,
        )
        a = Answer(question="?", answer="!", sources=[qr])
        assert len(a.sources) == 1
