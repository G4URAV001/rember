"""Unit tests for ChunkStage."""

from __future__ import annotations

import pytest

from rember.config import ChunkingConfig
from rember.models import ExtractedInfo
from rember.pipeline.chunk import ChunkStage, _split_into_chunks, _word_count


class TestWordCount:
    def test_empty_string(self):
        assert _word_count("") == 0

    def test_single_word(self):
        assert _word_count("hello") == 1

    def test_multiple_words(self):
        assert _word_count("hello world foo") == 3

    def test_extra_whitespace(self):
        assert _word_count("  hello   world  ") == 2


class TestSplitIntoChunks:
    def test_short_text_no_split(self):
        words = " ".join([f"word{i}" for i in range(5)])
        chunks = _split_into_chunks(words, max_chunk_size=10, chunk_overlap=2)
        assert len(chunks) == 1
        assert chunks[0] == words

    def test_longer_text_splits(self):
        words = " ".join([f"word{i}" for i in range(30)])
        chunks = _split_into_chunks(words, max_chunk_size=10, chunk_overlap=2)
        assert len(chunks) > 1

    def test_overlap_shared_content(self):
        words = " ".join([f"w{i}" for i in range(20)])
        chunks = _split_into_chunks(words, max_chunk_size=10, chunk_overlap=3)
        # The end of chunk N should overlap with the start of chunk N+1
        for i in range(len(chunks) - 1):
            end_words = set(chunks[i].split()[-3:])
            start_words = set(chunks[i + 1].split()[:3])
            assert len(end_words & start_words) > 0

    def test_empty_text(self):
        chunks = _split_into_chunks("", max_chunk_size=10, chunk_overlap=2)
        assert chunks == []

    def test_exact_chunk_size(self):
        words = " ".join([f"w{i}" for i in range(10)])
        chunks = _split_into_chunks(words, max_chunk_size=10, chunk_overlap=0)
        assert len(chunks) == 1


class TestChunkStage:
    def _make_extracted(self, doc_id: str, facts: list[str]) -> ExtractedInfo:
        return ExtractedInfo(
            document_id=doc_id,
            summary="A short summary.",
            key_facts=facts,
            topics=["test"],
            metadata={"source": "test"},
        )

    def test_short_content_single_chunk(self):
        """Content below adaptive_threshold should produce exactly 1 chunk."""
        config = ChunkingConfig(adaptive_threshold=500, max_chunk_size=1000, chunk_overlap=100)
        stage = ChunkStage(config)

        extracted = self._make_extracted("doc-1", ["Fact 1.", "Fact 2."])
        chunks = stage.chunk(extracted)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc-1"
        assert "Mock" not in chunks[0].content  # content is real

    def test_short_content_whole_strategy(self):
        config = ChunkingConfig(adaptive_threshold=500)
        stage = ChunkStage(config)
        extracted = self._make_extracted("doc-2", ["Fact 1."])
        chunks = stage.chunk(extracted)
        assert chunks[0].metadata.get("chunk_strategy") == "whole"

    def test_long_content_splits(self):
        """Content above adaptive_threshold should produce multiple chunks."""
        config = ChunkingConfig(
            adaptive_threshold=5,   # very low threshold
            max_chunk_size=10,
            chunk_overlap=2,
        )
        stage = ChunkStage(config)

        # Create enough facts to exceed the threshold
        facts = [f"This is fact number {i} with extra words to fill space." for i in range(10)]
        extracted = self._make_extracted("doc-3", facts)
        chunks = stage.chunk(extracted)

        assert len(chunks) > 1

    def test_split_strategy_metadata(self):
        config = ChunkingConfig(adaptive_threshold=5, max_chunk_size=10, chunk_overlap=2)
        stage = ChunkStage(config)
        facts = [f"Fact {i} with more words here." for i in range(10)]
        extracted = self._make_extracted("doc-4", facts)
        chunks = stage.chunk(extracted)
        for chunk in chunks:
            assert chunk.metadata.get("chunk_strategy") == "split"

    def test_chunks_inherit_topics(self):
        config = ChunkingConfig(adaptive_threshold=500)
        stage = ChunkStage(config)
        extracted = self._make_extracted("doc-5", ["Fact."])
        chunks = stage.chunk(extracted)
        assert "test" in chunks[0].metadata.get("topics", [])

    def test_chunk_token_count_positive(self):
        config = ChunkingConfig(adaptive_threshold=500)
        stage = ChunkStage(config)
        extracted = self._make_extracted("doc-6", ["Hello world. This is a test."])
        chunks = stage.chunk(extracted)
        assert all(c.token_count > 0 for c in chunks)

    def test_empty_extracted_returns_empty(self):
        config = ChunkingConfig(adaptive_threshold=500)
        stage = ChunkStage(config)
        extracted = ExtractedInfo(
            document_id="doc-empty",
            summary="",
            key_facts=[],
        )
        chunks = stage.chunk(extracted)
        assert chunks == []
