"""
Integration tests for the full ingestion pipeline.

Uses mock LLM + mock embedder (no API calls), but tests the full flow:
  IngestStage → ExtractStage → ChunkStage → Embed → Store

Run with:
    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rember.llm.registry import LLMRegistry
from rember.pipeline.chunk import ChunkStage
from rember.pipeline.extract import ExtractStage
from rember.pipeline.ingest import IngestStage
from rember.pipeline.orchestrator import Pipeline
from rember.models import StoredChunk


@pytest.mark.integration
class TestIngestionPipeline:
    """End-to-end pipeline tests using mock providers."""

    def _build_pipeline(self, tmp_settings, mock_llm, mock_embedder, tmp_path):
        """Build a Pipeline using mock providers (no API calls)."""
        from rember.storage.metadata import MetadataStore
        from rember.storage.vector import FAISSVectorStore

        registry = LLMRegistry(default_provider=mock_llm)
        vector_store = FAISSVectorStore(dimension=mock_embedder.dimension)
        metadata_store = MetadataStore(db_path=tmp_path / "test.db")

        pipeline = Pipeline(
            settings=tmp_settings,
            llm_registry=registry,
            embedding_provider=mock_embedder,
            vector_store=vector_store,
            metadata_store=metadata_store,
        )
        return pipeline

    def test_ingest_text_stores_document(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_text("Python was created by Guido van Rossum.")

        # Document should be in SQLite
        fetched = pipeline.metadata_store.get_document(doc.id)
        assert fetched is not None
        assert fetched.raw_content == "Python was created by Guido van Rossum."

    def test_ingest_text_stores_chunks(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_text("Some content here.")

        chunks = pipeline.metadata_store.list_chunks_for_document(doc.id)
        assert len(chunks) >= 1
        assert all(isinstance(c, StoredChunk) for c in chunks)

    def test_ingest_increments_vector_count(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        pipeline.ingest_text("First document.")
        count_after_1 = pipeline.vector_store.total_vectors

        pipeline.ingest_text("Second document.")
        count_after_2 = pipeline.vector_store.total_vectors

        assert count_after_2 > count_after_1

    def test_ingest_file_txt(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        sample = Path(__file__).parent.parent / "fixtures" / "sample.txt"
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(sample)

        assert doc.source_path is not None
        assert "sample.txt" in doc.source_path

    def test_ingest_file_json(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        json_file = tmp_path / "data.json"
        json_file.write_text('{"key": "value", "number": 42}')

        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(json_file)
        assert "key" in doc.raw_content

    def test_ingest_with_metadata_tags(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_text(
            "Tagged content.",
            metadata={"project": "rember", "author": "tester"},
        )
        fetched = pipeline.metadata_store.get_document(doc.id)
        assert fetched.metadata["project"] == "rember"

    def test_ingest_empty_text_raises(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        with pytest.raises(ValueError, match="empty"):
            pipeline.ingest_text("   ")

    def test_ingest_multiple_documents_independent(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc1 = pipeline.ingest_text("Document one.")
        doc2 = pipeline.ingest_text("Document two.")

        assert doc1.id != doc2.id
        docs = pipeline.metadata_store.list_documents()
        assert len(docs) == 2

    def test_get_stats_after_ingest(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        pipeline.ingest_text("Hello world.")
        stats = pipeline.get_stats()

        assert stats["document_count"] >= 1
        assert stats["chunk_count"] >= 1
        assert stats["vector_count"] >= 1

    def test_faiss_ids_are_unique(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline = self._build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        pipeline.ingest_text("First.")
        pipeline.ingest_text("Second.")

        all_chunks = []
        for doc in pipeline.metadata_store.list_documents():
            all_chunks.extend(
                pipeline.metadata_store.list_chunks_for_document(doc.id)
            )

        faiss_ids = [c.faiss_id for c in all_chunks]
        assert len(faiss_ids) == len(set(faiss_ids)), "FAISS IDs must be unique"
