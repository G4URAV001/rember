"""Unit tests for MetadataStore (SQLite)."""

from __future__ import annotations

from datetime import datetime

import pytest

from rember.models import Document, SourceType, StoredChunk
from rember.storage.metadata import MetadataStore


def _make_doc(raw: str = "hello world", source_type=SourceType.TEXT) -> Document:
    return Document(
        source_type=source_type,
        raw_content=raw,
        metadata={"test": "true"},
    )


def _make_chunk(doc_id: str, faiss_id: int, index: int = 0) -> StoredChunk:
    return StoredChunk(
        chunk_id=f"chunk-{faiss_id}",
        document_id=doc_id,
        faiss_id=faiss_id,
        content=f"Chunk content {faiss_id}",
        metadata={"chunk": str(faiss_id)},
        token_count=3,
        chunk_index=index,
    )


class TestFaissIdAllocation:
    def test_starts_at_zero(self, metadata_store):
        fid = metadata_store.get_next_faiss_id(1)
        assert fid == 0

    def test_sequential_allocation(self, metadata_store):
        id1 = metadata_store.get_next_faiss_id(3)  # reserves 0,1,2
        id2 = metadata_store.get_next_faiss_id(2)  # reserves 3,4
        assert id1 == 0
        assert id2 == 3

    def test_count_zero(self, metadata_store):
        id1 = metadata_store.get_next_faiss_id(0)
        id2 = metadata_store.get_next_faiss_id(1)
        assert id1 == 0
        assert id2 == 0  # counter didn't advance


class TestDocumentCRUD:
    def test_save_and_retrieve(self, metadata_store):
        doc = _make_doc("Python is great.")
        metadata_store.save_document(doc)
        fetched = metadata_store.get_document(doc.id)
        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.raw_content == "Python is great."
        assert fetched.metadata == {"test": "true"}

    def test_get_nonexistent_returns_none(self, metadata_store):
        result = metadata_store.get_document("does-not-exist")
        assert result is None

    def test_list_documents_empty(self, metadata_store):
        docs = metadata_store.list_documents()
        assert docs == []

    def test_list_documents_returns_all(self, metadata_store):
        d1 = _make_doc("first")
        d2 = _make_doc("second")
        metadata_store.save_document(d1)
        metadata_store.save_document(d2)
        docs = metadata_store.list_documents()
        assert len(docs) == 2

    def test_list_documents_newest_first(self, metadata_store):
        from datetime import timezone
        import time

        d1 = _make_doc("old")
        d1 = d1.model_copy(update={"created_at": datetime(2020, 1, 1)})
        d2 = _make_doc("new")
        d2 = d2.model_copy(update={"created_at": datetime(2025, 1, 1)})

        metadata_store.save_document(d1)
        metadata_store.save_document(d2)
        docs = metadata_store.list_documents()
        # newest first
        assert docs[0].id == d2.id

    def test_delete_document(self, metadata_store):
        doc = _make_doc("to delete")
        metadata_store.save_document(doc)
        deleted = metadata_store.delete_document(doc.id)
        assert deleted is True
        assert metadata_store.get_document(doc.id) is None

    def test_delete_nonexistent_returns_false(self, metadata_store):
        result = metadata_store.delete_document("no-such-id")
        assert result is False

    def test_upsert_document(self, metadata_store):
        doc = _make_doc("original")
        metadata_store.save_document(doc)
        updated = doc.model_copy(update={"raw_content": "updated content"})
        metadata_store.save_document(updated)
        fetched = metadata_store.get_document(doc.id)
        assert fetched.raw_content == "updated content"


class TestChunkCRUD:
    def test_save_and_retrieve_by_faiss_id(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)

        chunk = _make_chunk(doc.id, faiss_id=7)
        metadata_store.save_chunks([chunk])

        fetched = metadata_store.get_chunk_by_faiss_id(7)
        assert fetched is not None
        assert fetched.chunk_id == "chunk-7"
        assert fetched.content == "Chunk content 7"

    def test_get_chunk_nonexistent(self, metadata_store):
        result = metadata_store.get_chunk_by_faiss_id(9999)
        assert result is None

    def test_get_chunks_by_faiss_ids(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)
        chunks = [_make_chunk(doc.id, fid, i) for i, fid in enumerate([10, 20, 30])]
        metadata_store.save_chunks(chunks)

        results = metadata_store.get_chunks_by_faiss_ids([10, 30])
        assert len(results) == 2
        faiss_ids = [r.faiss_id for r in results]
        assert 10 in faiss_ids
        assert 30 in faiss_ids

    def test_get_chunks_preserves_order(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)
        chunks = [_make_chunk(doc.id, fid, i) for i, fid in enumerate([1, 2, 3])]
        metadata_store.save_chunks(chunks)

        results = metadata_store.get_chunks_by_faiss_ids([3, 1])  # reversed order
        assert results[0].faiss_id == 3
        assert results[1].faiss_id == 1

    def test_bulk_save_chunks(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)
        chunks = [_make_chunk(doc.id, i, i) for i in range(10)]
        metadata_store.save_chunks(chunks)

        results = metadata_store.get_chunks_by_faiss_ids(list(range(10)))
        assert len(results) == 10

    def test_delete_document_cascades_chunks(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)
        chunk = _make_chunk(doc.id, faiss_id=99)
        metadata_store.save_chunks([chunk])

        metadata_store.delete_document(doc.id)
        assert metadata_store.get_chunk_by_faiss_id(99) is None

    def test_list_chunks_for_document(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)
        chunks = [_make_chunk(doc.id, i, i) for i in range(3)]
        metadata_store.save_chunks(chunks)

        results = metadata_store.list_chunks_for_document(doc.id)
        assert len(results) == 3
        # Should be in chunk_index order
        assert [r.chunk_index for r in results] == [0, 1, 2]


class TestStats:
    def test_empty_stats(self, metadata_store):
        s = metadata_store.get_stats()
        assert s["document_count"] == 0
        assert s["chunk_count"] == 0
        assert s["next_faiss_id"] == 0

    def test_stats_after_insert(self, metadata_store):
        doc = _make_doc()
        metadata_store.save_document(doc)
        metadata_store.save_chunks([_make_chunk(doc.id, 0)])
        metadata_store.get_next_faiss_id(5)

        s = metadata_store.get_stats()
        assert s["document_count"] == 1
        assert s["chunk_count"] == 1
        assert s["next_faiss_id"] == 5
