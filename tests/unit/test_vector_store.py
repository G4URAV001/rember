"""Unit tests for FAISSVectorStore."""

from __future__ import annotations

import numpy as np
import pytest

from rember.storage.vector import FAISSVectorStore

DIM = 8


def _random_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, dim)).astype(np.float32)


class TestFAISSVectorStore:
    def test_empty_store(self):
        store = FAISSVectorStore(dimension=DIM)
        assert store.total_vectors == 0
        assert store.dimension == DIM

    def test_add_vectors(self):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(5)
        store.add(vecs, ids=[0, 1, 2, 3, 4])
        assert store.total_vectors == 5

    def test_search_returns_top_k(self):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(10)
        store.add(vecs, ids=list(range(10)))

        query = _random_vectors(1, seed=99)
        results = store.search(query, k=3)
        assert len(results) == 3

    def test_search_result_structure(self):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(5)
        store.add(vecs, ids=[10, 20, 30, 40, 50])

        query = _random_vectors(1)
        results = store.search(query, k=2)

        for fid, score in results:
            assert isinstance(fid, int)
            assert isinstance(score, float)
            assert fid in [10, 20, 30, 40, 50]

    def test_search_empty_store_returns_empty(self):
        store = FAISSVectorStore(dimension=DIM)
        query = _random_vectors(1)
        results = store.search(query, k=5)
        assert results == []

    def test_search_k_larger_than_vectors(self):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(3)
        store.add(vecs, ids=[0, 1, 2])
        query = _random_vectors(1)
        results = store.search(query, k=100)  # only 3 vectors exist
        assert len(results) <= 3

    def test_identical_vector_highest_score(self):
        """The exact same vector should score ~1.0 (cosine similarity = 1)."""
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(5)
        store.add(vecs, ids=list(range(5)))

        # Search with the exact first vector
        query = vecs[0:1].copy()
        results = store.search(query, k=1)
        assert len(results) == 1
        fid, score = results[0]
        assert fid == 0
        assert score > 0.99

    def test_save_and_load(self, tmp_path):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(5)
        store.add(vecs, ids=[100, 101, 102, 103, 104])

        index_path = tmp_path / "test.faiss"
        store.save(index_path)
        assert index_path.exists()

        # Load into a new store
        store2 = FAISSVectorStore(dimension=DIM)
        store2.load(index_path)
        assert store2.total_vectors == 5

    def test_from_file_class_method(self, tmp_path):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(3)
        store.add(vecs, ids=[0, 1, 2])
        path = tmp_path / "idx.faiss"
        store.save(path)

        loaded = FAISSVectorStore.from_file(path, dimension=DIM)
        assert loaded.total_vectors == 3

    def test_load_nonexistent_raises(self, tmp_path):
        store = FAISSVectorStore(dimension=DIM)
        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "nonexistent.faiss")

    def test_1d_query_vector(self):
        """Should accept a 1D vector (shape=(dim,)) as query."""
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(3)
        store.add(vecs, ids=[0, 1, 2])

        query_1d = _random_vectors(1).flatten()  # shape (DIM,)
        results = store.search(query_1d, k=2)
        assert len(results) == 2

    def test_save_creates_parent_dirs(self, tmp_path):
        store = FAISSVectorStore(dimension=DIM)
        vecs = _random_vectors(2)
        store.add(vecs, ids=[0, 1])

        nested_path = tmp_path / "a" / "b" / "c" / "index.faiss"
        store.save(nested_path)
        assert nested_path.exists()
