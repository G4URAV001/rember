"""
FAISS vector store wrapper.

Design decisions:
  - Uses IndexFlatIP (exact inner-product / cosine similarity) wrapped in
    IndexIDMap so we can assign stable int64 IDs that map to SQLite rows.
  - Vectors are L2-normalised before insertion so that inner-product == cosine sim.
  - The index is persisted to a single .faiss binary file.
  - Metadata is NOT stored here — that's SQLite's job.

HNSW support:
  - Set index_type="hnsw" in config for approximate search at larger scale.
  - HNSW does not support IndexIDMap2, so we fall back to a sequential ID mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Thin wrapper around a FAISS index that adds:
      - Cosine similarity via L2 normalisation + inner product
      - Custom int64 IDs via IndexIDMap
      - Persistence (save/load)
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat_ip",
    ) -> None:
        """
        Args:
            dimension: Embedding vector dimension (e.g. 3072 for Gemini).
            index_type: "flat_ip" (exact, default) or "hnsw" (approximate).
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            ) from e

        self._faiss = faiss
        self._dimension = dimension
        self._index_type = index_type
        self._index = self._build_index(dimension, index_type)

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, dimension: int, index_type: str):
        faiss = self._faiss
        if index_type == "hnsw":
            # HNSW: no IndexIDMap needed; IDs are implicit sequential ints
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            logger.info("FAISS: using IndexHNSWFlat (dim=%d)", dimension)
        else:
            # Default: exact cosine similarity via inner product + ID map
            base = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIDMap(base)
            logger.info("FAISS: using IndexFlatIP with IndexIDMap (dim=%d)", dimension)
        return index

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        """
        Add vectors to the index with custom int64 IDs.

        Args:
            vectors: Float32 array of shape (N, dimension).
            ids: List of N unique int64 IDs corresponding to each vector.
        """
        if len(vectors) == 0:
            return

        vectors = vectors.astype(np.float32)
        self._faiss.normalize_L2(vectors)  # in-place, enables cosine similarity

        ids_array = np.array(ids, dtype=np.int64)

        if self._index_type == "hnsw":
            # HNSW doesn't support add_with_ids in the same way
            self._index.add(vectors)
        else:
            self._index.add_with_ids(vectors, ids_array)

        logger.debug("Added %d vectors. Total: %d", len(vectors), self.total_vectors)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Search for the k nearest vectors.

        Args:
            query_vector: Float32 array of shape (1, dimension) or (dimension,).
            k: Number of nearest neighbours to return.

        Returns:
            List of (faiss_id, score) tuples, sorted by score descending.
            Scores are cosine similarities (0–1). Invalid results (id == -1) are excluded.
        """
        vec = query_vector.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        self._faiss.normalize_L2(vec)

        k = min(k, self.total_vectors)
        if k == 0:
            return []

        distances, indices = self._index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(dist)))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the FAISS index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path))
        logger.info("FAISS index saved to %s (%d vectors)", path, self.total_vectors)

    def load(self, path: str | Path) -> None:
        """Load the FAISS index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")
        self._index = self._faiss.read_index(str(path))
        logger.info("FAISS index loaded from %s (%d vectors)", path, self.total_vectors)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        dimension: int,
        index_type: str = "flat_ip",
    ) -> "FAISSVectorStore":
        """Create a FAISSVectorStore and immediately load an existing index."""
        store = cls(dimension=dimension, index_type=index_type)
        store.load(path)
        return store

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._dimension
