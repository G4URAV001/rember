"""
Retriever — embeds a query and finds relevant chunks via FAISS.

Flow:
  1. Embed query with RETRIEVAL_QUERY task type
  2. Search FAISS for top-K nearest neighbours
  3. Filter by min_score threshold
  4. Hydrate full chunk metadata from SQLite
  5. Return list[QueryResult] ordered by relevance

Re-ranking (Phase 3):
  After retrieval, a cross-encoder or LLM re-ranker will re-score
  and reorder the results for higher precision.
"""

from __future__ import annotations

import logging

from rember.config import QueryConfig
from rember.embeddings.base import EmbeddingProvider
from rember.models import QueryResult
from rember.storage.metadata import MetadataStore
from rember.storage.vector import FAISSVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Embeds queries and retrieves relevant chunks from FAISS + SQLite."""

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        metadata_store: MetadataStore,
        embedding_provider: EmbeddingProvider,
        config: QueryConfig,
    ) -> None:
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedding_provider = embedding_provider
        self._config = config

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[QueryResult]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question or search string.
            top_k: Number of results to retrieve (defaults to config value).
            min_score: Minimum cosine similarity score (defaults to config value).

        Returns:
            List of QueryResult sorted by score descending.
        """
        k = top_k if top_k is not None else self._config.top_k
        min_s = min_score if min_score is not None else self._config.min_score

        if self._vector_store.total_vectors == 0:
            logger.warning("Vector store is empty. No results to retrieve.")
            return []

        # Step 1: Embed the query
        query_vector = self._embedding_provider.embed_query(query)
        logger.debug("Query embedded: shape %s", query_vector.shape)

        # Step 2: FAISS similarity search
        raw_results = self._vector_store.search(query_vector, k=k)
        logger.debug("FAISS returned %d raw results", len(raw_results))

        if not raw_results:
            return []

        # Step 3: Filter by minimum score
        filtered = [(fid, score) for fid, score in raw_results if score >= min_s]
        logger.debug(
            "%d results passed min_score threshold (%.2f)", len(filtered), min_s
        )

        if not filtered:
            return []

        # Step 4: Hydrate metadata from SQLite
        faiss_ids = [fid for fid, _ in filtered]
        score_map = {fid: score for fid, score in filtered}

        chunks = self._metadata_store.get_chunks_by_faiss_ids(faiss_ids)

        # Step 5: Build QueryResult objects
        results = [
            QueryResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=score_map.get(chunk.faiss_id, 0.0),
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        # Ensure sorted by score descending (FAISS should already do this,
        # but SQLite rehydration may reorder)
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Retrieved %d results for query: '%s'",
            len(results),
            query[:60],
        )

        return results
