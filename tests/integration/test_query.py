"""
Integration tests for the query pipeline (Retriever + Answerer).

Uses mock providers — no API calls. Tests the full end-to-end flow:
  ingest → store → retrieve → answer

Run with:
    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import numpy as np
import pytest

from rember.llm.registry import LLMRegistry
from rember.models import Answer
from rember.pipeline.orchestrator import Pipeline
from rember.query.answerer import Answerer
from rember.query.retriever import Retriever
from rember.storage.metadata import MetadataStore
from rember.storage.vector import FAISSVectorStore


@pytest.mark.integration
class TestQueryPipeline:
    """End-to-end query tests using mock providers."""

    def _build_pipeline_and_query(self, tmp_settings, mock_llm, mock_embedder, tmp_path):
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

        retriever = Retriever(
            vector_store=vector_store,
            metadata_store=metadata_store,
            embedding_provider=mock_embedder,
            config=tmp_settings.query,
        )
        answerer = Answerer(llm_registry=registry)

        return pipeline, retriever, answerer

    def test_retrieve_after_ingest(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline, retriever, _ = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        pipeline.ingest_text("Python was created by Guido van Rossum.")

        results = retriever.retrieve("Who created Python?")
        assert len(results) > 0

    def test_retrieve_returns_query_result_objects(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        from rember.models import QueryResult

        pipeline, retriever, _ = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        pipeline.ingest_text("Test content.")
        results = retriever.retrieve("test")

        for r in results:
            assert isinstance(r, QueryResult)
            assert r.score >= 0.0
            assert r.content != ""

    def test_empty_store_returns_no_results(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        _, retriever, _ = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        results = retriever.retrieve("anything")
        assert results == []

    def test_answer_generated_from_context(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline, retriever, answerer = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        pipeline.ingest_text("Rember is a RAG pipeline tool.")

        results = retriever.retrieve("What is Rember?")
        answer = answerer.answer("What is Rember?", results)

        assert isinstance(answer, Answer)
        assert len(answer.answer) > 0
        assert answer.question == "What is Rember?"

    def test_answer_with_no_context(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        _, _, answerer = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        answer = answerer.answer("What is the meaning of life?", context=[])

        assert isinstance(answer, Answer)
        assert "ingest" in answer.answer.lower() or len(answer.answer) > 0
        assert answer.sources == []

    def test_sources_in_answer_match_retrieved(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline, retriever, answerer = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        pipeline.ingest_text("Some content about AI.")

        results = retriever.retrieve("AI")
        answer = answerer.answer("Tell me about AI.", results)

        assert len(answer.sources) == len(results)

    def test_retrieve_top_k_limits_results(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline, retriever, _ = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        # Ingest multiple docs to have enough vectors
        for i in range(5):
            pipeline.ingest_text(f"Document number {i} with unique content.")

        results = retriever.retrieve("document", top_k=2)
        assert len(results) <= 2

    def test_retrieve_min_score_filter(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline, retriever, _ = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        pipeline.ingest_text("Irrelevant content.")

        # Use very high min_score — likely nothing matches
        results = retriever.retrieve("anything", min_score=0.9999)
        # Results should be empty or only very high-score ones
        for r in results:
            assert r.score >= 0.9999

    def test_results_sorted_by_score_descending(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        pipeline, retriever, _ = self._build_pipeline_and_query(
            tmp_settings, mock_llm, mock_embedder, tmp_path
        )
        for i in range(3):
            pipeline.ingest_text(f"Content {i}.")

        results = retriever.retrieve("content", min_score=0.0)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
