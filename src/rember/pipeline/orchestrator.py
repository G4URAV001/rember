"""
Pipeline orchestrator — chains IngestStage → ExtractStage → ChunkStage →
Embed → Store into a single high-level API.

Public API:
    pipeline = Pipeline.from_settings(settings)
    doc = pipeline.ingest("path/to/file.txt", metadata={"tag": "work"})
    doc = pipeline.ingest_text("Python was created in 1991.", metadata={})
    doc = pipeline.ingest_file("notes.md")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rember.config import Settings, load_settings
from rember.embeddings.base import EmbeddingProvider
from rember.embeddings.gemini import GeminiEmbeddingProvider
from rember.llm.gemini import GeminiLLMProvider
from rember.llm.registry import LLMRegistry
from rember.models import Document, StoredChunk
from rember.pipeline.chunk import ChunkStage
from rember.pipeline.extract import ExtractStage
from rember.pipeline.ingest import IngestStage
from rember.storage.metadata import MetadataStore
from rember.storage.vector import FAISSVectorStore

logger = logging.getLogger(__name__)


class Pipeline:
    """
    High-level ingestion pipeline.

    Chains: Ingest → Extract → Chunk → Embed → Store
    """

    def __init__(
        self,
        settings: Settings,
        llm_registry: LLMRegistry,
        embedding_provider: EmbeddingProvider,
        vector_store: FAISSVectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self._settings = settings
        self._llm_registry = llm_registry
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._metadata_store = metadata_store

        self._ingest_stage = IngestStage()
        self._extract_stage = ExtractStage(llm_registry, media_config=settings.media)
        self._chunk_stage = ChunkStage(settings.chunking)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "Pipeline":
        """
        Build a Pipeline from a Settings object.

        If settings is None, loads from default config/env.
        Creates the data directory and initialises storage if needed.
        """
        if settings is None:
            settings = load_settings()

        api_key = settings.google_api_key.get_secret_value()
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Add it to your .env file or "
                "set the environment variable."
            )

        # Build LLM provider + registry
        llm_config = settings.get_llm_config("gemini")
        gemini_llm = GeminiLLMProvider(api_key=api_key, config=llm_config)
        llm_registry = LLMRegistry(default_provider=gemini_llm)
        llm_registry.load_routing_from_config(settings.task_routing)

        # Build embedding provider
        embed_config = settings.get_embedding_config("gemini")
        embedding_provider = GeminiEmbeddingProvider(api_key=api_key, config=embed_config)

        # Ensure data directory exists
        data_dir = settings.storage.resolved_data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Build storage
        index_path = settings.storage.vector_index_path
        dimension = embed_config.dimension or embedding_provider.dimension
        vector_store = FAISSVectorStore(
            dimension=dimension,
            index_type=settings.storage.vector.index_type,
        )
        if index_path.exists():
            vector_store.load(index_path)

        metadata_store = MetadataStore(db_path=settings.storage.db_path)

        return cls(
            settings=settings,
            llm_registry=llm_registry,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            metadata_store=metadata_store,
        )

    # ------------------------------------------------------------------
    # Public ingestion API
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """
        Ingest from a file path or raw text string.

        Args:
            source: A file path OR raw text content.
            metadata: Optional key-value tags to attach to the document.

        Returns:
            The stored Document.
        """
        return self._run_pipeline(source, metadata or {})

    def ingest_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Convenience: ingest a raw text string."""
        return self._run_pipeline(text, metadata or {})

    def ingest_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Convenience: ingest from a file path."""
        return self._run_pipeline(str(file_path), metadata or {})

    # ------------------------------------------------------------------
    # Internal pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(self, source: str, metadata: dict[str, Any]) -> Document:
        """Execute the full ingestion pipeline."""
        logger.info("Pipeline: starting ingestion from source '%s'", source[:80])

        # Stage 1: Ingest
        doc = self._ingest_stage.ingest(source, metadata)
        logger.info("Pipeline [ingest] → Document %s (%d chars)", doc.id, len(doc.raw_content))

        # Stage 2: Extract
        extracted = self._extract_stage.extract(doc)
        logger.info(
            "Pipeline [extract] → %d facts, %d topics",
            len(extracted.key_facts),
            len(extracted.topics),
        )

        # Stage 3: Chunk
        chunks = self._chunk_stage.chunk(extracted)
        logger.info("Pipeline [chunk] → %d chunks", len(chunks))

        if not chunks:
            logger.warning("Pipeline produced 0 chunks for document %s. Skipping storage.", doc.id)
            return doc

        # Stage 4: Embed
        texts = [c.content for c in chunks]
        embeddings = self._embedding_provider.embed(texts)
        logger.info("Pipeline [embed] → shape %s", embeddings.shape)

        # Stage 5: Store
        faiss_start = self._metadata_store.get_next_faiss_id(len(chunks))
        faiss_ids = list(range(faiss_start, faiss_start + len(chunks)))

        self._vector_store.add(embeddings, faiss_ids)

        stored_chunks = [
            StoredChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                faiss_id=fid,
                content=chunk.content,
                metadata=chunk.metadata,
                token_count=chunk.token_count,
                chunk_index=chunk.chunk_index,
            )
            for chunk, fid in zip(chunks, faiss_ids)
        ]

        self._metadata_store.save_document(doc)
        self._metadata_store.save_chunks(stored_chunks)

        # Persist FAISS index after every ingestion
        self._vector_store.save(self._settings.storage.vector_index_path)

        logger.info(
            "Pipeline [store] → saved document %s with %d chunks "
            "(FAISS IDs %d–%d)",
            doc.id,
            len(stored_chunks),
            faiss_start,
            faiss_start + len(chunks) - 1,
        )

        return doc

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return combined storage statistics."""
        meta_stats = self._metadata_store.get_stats()
        return {
            **meta_stats,
            "vector_count": self._vector_store.total_vectors,
            "embedding_dimension": self._embedding_provider.dimension,
            "index_path": str(self._settings.storage.vector_index_path),
        }

    @property
    def metadata_store(self) -> MetadataStore:
        return self._metadata_store

    @property
    def vector_store(self) -> FAISSVectorStore:
        return self._vector_store

    @property
    def llm_registry(self) -> LLMRegistry:
        return self._llm_registry

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider
