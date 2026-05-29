"""
Integration tests for the full media ingestion pipeline (Phase 2).

Tests the end-to-end flow for images and videos using:
  - Mock LLM (no API calls)
  - Mock embedder (no API calls)
  - Real Pillow for image operations
  - Mocked ffmpeg for video operations

Run with:
    pytest tests/integration/test_media_pipeline.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rember.config import MediaConfig
from rember.llm.registry import LLMRegistry
from rember.models import Document, ExtractedInfo, SourceType
from rember.pipeline.extract import ExtractStage
from rember.pipeline.ingest import IngestStage
from rember.pipeline.orchestrator import Pipeline
from rember.storage.metadata import MetadataStore
from rember.storage.vector import FAISSVectorStore


def _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path):
    """Build a test Pipeline with mock providers."""
    registry = LLMRegistry(default_provider=mock_llm)
    vector_store = FAISSVectorStore(dimension=mock_embedder.dimension)
    metadata_store = MetadataStore(db_path=tmp_path / "test.db")
    return Pipeline(
        settings=tmp_settings,
        llm_registry=registry,
        embedding_provider=mock_embedder,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )


@pytest.mark.integration
class TestImageIngestionPipeline:
    """Full pipeline tests for image ingestion."""

    def test_ingest_jpeg_stores_document(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Image should produce a Document with source_type=image."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(sample_jpeg_path)

        assert doc.source_type == "image"
        assert doc.source_path is not None
        assert "sample" in Path(doc.source_path).name

    def test_ingest_jpeg_metadata(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Image document should carry width, height, format metadata."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(sample_jpeg_path)

        assert doc.metadata["image_width"] == 100
        assert doc.metadata["image_height"] == 100
        assert doc.metadata["image_format"] == "JPEG"

    def test_ingest_jpeg_mime_type(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Document should have mime_type set."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(sample_jpeg_path)

        assert doc.mime_type == "image/jpeg"

    def test_ingest_jpeg_stores_chunks(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Image ingestion should store chunks in SQLite."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(sample_jpeg_path)

        chunks = pipeline.metadata_store.list_chunks_for_document(doc.id)
        assert len(chunks) >= 1

    def test_ingest_jpeg_increments_vector_count(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Image ingestion should add vectors to FAISS."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        assert pipeline.vector_store.total_vectors == 0

        pipeline.ingest_file(sample_jpeg_path)
        assert pipeline.vector_store.total_vectors > 0

    def test_ingest_jpeg_with_tags(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Metadata tags should be preserved through the pipeline."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(sample_jpeg_path, metadata={"album": "vacation"})

        fetched = pipeline.metadata_store.get_document(doc.id)
        assert fetched.metadata["album"] == "vacation"

    def test_ingest_invalid_image_raises(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        """A non-image file with .jpg extension should raise ValueError."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        fake_jpg = tmp_path / "fake.jpg"
        fake_jpg.write_bytes(b"this is not an image at all")

        with pytest.raises(ValueError, match="Cannot ingest image"):
            pipeline.ingest_file(fake_jpg)

    def test_stats_include_image_document(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        """Stats should count image documents."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        pipeline.ingest_file(sample_jpeg_path)

        stats = pipeline.get_stats()
        assert stats["document_count"] == 1
        assert stats["chunk_count"] >= 1

    def test_ingest_png_image(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        """PNG files should also be ingested correctly."""
        from PIL import Image

        png_path = tmp_path / "test.png"
        img = Image.new("RGB", (200, 150), color=(0, 128, 255))
        img.save(png_path, format="PNG")

        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        doc = pipeline.ingest_file(png_path)

        assert doc.source_type == "image"
        assert doc.mime_type == "image/png"
        assert doc.metadata["image_width"] == 200
        assert doc.metadata["image_height"] == 150


@pytest.mark.integration
class TestIngestStageImageOnly:
    """IngestStage unit-level integration: image without running full pipeline."""

    def test_ingest_stage_image(self, sample_jpeg_path):
        stage = IngestStage()
        doc = stage.ingest(str(sample_jpeg_path))

        assert doc.source_type == "image"
        assert doc.source_path is not None
        assert doc.mime_type == "image/jpeg"
        assert "Image file" in doc.raw_content

    def test_ingest_stage_image_raw_content_is_description(self, sample_jpeg_path):
        stage = IngestStage()
        doc = stage.ingest(str(sample_jpeg_path))
        # raw_content should be a human-readable description, not binary data
        assert doc.raw_content.startswith("Image file:")
        assert "100" in doc.raw_content  # dimensions appear in description


@pytest.mark.integration
class TestVideoIngestionPipeline:
    """Full pipeline tests for video ingestion — ffmpeg calls mocked."""

    MOCK_PROBE = {
        "format": {"duration": "10.0", "bit_rate": "1000000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 480,
                "avg_frame_rate": "24/1",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

    def test_ingest_video_stores_document(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        """Video ingestion should produce a Document with source_type=video."""
        fake_video = tmp_path / "clip.mp4"
        fake_video.write_bytes(b"fake mp4 content " * 50)

        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)

        with patch("ffmpeg.probe", return_value=self.MOCK_PROBE):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                # Use side_effect to capture the real doc.id from the Document argument
                def make_extracted(doc, provider):
                    return ExtractedInfo(
                        document_id=doc.id,
                        summary="A test video.",
                        key_facts=["It shows a demo."],
                        topics=["video", "test"],
                    )
                with patch.object(pipeline._extract_stage, "_extract_video_fallback", side_effect=make_extracted):
                    doc = pipeline.ingest_file(fake_video)

        assert doc.source_type == "video"
        assert doc.mime_type == "video/mp4"

    def test_ingest_video_metadata(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        """Video document should carry duration, dimensions, codec metadata."""
        fake_video = tmp_path / "clip.mp4"
        fake_video.write_bytes(b"fake " * 100)

        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)

        with patch("ffmpeg.probe", return_value=self.MOCK_PROBE):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                def make_extracted(doc, provider):
                    return ExtractedInfo(
                        document_id=doc.id,
                        summary="ok",
                        key_facts=[],
                        topics=[],
                    )
                with patch.object(pipeline._extract_stage, "_extract_video_fallback", side_effect=make_extracted):
                    doc = pipeline.ingest_file(fake_video)

        assert doc.metadata["video_width"] == 640
        assert doc.metadata["video_height"] == 480
        assert doc.metadata["video_duration"] == pytest.approx(10.0)
        assert doc.metadata["has_audio"] is True

    def test_ingest_unsupported_format_still_works_as_text(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path
    ):
        """An unsupported extension falls through to text ingestion."""
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        txt = tmp_path / "notes.txt"
        txt.write_text("Hello world")
        doc = pipeline.ingest_file(txt)
        assert doc.source_type == "file"


@pytest.mark.integration
class TestMixedPipelineQuery:
    """Verify that text and image documents coexist and can both be queried."""

    def test_text_and_image_both_stored(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)

        text_doc = pipeline.ingest_text("Python is a programming language.")
        img_doc = pipeline.ingest_file(sample_jpeg_path)

        docs = pipeline.metadata_store.list_documents()
        assert len(docs) == 2
        types = {d.source_type for d in docs}
        assert "text" in types
        assert "image" in types

    def test_query_returns_results_from_mixed_store(
        self, tmp_settings, mock_llm, mock_embedder, tmp_path, sample_jpeg_path
    ):
        from rember.query.answerer import Answerer
        from rember.query.retriever import Retriever

        pipeline = _build_pipeline(tmp_settings, mock_llm, mock_embedder, tmp_path)
        pipeline.ingest_text("Machine learning is a subset of AI.")
        pipeline.ingest_file(sample_jpeg_path)

        retriever = Retriever(
            vector_store=pipeline.vector_store,
            metadata_store=pipeline.metadata_store,
            embedding_provider=mock_embedder,
            config=tmp_settings.query,
        )
        answerer = Answerer(llm_registry=pipeline.llm_registry)

        results = retriever.retrieve("AI and machine learning")
        answer = answerer.answer("What is machine learning?", results)

        assert len(results) > 0
        assert len(answer.answer) > 0
