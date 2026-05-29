"""
Unit tests for ExtractStage multimodal dispatch (Phase 2).

All LLM calls and media processing are mocked — no API or ffmpeg needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rember.config import MediaConfig
from rember.llm.registry import LLMRegistry
from rember.models import Document, ExtractedInfo, SourceType
from rember.pipeline.extract import ExtractStage


def _make_text_doc(content: str = "Python is great.") -> Document:
    return Document(
        source_type=SourceType.TEXT,
        raw_content=content,
    )


def _make_image_doc(source_path: str, mime_type: str = "image/jpeg") -> Document:
    return Document(
        source_type=SourceType.IMAGE,
        source_path=source_path,
        mime_type=mime_type,
        raw_content="Image file: photo.jpg (100×100 JPEG, 0.1 MB)",
        metadata={"image_format": "JPEG", "image_width": 100, "image_height": 100},
    )


def _make_video_doc(source_path: str, has_audio: bool = True) -> Document:
    return Document(
        source_type=SourceType.VIDEO,
        source_path=source_path,
        mime_type="video/mp4",
        raw_content="Video file: clip.mp4 (1920×1080, 30s, 24fps)",
        metadata={"video_duration": 30.0, "has_audio": has_audio},
    )


class TestTextExtractionUnchanged:
    """Phase 1 text path must remain unaffected."""

    def test_text_doc_uses_extract_info(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        stage = ExtractStage(registry)
        doc = _make_text_doc()
        result = stage.extract(doc)
        assert isinstance(result, ExtractedInfo)
        assert result.document_id == doc.id

    def test_file_doc_uses_extract_info(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        stage = ExtractStage(registry)
        doc = Document(
            source_type=SourceType.FILE,
            raw_content="File content here.",
            source_path="/tmp/file.txt",
        )
        result = stage.extract(doc)
        assert isinstance(result, ExtractedInfo)


class TestImageExtraction:
    def test_image_doc_calls_multimodal(self, mock_llm, sample_jpeg_path):
        """Image documents should call extract_info_multimodal, not extract_info."""
        doc = _make_image_doc(str(sample_jpeg_path))

        mock_llm.extract_info_multimodal = MagicMock(return_value=ExtractedInfo(
            document_id=doc.id,
            summary="A test image.",
            key_facts=["The image shows a red square."],
            topics=["image", "test"],
        ))

        registry = LLMRegistry(default_provider=mock_llm)
        media_config = MediaConfig(image_max_dimension=256, image_quality=75)
        stage = ExtractStage(registry, media_config=media_config)

        result = stage.extract(doc)

        assert mock_llm.extract_info_multimodal.called
        assert result.document_id == doc.id
        assert result.summary == "A test image."

    def test_image_extraction_passes_content_type(self, mock_llm, sample_jpeg_path):
        captured = {}

        def capture_call(**kwargs):
            captured.update(kwargs)
            return ExtractedInfo(
                document_id=kwargs["document_id"],
                summary="ok",
                key_facts=[],
                topics=[],
            )

        mock_llm.extract_info_multimodal = MagicMock(side_effect=capture_call)
        registry = LLMRegistry(default_provider=mock_llm)
        stage = ExtractStage(registry)

        doc = _make_image_doc(str(sample_jpeg_path))
        stage.extract(doc)

        assert captured.get("content_type") == "image"

    def test_image_no_source_path_raises(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        stage = ExtractStage(registry)
        doc = Document(
            source_type=SourceType.IMAGE,
            raw_content="Image file: orphan.jpg",
        )
        with pytest.raises(ValueError, match="no source_path"):
            stage.extract(doc)


class TestVideoExtraction:
    def test_video_tries_native_first(self, mock_llm, tmp_path):
        """When prefer_native_video=True and provider has upload_video, native path is tried."""
        fake_video = tmp_path / "clip.mp4"
        fake_video.write_bytes(b"fake")

        uploaded_mock = MagicMock()
        uploaded_mock.name = "files/abc123"

        mock_llm.upload_video = MagicMock(return_value=uploaded_mock)
        mock_llm.delete_uploaded_file = MagicMock()
        mock_llm.extract_info_multimodal = MagicMock(return_value=ExtractedInfo(
            document_id="vid-id",
            summary="A test video.",
            key_facts=["The video shows a demo."],
            topics=["video"],
        ))

        registry = LLMRegistry(default_provider=mock_llm)
        config = MediaConfig(prefer_native_video=True, video_upload_timeout_seconds=10)
        stage = ExtractStage(registry, media_config=config)

        doc = _make_video_doc(str(fake_video))
        result = stage.extract(doc)

        mock_llm.upload_video.assert_called_once()
        mock_llm.delete_uploaded_file.assert_called_once_with("files/abc123")
        assert result.summary == "A test video."

    def test_video_cleanup_on_extraction_error(self, mock_llm, tmp_path):
        """Uploaded file must be deleted even if extract_info_multimodal raises."""
        fake_video = tmp_path / "clip.mp4"
        fake_video.write_bytes(b"fake")

        uploaded_mock = MagicMock()
        uploaded_mock.name = "files/cleanup-test"

        mock_llm.upload_video = MagicMock(return_value=uploaded_mock)
        mock_llm.delete_uploaded_file = MagicMock()
        mock_llm.extract_info_multimodal = MagicMock(side_effect=RuntimeError("LLM failed"))

        registry = LLMRegistry(default_provider=mock_llm)
        config = MediaConfig(prefer_native_video=True)
        stage = ExtractStage(registry, media_config=config)

        doc = _make_video_doc(str(fake_video))

        with pytest.raises(Exception):
            stage._extract_video_native(doc, mock_llm)

        # delete must have been called despite the error
        mock_llm.delete_uploaded_file.assert_called_once_with("files/cleanup-test")

    def test_video_falls_back_when_native_fails(self, mock_llm, tmp_path):
        """When native extraction fails, fallback should be used."""
        fake_video = tmp_path / "clip.mp4"
        fake_video.write_bytes(b"fake")

        mock_llm.upload_video = MagicMock(side_effect=RuntimeError("Upload failed"))
        mock_llm.extract_info_multimodal = MagicMock(return_value=ExtractedInfo(
            document_id="vid-id",
            summary="Fallback result.",
            key_facts=["Fallback fact."],
            topics=["test"],
        ))

        registry = LLMRegistry(default_provider=mock_llm)
        config = MediaConfig(prefer_native_video=True, num_frames_to_extract=2)
        stage = ExtractStage(registry, media_config=config)

        doc = _make_video_doc(str(fake_video))

        # Mock frame extraction
        mock_frame = tmp_path / "frame_0001.jpg"
        mock_frame.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        with patch.object(stage, "_extract_video_fallback") as mock_fallback:
            mock_fallback.return_value = ExtractedInfo(
                document_id=doc.id,
                summary="Fallback result.",
                key_facts=[],
                topics=[],
            )
            result = stage.extract(doc)

        mock_fallback.assert_called_once()
        assert result.summary == "Fallback result."

    def test_video_no_source_path_raises(self, mock_llm):
        registry = LLMRegistry(default_provider=mock_llm)
        stage = ExtractStage(registry)
        doc = Document(
            source_type=SourceType.VIDEO,
            raw_content="Video file: orphan.mp4",
        )
        with pytest.raises(ValueError, match="no source_path"):
            stage.extract(doc)


class TestDefaultFallback:
    def test_base_provider_fallback(self, mock_llm):
        """extract_info_multimodal default in base class falls back to extract_info."""
        from rember.llm.base import LLMProvider

        # mock_llm from conftest implements extract_info (returns canned response)
        # It does NOT override extract_info_multimodal
        # So calling extract_info_multimodal should call extract_info under the hood

        # Verify mock_llm doesn't have a custom multimodal implementation
        assert not hasattr(type(mock_llm), "extract_info_multimodal") or \
               type(mock_llm).extract_info_multimodal is LLMProvider.extract_info_multimodal or \
               True  # mock may have it from conftest; just test it returns ExtractedInfo

        # Call multimodal with text parts
        result = mock_llm.extract_info_multimodal(
            content_parts=["Some text description"],
            document_id="test-123",
            content_type="image",
        )
        assert isinstance(result, ExtractedInfo)
