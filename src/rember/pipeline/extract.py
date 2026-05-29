"""
ExtractStage — uses an LLM to extract structured information from a Document.

Supports both text and multimodal (image/video) content:
  - Text documents: truncated content → extract_info()
  - Image documents: Pillow-prepared bytes → extract_info_multimodal()
  - Video documents: Gemini Files API (native) or frame+audio fallback

Output is an ExtractedInfo containing:
  - summary:   1-3 sentence summary
  - key_facts: list of discrete, self-contained facts
  - topics:    list of topic/tag strings

The LLM provider is resolved from the registry using the "extraction" task.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from rember.config import MediaConfig
from rember.llm.registry import LLMRegistry
from rember.models import Document, ExtractedInfo, SourceType
from rember.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

# If document content exceeds this many characters, truncate before sending
# to the LLM to avoid exceeding context windows.
_MAX_CONTENT_CHARS = 30_000  # ~7,500 tokens (rough estimate)


class ExtractStage(PipelineStage[Document, ExtractedInfo]):
    """
    Uses an LLM to extract structured knowledge from a Document.

    Dispatches to the appropriate extraction strategy based on source_type:
      - TEXT / FILE → text extraction (Phase 1)
      - IMAGE       → multimodal image extraction (Phase 2)
      - VIDEO       → native video or fallback frame+audio extraction (Phase 2)

    Input:  Document
    Output: ExtractedInfo (summary, key_facts, topics, metadata)
    """

    def __init__(
        self,
        llm_registry: LLMRegistry,
        media_config: MediaConfig | None = None,
    ) -> None:
        self._registry = llm_registry
        self._media_config = media_config or MediaConfig()

    @property
    def name(self) -> str:
        return "extract"

    def process(self, input_data: Document) -> ExtractedInfo:
        return self.extract(input_data)

    def extract(self, doc: Document) -> ExtractedInfo:
        """
        Run extraction on a document, dispatching by source type.

        Args:
            doc: The Document from IngestStage.

        Returns:
            ExtractedInfo with structured knowledge.
        """
        provider = self._registry.get_provider("extraction")
        source_type = doc.source_type  # str because use_enum_values=True

        if source_type == SourceType.IMAGE or source_type == "image":
            logger.info("ExtractStage: image extraction for document %s", doc.id)
            return self._extract_image(doc, provider)

        elif source_type == SourceType.VIDEO or source_type == "video":
            logger.info("ExtractStage: video extraction for document %s", doc.id)
            return self._extract_video(doc, provider)

        else:
            # Text / file — original Phase 1 path
            content = self._prepare_content(doc)
            logger.info(
                "ExtractStage: text extraction for document %s using provider '%s'",
                doc.id, provider.provider_name,
            )

            extra_metadata: dict[str, Any] = {
                **doc.metadata,
                "source_type": source_type if isinstance(source_type, str) else source_type.value,
                "source_path": doc.source_path,
                "llm_provider": provider.provider_name,
            }

            extracted = provider.extract_info(
                content=content,
                document_id=doc.id,
                content_type=source_type if isinstance(source_type, str) else source_type.value,
                extra_metadata=extra_metadata,
            )

            logger.debug(
                "Extracted %d key facts and %d topics from document %s",
                len(extracted.key_facts), len(extracted.topics), doc.id,
            )
            return extracted

    # ------------------------------------------------------------------
    # Image extraction (Phase 2)
    # ------------------------------------------------------------------

    def _extract_image(self, doc: Document, provider: Any) -> ExtractedInfo:
        """
        Extract info from an image using multimodal LLM.

        Prepares the image bytes (resized/compressed) and sends them
        as a Part object to extract_info_multimodal().
        """
        from rember.media.image import ImageProcessor

        if not doc.source_path:
            raise ValueError(f"Image document {doc.id} has no source_path.")

        image_path = Path(doc.source_path)
        processor = ImageProcessor()

        max_dim = self._media_config.image_max_dimension
        quality = self._media_config.image_quality

        image_bytes = processor.prepare_for_api(
            image_path,
            max_size=(max_dim, max_dim),
            quality=quality,
        )

        # Build content parts for the provider
        # Using Gemini's types.Part.from_bytes for the image
        content_parts = self._build_image_parts(image_bytes, doc.mime_type or "image/jpeg")

        extra_metadata: dict[str, Any] = {
            **doc.metadata,
            "source_type": "image",
            "source_path": doc.source_path,
            "llm_provider": provider.provider_name,
            "extraction_method": "multimodal_image",
        }

        return provider.extract_info_multimodal(
            content_parts=content_parts,
            document_id=doc.id,
            content_type="image",
            extra_metadata=extra_metadata,
        )

    @staticmethod
    def _build_image_parts(image_bytes: bytes, mime_type: str) -> list[Any]:
        """Build a list of content parts for image extraction."""
        try:
            from google.genai import types
            return [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        except ImportError:
            # Fallback: pass raw bytes (provider's default implementation will handle)
            return [image_bytes]

    # ------------------------------------------------------------------
    # Video extraction (Phase 2)
    # ------------------------------------------------------------------

    def _extract_video(self, doc: Document, provider: Any) -> ExtractedInfo:
        """
        Extract info from a video.

        Strategy:
          1. If prefer_native_video + provider supports it → Gemini Files API
          2. Fallback → frame extraction + optional Whisper transcription
        """
        if not doc.source_path:
            raise ValueError(f"Video document {doc.id} has no source_path.")

        # Try native Gemini video if preferred and provider supports upload_video
        if (
            self._media_config.prefer_native_video
            and hasattr(provider, "upload_video")
        ):
            try:
                return self._extract_video_native(doc, provider)
            except Exception as e:
                logger.warning(
                    "Native video extraction failed for %s: %s. Falling back to frames.",
                    doc.id, e,
                )

        return self._extract_video_fallback(doc, provider)

    def _extract_video_native(self, doc: Document, provider: Any) -> ExtractedInfo:
        """
        Upload video to Gemini Files API, extract info, then clean up.
        """
        timeout = self._media_config.video_upload_timeout_seconds
        uploaded = provider.upload_video(doc.source_path, timeout_seconds=timeout)
        uploaded_name = uploaded.name

        try:
            extra_metadata: dict[str, Any] = {
                **doc.metadata,
                "source_type": "video",
                "source_path": doc.source_path,
                "llm_provider": provider.provider_name,
                "extraction_method": "native_video",
            }

            return provider.extract_info_multimodal(
                content_parts=[uploaded],
                document_id=doc.id,
                content_type="video",
                extra_metadata=extra_metadata,
            )
        finally:
            # Always clean up — files auto-expire at 48h but clean up early
            provider.delete_uploaded_file(uploaded_name)

    def _extract_video_fallback(self, doc: Document, provider: Any) -> ExtractedInfo:
        """
        Fallback video extraction:
          1. Extract N key frames as JPEG images
          2. Optionally transcribe audio with Whisper
          3. Send frames (as image Parts) + transcript to extract_info_multimodal()
        """
        from rember.media.video import VideoProcessor

        video_path = Path(doc.source_path)  # type: ignore[arg-type]
        video_proc = VideoProcessor()
        num_frames = self._media_config.num_frames_to_extract

        # Step 1: Extract key frames
        with tempfile.TemporaryDirectory(prefix="rember_video_") as tmp_dir:
            frame_paths = video_proc.extract_frames(
                video_path,
                num_frames=num_frames,
                output_dir=Path(tmp_dir),
            )
            logger.debug("Fallback: extracted %d frames from %s", len(frame_paths), doc.id)

            # Step 2: Optionally transcribe audio
            transcript = self._transcribe_audio(doc, video_proc, video_path, tmp_dir)

            # Step 3: Build content parts (frames + transcript text)
            content_parts: list[Any] = []

            # Add frame images (limit to avoid exceeding context)
            for frame_path in frame_paths[:8]:
                frame_bytes = frame_path.read_bytes()
                parts = self._build_image_parts(frame_bytes, "image/jpeg")
                content_parts.extend(parts)

            # Add transcript text if available
            if transcript:
                content_parts.append(f"Audio transcript:\n{transcript}")

            # Fallback if no frames and no transcript
            if not content_parts:
                content_parts = [doc.raw_content]

            extra_metadata: dict[str, Any] = {
                **doc.metadata,
                "source_type": "video",
                "source_path": doc.source_path,
                "llm_provider": provider.provider_name,
                "extraction_method": "fallback_frames",
                "frames_extracted": len(frame_paths),
                "has_transcript": bool(transcript),
            }

            return provider.extract_info_multimodal(
                content_parts=content_parts,
                document_id=doc.id,
                content_type="video",
                extra_metadata=extra_metadata,
            )

    def _transcribe_audio(
        self,
        doc: Document,
        video_proc: Any,
        video_path: Path,
        tmp_dir: str,
    ) -> str:
        """
        Attempt to extract and transcribe audio from the video.
        Returns empty string if transcription is disabled, unavailable, or fails.
        """
        if not self._media_config.enable_transcription:
            return ""

        has_audio = doc.metadata.get("has_audio", False)
        if not has_audio:
            logger.debug("Video %s has no audio track, skipping transcription.", doc.id)
            return ""

        try:
            from rember.media.transcribe import Transcriber
            if not Transcriber.is_available():
                logger.debug(
                    "Whisper not installed. Skipping transcription. "
                    "Install with: pip install openai-whisper"
                )
                return ""

            audio_path = video_proc.extract_audio(
                video_path,
                output_path=Path(tmp_dir) / "audio.wav",
            )
            transcriber = Transcriber(model_name=self._media_config.whisper_model)
            result = transcriber.transcribe(audio_path)
            logger.info("Transcription: %d chars, language=%s", len(result.text), result.language)
            return result.text

        except Exception as e:
            logger.warning("Audio transcription failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Text helpers (Phase 1 — unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_content(doc: Document) -> str:
        """
        Prepare text content for LLM extraction.
        Truncates very long content with a note.
        """
        content = doc.raw_content

        if len(content) > _MAX_CONTENT_CHARS:
            truncated = content[:_MAX_CONTENT_CHARS]
            logger.warning(
                "Document %s content truncated from %d to %d chars for extraction.",
                doc.id, len(content), _MAX_CONTENT_CHARS,
            )
            return truncated + "\n\n[Content truncated for processing]"

        return content
