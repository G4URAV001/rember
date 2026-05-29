"""
IngestStage — reads raw input and produces a Document.

Supported input formats:
  Phase 1 (text):
    - Raw text string (passed directly)
    - .txt  plain text
    - .md   markdown (treated as plain text)
    - .json JSON file (pretty-printed as text)
    - .csv  CSV file (rows joined as text)

  Phase 2 (media):
    - Image files: .jpg, .jpeg, .png, .webp, .gif, .bmp, .heic, .heif
    - Video files: .mp4, .mov, .avi, .mkv, .webm, .mpeg, .wmv
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from rember.models import Document, SourceType
from rember.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

# Supported text-based file extensions
_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".text"}
_JSON_EXTENSIONS = {".json", ".jsonl"}
_CSV_EXTENSIONS  = {".csv", ".tsv"}

# Phase 2: image and video extensions (sourced from media module)
# Kept here as well for fast extension-based routing in _ingest_file()
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".wmv"}


class IngestStage(PipelineStage[tuple[str, dict[str, Any]], Document]):
    """
    Reads source content and creates a Document model.

    Input:  (source, metadata)
              source — either a file path string or raw text content
              metadata — user-supplied tags / annotations
    Output: Document
    """

    @property
    def name(self) -> str:
        return "ingest"

    def process(self, input_data: tuple[str, dict[str, Any]]) -> Document:
        source, metadata = input_data
        return self.ingest(source, metadata)

    def ingest(self, source: str, metadata: dict[str, Any] | None = None) -> Document:
        """
        Ingest from a file path or raw text string.

        Args:
            source: Either an absolute/relative file path OR raw text.
            metadata: Optional key-value tags to attach to the document.
        """
        meta = dict(metadata or {})
        path = Path(source)

        if path.exists() and path.is_file():
            return self._ingest_file(path, meta)
        else:
            # Treat as raw text
            return self._ingest_text(source, meta)

    # ------------------------------------------------------------------
    # Internal ingestion methods
    # ------------------------------------------------------------------

    def _ingest_text(self, text: str, metadata: dict[str, Any]) -> Document:
        """Ingest a raw text string."""
        if not text.strip():
            raise ValueError("Cannot ingest empty text.")

        logger.debug("Ingesting raw text (%d chars)", len(text))
        return Document(
            source_type=SourceType.TEXT,
            source_path=None,
            raw_content=text.strip(),
            metadata=metadata,
        )

    def _ingest_file(self, path: Path, metadata: dict[str, Any]) -> Document:
        """Ingest a file, detecting format by extension."""
        ext = path.suffix.lower()
        metadata.setdefault("filename", path.name)
        metadata.setdefault("file_extension", ext)

        if ext in _IMAGE_EXTENSIONS:
            return self._ingest_image(path, metadata)
        elif ext in _VIDEO_EXTENSIONS:
            return self._ingest_video(path, metadata)

        logger.info("Ingesting file: %s", path)

        if ext in _TEXT_EXTENSIONS:
            content = self._read_text_file(path)
            source_type = SourceType.FILE
        elif ext in _JSON_EXTENSIONS:
            content = self._read_json_file(path)
            source_type = SourceType.FILE
        elif ext in _CSV_EXTENSIONS:
            content = self._read_csv_file(path, ext)
            source_type = SourceType.FILE
        else:
            # Attempt to read as plain text with a warning
            logger.warning(
                "Unknown file extension '%s'. Attempting to read as plain text.", ext
            )
            content = self._read_text_file(path)
            source_type = SourceType.FILE

        if not content.strip():
            raise ValueError(f"File '{path}' is empty or produced no content.")

        return Document(
            source_type=source_type,
            source_path=str(path.resolve()),
            raw_content=content,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Media ingestion methods (Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _ingest_image(path: Path, metadata: dict[str, Any]) -> Document:
        """
        Ingest an image file.

        Validates the image with Pillow and extracts metadata (dimensions, format).
        Binary data is NOT stored in the Document — ExtractStage reads it from
        source_path on demand.
        """
        from rember.media.image import ImageProcessor

        processor = ImageProcessor()
        try:
            img_meta = processor.validate(path)
        except (ValueError, FileNotFoundError) as e:
            raise ValueError(f"Cannot ingest image '{path.name}': {e}") from e

        metadata.update({
            "image_format": img_meta.format,
            "image_width": img_meta.width,
            "image_height": img_meta.height,
            "file_size_bytes": img_meta.file_size_bytes,
        })

        logger.info(
            "Ingesting image: %s (%dx%d %s, %.1f KB)",
            path.name, img_meta.width, img_meta.height,
            img_meta.format, img_meta.file_size_bytes / 1024,
        )

        return Document(
            source_type=SourceType.IMAGE,
            source_path=str(path.resolve()),
            mime_type=img_meta.mime_type,
            raw_content=processor.get_description(path),
            metadata=metadata,
        )

    @staticmethod
    def _ingest_video(path: Path, metadata: dict[str, Any]) -> Document:
        """
        Ingest a video file.

        Probes the video with ffmpeg and extracts metadata (duration, dimensions,
        fps, codec, audio). Binary data is NOT stored — ExtractStage reads from
        source_path on demand.
        """
        from rember.media.video import VideoProcessor

        processor = VideoProcessor()
        try:
            vid_meta = processor.get_metadata(path)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise ValueError(f"Cannot ingest video '{path.name}': {e}") from e

        # Validate against reasonable limits
        max_size_mb = 500  # hard ceiling; soft limits are in MediaConfig
        file_mb = vid_meta.file_size_bytes / (1024 * 1024)
        if file_mb > max_size_mb:
            raise ValueError(
                f"Video '{path.name}' is too large ({file_mb:.1f} MB, max {max_size_mb} MB)."
            )

        metadata.update({
            "video_duration": vid_meta.duration_seconds,
            "video_width": vid_meta.width,
            "video_height": vid_meta.height,
            "video_fps": vid_meta.fps,
            "video_codec": vid_meta.codec,
            "has_audio": vid_meta.has_audio,
            "file_size_bytes": vid_meta.file_size_bytes,
        })

        logger.info(
            "Ingesting video: %s (%.1fs, %dx%d, %s, audio=%s)",
            path.name, vid_meta.duration_seconds,
            vid_meta.width, vid_meta.height,
            vid_meta.codec, vid_meta.has_audio,
        )

        return Document(
            source_type=SourceType.VIDEO,
            source_path=str(path.resolve()),
            mime_type=vid_meta.mime_type,
            raw_content=processor.get_description(path),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # File readers (text formats)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text_file(path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _read_json_file(path: Path) -> str:
        """Read a JSON file and convert to pretty-printed text."""
        raw = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            # JSONL: one JSON object per line
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            objects = []
            for line in lines:
                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return json.dumps(objects, indent=2, ensure_ascii=False)
        else:
            try:
                data = json.loads(raw)
                return json.dumps(data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                return raw  # fallback: treat as plain text

    @staticmethod
    def _read_csv_file(path: Path, ext: str) -> str:
        """Read a CSV/TSV file and join rows into readable text."""
        delimiter = "\t" if ext == ".tsv" else ","
        lines = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames:
                lines.append("Fields: " + ", ".join(reader.fieldnames))
                lines.append("")
            for i, row in enumerate(reader):
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
                lines.append(f"Row {i + 1}: {row_text}")
        return "\n".join(lines)
