"""
Image processing utilities using Pillow.

Responsibilities:
  - Validate image file integrity
  - Extract image metadata (format, dimensions, file size)
  - Prepare images for Gemini API (resize, compress, convert to bytes)
  - Map file extensions to MIME types
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported image formats and their MIME types
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif"}

MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".gif":  "image/gif",
    ".bmp":  "image/bmp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


@dataclass
class ImageMetadata:
    """Metadata extracted from an image file."""
    format: str          # "JPEG", "PNG", "WEBP", "GIF", "BMP"
    width: int
    height: int
    mode: str            # "RGB", "RGBA", "L", "P", etc.
    file_size_bytes: int
    mime_type: str


class ImageProcessor:
    """Validate, inspect, and prepare images for LLM processing."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, path: Path) -> ImageMetadata:
        """
        Open the image with Pillow, verify integrity, and return metadata.

        Raises:
            ValueError: If the file is not a valid image or not supported.
            ImportError: If Pillow is not installed.
        """
        self._require_pillow()
        from PIL import Image, UnidentifiedImageError

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        try:
            with Image.open(path) as img:
                img.verify()  # verify integrity (can't use img after this)
        except (UnidentifiedImageError, Exception) as e:
            raise ValueError(f"Invalid image file '{path.name}': {e}") from e

        # Re-open after verify (verify consumes the file pointer)
        try:
            with Image.open(path) as img:
                fmt = img.format or ext.lstrip(".").upper()
                width, height = img.size
                mode = img.mode
        except Exception as e:
            raise ValueError(f"Could not read image '{path.name}': {e}") from e

        mime_type = self.get_mime_type(path)
        file_size = path.stat().st_size

        logger.debug(
            "Image validated: %s (%dx%d %s, %.1f KB)",
            path.name, width, height, fmt, file_size / 1024,
        )

        return ImageMetadata(
            format=fmt,
            width=width,
            height=height,
            mode=mode,
            file_size_bytes=file_size,
            mime_type=mime_type,
        )

    def get_mime_type(self, path: Path) -> str:
        """Return the MIME type for a given image file path."""
        ext = Path(path).suffix.lower()
        return MIME_MAP.get(ext, "image/jpeg")

    def prepare_for_api(
        self,
        path: Path,
        max_size: tuple[int, int] = (2048, 2048),
        quality: int = 85,
    ) -> bytes:
        """
        Prepare an image for the Gemini API.

        Steps:
          1. Open with Pillow
          2. Convert RGBA/P → RGB (required for JPEG output)
          3. Resize to fit within max_size (preserving aspect ratio)
          4. Encode as JPEG bytes

        Args:
            path: Path to the image file.
            max_size: Maximum (width, height) tuple.
            quality: JPEG compression quality (1-95).

        Returns:
            JPEG bytes ready for types.Part.from_bytes().
        """
        self._require_pillow()
        from PIL import Image

        with Image.open(path) as img:
            # Convert modes incompatible with JPEG
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Resize if larger than max_size (thumbnail is in-place)
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                logger.debug("Resized image to %s", img.size)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue()

    def read_bytes(self, path: Path) -> bytes:
        """Read raw image bytes without transformation."""
        return Path(path).read_bytes()

    def get_description(self, path: Path) -> str:
        """
        Return a human-readable description for Document.raw_content.

        Example: "Image file: photo.jpg (1920×1080 JPEG, 2.3 MB)"
        """
        path = Path(path)
        try:
            meta = self.validate(path)
            size_mb = meta.file_size_bytes / (1024 * 1024)
            return (
                f"Image file: {path.name} "
                f"({meta.width}×{meta.height} {meta.format}, {size_mb:.1f} MB)"
            )
        except Exception:
            size_bytes = path.stat().st_size if path.exists() else 0
            return f"Image file: {path.name} ({size_bytes / 1024:.1f} KB)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_pillow() -> None:
        try:
            import PIL  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Pillow is not installed. Run: pip install Pillow"
            ) from e

    @classmethod
    def is_supported(cls, path: Path) -> bool:
        """Return True if the file extension is a supported image format."""
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
