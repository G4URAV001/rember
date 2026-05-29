"""
Unit tests for ImageProcessor (media/image.py).

All tests mock the Pillow calls — no actual image files required
(except the programmatically generated sample.jpg in conftest).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rember.media.image import (
    MIME_MAP,
    SUPPORTED_EXTENSIONS,
    ImageMetadata,
    ImageProcessor,
)


class TestMimeMap:
    def test_jpeg_extensions(self):
        proc = ImageProcessor()
        assert proc.get_mime_type(Path("photo.jpg")) == "image/jpeg"
        assert proc.get_mime_type(Path("photo.jpeg")) == "image/jpeg"

    def test_png_mime(self):
        assert ImageProcessor().get_mime_type(Path("img.png")) == "image/png"

    def test_webp_mime(self):
        assert ImageProcessor().get_mime_type(Path("img.webp")) == "image/webp"

    def test_unknown_extension_defaults_jpeg(self):
        assert ImageProcessor().get_mime_type(Path("img.xyz")) == "image/jpeg"


class TestIsSupportedClassMethod:
    def test_jpg_supported(self):
        assert ImageProcessor.is_supported(Path("img.jpg"))

    def test_mp4_not_supported(self):
        assert not ImageProcessor.is_supported(Path("video.mp4"))

    def test_txt_not_supported(self):
        assert not ImageProcessor.is_supported(Path("doc.txt"))


class TestValidate:
    def test_validates_real_jpeg(self, sample_jpeg_path):
        """Tests against the programmatically generated test JPEG."""
        proc = ImageProcessor()
        meta = proc.validate(sample_jpeg_path)
        assert meta.width == 100
        assert meta.height == 100
        assert meta.format == "JPEG"
        assert meta.mime_type == "image/jpeg"
        assert meta.file_size_bytes > 0

    def test_file_not_found(self, tmp_path):
        proc = ImageProcessor()
        with pytest.raises(FileNotFoundError):
            proc.validate(tmp_path / "nonexistent.jpg")

    def test_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "file.xyz"
        bad_file.write_bytes(b"fake")
        proc = ImageProcessor()
        with pytest.raises(ValueError, match="Unsupported image format"):
            proc.validate(bad_file)

    def test_invalid_image_content(self, tmp_path):
        bad_file = tmp_path / "fake.jpg"
        bad_file.write_bytes(b"this is not an image")
        proc = ImageProcessor()
        with pytest.raises(ValueError, match="Invalid image"):
            proc.validate(bad_file)


class TestPrepareForApi:
    def test_returns_bytes(self, sample_jpeg_path):
        proc = ImageProcessor()
        result = proc.prepare_for_api(sample_jpeg_path)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_output_is_jpeg(self, sample_jpeg_path):
        """JPEG starts with FF D8 FF magic bytes."""
        proc = ImageProcessor()
        result = proc.prepare_for_api(sample_jpeg_path)
        assert result[:3] == b"\xff\xd8\xff"

    def test_respects_max_size(self, sample_jpeg_path):
        """After prepare_for_api, the image should fit within max_size."""
        from PIL import Image
        proc = ImageProcessor()
        # Prepare with a tiny max size
        result = proc.prepare_for_api(sample_jpeg_path, max_size=(50, 50))
        img = Image.open(io.BytesIO(result))
        assert img.width <= 50
        assert img.height <= 50

    def test_rgba_converted_to_rgb(self, tmp_path):
        """RGBA images should be converted to RGB for JPEG compatibility."""
        from PIL import Image
        rgba_path = tmp_path / "rgba.png"
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        img.save(rgba_path)

        proc = ImageProcessor()
        result = proc.prepare_for_api(rgba_path)
        assert isinstance(result, bytes)
        assert result[:3] == b"\xff\xd8\xff"  # valid JPEG


class TestGetDescription:
    def test_description_format(self, sample_jpeg_path):
        proc = ImageProcessor()
        desc = proc.get_description(sample_jpeg_path)
        assert "sample" in desc.lower() or "image" in desc.lower()
        assert "100" in desc  # dimensions

    def test_description_on_missing_file(self, tmp_path):
        """Should return a safe fallback even if file doesn't exist."""
        proc = ImageProcessor()
        # Missing file — should not raise
        missing = tmp_path / "missing.jpg"
        desc = proc.get_description(missing)
        assert "missing" in desc.lower() or "image" in desc.lower()


class TestReadBytes:
    def test_reads_raw_bytes(self, sample_jpeg_path):
        proc = ImageProcessor()
        data = proc.read_bytes(sample_jpeg_path)
        assert isinstance(data, bytes)
        assert len(data) > 0
