"""
Unit tests for VideoProcessor (media/video.py).

ffmpeg calls are mocked — no actual video files or ffmpeg binary needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rember.media.video import (
    MIME_MAP,
    SUPPORTED_EXTENSIONS,
    VideoMetadata,
    VideoProcessor,
)


# Reusable mock ffmpeg probe response
MOCK_PROBE = {
    "format": {
        "duration": "30.5",
        "bit_rate": "2000000",
    },
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "30/1",
            "duration": "30.5",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
        },
    ],
}

MOCK_PROBE_NO_AUDIO = {
    "format": {"duration": "10.0"},
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "vp9",
            "width": 1280,
            "height": 720,
            "avg_frame_rate": "24/1",
        }
    ],
}


def _make_fake_video(tmp_path: Path, name: str = "video.mp4") -> Path:
    """Create a fake video file (not real video, just for path operations)."""
    path = tmp_path / name
    path.write_bytes(b"fake video content " * 100)
    return path


class TestMimeMap:
    def test_mp4_mime(self):
        assert VideoProcessor().get_mime_type(Path("v.mp4")) == "video/mp4"

    def test_mov_mime(self):
        assert VideoProcessor().get_mime_type(Path("v.mov")) == "video/quicktime"

    def test_avi_mime(self):
        assert VideoProcessor().get_mime_type(Path("v.avi")) == "video/avi"

    def test_unknown_defaults_mp4(self):
        assert VideoProcessor().get_mime_type(Path("v.xyz")) == "video/mp4"


class TestIsSupported:
    def test_mp4_supported(self):
        assert VideoProcessor.is_supported(Path("v.mp4"))

    def test_jpg_not_supported(self):
        assert not VideoProcessor.is_supported(Path("img.jpg"))


class TestGetMetadata:
    def test_metadata_parsing(self, tmp_path):
        fake = _make_fake_video(tmp_path)

        with patch("ffmpeg.probe", return_value=MOCK_PROBE):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                proc = VideoProcessor()
                meta = proc.get_metadata(fake)

        assert meta.duration_seconds == 30.5
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.fps == 30.0
        assert meta.codec == "h264"
        assert meta.has_audio is True
        assert meta.mime_type == "video/mp4"

    def test_no_audio(self, tmp_path):
        fake = _make_fake_video(tmp_path, "clip.webm")

        with patch("ffmpeg.probe", return_value=MOCK_PROBE_NO_AUDIO):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                proc = VideoProcessor()
                meta = proc.get_metadata(fake)

        assert meta.has_audio is False
        assert meta.fps == 24.0

    def test_file_not_found(self, tmp_path):
        proc = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                proc.get_metadata(tmp_path / "missing.mp4")

    def test_no_video_stream_raises(self, tmp_path):
        fake = _make_fake_video(tmp_path)
        probe_no_video = {"format": {"duration": "0"}, "streams": []}

        with patch("ffmpeg.probe", return_value=probe_no_video):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                proc = VideoProcessor()
                with pytest.raises(ValueError, match="No video stream"):
                    proc.get_metadata(fake)

    def test_ffmpeg_not_installed(self):
        proc = VideoProcessor()
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg binary not found"):
                proc.get_metadata(Path("video.mp4"))


class TestParseFraction:
    def test_simple_fraction(self):
        assert VideoProcessor._parse_fraction("30/1") == 30.0

    def test_ntsc_fraction(self):
        fps = VideoProcessor._parse_fraction("30000/1001")
        assert abs(fps - 29.97) < 0.01

    def test_zero_denominator(self):
        assert VideoProcessor._parse_fraction("0/0") == 0.0

    def test_plain_number(self):
        assert VideoProcessor._parse_fraction("25") == 25.0


class TestGetDescription:
    def test_description_contains_filename(self, tmp_path):
        fake = _make_fake_video(tmp_path, "myvideo.mp4")

        with patch("ffmpeg.probe", return_value=MOCK_PROBE):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                proc = VideoProcessor()
                desc = proc.get_description(fake)

        assert "myvideo.mp4" in desc
        assert "1920" in desc
        assert "1080" in desc

    def test_description_fallback_on_error(self, tmp_path):
        fake = _make_fake_video(tmp_path)
        proc = VideoProcessor()

        with patch("ffmpeg.probe", side_effect=Exception("probe failed")):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                desc = proc.get_description(fake)

        assert "video.mp4" in desc or "Video file" in desc
