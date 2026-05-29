"""
Video processing utilities using ffmpeg-python.

Responsibilities:
  - Probe video metadata (duration, resolution, fps, codec, audio)
  - Extract N evenly-spaced key frames as JPEG files
  - Extract audio track as 16kHz mono WAV (Whisper-compatible)
  - Map file extensions to MIME types
  - Generate human-readable descriptions for Document.raw_content

Requires: ffmpeg-python (pip install ffmpeg-python)
Requires: ffmpeg binary on PATH (sudo apt install ffmpeg)
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported video formats and their MIME types
SUPPORTED_EXTENSIONS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg", ".wmv"}

MIME_MAP: dict[str, str] = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".avi":  "video/avi",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
    ".mpeg": "video/mpeg",
    ".mpg":  "video/mpeg",
    ".wmv":  "video/x-ms-wmv",
    ".3gp":  "video/3gpp",
}


@dataclass
class VideoMetadata:
    """Metadata probed from a video file."""
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    file_size_bytes: int
    has_audio: bool
    mime_type: str
    bitrate: str = ""


class VideoProcessor:
    """Extract metadata, key frames, and audio from video files using ffmpeg."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_metadata(self, path: Path) -> VideoMetadata:
        """
        Probe video file with ffmpeg and return structured metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If no video stream is found or ffmpeg fails.
            ImportError: If ffmpeg-python is not installed.
            RuntimeError: If ffmpeg binary is not on PATH.
        """
        self._require_ffmpeg()
        import ffmpeg

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        try:
            probe = ffmpeg.probe(str(path))
        except ffmpeg.Error as e:
            raise ValueError(
                f"ffmpeg could not probe '{path.name}': {e.stderr.decode() if e.stderr else e}"
            ) from e

        # Find video stream
        video_streams = [s for s in probe["streams"] if s.get("codec_type") == "video"]
        if not video_streams:
            raise ValueError(f"No video stream found in '{path.name}'")

        video = video_streams[0]

        # Parse FPS (often "24/1" or "30000/1001")
        fps = self._parse_fraction(video.get("avg_frame_rate", "0/1"))

        # Duration: prefer format-level, fall back to stream-level
        duration = float(probe["format"].get("duration", 0))
        if duration == 0:
            duration = float(video.get("duration", 0))

        # Check for audio stream
        has_audio = any(s.get("codec_type") == "audio" for s in probe["streams"])

        mime_type = self.get_mime_type(path)
        file_size = path.stat().st_size

        meta = VideoMetadata(
            duration_seconds=duration,
            width=video.get("width", 0),
            height=video.get("height", 0),
            fps=round(fps, 2),
            codec=video.get("codec_name", "unknown"),
            file_size_bytes=file_size,
            has_audio=has_audio,
            mime_type=mime_type,
            bitrate=probe["format"].get("bit_rate", ""),
        )

        logger.debug(
            "Video probed: %s (%dx%d, %.1fs, %.1f fps, %s, audio=%s)",
            path.name, meta.width, meta.height, meta.duration_seconds,
            meta.fps, meta.codec, meta.has_audio,
        )

        return meta

    def get_mime_type(self, path: Path) -> str:
        """Return the MIME type for a given video file path."""
        ext = Path(path).suffix.lower()
        return MIME_MAP.get(ext, "video/mp4")

    def extract_frames(
        self,
        path: Path,
        num_frames: int = 10,
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Extract N evenly-spaced key frames from the video as JPEG files.

        Uses a single-pass ffmpeg fps filter for efficiency.

        Args:
            path: Path to the video file.
            num_frames: Number of frames to extract.
            output_dir: Directory to save frames. Defaults to a temp directory.

        Returns:
            List of Paths to the extracted JPEG frame files.
        """
        self._require_ffmpeg()
        import ffmpeg

        path = Path(path)
        meta = self.get_metadata(path)

        if meta.duration_seconds <= 0:
            logger.warning("Video has zero duration, cannot extract frames.")
            return []

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="rember_frames_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate fps filter rate to get exactly num_frames over the duration
        fps_rate = num_frames / meta.duration_seconds
        output_pattern = str(output_dir / "frame_%04d.jpg")

        try:
            (
                ffmpeg
                .input(str(path))
                .filter("fps", fps=fps_rate)
                .output(output_pattern, **{"qscale:v": 2, "vframes": num_frames})
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Frame extraction failed for '{path.name}': {stderr}") from e

        # Collect output frames in order
        frames = sorted(output_dir.glob("frame_*.jpg"))
        logger.debug("Extracted %d frames from '%s' → %s", len(frames), path.name, output_dir)
        return frames

    def extract_audio(
        self,
        path: Path,
        output_path: Path | None = None,
    ) -> Path:
        """
        Extract the audio track as a 16kHz mono WAV file (Whisper-compatible).

        Args:
            path: Path to the video file.
            output_path: Path for output WAV file. Defaults to a temp file.

        Returns:
            Path to the extracted WAV file.
        """
        self._require_ffmpeg()
        import ffmpeg

        path = Path(path)

        if output_path is None:
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="rember_audio_")
            os.close(fd)
            output_path = Path(tmp)
        else:
            output_path = Path(output_path)

        try:
            (
                ffmpeg
                .input(str(path))
                .output(
                    str(output_path),
                    acodec="pcm_s16le",  # uncompressed WAV
                    ac=1,                # mono
                    ar=16000,            # 16kHz — what Whisper expects
                )
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"Audio extraction failed for '{path.name}': {stderr}") from e

        logger.debug("Extracted audio from '%s' → %s", path.name, output_path)
        return output_path

    def get_description(self, path: Path) -> str:
        """
        Return a human-readable description for Document.raw_content.

        Example: "Video file: reel.mp4 (1080×1920, 30.0s, 24.0fps, h264, 15.2 MB)"
        """
        path = Path(path)
        try:
            meta = self.get_metadata(path)
            size_mb = meta.file_size_bytes / (1024 * 1024)
            audio_info = " [audio]" if meta.has_audio else ""
            return (
                f"Video file: {path.name} "
                f"({meta.width}×{meta.height}, {meta.duration_seconds:.1f}s, "
                f"{meta.fps}fps, {meta.codec}{audio_info}, {size_mb:.1f} MB)"
            )
        except Exception:
            size_bytes = path.stat().st_size if path.exists() else 0
            return f"Video file: {path.name} ({size_bytes / (1024*1024):.1f} MB)"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_fraction(fraction_str: str) -> float:
        """Parse a fraction string like '30000/1001' into a float."""
        try:
            parts = fraction_str.split("/")
            if len(parts) == 2:
                num, den = int(parts[0]), int(parts[1])
                return num / den if den else 0.0
            return float(fraction_str)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _require_ffmpeg() -> None:
        """Check ffmpeg-python is installed and ffmpeg binary is on PATH."""
        try:
            import ffmpeg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ffmpeg-python is not installed. Run: pip install ffmpeg-python"
            ) from e

        import shutil
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg binary not found on PATH. "
                "Install it with: sudo apt install ffmpeg  "
                "(or brew install ffmpeg on macOS)"
            )

    @classmethod
    def is_supported(cls, path: Path) -> bool:
        """Return True if the file extension is a supported video format."""
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
