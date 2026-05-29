"""
Optional Whisper transcription for audio/video fallback processing.

This module is an optional dependency — openai-whisper is NOT installed by default.
Install with: pip install "rember[whisper]"  or  pip install openai-whisper

Usage:
    if Transcriber.is_available():
        t = Transcriber(model_name="base")
        result = t.transcribe(Path("audio.wav"))
        print(result.text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of a Whisper transcription."""
    text: str
    language: str | None = None
    segments: list[dict] = field(default_factory=list)


class Transcriber:
    """
    Transcribe audio files using OpenAI Whisper (local, no API calls).

    Model sizes (speed vs quality):
      tiny  → fastest, lowest quality  (~39M params)
      base  → good balance             (~74M params)  ← default
      small → better quality           (~244M params)
      medium→ high quality             (~769M params)
      large → best quality             (~1.55B params)

    Models are downloaded on first use to ~/.cache/whisper/
    """

    def __init__(self, model_name: str = "base") -> None:
        """
        Load the Whisper model.

        Args:
            model_name: Whisper model size (tiny/base/small/medium/large).

        Raises:
            ImportError: If openai-whisper is not installed.
        """
        if not self.is_available():
            raise ImportError(
                "openai-whisper is not installed. "
                "Install with: pip install openai-whisper\n"
                "Or: pip install \"rember[whisper]\""
            )

        import whisper  # noqa: F401
        self._model_name = model_name
        self._model = None  # lazy-load on first transcription

    def _load_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            import whisper
            logger.info("Loading Whisper model '%s'…", self._model_name)
            self._model = whisper.load_model(self._model_name)
            logger.info("Whisper model '%s' loaded.", self._model_name)
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (WAV, MP3, M4A, etc.).
                       For best results, use 16kHz mono WAV.

        Returns:
            TranscriptionResult with text, detected language, and segments.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()
        logger.info("Transcribing '%s'…", audio_path.name)

        result = model.transcribe(str(audio_path))

        text = result.get("text", "").strip()
        language = result.get("language")
        segments = [
            {
                "text": seg.get("text", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            }
            for seg in result.get("segments", [])
        ]

        logger.info(
            "Transcription complete: %d chars, language=%s", len(text), language
        )

        return TranscriptionResult(text=text, language=language, segments=segments)

    @staticmethod
    def is_available() -> bool:
        """Return True if openai-whisper is installed."""
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False
