"""
Gemini LLM provider implementation.

Uses the `google-genai` SDK (google.genai.Client) to call:
  - gemini-2.0-flash (or configured model) for text generation
  - Structured extraction via a carefully designed prompt
  - Native multimodal: image inline bytes + video via Files API (Phase 2)

Implements exponential backoff on rate-limit errors (429).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from rember.config import LLMProviderConfig
from rember.llm.base import LLMProvider
from rember.models import ExtractedInfo

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge extraction assistant.
Given content, you MUST respond with ONLY valid JSON (no markdown, no code fences).

The JSON must have exactly these keys:
{
  "summary": "A concise 1-3 sentence summary of the content.",
  "key_facts": ["fact 1", "fact 2", "..."],
  "topics": ["topic1", "topic2", "..."]
}

Rules:
- summary: concise, informative, 1-3 sentences
- key_facts: list of discrete, self-contained factual statements (5-15 facts)
- topics: list of 2-6 short topic/tag strings (e.g. "machine learning", "python", "history")
- For images: describe what you see as facts (objects, text, scene, people, actions)
- For videos: describe key moments, topics covered, and any visible or spoken content
- Respond with ONLY the JSON object. No extra text."""

_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0  # seconds


class GeminiLLMProvider(LLMProvider):
    """LLM provider backed by Google Gemini via the google-genai SDK."""

    def __init__(self, api_key: str, config: LLMProviderConfig) -> None:
        """
        Args:
            api_key: Google API key.
            config: LLM provider config (model, temperature, max_output_tokens).
        """
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from e

        self._genai = genai
        self._types = genai_types
        self._client = genai.Client(api_key=api_key)
        self._config = config

    @property
    def provider_name(self) -> str:
        return "gemini"

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate text with optional system instruction and exponential backoff."""
        gen_config = self._types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            system_instruction=system_instruction,
        )

        contents = [prompt]
        return self._call_with_backoff(
            lambda: self._client.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=gen_config,
            ).text or ""
        )

    def extract_info(
        self,
        content: str,
        document_id: str,
        content_type: str = "text",
        extra_metadata: dict | None = None,
    ) -> ExtractedInfo:
        """Extract structured info (summary, key facts, topics) from text content."""
        prompt = (
            f"Content type: {content_type}\n\n"
            f"Content:\n{content}\n\n"
            "Extract the information as instructed."
        )

        raw = self.generate(prompt, system_instruction=_EXTRACTION_SYSTEM_PROMPT)
        parsed = self._parse_extraction(raw)

        return ExtractedInfo(
            document_id=document_id,
            summary=parsed.get("summary", ""),
            key_facts=parsed.get("key_facts", []),
            topics=parsed.get("topics", []),
            metadata=extra_metadata or {},
        )

    def extract_info_multimodal(
        self,
        content_parts: list[Any],
        document_id: str,
        content_type: str = "image",
        extra_metadata: dict | None = None,
    ) -> ExtractedInfo:
        """
        Extract structured info from multimodal content using Gemini's vision.

        Args:
            content_parts: List containing text strings and/or google.genai.types.Part
                          objects (image bytes) or uploaded file objects (video).
            document_id: Parent document ID.
            content_type: "image" or "video".
            extra_metadata: Metadata to carry forward.
        """
        # Prepend an extraction instruction as the first text part
        extraction_instruction = (
            f"Content type: {content_type}\n\n"
            "Analyze the provided media content and extract the information as instructed."
        )

        contents = [extraction_instruction] + list(content_parts)

        gen_config = self._types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            system_instruction=_EXTRACTION_SYSTEM_PROMPT,
        )

        logger.info(
            "Multimodal extraction: %s content, %d parts, model=%s",
            content_type, len(content_parts), self._config.model,
        )

        raw = self._call_with_backoff(
            lambda: self._client.models.generate_content(
                model=self._config.model,
                contents=contents,
                config=gen_config,
            ).text or ""
        )

        parsed = self._parse_extraction(raw)

        return ExtractedInfo(
            document_id=document_id,
            summary=parsed.get("summary", ""),
            key_facts=parsed.get("key_facts", []),
            topics=parsed.get("topics", []),
            metadata=extra_metadata or {},
        )

    def upload_video(self, video_path: str, timeout_seconds: int = 300) -> Any:
        """
        Upload a video to the Gemini Files API and wait for processing.

        Args:
            video_path: Path to the video file.
            timeout_seconds: Maximum seconds to wait for processing.

        Returns:
            The uploaded file object (can be passed directly to generate_content).

        Raises:
            RuntimeError: If processing fails or times out.
        """
        logger.info("Uploading video to Gemini Files API: %s", video_path)
        uploaded = self._client.files.upload(file=video_path)
        logger.info("Video uploaded: %s (state=%s)", uploaded.name, uploaded.state.name)

        elapsed = 0
        poll_interval = 5

        while uploaded.state.name == "PROCESSING" and elapsed < timeout_seconds:
            logger.debug("Video processing… (%.0fs elapsed)", elapsed)
            time.sleep(poll_interval)
            elapsed += poll_interval
            uploaded = self._client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            error = getattr(uploaded, "error", "unknown error")
            raise RuntimeError(
                f"Gemini video processing failed for '{video_path}': {error}"
            )

        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(
                f"Gemini video processing timed out after {timeout_seconds}s "
                f"(state: {uploaded.state.name})"
            )

        logger.info("Video ready: %s", uploaded.name)
        return uploaded

    def delete_uploaded_file(self, file_name: str) -> None:
        """
        Delete an uploaded file from the Gemini Files API.

        Args:
            file_name: The file name returned by upload_video() (e.g. "files/abc123").
        """
        try:
            self._client.files.delete(name=file_name)
            logger.debug("Deleted uploaded file: %s", file_name)
        except Exception as e:
            # Non-fatal — files auto-expire after 48h anyway
            logger.warning("Failed to delete uploaded file '%s': %s", file_name, e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_backoff(self, fn: Any) -> str:
        """Call fn(), retrying with exponential backoff on rate-limit errors."""
        backoff = _INITIAL_BACKOFF
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:
                exc_str = str(exc).lower()
                is_rate_limit = "429" in exc_str or "quota" in exc_str or "rate" in exc_str

                if not is_rate_limit:
                    raise

                last_exc = exc
                wait = backoff * (2 ** attempt)
                logger.warning(
                    "Gemini rate limit hit (attempt %d/%d). Retrying in %.1fs…",
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Gemini API failed after {_MAX_RETRIES} retries."
        ) from last_exc

    @staticmethod
    def _parse_extraction(raw: str) -> dict[str, Any]:
        """Parse JSON from the LLM extraction response, with fallback."""
        # Strip potential markdown fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction JSON. Raw response:\n%s", raw)
            # Return a best-effort fallback
            return {
                "summary": raw[:500] if raw else "Unable to extract summary.",
                "key_facts": [],
                "topics": [],
            }
