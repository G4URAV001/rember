"""
ChunkStage — splits ExtractedInfo into embeddable Chunks.

Strategy (adaptive):
  1. Build "embeddable content" from key_facts + summary.
  2. If total token count < adaptive_threshold → store as ONE chunk.
  3. Otherwise → split into overlapping fixed-size chunks.

"Token count" here uses a simple whitespace-based word count
(1 word ≈ 1 token). This is good enough for chunking decisions;
you can upgrade to tiktoken in Phase 3 for precision.

Each chunk inherits metadata from ExtractedInfo and gets:
  - Its own UUID
  - chunk_index (position within the document)
  - token_count
"""

from __future__ import annotations

import logging

from rember.config import ChunkingConfig
from rember.models import Chunk, ExtractedInfo
from rember.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    """Rough token estimate: whitespace-split word count."""
    return len(text.split())


def _split_into_chunks(
    text: str,
    max_chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into overlapping fixed-size word-count chunks.

    Args:
        text: Input text.
        max_chunk_size: Maximum words per chunk.
        chunk_overlap: Number of words to overlap between adjacent chunks.

    Returns:
        List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, max_chunk_size - chunk_overlap)
    start = 0

    while start < len(words):
        end = min(start + max_chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks


class ChunkStage(PipelineStage[ExtractedInfo, list[Chunk]]):
    """
    Splits ExtractedInfo into embeddable Chunk objects.

    Input:  ExtractedInfo
    Output: list[Chunk]
    """

    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "chunk"

    def process(self, input_data: ExtractedInfo) -> list[Chunk]:
        return self.chunk(input_data)

    def chunk(self, extracted: ExtractedInfo) -> list[Chunk]:
        """
        Convert ExtractedInfo into a list of Chunks using the adaptive strategy.

        The embeddable content is built as:
          "{summary}\n\n{fact1}\n{fact2}\n..."
        This ensures each chunk contains rich, self-contained context.
        """
        embeddable_content = self._build_embeddable_content(extracted)
        if not embeddable_content.strip():
            return []

        total_tokens = _word_count(embeddable_content)

        base_metadata = {
            **extracted.metadata,
            "topics": extracted.topics,
            "document_id": extracted.document_id,
        }

        if total_tokens <= self._config.adaptive_threshold:
            # Short content: store whole
            logger.debug(
                "Document %s: %d tokens ≤ threshold %d → single chunk",
                extracted.document_id,
                total_tokens,
                self._config.adaptive_threshold,
            )
            return [
                Chunk(
                    document_id=extracted.document_id,
                    content=embeddable_content,
                    metadata={**base_metadata, "chunk_strategy": "whole"},
                    token_count=total_tokens,
                    chunk_index=0,
                )
            ]
        else:
            # Long content: split into overlapping chunks
            logger.debug(
                "Document %s: %d tokens > threshold %d → splitting into chunks",
                extracted.document_id,
                total_tokens,
                self._config.adaptive_threshold,
            )
            raw_chunks = _split_into_chunks(
                text=embeddable_content,
                max_chunk_size=self._config.max_chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

            chunks = []
            for i, text in enumerate(raw_chunks):
                chunks.append(
                    Chunk(
                        document_id=extracted.document_id,
                        content=text,
                        metadata={
                            **base_metadata,
                            "chunk_strategy": "split",
                            "total_chunks": len(raw_chunks),
                        },
                        token_count=_word_count(text),
                        chunk_index=i,
                    )
                )

            logger.debug(
                "Document %s → %d chunks", extracted.document_id, len(chunks)
            )
            return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_embeddable_content(extracted: ExtractedInfo) -> str:
        """
        Combine summary and key facts into a single embeddable text block.

        Format:
          Summary: {summary}

          Key Facts:
          - {fact1}
          - {fact2}
          ...
        """
        parts = []

        if extracted.summary:
            parts.append(f"Summary: {extracted.summary}")

        if extracted.key_facts:
            facts_text = "\n".join(f"- {fact}" for fact in extracted.key_facts)
            parts.append(f"Key Facts:\n{facts_text}")

        if not parts:
            return ""

        return "\n\n".join(parts)
