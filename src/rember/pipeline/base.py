"""
Abstract base class for pipeline stages.

Each stage transforms one type into another:
  IngestStage:  (str | Path, metadata)  → Document
  ExtractStage: Document                → ExtractedInfo
  ChunkStage:   ExtractedInfo           → list[Chunk]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class PipelineStage(ABC, Generic[TInput, TOutput]):
    """Abstract pipeline stage."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name (used in logging)."""

    @abstractmethod
    def process(self, input_data: TInput) -> TOutput:
        """Transform input_data into the output type."""
