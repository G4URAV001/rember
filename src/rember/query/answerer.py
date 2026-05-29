"""
Answerer — generates a grounded answer from retrieved context using an LLM.

The LLM is instructed to:
  - Only use the provided context
  - Cite sources by number
  - Admit when context is insufficient

The LLM provider is resolved via the registry using the "query_answering" task.
"""

from __future__ import annotations

import logging

from rember.llm.registry import LLMRegistry
from rember.models import Answer, QueryResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a knowledgeable assistant with access to a personal knowledge base.

Answer the user's question using ONLY the context provided below.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- Cite sources by referencing them as [Source 1], [Source 2], etc.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and factual. Avoid padding or repetition.
- If multiple sources support the same point, cite them all."""


class Answerer:
    """Generates answers from retrieved context using an LLM."""

    def __init__(self, llm_registry: LLMRegistry) -> None:
        self._registry = llm_registry

    def answer(
        self,
        question: str,
        context: list[QueryResult],
    ) -> Answer:
        """
        Generate an answer for a question given retrieved context chunks.

        Args:
            question: The user's question.
            context: List of QueryResult objects from the Retriever.

        Returns:
            Answer with the generated text and source attribution.
        """
        if not context:
            logger.warning("Answerer received empty context for question: '%s'", question)
            return Answer(
                question=question,
                answer=(
                    "I don't have any relevant information stored to answer that question. "
                    "Try ingesting some content first with `rember ingest`."
                ),
                sources=[],
            )

        provider = self._registry.get_provider("query_answering")

        context_text = self._build_context(context)
        prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        logger.info(
            "Generating answer for '%s' using provider '%s' with %d sources",
            question[:60],
            provider.provider_name,
            len(context),
        )

        response_text = provider.generate(
            prompt=prompt,
            system_instruction=_SYSTEM_PROMPT,
        )

        return Answer(
            question=question,
            answer=response_text.strip(),
            sources=context,
            model_used=provider.provider_name,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(results: list[QueryResult]) -> str:
        """Format QueryResults into a numbered source context block."""
        parts = []
        for i, result in enumerate(results, start=1):
            score_pct = int(result.score * 100)
            source_hint = result.metadata.get("source_path") or result.metadata.get(
                "filename", f"Document {result.document_id[:8]}"
            )
            parts.append(
                f"[Source {i}] (relevance: {score_pct}%, from: {source_hint})\n"
                f"{result.content}"
            )
        return "\n\n---\n\n".join(parts)
