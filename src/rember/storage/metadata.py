"""
SQLite metadata store.

Stores all non-vector data: documents, chunks, and a FAISS ID counter.

Schema:
  documents  — one row per ingested document
  chunks     — one row per stored chunk (many-to-one with documents)
  faiss_id_counter — single-row table to allocate sequential FAISS int64 IDs

The faiss_id_counter is the source of truth for FAISS IDs. It's incremented
atomically inside a transaction so concurrent ingestion won't produce duplicate IDs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from rember.models import Document, StoredChunk

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_path TEXT,
    raw_content TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    faiss_id    INTEGER UNIQUE NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    token_count INTEGER NOT NULL DEFAULT 0,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faiss_id_counter (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    next_id INTEGER NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO faiss_id_counter (id, next_id) VALUES (1, 0);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id    ON chunks(faiss_id);
"""


class MetadataStore:
    """SQLite-backed store for document and chunk metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
        logger.debug("Metadata DB initialised at %s", self._db_path)

    # ------------------------------------------------------------------
    # FAISS ID allocation
    # ------------------------------------------------------------------

    def get_next_faiss_id(self, count: int = 1) -> int:
        """
        Atomically allocate `count` sequential FAISS IDs.

        Returns the first ID in the allocated range.
        e.g. get_next_faiss_id(3) → 5 means IDs 5, 6, 7 are reserved.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT next_id FROM faiss_id_counter WHERE id = 1").fetchone()
            start_id = row["next_id"]
            conn.execute(
                "UPDATE faiss_id_counter SET next_id = ? WHERE id = 1",
                (start_id + count,),
            )
        return start_id

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def save_document(self, doc: Document) -> None:
        """Insert or replace a document row."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (id, source_type, source_path, raw_content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.source_type if isinstance(doc.source_type, str) else doc.source_type.value,
                    doc.source_path,
                    doc.raw_content,
                    json.dumps(doc.metadata),
                    doc.created_at.isoformat(),
                ),
            )

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a document by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        """Return all documents, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks (CASCADE).
        Returns True if a document was deleted, False if not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE id = ?", (doc_id,)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: list[StoredChunk]) -> None:
        """Bulk-insert stored chunks."""
        if not chunks:
            return

        rows = [
            (
                c.chunk_id,
                c.document_id,
                c.faiss_id,
                c.content,
                json.dumps(c.metadata),
                c.token_count,
                c.chunk_index,
                c.created_at.isoformat(),
            )
            for c in chunks
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks
                    (id, document_id, faiss_id, content, metadata, token_count, chunk_index, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_chunk_by_faiss_id(self, faiss_id: int) -> StoredChunk | None:
        """Look up a single chunk by its FAISS integer ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE faiss_id = ?", (faiss_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks_by_faiss_ids(self, faiss_ids: list[int]) -> list[StoredChunk]:
        """
        Bulk-retrieve chunks by a list of FAISS IDs.
        Preserves the order of faiss_ids.
        """
        if not faiss_ids:
            return []

        placeholders = ",".join("?" * len(faiss_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE faiss_id IN ({placeholders})",
                faiss_ids,
            ).fetchall()

        # Build id→chunk map for order preservation
        chunk_map: dict[int, StoredChunk] = {}
        for row in rows:
            chunk = self._row_to_chunk(row)
            chunk_map[chunk.faiss_id] = chunk

        return [chunk_map[fid] for fid in faiss_ids if fid in chunk_map]

    def list_chunks_for_document(self, doc_id: str) -> list[StoredChunk]:
        """Return all chunks for a given document, in chunk_index order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return basic storage statistics."""
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            counter = conn.execute(
                "SELECT next_id FROM faiss_id_counter WHERE id = 1"
            ).fetchone()
            next_faiss_id = counter["next_id"] if counter else 0

        db_size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0

        return {
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "next_faiss_id": next_faiss_id,
            "db_size_bytes": db_size_bytes,
            "db_path": str(self._db_path),
        }

    # ------------------------------------------------------------------
    # Row → model converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            source_type=row["source_type"],
            source_path=row["source_path"],
            raw_content=row["raw_content"] or "",
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> StoredChunk:
        return StoredChunk(
            chunk_id=row["id"],
            document_id=row["document_id"],
            faiss_id=row["faiss_id"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            token_count=row["token_count"],
            chunk_index=row["chunk_index"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
