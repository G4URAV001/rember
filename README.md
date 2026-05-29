# Rember

> A modular, provider-agnostic RAG pipeline that remembers everything you feed it.

Feed Rember text, files, images, or videos — ask it anything about what you've shared.

## Features

- 📥 **Multi-format ingestion**: Text, files (`.txt`, `.md`, `.json`, `.csv`), images & videos (Phase 2)
- 🧠 **LLM-powered extraction**: Extracts structured facts and summaries from raw content
- 🔍 **Semantic search**: FAISS-powered vector similarity search with re-ranking
- 🤖 **Provider-agnostic**: Swap LLM and embedding providers via config (Gemini default)
- 🗄️ **Persistent storage**: FAISS index + SQLite metadata, all local in `~/.rember/`
- 🔧 **Per-task routing**: Different LLMs for extraction, summarization, and query answering

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourname/rember
cd rember
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
# Set your API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# (Optional) Customize pipeline config
cp config.default.yaml config.yaml
```

### 3. Initialize

```bash
rember init
```

### 4. Use

```bash
# Ingest raw text
rember ingest --text "Python was created by Guido van Rossum in 1991."

# Ingest a file
rember ingest ./notes.txt --tag project=python --tag source=notes

# Ask a question
rember query "Who created Python?"

# List what's been ingested
rember list

# Show storage stats
rember stats
```

## Architecture

```
Input (text/file)
    ↓
IngestStage       → reads content, creates Document
    ↓
ExtractStage      → LLM extracts key facts + summary
    ↓
ChunkStage        → adaptive chunking (short → whole, long → split)
    ↓
EmbeddingProvider → batch embed with gemini-embedding-001
    ↓
FAISSVectorStore  → save vectors (~/.rember/index.faiss)
SQLiteMetadata    → save metadata (~/.rember/rember.db)

Query:
Question → embed → FAISS search → hydrate metadata → LLM answer
```

## Configuration

Pipeline settings live in `config.yaml` (non-secret), secrets in `.env`.

| Setting | Default | Description |
|---|---|---|
| `pipeline.default_llm` | `gemini` | Default LLM provider |
| `llm.gemini.model` | `gemini-2.0-flash` | Gemini model for generation |
| `embeddings.gemini.model` | `gemini-embedding-001` | Embedding model (3072 dims) |
| `storage.data_dir` | `~/.rember` | Where to store index + DB |
| `chunking.adaptive_threshold` | `500` | Token threshold for adaptive chunking |
| `query.top_k` | `10` | Number of results to retrieve |

## Development

```bash
# Run unit tests (no API keys needed)
pytest tests/unit/ -v

# Run integration tests (requires GOOGLE_API_KEY)
pytest tests/integration/ -m integration -v

# Run with coverage
pytest --cov=rember --cov-report=term-missing

# Lint
ruff check src/ tests/
```

## Roadmap

- **Phase 1** (current): Text ingestion, Gemini LLM/embeddings, FAISS+SQLite, CLI
- **Phase 2**: Image and video ingestion (Gemini vision + frame extraction)
- **Phase 3**: Re-ranking, query expansion, multi-provider routing UI
