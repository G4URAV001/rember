"""
Configuration system for Rember.

Priority (highest → lowest):
  1. Environment variables (GOOGLE_API_KEY, REMBER_DATA_DIR, etc.)
  2. .env file
  3. config.yaml (user customised)
  4. config.default.yaml (bundled defaults)
  5. Hard-coded defaults in this module

Usage:
    from rember.config import get_settings
    settings = get_settings()
    print(settings.google_api_key.get_secret_value())
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr


# ---------------------------------------------------------------------------
# Sub-config models (plain Pydantic, not BaseSettings)
# ---------------------------------------------------------------------------


class LLMProviderConfig(BaseModel):
    model: str = "gemini-2.0-flash"
    temperature: float = 0.3
    max_output_tokens: int = 2048


class EmbeddingProviderConfig(BaseModel):
    model: str = "gemini-embedding-001"
    task_type: str = "RETRIEVAL_DOCUMENT"
    dimension: int | None = None  # None → use model default (3072 for Gemini)


class VectorStorageConfig(BaseModel):
    index_type: str = "flat_ip"  # flat_ip | hnsw
    normalize: bool = True


class MetadataStorageConfig(BaseModel):
    db_name: str = "rember.db"


class StorageConfig(BaseModel):
    data_dir: str = "~/.rember"
    vector: VectorStorageConfig = Field(default_factory=VectorStorageConfig)
    metadata: MetadataStorageConfig = Field(default_factory=MetadataStorageConfig)

    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def vector_index_path(self) -> Path:
        return self.resolved_data_dir / "index.faiss"

    @property
    def db_path(self) -> Path:
        return self.resolved_data_dir / self.metadata.db_name


class PipelineConfig(BaseModel):
    default_llm: str = "gemini"
    default_embedding: str = "gemini"


class ChunkingConfig(BaseModel):
    adaptive_threshold: int = 500  # token count below which content is stored whole
    max_chunk_size: int = 1000
    chunk_overlap: int = 100


class MediaConfig(BaseModel):
    """Configuration for image and video processing (Phase 2)."""
    # Image processing
    max_image_size_mb: float = 20.0         # files larger than this → resize before API
    image_max_dimension: int = 2048         # max width/height after resize
    image_quality: int = 85                 # JPEG compression quality (1-95)

    # Video processing
    max_video_size_mb: float = 100.0        # max video file size accepted
    max_video_duration_seconds: int = 600   # 10 minutes max
    num_frames_to_extract: int = 10         # frames extracted in fallback mode
    video_upload_timeout_seconds: int = 300 # Gemini Files API polling timeout

    # Audio transcription (fallback, requires openai-whisper)
    whisper_model: str = "base"             # tiny | base | small | medium | large
    enable_transcription: bool = True       # whether to transcribe audio in fallback

    # Strategy
    prefer_native_video: bool = True        # try Gemini native video first


class QueryConfig(BaseModel):
    top_k: int = 10
    min_score: float = 0.3


# ---------------------------------------------------------------------------
# Top-level Settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """
    Top-level application settings.
    Built manually from layered YAML + env vars for maximum flexibility.
    """

    # Secrets (read from env / .env)
    google_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    # Config path overrides
    config_path: str = ""  # if set, load from this path instead of defaults

    # Sub-configs
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm: dict[str, LLMProviderConfig] = Field(
        default_factory=lambda: {"gemini": LLMProviderConfig()}
    )
    task_routing: dict[str, str] = Field(default_factory=dict)
    embeddings: dict[str, EmbeddingProviderConfig] = Field(
        default_factory=lambda: {"gemini": EmbeddingProviderConfig()}
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    media: MediaConfig = Field(default_factory=MediaConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)

    def get_llm_config(self, provider: str | None = None) -> LLMProviderConfig:
        name = provider or self.pipeline.default_llm
        return self.llm.get(name, LLMProviderConfig())

    def get_embedding_config(self, provider: str | None = None) -> EmbeddingProviderConfig:
        name = provider or self.pipeline.default_embedding
        return self.embeddings.get(name, EmbeddingProviderConfig())

    def get_task_provider(self, task: str) -> str:
        """Return the provider name for a given task, falling back to default."""
        return self.task_routing.get(task, self.pipeline.default_llm)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning {} if the file doesn't exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. override wins on conflicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(
    config_path: str | Path | None = None,
    env_file: str | Path | None = ".env",
) -> Settings:
    """
    Build Settings by merging:
      1. config.default.yaml  (bundled defaults)
      2. config.yaml          (user's custom config, if present)
      3. custom config_path   (optional explicit override)
      4. Environment variables / .env file
    """
    # Load .env file if it exists
    _dotenv_path = Path(env_file) if env_file else None
    if _dotenv_path and _dotenv_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(_dotenv_path, override=False)
        except ImportError:
            pass

    # Locate bundled default config
    _here = Path(__file__).parent.parent.parent  # repo root
    default_yaml_path = _here / "config.default.yaml"

    # Load YAML layers
    data: dict[str, Any] = {}
    data = _deep_merge(data, _load_yaml(default_yaml_path))
    data = _deep_merge(data, _load_yaml(Path("config.yaml")))
    if config_path:
        data = _deep_merge(data, _load_yaml(Path(config_path)))

    # Inject secrets from environment
    google_key = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    # Override data_dir from env if set
    if data_dir_env := os.environ.get("REMBER_DATA_DIR"):
        if "storage" not in data:
            data["storage"] = {}
        data["storage"]["data_dir"] = data_dir_env

    return Settings(
        google_api_key=SecretStr(google_key),
        openai_api_key=SecretStr(openai_key),
        **data,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call invalidate_settings() to reset."""
    return load_settings()


def invalidate_settings() -> None:
    """Clear the cached settings (useful in tests)."""
    get_settings.cache_clear()
