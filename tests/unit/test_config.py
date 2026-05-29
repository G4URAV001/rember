"""Unit tests for config loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from rember.config import (
    ChunkingConfig,
    Settings,
    StorageConfig,
    _deep_merge,
    _load_yaml,
    invalidate_settings,
    load_settings,
)


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result["a"] == {"x": 1, "y": 99, "z": 100}
        assert result["b"] == 3

    def test_override_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 2}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1  # unchanged


class TestLoadYaml:
    def test_returns_empty_for_nonexistent_file(self, tmp_path):
        result = _load_yaml(tmp_path / "missing.yaml")
        assert result == {}

    def test_loads_valid_yaml(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        result = _load_yaml(f)
        assert result["key"] == "value"
        assert result["nested"]["a"] == 1


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.pipeline.default_llm == "gemini"
        assert s.chunking.adaptive_threshold == 500
        assert s.query.top_k == 10

    def test_resolved_data_dir(self, tmp_path):
        s = Settings(storage=StorageConfig(data_dir=str(tmp_path / "data")))
        assert s.storage.resolved_data_dir == tmp_path / "data"

    def test_home_expansion(self):
        s = Settings(storage=StorageConfig(data_dir="~/.rember"))
        resolved = s.storage.resolved_data_dir
        assert not str(resolved).startswith("~")

    def test_get_llm_config_default(self):
        s = Settings()
        cfg = s.get_llm_config()
        assert cfg.model == "gemini-2.0-flash"

    def test_get_embedding_config_default(self):
        s = Settings()
        cfg = s.get_embedding_config()
        assert cfg.model == "gemini-embedding-001"

    def test_get_task_provider_routing(self):
        s = Settings(task_routing={"extraction": "openai"})
        assert s.get_task_provider("extraction") == "openai"
        assert s.get_task_provider("unknown") == "gemini"  # falls back to default


class TestLoadSettings:
    def test_load_from_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {
            "pipeline": {"default_llm": "gemini"},
            "chunking": {"adaptive_threshold": 999},
        }
        (tmp_path / "config.yaml").write_text(yaml.dump(config))

        invalidate_settings()
        s = load_settings(env_file=None)
        assert s.chunking.adaptive_threshold == 999
        invalidate_settings()

    def test_env_var_injects_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-xyz")
        invalidate_settings()
        s = load_settings(env_file=None)
        assert s.google_api_key.get_secret_value() == "test-key-xyz"
        invalidate_settings()

    def test_rember_data_dir_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REMBER_DATA_DIR", str(tmp_path / "custom"))
        invalidate_settings()
        s = load_settings(env_file=None)
        assert s.storage.data_dir == str(tmp_path / "custom")
        invalidate_settings()
