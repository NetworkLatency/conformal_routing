"""Helpers for loading experiment configs and resolving local paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


PATH_FIELDS = {
    "model_name_or_path",
    "tokenizer_name_or_path",
    "download_dir",
    "chat_template_path",
}

REMOTE_BACKENDS = {"openai", "openai_compatible", "remote_vllm"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if path.exists() or path.is_absolute():
        return path
    project_path = _project_root() / path
    return project_path if project_path.exists() else path


def _resolve_relative_path(value: str, config_path: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    project_root = config_path.parent.parent
    candidates = (Path.cwd() / path, config_path.parent / path, project_root / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(project_root / path)


def resolve_local_paths_in_config(cfg: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Resolve model and dataset paths relative to the project config.

    Configs live under ``configs/`` by default. A value such as
    ``data/aime24.jsonl`` should therefore point at ``<project>/data/aime24.jsonl``
    even when the script is launched from another working directory.
    """
    config_path = Path(config_path).resolve()

    dataset_paths = cfg.get("dataset_paths")
    if isinstance(dataset_paths, dict):
        cfg["dataset_paths"] = {
            key: _resolve_relative_path(value, config_path)
            for key, value in dataset_paths.items()
            if value is not None
        }

    output_dirs = cfg.get("output_dirs")
    if isinstance(output_dirs, dict):
        cfg["output_dirs"] = {
            key: _resolve_relative_path(value, config_path)
            for key, value in output_dirs.items()
            if value is not None
        }

    for model_key in ("small_model", "large_model"):
        model_cfg = cfg.get(model_key)
        if not isinstance(model_cfg, dict):
            continue
        backend = str(model_cfg.get("backend") or model_cfg.get("model_backend") or "").lower()
        is_remote = bool(model_cfg.get("api_base_url")) or backend in REMOTE_BACKENDS
        for field in PATH_FIELDS:
            value = model_cfg.get(field)
            if isinstance(value, str):
                if is_remote and field == "model_name_or_path":
                    continue
                model_cfg[field] = _resolve_relative_path(value, config_path)

    return cfg


def apply_model_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge global vLLM defaults into small/large model configs."""
    defaults = cfg.get("vllm_defaults") or cfg.get("model_defaults") or {}
    if not isinstance(defaults, dict):
        raise TypeError("vllm_defaults/model_defaults must be a mapping if provided.")

    for model_key in ("small_model", "large_model"):
        model_cfg = cfg.get(model_key)
        if not isinstance(model_cfg, dict):
            continue
        merged = dict(defaults)
        merged.update(model_cfg)
        cfg[model_key] = merged
    return cfg


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML/JSON config, apply vLLM defaults, and resolve local paths."""
    path = _resolve_config_path(config_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    cfg = apply_model_defaults(cfg)
    cfg = resolve_local_paths_in_config(cfg, path)
    return cfg


def configured_output_dir(
    cfg: dict[str, Any],
    key: str,
    fallback: str,
    override: str | Path | None = None,
) -> Path:
    """Resolve an output directory from CLI override, config, or fallback."""
    value = override
    if value is None:
        output_dirs = cfg.get("output_dirs") or {}
        value = output_dirs.get(key) if isinstance(output_dirs, dict) else None
    if value is None:
        value = fallback
    return Path(value)
