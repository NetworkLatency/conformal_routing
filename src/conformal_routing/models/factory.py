"""Model factory helpers."""

from __future__ import annotations

from typing import Any

from conformal_routing.models.base import ModelWrapper
from conformal_routing.models.openai_compatible import OpenAICompatibleWrapper
from conformal_routing.models.vllm_wrapper import VLLMWrapper


def build_model(cfg: dict[str, Any]) -> ModelWrapper:
    backend = str(cfg.get("backend") or cfg.get("model_backend") or "vllm").lower()
    if cfg.get("api_base_url"):
        backend = "openai"

    if backend in {"openai", "openai_compatible", "remote_vllm"}:
        return OpenAICompatibleWrapper(**cfg)
    if backend == "vllm":
        return VLLMWrapper(**cfg)
    raise ValueError(f"Unknown model backend {backend!r}")
