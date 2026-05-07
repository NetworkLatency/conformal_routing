"""Model factory helpers."""

from __future__ import annotations

from typing import Any

from conformal_routing.models.base import ModelWrapper
from conformal_routing.models.openai_compatible import OpenAICompatibleWrapper
from conformal_routing.models.vllm_wrapper import VLLMWrapper


def build_model(cfg: dict[str, Any]) -> ModelWrapper:
    wrapper_cfg = dict(cfg)
    backend = str(wrapper_cfg.pop("backend", None) or wrapper_cfg.pop("model_backend", None) or "vllm").lower()
    if wrapper_cfg.get("api_base_url"):
        backend = "openai"

    if backend in {"openai", "openai_compatible", "remote_vllm"}:
        return OpenAICompatibleWrapper(**wrapper_cfg)
    if backend == "vllm":
        return VLLMWrapper(**wrapper_cfg)
    raise ValueError(f"Unknown model backend {backend!r}")

