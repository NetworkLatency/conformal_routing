"""Inspect tokenizer chat-template rendering without starting vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.conformal_routing.config_paths import configured_output_dir, load_experiment_config
from src.conformal_routing.models.render import (
    apply_chat_template_override,
    chat_template_hash,
    render_for_continuation,
    rendered_initial_assistant_marker,
)


def parse_args():
    p = argparse.ArgumentParser(description="Check chat-template continuation rendering.")
    p.add_argument("--config", default="configs/default.yaml", type=Path)
    p.add_argument("--out", default=None, type=Path)
    p.add_argument(
        "--problem",
        default="What is 1+1? Put the final answer in \\boxed{}.",
    )
    return p.parse_args()


def inspect_model(name: str, cfg: dict[str, Any], problem: str) -> dict[str, Any]:
    tokenizer_path = cfg.get("tokenizer_name_or_path") or cfg.get("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=cfg.get("trust_remote_code", True),
        use_fast=True,
    )
    apply_chat_template_override(
        tokenizer,
        cfg.get("chat_template"),
        cfg.get("chat_template_path"),
    )
    rendered_empty = render_for_continuation(problem, "", tokenizer)
    rendered_close = render_for_continuation(problem, "</think>\n\n", tokenizer)
    return {
        "name": name,
        "model_name_or_path": cfg.get("model_name_or_path"),
        "tokenizer_name_or_path": tokenizer_path,
        "chat_template_path": cfg.get("chat_template_path"),
        "chat_template_hash": chat_template_hash(tokenizer),
        "rendered_initial_assistant_marker": rendered_initial_assistant_marker(problem, tokenizer),
        "contains_think_marker_with_empty_prefix": "<think>" in rendered_empty,
        "contains_double_think_with_empty_prefix": rendered_empty.count("<think>") > 1,
        "empty_prefix_preview": rendered_empty[-500:],
        "close_think_prefix_preview": rendered_close[-500:],
    }


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = load_experiment_config(args.config)
    out_dir = configured_output_dir(cfg, "diagnostics", "outputs/diagnostics", args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        inspect_model("small", cfg["small_model"], args.problem),
        inspect_model("large", cfg["large_model"], args.problem),
    ]
    out_path = out_dir / "render_sanity.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
