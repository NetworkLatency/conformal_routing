"""Chat-template rendering helpers.

This mirrors the continuation strategy used by the related GlimpRouter BPA
code: render a fresh user turn with ``add_generation_prompt=True``; once some
assistant text exists, render it as the assistant message and ask the tokenizer
to continue that final message.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any


LOG = logging.getLogger(__name__)


def apply_chat_template_override(
    tokenizer: Any,
    chat_template: str | None = None,
    chat_template_path: str | None = None,
) -> None:
    if chat_template_path:
        chat_template = Path(chat_template_path).read_text(encoding="utf-8")
    if chat_template:
        tokenizer.chat_template = chat_template


def render_for_continuation(
    question: str,
    assistant_prefix_text: str,
    tokenizer: Any,
    *,
    add_generation_prompt: bool = True,
    continue_final_message: bool = True,
) -> str:
    generation_prompt = render_generation_prompt(
        question,
        tokenizer,
        add_generation_prompt=add_generation_prompt,
    )
    if not assistant_prefix_text:
        return generation_prompt

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_prefix_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=continue_final_message,
            add_generation_prompt=False,
        )
    except ValueError as exc:
        if "continue_final_message" not in str(exc):
            raise
        LOG.debug(
            "continue_final_message unsupported; falling back to concat: %s",
            exc,
        )
        return generation_prompt + assistant_prefix_text


def render_generation_prompt(
    question: str,
    tokenizer: Any,
    *,
    add_generation_prompt: bool = True,
) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        continue_final_message=False,
        add_generation_prompt=add_generation_prompt,
    )


def chat_template_hash(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", "") or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


def rendered_initial_assistant_marker(question: str, tokenizer: Any, width: int = 200) -> str:
    rendered = render_for_continuation(question, "", tokenizer)
    user_tail = question[-80:]
    idx = rendered.rfind(user_tail)
    if idx >= 0:
        return rendered[idx + len(user_tail) : idx + len(user_tail) + width]
    return rendered[-width:]
