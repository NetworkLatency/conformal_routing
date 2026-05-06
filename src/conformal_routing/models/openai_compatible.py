"""OpenAI-compatible completion backend for remote vLLM servers."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
from transformers import AutoTokenizer

from conformal_routing.models.base import FirstTokenProbe, ModelWrapper, StepOutput


FINAL_ANSWER_MARKERS = ("\\boxed{", "</think>")
LOG = logging.getLogger(__name__)


class OpenAICompatibleWrapper(ModelWrapper):
    """ModelWrapper for a remote OpenAI-compatible vLLM `/v1/completions` server."""

    def __init__(
        self,
        model_name_or_path: str,
        n_params_billion: float,
        api_base_url: str,
        api_key: str = "EMPTY",
        api_model: str | None = None,
        tokenizer_name_or_path: str | None = None,
        trust_remote_code: bool = True,
        probe_logprobs: int = 20,
        use_chat_template: bool = False,
        chat_template: str | None = None,
        chat_template_path: str | None = None,
        assistant_prefix_start: str = "",
        continue_final_message: bool = True,
        add_generation_prompt: bool = True,
        timeout_s: float = 600.0,
        **_: Any,
    ):
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        self._set_chat_template(chat_template, chat_template_path)
        self._model_name = api_model or model_name_or_path
        self._api_base_url = api_base_url.rstrip("/")
        self._api_key = api_key
        self._n_params_billion = n_params_billion
        self._probe_logprobs = probe_logprobs
        self._use_chat_template = use_chat_template
        self._assistant_prefix_start = assistant_prefix_start
        self._continue_final_message = continue_final_message
        self._add_generation_prompt = add_generation_prompt
        self._timeout_s = timeout_s

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vocab_size(self) -> int:
        try:
            return int(len(self.tokenizer))
        except TypeError:
            pass
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
        return len(self.tokenizer.get_vocab())

    def render_prompt(self, question: str, history: str = "") -> str:
        if not self._use_chat_template:
            return question + history
        return self._render_chat_continuation(question, history)

    def probe_first_token(self, context: str) -> FirstTokenProbe:
        data = self._completion(
            context,
            max_tokens=1,
            temperature=0.0,
            logprobs=self._probe_logprobs,
        )
        choice = data["choices"][0]
        top = self._top_logprobs_at(choice, 0)
        logits = self._logprobs_to_vector(top)
        top_k_ids, top_k_probs = self._top_k(logits, k=20)
        return FirstTokenProbe(
            logits=logits,
            top_k_ids=top_k_ids,
            top_k_probs=top_k_probs,
            extra={"backend": "openai_compatible"},
        )

    def generate_step(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        step_delimiters: tuple[str, ...] = ("\n\n",),
        prefix_token_ids: Optional[list[int]] = None,
    ) -> StepOutput:
        forced_prefix = list(prefix_token_ids or [])
        if forced_prefix:
            context = context + self.decode(forced_prefix)
        data = self._completion(
            context,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=list(step_delimiters),
            logprobs=1,
        )
        return self._choice_to_step(data["choices"][0], forced_prefix, step_delimiters)

    def generate_full(
        self,
        context: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> StepOutput:
        data = self._completion(
            context,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=1,
        )
        return self._choice_to_step(data["choices"][0], [], ())

    def sample_prefixes(
        self,
        prompt: str,
        n: int,
        k: int,
        temperature: float,
    ) -> list[list[int]]:
        if n <= 0:
            return []
        data = self._completion(prompt, max_tokens=k, temperature=temperature, n=n)
        return [self.encode(choice.get("text", ""))[:k] for choice in data["choices"]]

    def estimate_flops(self, n_input_tokens: int, n_output_tokens: int) -> float:
        n_params = self._n_params_billion * 1e9
        return 6.0 * n_params * (n_input_tokens + n_output_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _completion(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        body = {"model": self._model_name, "prompt": prompt}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        req = urllib.request.Request(
            f"{self._api_base_url}/v1/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote vLLM completion failed: {exc.code} {detail}") from exc

    def _choice_to_step(
        self,
        choice: dict[str, Any],
        forced_prefix: list[int],
        step_delimiters: tuple[str, ...],
    ) -> StepOutput:
        generated_text = str(choice.get("text") or "")
        generated_ids = self.encode(generated_text)
        text = (self.decode(forced_prefix) if forced_prefix else "") + generated_text
        token_ids = forced_prefix + generated_ids
        logprobs = [float("nan")] * len(forced_prefix) + self._chosen_logprobs(choice)
        first_logits = self._logprobs_to_vector(self._top_logprobs_at(choice, 0))
        if forced_prefix:
            first_logits = np.full(self.vocab_size, -float("inf"), dtype=np.float32)
            first_logits[forced_prefix[0]] = 0.0
        finished = self._is_finished(choice, text, step_delimiters)
        return StepOutput(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            first_token_logits=first_logits,
            n_tokens=len(token_ids),
            finished=finished,
            extra={"finish_reason": choice.get("finish_reason"), "backend": "openai_compatible"},
        )

    @staticmethod
    def _top_logprobs_at(choice: dict[str, Any], idx: int) -> dict[str, float]:
        logprobs = choice.get("logprobs") or {}
        top_logprobs = logprobs.get("top_logprobs") or []
        if idx >= len(top_logprobs) or top_logprobs[idx] is None:
            return {}
        return dict(top_logprobs[idx])

    def _chosen_logprobs(self, choice: dict[str, Any]) -> list[float]:
        logprobs = choice.get("logprobs") or {}
        return [
            float(value) if value is not None else float("nan")
            for value in (logprobs.get("token_logprobs") or [])
        ]

    def _logprobs_to_vector(self, token_logprobs: dict[str, float]) -> np.ndarray:
        logits = np.full(self.vocab_size, -float("inf"), dtype=np.float32)
        for token, logprob in token_logprobs.items():
            ids = self.encode(str(token))
            if not ids:
                continue
            idx = int(ids[0])
            if 0 <= idx < self.vocab_size:
                logits[idx] = max(float(logprob), float(logits[idx]))
        return logits

    @staticmethod
    def _top_k(logits: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        finite = np.flatnonzero(np.isfinite(logits))
        if len(finite) == 0:
            return [], []
        k = min(k, len(finite))
        local = np.argpartition(-logits[finite], k - 1)[:k]
        ids = finite[local]
        ids = ids[np.argsort(-logits[ids])]
        return ids.astype(int).tolist(), np.exp(logits[ids]).astype(float).tolist()

    @staticmethod
    def _is_finished(
        choice: dict[str, Any],
        text: str,
        step_delimiters: tuple[str, ...],
    ) -> bool:
        if any(marker in text for marker in FINAL_ANSWER_MARKERS):
            return True
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            return False
        if step_delimiters and finish_reason == "stop":
            return False
        return finish_reason in {"stop", "eos"}

    def _set_chat_template(
        self,
        chat_template: str | None,
        chat_template_path: str | None,
    ) -> None:
        if chat_template_path:
            chat_template = Path(chat_template_path).read_text(encoding="utf-8")
        if chat_template:
            self.tokenizer.chat_template = chat_template

    def _assistant_prefix(self, history: str) -> str:
        if not self._assistant_prefix_start:
            return history
        if history.startswith(self._assistant_prefix_start):
            return history
        return self._assistant_prefix_start + history

    def _render_chat_continuation(self, question: str, history: str) -> str:
        assistant_prefix = self._assistant_prefix(history)
        if assistant_prefix:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_prefix},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    continue_final_message=self._continue_final_message,
                    add_generation_prompt=False,
                )
            except ValueError as exc:
                if "continue_final_message" not in str(exc):
                    raise
                LOG.debug(
                    "continue_final_message unsupported; falling back to concat: %s",
                    exc,
                )
                return self._render_generation_prompt(question) + assistant_prefix
        return self._render_generation_prompt(question)

    def _render_generation_prompt(self, question: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            continue_final_message=False,
            add_generation_prompt=self._add_generation_prompt,
        )
