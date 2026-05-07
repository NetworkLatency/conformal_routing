"""vLLM-based concrete implementation of ModelWrapper.

Key requirements:
  - Enable prefix caching (`enable_prefix_caching=True`) so repeated probing /
    generation on contexts with shared prefixes is fast.
  - Probe first token via `SamplingParams(max_tokens=1, logprobs=probe_logprobs)`.
  - For step generation, use `stop=step_delimiters` to halt at delimiter.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from conformal_routing.models.render import (
    apply_chat_template_override,
    render_for_continuation,
)
from conformal_routing.models.base import (
    FirstTokenProbe,
    ModelWrapper,
    StepOutput,
)


FINAL_ANSWER_MARKERS = ("\\boxed{",)
LOG = logging.getLogger(__name__)


class VLLMWrapper(ModelWrapper):
    """vLLM-backed model wrapper. Both small and large models use this class."""

    def __init__(
        self,
        model_name_or_path: str,
        n_params_billion: float,
        gpu_memory_utilization: float = 0.45,
        dtype: str = "bfloat16",
        max_model_len: int = 32768,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        seed: int = 42,
        probe_logprobs: int = 200,
        trust_remote_code: bool = True,
        tokenizer_name_or_path: str | None = None,
        download_dir: str | None = None,
        offline: bool = True,
        require_local_model: bool = True,
        use_chat_template: bool = False,
        chat_template: str | None = None,
        chat_template_path: str | None = None,
        assistant_prefix_start: str = "",
        continue_final_message: bool = True,
        add_generation_prompt: bool = True,
    ):
        if offline:
            self._enable_hf_offline_mode()

        model_path = self._validate_local_path(
            model_name_or_path,
            field_name="model_name_or_path",
            required=require_local_model,
        )
        tokenizer_path = self._validate_local_path(
            tokenizer_name_or_path or model_name_or_path,
            field_name="tokenizer_name_or_path",
            required=require_local_model,
        )

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - exercised only on GPU setups
            raise ImportError(
                "VLLMWrapper requires vllm. Install project dependencies with "
                '`pip install -e ".[dev]"` before constructing it.'
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": enable_prefix_caching,
            "seed": seed,
            "trust_remote_code": trust_remote_code,
            "max_logprobs": max(1, int(probe_logprobs)),
        }
        if download_dir is not None:
            llm_kwargs["download_dir"] = download_dir

        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError as exc:
            if "max_logprobs" not in str(exc):
                raise
            LOG.warning(
                "Installed vLLM does not accept max_logprobs in LLM(...); "
                "falling back to engine defaults."
            )
            llm_kwargs.pop("max_logprobs", None)
            self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        apply_chat_template_override(self.tokenizer, chat_template, chat_template_path)
        self._SamplingParams = SamplingParams
        self._model_name = model_path
        self._n_params_billion = n_params_billion
        self._vocab_size: Optional[int] = self._infer_vocab_size()
        self._probe_logprobs = min(max(1, int(probe_logprobs)), self.vocab_size)
        self._use_chat_template = use_chat_template
        self._continue_final_message = continue_final_message
        self._add_generation_prompt = add_generation_prompt
        if assistant_prefix_start:
            LOG.warning(
                "assistant_prefix_start=%r is ignored. Prompt continuation now follows "
                "the tokenizer chat template directly, matching GlimpRouter's BPA renderer.",
                assistant_prefix_start,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vocab_size(self) -> int:
        assert self._vocab_size is not None
        return self._vocab_size

    def render_prompt(self, question: str, history: str = "") -> str:
        if not self._use_chat_template:
            return question + history
        return render_for_continuation(
            question,
            history,
            self.tokenizer,
            add_generation_prompt=self._add_generation_prompt,
            continue_final_message=self._continue_final_message,
        )

    def probe_first_token(self, context: str) -> FirstTokenProbe:
        """Single forward pass returning next-token logits.

        vLLM returns the requested top-k logprobs for the one generated token.
        Non-returned vocabulary positions stay at ``-inf``.
        """
        sp = self._SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=self._probe_logprobs,
        )
        out = self.llm.generate([context], sampling_params=sp, use_tqdm=False)[0]
        completion = out.outputs[0]
        first_logprobs = self._get_logprobs_at(completion, 0)
        logits = self._logprobs_to_vector(first_logprobs)
        top_k_ids, top_k_probs = self._top_k(logits, k=20)
        return FirstTokenProbe(
            logits=logits,
            top_k_ids=top_k_ids,
            top_k_probs=top_k_probs,
            extra={"logprobs_returned": len(first_logprobs or {})},
        )

    def generate_step(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        step_delimiters: tuple[str, ...] = ("\n\n",),
        prefix_token_ids: Optional[list[int]] = None,
    ) -> StepOutput:
        """Generate one step until delimiter / EOS.

        If ``prefix_token_ids`` is provided, those ids are appended to the prompt
        token ids before decoding and then prepended to the returned step text.
        """
        sp = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=list(step_delimiters),
            logprobs=1,
        )
        return self._generate_single(
            context=context,
            sampling_params=sp,
            step_delimiters=step_delimiters,
            prefix_token_ids=prefix_token_ids,
        )

    def generate_full(
        self,
        context: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> StepOutput:
        sp = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,
        )
        return self._generate_single(
            context=context,
            sampling_params=sp,
            step_delimiters=(),
            prefix_token_ids=None,
        )

    def sample_prefixes(
        self,
        prompt: str,
        n: int,
        k: int,
        temperature: float,
    ) -> list[list[int]]:
        """Sample short prefixes with one vLLM request using SamplingParams(n=N)."""
        if n <= 0:
            return []
        sp = self._SamplingParams(
            temperature=temperature,
            max_tokens=k,
            n=n,
        )
        out = self.llm.generate([prompt], sampling_params=sp, use_tqdm=False)[0]
        return [list(completion.token_ids)[:k] for completion in out.outputs]

    def estimate_flops(self, n_input_tokens: int, n_output_tokens: int) -> float:
        """6 * N * (n_in + n_out) approximation (Hoffmann et al. 2022)."""
        n_params = self._n_params_billion * 1e9
        return 6.0 * n_params * (n_input_tokens + n_output_tokens)

    @staticmethod
    def _enable_hf_offline_mode() -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    @staticmethod
    def _validate_local_path(
        value: str,
        field_name: str,
        required: bool,
    ) -> str:
        path = Path(value).expanduser()
        if required and not path.exists():
            raise FileNotFoundError(
                f"{field_name} must point to a local path in offline mode: {value!r}. "
                "Put the model files on the server and update the YAML config, or set "
                "require_local_model: false if you intentionally rely on a preconfigured "
                "non-path model resolver."
            )
        return str(path) if path.exists() else value

    def _generate_single(
        self,
        context: str,
        sampling_params: Any,
        step_delimiters: tuple[str, ...],
        prefix_token_ids: Optional[list[int]] = None,
    ) -> StepOutput:
        forced_prefix = list(prefix_token_ids or [])
        if forced_prefix:
            prompt_token_ids = self._encode(context) + forced_prefix
            request_out = self.llm.generate(
                prompt_token_ids=[prompt_token_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
            )[0]
        else:
            request_out = self.llm.generate(
                [context],
                sampling_params=sampling_params,
                use_tqdm=False,
            )[0]

        completion = request_out.outputs[0]
        generated_ids = list(completion.token_ids)
        generated_logprobs = self._chosen_logprobs(generated_ids, completion)
        generated_text = completion.text or ""

        if forced_prefix:
            prefix_text = self._decode(forced_prefix)
            text = prefix_text + generated_text
            token_ids = forced_prefix + generated_ids
            logprobs = [float("nan")] * len(forced_prefix) + generated_logprobs
            first_token_logits = np.full(self.vocab_size, -float("inf"), dtype=np.float32)
            first_id = forced_prefix[0]
            if 0 <= first_id < self.vocab_size:
                first_token_logits[first_id] = 0.0
        else:
            text = generated_text
            token_ids = generated_ids
            logprobs = generated_logprobs
            first_token_logits = self._logprobs_to_vector(self._get_logprobs_at(completion, 0))

        finished = self._is_finished(completion, text, step_delimiters)
        return StepOutput(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            first_token_logits=first_token_logits,
            n_tokens=len(token_ids),
            finished=finished,
            extra={
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
                "forced_prefix_len": len(forced_prefix),
            },
        )

    def _infer_vocab_size(self) -> int:
        try:
            return int(len(self.tokenizer))
        except TypeError:
            pass
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
        return len(self.tokenizer.get_vocab())

    def _encode(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer.encode(text))

    def _decode(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        except TypeError:
            return self.tokenizer.decode(token_ids)

    def _logprobs_to_vector(self, logprobs: Any) -> np.ndarray:
        logits = np.full(self.vocab_size, -float("inf"), dtype=np.float32)
        if not logprobs:
            return logits
        for tok_id, lp_obj in logprobs.items():
            idx = int(tok_id)
            if idx < 0 or idx >= self.vocab_size:
                continue
            logits[idx] = float(getattr(lp_obj, "logprob", lp_obj))
        return logits

    @staticmethod
    def _get_logprobs_at(completion: Any, idx: int) -> Any:
        all_logprobs = getattr(completion, "logprobs", None)
        if not all_logprobs or idx >= len(all_logprobs):
            return None
        return all_logprobs[idx]

    def _chosen_logprobs(self, token_ids: list[int], completion: Any) -> list[float]:
        out: list[float] = []
        all_logprobs = getattr(completion, "logprobs", None) or []
        for i, tok_id in enumerate(token_ids):
            lp_dict = all_logprobs[i] if i < len(all_logprobs) else None
            lp_obj = lp_dict.get(tok_id) if lp_dict else None
            if lp_obj is None and lp_dict:
                lp_obj = next(iter(lp_dict.values()))
            if lp_obj is None:
                out.append(float("nan"))
            else:
                out.append(float(getattr(lp_obj, "logprob", lp_obj)))
        return out

    @staticmethod
    def _top_k(logits: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        finite = np.flatnonzero(np.isfinite(logits))
        if len(finite) == 0:
            return [], []
        k = min(k, len(finite))
        local = np.argpartition(-logits[finite], k - 1)[:k]
        ids = finite[local]
        ids = ids[np.argsort(-logits[ids])]
        probs = np.exp(logits[ids])
        return ids.astype(int).tolist(), probs.astype(float).tolist()

    @staticmethod
    def _is_finished(completion: Any, text: str, step_delimiters: tuple[str, ...]) -> bool:
        if any(marker in text for marker in FINAL_ANSWER_MARKERS):
            return True

        finish_reason = getattr(completion, "finish_reason", None)
        stop_reason = getattr(completion, "stop_reason", None)
        if finish_reason in {"length", None}:
            return False
        if finish_reason == "eos":
            return True
        if finish_reason == "stop":
            if isinstance(stop_reason, str) and stop_reason in step_delimiters:
                return False
            return True
        return False

