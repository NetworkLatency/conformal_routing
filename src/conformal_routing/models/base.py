"""Model wrapper interfaces.

A wrapper exposes two key capabilities required by the routing pipeline:
  1. `generate_step(context, ...)` — generate one reasoning step (until step delimiter)
  2. `score_first_token(context)` — return logits/probs for the FIRST token of a step
                                    without consuming generation budget for the rest

Both small and large models implement the same interface so the pipeline is symmetric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StepOutput:
    """Result of generating a single reasoning step."""
    text: str                       # The generated step text (without trailing delimiter)
    token_ids: list[int]            # Token IDs of the step
    logprobs: list[float]           # Per-token logprobs of the chosen tokens
    first_token_logits: np.ndarray  # Full vocabulary logits for the FIRST token (shape: [V])
    n_tokens: int                   # Length in tokens
    finished: bool                  # True if this is the last step (EOS or final answer)
    extra: dict = field(default_factory=dict)


@dataclass
class FirstTokenProbe:
    """Result of probing only the first token of an upcoming step.

    Used for cheap signal extraction (e.g. H_init, logit confidence) without
    generating the full step.
    """
    logits: np.ndarray   # Full vocabulary logits (shape: [V])
    top_k_ids: list[int]
    top_k_probs: list[float]
    extra: dict = field(default_factory=dict)


class ModelWrapper(ABC):
    """Abstract interface for an LLM that can generate reasoning step-by-step."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    def render_prompt(self, question: str, history: str = "") -> str:
        """Render the model-specific prompt for continuing an assistant answer.

        Plain wrappers use the historical behavior: concatenate question and
        already-generated assistant text. Chat-model wrappers can override this
        to apply their tokenizer chat template.
        """
        return question + history

    @abstractmethod
    def probe_first_token(self, context: str) -> FirstTokenProbe:
        """Run a single forward pass on `context` and return logits over the next token.

        This must NOT continue generation — it is a cheap probe used by routing signals.
        Implementations should leverage prefix caching so repeated probing on extending
        contexts is cheap (cf. GlimpRouter §3.5).
        """
        ...

    @abstractmethod
    def generate_step(
        self,
        context: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        step_delimiters: tuple[str, ...] = ("\n\n",),
        prefix_token_ids: Optional[list[int]] = None,
    ) -> StepOutput:
        """Generate one reasoning step starting from `context`.

        A step ends when:
          - any of `step_delimiters` is generated, OR
          - EOS / max_tokens is reached, OR
          - the model emits a final-answer marker, e.g. \\boxed{...}

        If `prefix_token_ids` is provided, those tokens are forced as the start of the
        step (used when we already probed the first token and want to continue with it).
        """
        ...

    @abstractmethod
    def generate_full(
        self,
        context: str,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> StepOutput:
        """Generate the full reasoning trace + answer, ignoring step structure.
        Used for the 'small only' / 'large only' baselines.
        """
        ...

    def sample_prefixes(
        self,
        prompt: str,
        n: int,
        k: int,
        temperature: float,
    ) -> list[list[int]]:
        """Sample ``n`` short token prefixes from the same prompt.

        Concrete backends can override this with a single batched call. The
        default keeps tests and non-vLLM wrappers working by looping through
        ``generate_step``.
        """
        prefixes: list[list[int]] = []
        for _ in range(n):
            out = self.generate_step(
                prompt,
                max_tokens=k,
                temperature=temperature,
                step_delimiters=(),
            )
            prefixes.append(out.token_ids[:k])
        return prefixes

    @abstractmethod
    def estimate_flops(self, n_input_tokens: int, n_output_tokens: int) -> float:
        """Rough FLOPs estimate for this model on given token counts.
        Used for cost reporting. Use 6 * N_params * n_tokens as standard approx.
        """
        ...
