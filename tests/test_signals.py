"""Signal unit tests using a fake ModelWrapper (no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_routing.models.base import (
    FirstTokenProbe,
    ModelWrapper,
    StepOutput,
)
from conformal_routing.signals import build_signal
from conformal_routing.signals.base import SignalContext


class FakeModel(ModelWrapper):
    """Returns a fixed logit distribution; useful for unit tests."""

    def __init__(self, vocab_size: int = 100, peakedness: float = 5.0, seed: int = 0):
        self._V = vocab_size
        self.peakedness = peakedness
        self._rng = np.random.default_rng(seed)

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def vocab_size(self) -> int:
        return self._V

    def probe_first_token(self, context: str) -> FirstTokenProbe:
        # Sharp distribution centered on a deterministic id derived from context len.
        peak_id = (len(context) * 7) % self._V
        logits = np.full(self._V, -10.0)
        logits[peak_id] = self.peakedness
        # Spread small mass to neighbors
        for offset in (1, -1):
            logits[(peak_id + offset) % self._V] = self.peakedness - 2
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top_k_ids = list(np.argsort(probs)[-5:][::-1].astype(int))
        top_k_probs = [float(probs[i]) for i in top_k_ids]
        return FirstTokenProbe(logits=logits, top_k_ids=top_k_ids, top_k_probs=top_k_probs)

    def generate_step(self, context, max_tokens=1024, temperature=0.0,
                      step_delimiters=("\n\n",), prefix_token_ids=None):
        token_ids = list(self._rng.integers(0, self._V, size=10))
        if prefix_token_ids:
            token_ids = list(prefix_token_ids) + token_ids[len(prefix_token_ids):]
        return StepOutput(
            text="fake step text",
            token_ids=token_ids,
            logprobs=[-1.0] * len(token_ids),
            first_token_logits=self.probe_first_token(context).logits,
            n_tokens=len(token_ids),
            finished=False,
        )

    def generate_full(self, context, max_tokens=8192, temperature=0.0):
        return self.generate_step(context)

    def estimate_flops(self, n_input_tokens, n_output_tokens):
        return 6.0 * 1e9 * (n_input_tokens + n_output_tokens)


def test_h_init_returns_negative_value():
    sig = build_signal("h_init")
    fake = FakeModel(peakedness=5.0)
    ctx = SignalContext(question="q", history="", step_idx=0, small_model=fake)
    score = sig.extract(ctx)
    # Negative entropy: peaky distribution => entropy small => -entropy near 0 (but negative).
    assert score < 0
    assert score > -10  # not pathological


def test_logit_confidence_in_unit_interval():
    sig = build_signal("logit_confidence", mode="max_prob")
    fake = FakeModel(peakedness=5.0)
    ctx = SignalContext(question="q", history="", step_idx=0, small_model=fake)
    score = sig.extract(ctx)
    assert 0.0 < score <= 1.0


def test_logit_confidence_peakier_higher():
    sig = build_signal("logit_confidence", mode="max_prob")
    flat = FakeModel(peakedness=0.5)
    sharp = FakeModel(peakedness=10.0)
    ctx_flat = SignalContext(question="q", history="", step_idx=0, small_model=flat)
    ctx_sharp = SignalContext(question="q", history="", step_idx=0, small_model=sharp)
    assert sig.extract(ctx_sharp) > sig.extract(ctx_flat)
