"""H_init signal from GlimpRouter (Zeng et al. 2026).

Computes the entropy of the next-token distribution at the START of an upcoming step.
LOWER entropy => HIGHER confidence => keep small model.

We return NEGATIVE entropy so HIGHER score = MORE confident (matches global convention).
"""

from __future__ import annotations

import numpy as np

from src.conformal_routing.signals.base import SignalContext, SignalExtractor


class HInitSignal(SignalExtractor):
    name = "h_init"

    def __init__(self, top_k_for_entropy: int | None = None):
        # If set, compute entropy only over top-k tokens (cheaper, slight bias).
        # GlimpRouter uses full vocab; keep None to match.
        self.top_k = top_k_for_entropy

    def extract(self, ctx: SignalContext) -> float:
        prompt = self._build_prompt(ctx)
        probe = ctx.cached_probe or ctx.small_model.probe_first_token(prompt)

        logits = probe.logits  # (V,)
        # numerically stable softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()

        if self.top_k is not None:
            idx = np.argpartition(probs, -self.top_k)[-self.top_k :]
            p = probs[idx]
            p = p / p.sum()
        else:
            p = probs

        # Entropy in nats. Mask zeros to avoid log(0).
        p_nonzero = p[p > 0]
        entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))

        # Convention: higher = more confident, so return negative entropy.
        return -entropy

    @staticmethod
    def _build_prompt(ctx: SignalContext) -> str:
        # The probe input is question + history with no extra step delimiter at end —
        # the small model is about to emit the first token of step (step_idx).
        return ctx.small_model.render_prompt(ctx.question, ctx.history)
