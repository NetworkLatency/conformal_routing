"""Logit-based confidence signal — STEER's signal (arXiv 2511.06190).

STEER uses the smaller model's logits BEFORE generating a step. The exact form
in the paper is the max softmax probability over the next-token distribution
(or equivalently 1 - margin). We expose both variants.

HIGHER score = MORE confident.
"""

from __future__ import annotations

import numpy as np

from conformal_routing.signals.base import SignalContext, SignalExtractor


class LogitConfidenceSignal(SignalExtractor):
    name = "logit_confidence"

    def __init__(self, mode: str = "max_prob"):
        """
        mode:
          - "max_prob": p_max  (STEER default-style)
          - "margin":  p_top1 - p_top2
          - "neg_entropy": same as H_init but reported as confidence direction
        """
        assert mode in {"max_prob", "margin", "neg_entropy"}
        self.mode = mode

    def extract(self, ctx: SignalContext) -> float:
        probe = ctx.cached_probe or ctx.small_model.probe_first_token(
            ctx.small_model.render_prompt(ctx.question, ctx.history)
        )
        logits = probe.logits - np.max(probe.logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()

        if self.mode == "max_prob":
            return float(np.max(probs))
        if self.mode == "margin":
            top2 = np.partition(probs, -2)[-2:]  # ascending
            return float(top2[1] - top2[0])
        # neg_entropy
        nz = probs[probs > 0]
        return float(np.sum(nz * np.log(nz)))  # = -H, higher = more confident
