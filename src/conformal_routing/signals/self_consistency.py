"""Self-consistency signal (candidate for the proposed method).

Idea: at the start of a step, sample N short prefixes (k tokens each) from the small
model with temperature T > 0. If those prefixes agree (in token-id sequence or in
embedding-space cosine), the small model has a stable view of how to start the step
=> HIGH confidence. If they diverge, route to large model.

This is a step-level analogue of self-consistency (Wang et al. 2022) but using only
SHORT prefixes so latency overhead is small.

HIGHER score = MORE confident.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from conformal_routing.signals.base import SignalContext, SignalExtractor


class SelfConsistencySignal(SignalExtractor):
    name = "self_consistency"

    def __init__(
        self,
        n_samples: int = 4,
        prefix_tokens: int = 12,
        temperature: float = 0.7,
        agreement_metric: str = "exact_match",  # or "token_iou" or "first_token_agreement"
    ):
        self.n_samples = n_samples
        self.prefix_tokens = prefix_tokens
        self.temperature = temperature
        self.agreement_metric = agreement_metric

    def extract(self, ctx: SignalContext) -> float:
        prompt = ctx.small_model.render_prompt(ctx.question, ctx.history)

        prefixes = ctx.small_model.sample_prefixes(
            prompt=prompt,
            n=self.n_samples,
            k=self.prefix_tokens,
            temperature=self.temperature,
        )

        return self._agreement(prefixes)

    def _agreement(self, prefixes: list[list[int]]) -> float:
        if self.agreement_metric == "first_token_agreement":
            # Frequency of mode of first tokens.
            firsts = [p[0] for p in prefixes if p]
            if not firsts:
                return 0.0
            most_common_count = Counter(firsts).most_common(1)[0][1]
            return most_common_count / len(firsts)

        if self.agreement_metric == "exact_match":
            tup_prefixes = [tuple(p) for p in prefixes]
            most_common_count = Counter(tup_prefixes).most_common(1)[0][1]
            return most_common_count / len(tup_prefixes)

        if self.agreement_metric == "token_iou":
            sets = [set(p) for p in prefixes]
            n = len(sets)
            if n < 2:
                return 1.0
            ious = []
            for i in range(n):
                for j in range(i + 1, n):
                    inter = len(sets[i] & sets[j])
                    union = len(sets[i] | sets[j])
                    ious.append(inter / union if union > 0 else 0.0)
            return float(np.mean(ious))

        raise ValueError(f"Unknown metric {self.agreement_metric}")

