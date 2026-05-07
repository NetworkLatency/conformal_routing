"""GMM calibrator — reproduces STEER's calibration step.

Approach (per our reading of arXiv 2511.06190):
  1. Fit a 2-component GMM on the raw signal scores from the calibration set.
  2. The component with HIGHER mean is interpreted as "high confidence" (small ok),
     the other as "low confidence" (route to large).
  3. At inference, for a given score s, compute posterior P(high | s) under the GMM.
  4. Decision: SMALL if P(high | s) >= threshold (default 0.5).

Optionally, weight components by which side actually had higher small_correct rate.
We expose `weight_by_correctness=True` to do this.
"""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from src.conformal_routing.calibration.base import Calibrator, FitInputs, RouteDecision


class GMMCalibrator(Calibrator):
    name = "gmm"

    def __init__(
        self,
        n_components: int = 2,
        threshold: float = 0.5,
        weight_by_correctness: bool = True,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.weight_by_correctness = weight_by_correctness
        self.random_state = random_state
        self.gmm: GaussianMixture | None = None
        # index of the "high-confidence / small-trustworthy" component
        self.trust_component_: int | None = None

    def fit(self, data: FitInputs) -> None:
        s = data.scores.reshape(-1, 1).astype(np.float64)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type="full",
        ).fit(s)

        if self.weight_by_correctness:
            # For each component, take the soft-assignment-weighted mean of small_correct.
            # The component with the HIGHEST weighted accuracy is the trust component.
            resp = self.gmm.predict_proba(s)  # (N, K)
            comp_acc = (resp * data.small_correct[:, None]).sum(0) / (resp.sum(0) + 1e-12)
            self.trust_component_ = int(np.argmax(comp_acc))
        else:
            # Fallback: highest-mean component is the trust component.
            self.trust_component_ = int(np.argmax(self.gmm.means_.ravel()))

    def confidence(self, score: float, question_embed=None) -> float:
        assert self.gmm is not None and self.trust_component_ is not None
        post = self.gmm.predict_proba(np.array([[score]], dtype=np.float64))[0]
        return float(post[self.trust_component_])

    def decide(self, score: float, question_embed=None) -> RouteDecision:
        return (
            RouteDecision.SMALL
            if self.confidence(score) >= self.threshold
            else RouteDecision.LARGE
        )
