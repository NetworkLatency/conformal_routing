"""Conformal calibrator  - proposed method (Route 1).

Background
----------
Split Conformal Prediction (Vovk et al. 2005; Angelopoulos & Bates 2023) gives
finite-sample coverage guarantees without distributional assumptions.

Adaptation to stepwise routing  - clean one-sided formulation
------------------------------------------------------------
We route SMALL when we are confident the small model is correct. Frame it as:

    Find tau such that P(s >= tau | small wrong) <= alpha
                   <=> tau is the (1 - alpha) quantile of {scores of WRONG examples}

Then at inference: route SMALL iff s >= tau.

Coverage statement (joint form)
-------------------------------
    P(small wrong AND we route small)
       = P(small wrong) * P(s >= tau | small wrong)
       <= P(small wrong) * alpha       (by construction of tau)

So the rate of "we trusted small but it was wrong" is bounded by alpha times the
base error rate. This is the SELLING POINT vs. GMM, which has no such guarantee.

Why this formulation gives MONOTONE behavior in alpha
-----------------------------------------------------
Smaller alpha => higher (1 - alpha) quantile => higher tau => fewer routes to small.
Larger alpha => more permissive => more routes to small.

Notes
-----
- Exchangeability is technically violated because steps within a question are
  correlated. Mitigate by using QUESTION-level splits (calibration vs test
  questions disjoint). Standard practice in conformal-for-LLMs.
- `finite_sample_correction=True` uses the conformal quantile
  `ceil((n+1)(1-alpha))/n`-th order statistic for the formal guarantee.
- `alpha` is the routing operating point; sweep it to draw the Pareto curve.
"""

from __future__ import annotations

import numpy as np

from conformal_routing.calibration.base import Calibrator, FitInputs, RouteDecision


class ConformalCalibrator(Calibrator):
    name = "conformal"

    def __init__(
        self,
        alpha: float = 0.1,
        finite_sample_correction: bool = True,
        random_state: int = 0,
    ):
        """
        Args:
            alpha: target rate at which "wrong" examples are mistakenly routed to small.
                   Smaller alpha => more cautious (fewer SMALL routes). Sweep this
                   to get the Pareto curve.
            finite_sample_correction: use ceil((n+1)(1-alpha))/n quantile rather than
                   np.quantile (which uses linear interpolation). The former carries
                   the conformal coverage guarantee.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.finite_sample_correction = finite_sample_correction
        self.rng = np.random.default_rng(random_state)
        self.tau_: float | None = None
        self.n_calib_wrong_: int | None = None

    def fit(self, data: FitInputs) -> None:
        assert data.scores.ndim == 1
        assert data.small_correct.ndim == 1
        assert data.scores.shape == data.small_correct.shape

        wrong_scores = data.scores[data.small_correct == 0]
        n = len(wrong_scores)
        if n < 5:
            raise RuntimeError(
                f"Too few wrong calibration examples ({n}). Need >= 5. "
                "Increase calibration set or pick a benchmark where small model "
                "actually fails some questions."
            )
        self.n_calib_wrong_ = n

        if self.finite_sample_correction:
            # Conformal quantile: ceil((n+1)(1-alpha))/n -th order statistic
            rank = int(np.ceil((n + 1) * (1 - self.alpha)))
            rank = min(max(rank, 1), n)
            sorted_wrong = np.sort(wrong_scores)
            self.tau_ = float(sorted_wrong[rank - 1])
        else:
            self.tau_ = float(np.quantile(wrong_scores, 1 - self.alpha))

    def decide(self, score: float, question_embed=None) -> RouteDecision:
        assert self.tau_ is not None, "Call fit() before decide()."
        return RouteDecision.SMALL if score >= self.tau_ else RouteDecision.LARGE

    def confidence(self, score: float, question_embed=None) -> float:
        """Soft confidence in [0, 1] reflecting margin to threshold (diagnostics only)."""
        assert self.tau_ is not None
        return float(1.0 / (1.0 + np.exp(-(score - self.tau_))))

    def threshold(self) -> float:
        assert self.tau_ is not None
        return self.tau_

    @classmethod
    def fit_with_alpha_sweep(
        cls,
        data: FitInputs,
        alphas: list[float],
        finite_sample_correction: bool = True,
    ) -> dict[float, "ConformalCalibrator"]:
        """Fit one calibrator per alpha to draw the Pareto curve."""
        out: dict[float, ConformalCalibrator] = {}
        for a in alphas:
            cal = cls(alpha=a, finite_sample_correction=finite_sample_correction)
            cal.fit(data)
            out[a] = cal
        return out

