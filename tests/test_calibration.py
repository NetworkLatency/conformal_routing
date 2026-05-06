"""Calibrator unit tests (no GPU / models needed).

These verify the *logic* of GMM, Conformal, and QCond calibrators on synthetic data.
Codex should make sure these all pass before moving on.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_routing.calibration import (
    ConformalCalibrator,
    FitInputs,
    GMMCalibrator,
    QuestionConditionalCalibrator,
    RouteDecision,
    build_calibrator,
)
from conformal_routing.calibration.collect import collect_with_agreement
from conformal_routing.models.base import FirstTokenProbe, ModelWrapper, StepOutput
from conformal_routing.signals.base import SignalContext, SignalExtractor


def make_synthetic(n: int = 1000, seed: int = 0) -> FitInputs:
    """Synthetic data: high-confidence steps are mostly correct, low-confidence mostly wrong.
    Score is a noisy version of (small_correct - 0.5) + N(0, 0.5).
    """
    rng = np.random.default_rng(seed)
    correct = rng.binomial(1, 0.65, size=n)
    scores = (correct - 0.5) * 1.5 + rng.normal(0, 0.5, size=n)
    return FitInputs(scores=scores, small_correct=correct)


def test_gmm_basic():
    data = make_synthetic()
    cal = GMMCalibrator(n_components=2)
    cal.fit(data)
    # High score should route SMALL.
    assert cal.decide(2.0) == RouteDecision.SMALL
    # Very low score should route LARGE.
    assert cal.decide(-2.0) == RouteDecision.LARGE
    # Confidence is in [0, 1].
    for s in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        c = cal.confidence(s)
        assert 0.0 <= c <= 1.0


def test_conformal_threshold_monotone_in_alpha():
    """More cautious (smaller alpha) => higher threshold => fewer routes to small."""
    data = make_synthetic()
    cals = ConformalCalibrator.fit_with_alpha_sweep(
        data, alphas=[0.05, 0.10, 0.20, 0.30]
    )
    taus = [cals[a].tau_ for a in [0.05, 0.10, 0.20, 0.30]]
    # Smaller alpha => threshold tau LARGER (more cautious about choosing small).
    # tau values should be in non-increasing order as alpha increases.
    for i in range(len(taus) - 1):
        assert taus[i] >= taus[i + 1] - 1e-9, f"Non-monotone tau: {taus}"


def test_conformal_coverage_on_synthetic():
    """Realized error rate among WRONG examples that get routed to small should be ~ alpha."""
    rng = np.random.default_rng(123)
    full = make_synthetic(n=5000, seed=1)
    perm = rng.permutation(len(full.scores))
    cal_idx, test_idx = perm[:3000], perm[3000:]
    calib = FitInputs(scores=full.scores[cal_idx], small_correct=full.small_correct[cal_idx])
    test_scores = full.scores[test_idx]
    test_correct = full.small_correct[test_idx]

    alpha = 0.10
    cal = ConformalCalibrator(alpha=alpha)
    cal.fit(calib)

    routes_small = test_scores >= cal.tau_
    # Among WRONG test examples, what fraction were mistakenly routed to small?
    # By construction, this should be approximately <= alpha.
    wrong_mask = test_correct == 0
    if wrong_mask.sum() == 0:
        pytest.skip("No wrong test examples.")
    realized_among_wrong = (routes_small & wrong_mask).sum() / wrong_mask.sum()
    print(f"P(s>=tau | wrong) = {realized_among_wrong:.3f} (target alpha={alpha})")
    # Should be in a reasonable neighborhood of alpha (allow 2x slack for finite-sample noise).
    assert realized_among_wrong < 2 * alpha, (
        f"Conformal coverage violated: P(s>=tau | wrong) = {realized_among_wrong}"
    )


def test_qcond_passes_through():
    """QCond with n_clusters=1 should behave like its base calibrator."""
    rng = np.random.default_rng(0)
    n = 500
    scores = rng.normal(size=n)
    correct = (scores > 0).astype(int)
    embeds = rng.normal(size=(n, 8)).astype(np.float32)
    qids = np.arange(n)
    data = FitInputs(scores=scores, small_correct=correct,
                     question_embeds=embeds, question_ids=qids)

    cal = QuestionConditionalCalibrator(
        base_factory=lambda: ConformalCalibrator(alpha=0.10),
        n_clusters=1,
        min_per_cluster=10,
    )
    cal.fit(data)
    # Should not error; has one cluster covering everything.
    assert cal.decide(0.5, question_embed=embeds[0]) in (RouteDecision.SMALL, RouteDecision.LARGE)


def test_registry():
    cal = build_calibrator("conformal", alpha=0.1)
    assert isinstance(cal, ConformalCalibrator)
    cal = build_calibrator("gmm", n_components=2)
    assert isinstance(cal, GMMCalibrator)


class ToyRolloutModel(ModelWrapper):
    def __init__(self, final_text: str):
        self.final_text = final_text

    @property
    def model_name(self) -> str:
        return "toy"

    @property
    def vocab_size(self) -> int:
        return 4

    def probe_first_token(self, context: str) -> FirstTokenProbe:
        logits = np.array([0.0, -1.0, -2.0, -3.0])
        return FirstTokenProbe(logits=logits, top_k_ids=[0], top_k_probs=[1.0])

    def generate_step(self, context, max_tokens=1024, temperature=0.0,
                      step_delimiters=("\n\n",), prefix_token_ids=None):
        if "scratch" in context:
            return StepOutput(
                text=self.final_text,
                token_ids=[2],
                logprobs=[-0.1],
                first_token_logits=self.probe_first_token(context).logits,
                n_tokens=1,
                finished=True,
            )
        return StepOutput(
            text="scratch",
            token_ids=[1],
            logprobs=[-0.1],
            first_token_logits=self.probe_first_token(context).logits,
            n_tokens=1,
            finished=False,
        )

    def generate_full(self, context, max_tokens=8192, temperature=0.0):
        return StepOutput(
            text=self.final_text,
            token_ids=[2],
            logprobs=[-0.1],
            first_token_logits=self.probe_first_token(context).logits,
            n_tokens=1,
            finished=True,
        )

    def estimate_flops(self, n_input_tokens, n_output_tokens):
        return 1.0


class StepIndexSignal(SignalExtractor):
    name = "step_idx"

    def extract(self, ctx: SignalContext) -> float:
        return float(ctx.step_idx)


def test_collect_with_agreement_propagates_small_large_label():
    questions = [{"id": "q1", "question": "Solve: ", "answer": "7"}]
    examples = collect_with_agreement(
        questions=questions,
        small_model=ToyRolloutModel("\\boxed{7}"),
        large_model=ToyRolloutModel("\\boxed{7}"),
        signal=StepIndexSignal(),
        max_steps=4,
    )

    assert [e.step_idx for e in examples] == [0, 1]
    assert [e.score for e in examples] == [0.0, 1.0]
    assert [e.small_correct for e in examples] == [1, 1]

    disagree = collect_with_agreement(
        questions=questions,
        small_model=ToyRolloutModel("\\boxed{7}"),
        large_model=ToyRolloutModel("\\boxed{8}"),
        signal=StepIndexSignal(),
        max_steps=4,
    )
    assert [e.small_correct for e in disagree] == [0, 0]
