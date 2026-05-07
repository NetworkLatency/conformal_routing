from __future__ import annotations

import numpy as np

from conformal_routing.calibration.base import Calibrator, FitInputs, RouteDecision
from conformal_routing.models.base import FirstTokenProbe, ModelWrapper, StepOutput
from conformal_routing.routing import PipelineConfig, RoutingPipeline
from conformal_routing.routing.safety import RepetitionState, update_strict_step_repetition
from conformal_routing.signals.base import SignalContext, SignalExtractor


def test_strict_repetition_detects_duplicate_and_alternating_steps():
    state = RepetitionState()

    assert update_strict_step_repetition(state, "First useful sentence.\n\n") is None
    assert (
        update_strict_step_repetition(state, "First useful sentence.\n\n")
        == "duplicate_step"
    )

    state = RepetitionState()
    assert update_strict_step_repetition(state, "Step A is long enough.\n\n") is None
    assert update_strict_step_repetition(state, "Step B is long enough.\n\n") is None
    assert (
        update_strict_step_repetition(state, "Step A is long enough.\n\n")
        == "alternating_step"
    )


class ConstantStepModel(ModelWrapper):
    @property
    def model_name(self) -> str:
        return "constant"

    @property
    def vocab_size(self) -> int:
        return 4

    def probe_first_token(self, context: str) -> FirstTokenProbe:
        logits = np.array([0.0, -1.0, -2.0, -3.0])
        return FirstTokenProbe(logits=logits, top_k_ids=[0], top_k_probs=[1.0])

    def generate_step(
        self,
        context,
        max_tokens=1024,
        temperature=0.0,
        step_delimiters=("\n\n",),
        prefix_token_ids=None,
    ):
        return StepOutput(
            text="This step repeats exactly.",
            token_ids=[0],
            logprobs=[0.0],
            first_token_logits=self.probe_first_token(context).logits,
            n_tokens=1,
            finished=False,
        )

    def generate_full(self, context, max_tokens=8192, temperature=0.0):
        return self.generate_step(context)

    def estimate_flops(self, n_input_tokens, n_output_tokens):
        return 1.0


class ZeroSignal(SignalExtractor):
    name = "zero"

    def extract(self, ctx: SignalContext) -> float:
        return 0.0


class AlternatingCalibrator(Calibrator):
    def __init__(self):
        self.calls = 0

    def fit(self, data: FitInputs) -> None:
        pass

    def decide(self, score: float, question_embed=None) -> RouteDecision:
        self.calls += 1
        return RouteDecision.SMALL if self.calls % 2 == 1 else RouteDecision.LARGE

    def confidence(self, score: float, question_embed=None) -> float:
        return 0.5


def test_pipeline_stops_when_alternating_models_repeat_same_step():
    model = ConstantStepModel()
    pipe = RoutingPipeline(
        small_model=model,
        large_model=model,
        signal=ZeroSignal(),
        calibrator=AlternatingCalibrator(),
        config=PipelineConfig(max_steps=8, stop_on_repetition=True),
    )

    trace = pipe.run("q1", "Solve this.")

    assert [step.decision for step in trace.steps] == ["small", "large"]
    assert trace.stop_reason == "duplicate_step"
    assert trace.steps[-1].finished is True
    assert trace.steps[-1].stop_reason == "duplicate_step"

