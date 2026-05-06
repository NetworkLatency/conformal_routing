"""Stepwise routing pipeline.

Workflow per question:

    context = question
    history = ""
    cost = 0
    while not finished:
        # 1. Probe small model for the upcoming step's first token (single forward).
        probe = small.probe_first_token(small.render_prompt(question, history))

        # 2. Extract signal from the probe (or via additional sampling if needed).
        score = signal.extract(SignalContext(question, history, step_idx, small,
                                             cached_probe=probe))

        # 3. Calibrator decides which model generates this step.
        decision = calibrator.decide(score, question_embed=q_embed)

        # 4. Generate the step with the chosen model. We can re-use the probed
        #    first token (commit it) so the small probe is not wasted even when
        #    we delegate to small. When routing to large, the probe IS wasted
        #    (acceptable: it's just one forward of the small model).
        if decision == SMALL:
            step_out = small.generate_step(small.render_prompt(question, history),
                                           prefix_token_ids=[probe.top_k_ids[0]])
        else:
            step_out = large.generate_step(large.render_prompt(question, history))

        history += step_out.text
        cost += flops_of(step_out)
        if step_out.finished:
            break

The pipeline returns a `RoutingTrace` containing the full chain, per-step decisions,
per-step signals, and cost breakdown — everything needed for evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from conformal_routing.calibration.base import Calibrator, RouteDecision
from conformal_routing.models.base import ModelWrapper
from conformal_routing.signals.base import SignalContext, SignalExtractor


@dataclass
class StepRecord:
    step_idx: int
    score: float
    decision: str  # "small" | "large"
    text: str
    n_tokens: int
    flops: float
    latency_s: float
    finished: bool


@dataclass
class RoutingTrace:
    question_id: str
    question: str
    final_answer_text: str
    steps: list[StepRecord] = field(default_factory=list)
    total_flops: float = 0.0
    total_latency_s: float = 0.0
    intervention_rate: float = 0.0  # fraction of steps routed to large

    def summary(self) -> dict:
        return {
            "question_id": self.question_id,
            "n_steps": len(self.steps),
            "intervention_rate": self.intervention_rate,
            "total_flops": self.total_flops,
            "total_latency_s": self.total_latency_s,
        }


@dataclass
class PipelineConfig:
    max_steps: int = 64
    max_tokens_per_step: int = 1024
    step_delimiters: tuple[str, ...] = ("\n\n",)
    temperature_small: float = 0.0
    temperature_large: float = 0.0
    final_answer_markers: tuple[str, ...] = ("\\boxed{", "</think>")
    reuse_probe_first_token: bool = True


class RoutingPipeline:
    def __init__(
        self,
        small_model: ModelWrapper,
        large_model: ModelWrapper,
        signal: SignalExtractor,
        calibrator: Calibrator,
        config: Optional[PipelineConfig] = None,
    ):
        self.small = small_model
        self.large = large_model
        self.signal = signal
        self.calibrator = calibrator
        self.cfg = config or PipelineConfig()

    def run(
        self,
        question_id: str,
        question: str,
        question_embed: np.ndarray | None = None,
    ) -> RoutingTrace:
        history = ""
        trace = RoutingTrace(question_id=question_id, question=question, final_answer_text="")
        n_large = 0

        for step_idx in range(self.cfg.max_steps):
            t0 = time.time()
            small_prompt = self.small.render_prompt(question, history)

            # --- 1. Probe small model ---
            probe = self.small.probe_first_token(small_prompt)

            # --- 2. Extract signal ---
            ctx = SignalContext(
                question=question,
                history=history,
                step_idx=step_idx,
                small_model=self.small,
                cached_probe=probe,
            )
            score = self.signal.extract(ctx)

            # --- 3. Routing decision ---
            decision = self.calibrator.decide(score, question_embed=question_embed)

            # --- 4. Generate step with chosen model ---
            if decision == RouteDecision.SMALL:
                # Reuse the probed first token: force it to be the small model's argmax.
                prefix_ids = [int(np.argmax(probe.logits))] if self.cfg.reuse_probe_first_token else None
                step_out = self.small.generate_step(
                    small_prompt,
                    max_tokens=self.cfg.max_tokens_per_step,
                    temperature=self.cfg.temperature_small,
                    step_delimiters=self.cfg.step_delimiters,
                    prefix_token_ids=prefix_ids,
                )
                model_used = "small"
            else:
                large_prompt = self.large.render_prompt(question, history)
                step_out = self.large.generate_step(
                    large_prompt,
                    max_tokens=self.cfg.max_tokens_per_step,
                    temperature=self.cfg.temperature_large,
                    step_delimiters=self.cfg.step_delimiters,
                )
                model_used = "large"
                n_large += 1

            latency = time.time() - t0

            prompt_for_cost = small_prompt if decision == RouteDecision.SMALL else large_prompt
            n_input = self._approx_token_count(prompt_for_cost)
            flops_used = (
                self.small if decision == RouteDecision.SMALL else self.large
            ).estimate_flops(n_input, step_out.n_tokens)
            # Add probe overhead (always 1 token of small).
            flops_used += self.small.estimate_flops(n_input, 1)

            history += step_out.text + (
                self.cfg.step_delimiters[0] if not step_out.finished else ""
            )

            trace.steps.append(
                StepRecord(
                    step_idx=step_idx,
                    score=score,
                    decision=model_used,
                    text=step_out.text,
                    n_tokens=step_out.n_tokens,
                    flops=flops_used,
                    latency_s=latency,
                    finished=step_out.finished,
                )
            )
            trace.total_flops += flops_used
            trace.total_latency_s += latency

            if step_out.finished or self._has_final_answer(history):
                break

        trace.final_answer_text = history
        if trace.steps:
            trace.intervention_rate = n_large / len(trace.steps)
        return trace

    def _has_final_answer(self, text: str) -> bool:
        return any(m in text for m in self.cfg.final_answer_markers)

    @staticmethod
    def _approx_token_count(text: str) -> int:
        # Cheap heuristic; for exact counts pass tokenizer in.
        # 4 chars/token is the standard rough rule.
        return max(1, len(text) // 4)
