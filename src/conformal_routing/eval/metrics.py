"""Evaluation metrics for routing.

Per-question:
  - correct: 0/1 (final answer matches gold)
  - total_flops, total_latency_s, intervention_rate

Aggregate:
  - Pass@1 accuracy
  - Mean FLOPs
  - Mean latency
  - Mean intervention rate
  - Cost-vs-accuracy Pareto curve (sweep alpha or threshold)

For coverage diagnosis (Conformal-specific):
  - Realized small-error rate: (# steps routed to small AND small wrong) / (# steps routed to small)
    Should be -alpha by the conformal guarantee (under exchangeability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_routing.routing.pipeline import RoutingTrace


@dataclass
class AggregateMetrics:
    pass_at_1: float
    mean_flops: float
    mean_latency_s: float
    mean_intervention_rate: float
    n_questions: int

    def as_dict(self) -> dict:
        return {
            "pass_at_1": self.pass_at_1,
            "mean_flops": self.mean_flops,
            "mean_latency_s": self.mean_latency_s,
            "mean_intervention_rate": self.mean_intervention_rate,
            "n_questions": self.n_questions,
        }


def aggregate(traces: list[RoutingTrace], correctness: list[int]) -> AggregateMetrics:
    assert len(traces) == len(correctness)
    n = len(traces)
    return AggregateMetrics(
        pass_at_1=float(np.mean(correctness)),
        mean_flops=float(np.mean([t.total_flops for t in traces])),
        mean_latency_s=float(np.mean([t.total_latency_s for t in traces])),
        mean_intervention_rate=float(np.mean([t.intervention_rate for t in traces])),
        n_questions=n,
    )


def realized_small_error_rate(
    traces: list[RoutingTrace],
    correctness: list[int],
) -> float:
    """For Conformal: estimate the empirical small-model error rate AT THE QUESTION LEVEL.

    Limitation: we don't have step-level oracle labels at test time. This function
    therefore estimates a *question-level* proxy:
      Among questions whose every step was routed to small, what fraction are wrong?
    True step-level realized error needs an oracle (rollout)  - see analysis script.
    """
    small_only = [t for t, c in zip(traces, correctness) if t.intervention_rate == 0.0]
    if not small_only:
        return float("nan")
    # find indices of those traces
    idxs = [i for i, t in enumerate(traces) if t.intervention_rate == 0.0]
    n_wrong = sum(1 for i in idxs if correctness[i] == 0)
    return n_wrong / len(idxs)


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute Pareto frontier of (cost, accuracy) points.
    A point dominates another if it has lower cost AND higher accuracy.
    """
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    best_acc = -float("inf")
    for cost, acc in pts:
        if acc > best_acc:
            frontier.append((cost, acc))
            best_acc = acc
    return frontier

