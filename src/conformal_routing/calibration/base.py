"""Calibrator: maps raw signal score -> route decision (small | large).

Two phases:
  1. fit(scores, labels): offline fitting on a calibration set
       - scores: array of raw signal scores from small model on calibration steps
       - labels: array of binary labels (1 = small model would have been correct on this step,
                 0 = small model would have been wrong; this is the "small_correct" label)
  2. decide(score) -> RouteDecision: at inference time, given a single signal score,
       return SMALL or LARGE.

Different calibrators answer the question "should we trust the small model on this step?"
in different ways:

  - GMMCalibrator (STEER baseline): two-component GMM on calibration scores, threshold
    at posterior > 0.5 of the "high-confidence" component.
  - ConformalCalibrator (ours, route 1): split conformal prediction with target
    miscoverage alpha => routes such that small-model error rate <= alpha (with
    calibration coverage guarantee).
  - QuestionConditionalCalibrator (ours, route 2): k-means clusters on question
    embeddings; per-cluster calibration of an isotonic / GMM / threshold model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class RouteDecision(Enum):
    SMALL = "small"
    LARGE = "large"


@dataclass
class FitInputs:
    """Per-step calibration data.

    Attributes:
        scores: shape (N,) raw signal values
        small_correct: shape (N,) binary, 1 if small model produced a correct step here
        question_embeds: optional (N, D) embeddings of parent question (for QCond)
        question_ids: optional (N,) which question each step belongs to (for grouped CV)
    """
    scores: np.ndarray
    small_correct: np.ndarray
    question_embeds: np.ndarray | None = None
    question_ids: np.ndarray | None = None


class Calibrator(ABC):
    name: str = "abstract"

    @abstractmethod
    def fit(self, data: FitInputs) -> None: ...

    @abstractmethod
    def decide(self, score: float, question_embed: np.ndarray | None = None) -> RouteDecision: ...

    @abstractmethod
    def confidence(self, score: float, question_embed: np.ndarray | None = None) -> float:
        """Return the calibrator's posterior P(small_correct | score).
        Useful for diagnostics and for sweeping the operating point.
        """
        ...
