"""Signal extractors: produce a scalar score per upcoming reasoning step.

A larger score conventionally means "small model is MORE confident / step is EASIER".
Calibrators can flip sign internally if they need; signal extractors just expose a
consistent direction (up = easier, down = harder) for clarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from conformal_routing.models.base import ModelWrapper


@dataclass
class SignalContext:
    """All info a signal extractor may need to compute its score."""
    question: str
    history: str       # All previously generated steps (already committed)
    step_idx: int      # 0-based index of upcoming step
    small_model: ModelWrapper
    # Optional cached probe (avoid re-running forward if upstream already did it)
    cached_probe: object | None = None


class SignalExtractor(ABC):
    """Abstract: produce a scalar signal for the upcoming step.

    Convention: HIGHER score = small model is MORE confident.
    """

    name: str = "abstract"

    @abstractmethod
    def extract(self, ctx: SignalContext) -> float: ...

    def batch_extract(self, ctxs: list[SignalContext]) -> list[float]:
        """Default: just loop. Override for batched efficiency."""
        return [self.extract(c) for c in ctxs]

