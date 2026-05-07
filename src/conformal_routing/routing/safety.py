"""Runtime stopping guards for stepwise generation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepetitionState:
    recent_steps: list[str] = field(default_factory=list)
    triggered: bool = False
    trigger_reason: str | None = None


def normalize_step_text(step_text: str) -> str:
    return step_text.rstrip("\n").rstrip()


def update_strict_step_repetition(
    state: RepetitionState,
    new_step_text: str,
    min_chars: int = 10,
) -> str | None:
    """Stop on exact duplicate or alternating generated steps.

    This mirrors GlimpRouter's strict repetition guard. It is intentionally
    conservative: only exact repeated step text after right-trimming whitespace
    triggers a stop.
    """
    normalized = normalize_step_text(new_step_text)
    if len(normalized) < min_chars:
        state.recent_steps.append(normalized)
        return None

    if state.recent_steps and state.recent_steps[-1] == normalized:
        state.triggered = True
        state.trigger_reason = "duplicate_step"
        return state.trigger_reason

    if len(state.recent_steps) >= 2 and state.recent_steps[-2] == normalized:
        state.triggered = True
        state.trigger_reason = "alternating_step"
        return state.trigger_reason

    state.recent_steps.append(normalized)
    return None
