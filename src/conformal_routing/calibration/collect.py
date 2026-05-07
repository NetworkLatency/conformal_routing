"""Calibration data collection.

Goal: produce a labeled dataset for fitting calibrators:
    { (question_id, step_idx, score, small_correct, embed) }

This requires a "ground truth" oracle for `small_correct` per step. There is no
universal one for stepwise routing  - the field uses several proxies:

  (A) ROLLOUT-BASED (recommended): for each candidate step from the small model,
      complete the rest of the trajectory K times with the LARGE model, check
      if the final answer is correct in majority of the K rollouts. Label
      `small_correct = 1` if so. Expensive but matches STEER's setup.

  (B) AGREEMENT-BASED (cheap proxy): generate the step with both small and
      large models. Label `small_correct = 1` if their resulting final answers
      agree. Cheaper but biased.

  (C) FINAL-OUTCOME-PROPAGATION (cheapest): just run the small model end-to-end,
      check if the FINAL answer is correct, propagate the binary label to ALL
      steps in that trajectory. Very noisy but gives a directional signal.

We implement (B) and (C); leave hooks for (A).

For first prototype, START WITH (C). It's noisy but gets you to a fitted
calibrator in a few hours rather than days. Upgrade to (A) for the final paper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from tqdm import tqdm

from conformal_routing.calibration.base import FitInputs
from conformal_routing.eval.answer_check import (
    check_answer,
    clean_latex_answer,
    extract_answer,
    extract_boxed_answer,
    extract_choice_answer,
    extract_final_number,
)
from conformal_routing.models.base import ModelWrapper
from conformal_routing.routing.safety import RepetitionState, update_strict_step_repetition
from conformal_routing.signals.base import SignalContext, SignalExtractor


@dataclass
class CalibrationExample:
    question_id: str
    step_idx: int
    score: float
    small_correct: int
    embed: np.ndarray | None = None


def collect_with_outcome_propagation(
    questions: list[dict],          # each: {"id": str, "question": str, "answer": str}
    small_model: ModelWrapper,
    large_model: ModelWrapper,      # not strictly needed for (C) but kept for signature parity
    signal: SignalExtractor,
    answer_checker: Callable[[str, str], bool],   # (predicted_text, gold) -> bool
    max_steps: int = 32,
    step_delimiters: tuple[str, ...] = ("\n\n",),
    embed_fn: Callable[[str], np.ndarray] | None = None,
    debug_path: str | Path | None = None,
    stop_on_repetition: bool = True,
    repetition_min_chars: int = 10,
) -> list[CalibrationExample]:
    """Strategy (C): outcome propagation.

    For each calibration question:
      1. Run small model end-to-end, recording per-step signal scores.
      2. Check if the final answer matches gold.
      3. Label every step in this trajectory with the same `small_correct`.
    """
    out: list[CalibrationExample] = []
    debug_rows = []

    for q in tqdm(questions, desc="Calibration (outcome propagation)"):
        history = ""
        per_step = []
        last_step_extra = {}
        stop_reason = None
        repetition = RepetitionState()
        for step_idx in range(max_steps):
            prompt = small_model.render_prompt(q["question"], history)
            probe = small_model.probe_first_token(prompt)
            ctx = SignalContext(
                question=q["question"],
                history=history,
                step_idx=step_idx,
                small_model=small_model,
                cached_probe=probe,
            )
            score = signal.extract(ctx)

            step_out = small_model.generate_step(
                prompt,
                step_delimiters=step_delimiters,
            )
            per_step.append((step_idx, score))
            committed_text = step_out.text + (step_delimiters[0] if not step_out.finished else "")
            history += committed_text
            last_step_extra = dict(step_out.extra)
            if stop_on_repetition:
                stop_reason = update_strict_step_repetition(
                    repetition,
                    committed_text,
                    min_chars=repetition_min_chars,
                )
                if stop_reason is not None:
                    last_step_extra["stop_reason"] = stop_reason
                    break
            if step_out.finished:
                stop_reason = str(step_out.extra.get("finish_reason") or "finished")
                break

        small_correct = int(answer_checker(history, q["answer"]))
        embed = embed_fn(q["question"]) if embed_fn is not None else None
        if debug_path is not None:
            debug_rows.append(
                {
                    "question_id": q["id"],
                    "question_preview": _question_preview(q),
                    "question_meta": dict(q.get("meta", {})),
                    "n_steps": len(per_step),
                    "small_correct": small_correct,
                    "gold": q["answer"],
                    "small_extracted": extract_answer(history),
                    "small_tail": history[-1000:],
                    "stop_reason": stop_reason,
                    "last_step_extra": last_step_extra,
                }
            )

        for step_idx, score in per_step:
            out.append(
                CalibrationExample(
                    question_id=q["id"],
                    step_idx=step_idx,
                    score=score,
                    small_correct=small_correct,
                    embed=embed,
                )
            )
    _write_debug_rows(debug_path, debug_rows)
    return out


def collect_with_agreement(
    questions: list[dict],
    small_model: ModelWrapper,
    large_model: ModelWrapper,
    signal: SignalExtractor,
    max_steps: int = 32,
    step_delimiters: tuple[str, ...] = ("\n\n",),
    embed_fn: Callable[[str], np.ndarray] | None = None,
    debug_path: str | Path | None = None,
    stop_on_repetition: bool = True,
    repetition_min_chars: int = 10,
) -> list[CalibrationExample]:
    """Strategy (B): label step as correct if small and large agree on FINAL answer
    when each runs solo from the same question.

    The small rollout is generated step-by-step so we can record the signal for
    each step. The large model is run once end-to-end and its extracted final
    answer is used as the reference for ``check_answer``.
    """
    out: list[CalibrationExample] = []
    debug_rows = []

    for q in tqdm(questions, desc="Calibration (small-large agreement)"):
        history = ""
        per_step = []
        last_step_extra = {}
        stop_reason = None
        repetition = RepetitionState()
        for step_idx in range(max_steps):
            prompt = small_model.render_prompt(q["question"], history)
            probe = small_model.probe_first_token(prompt)
            ctx = SignalContext(
                question=q["question"],
                history=history,
                step_idx=step_idx,
                small_model=small_model,
                cached_probe=probe,
            )
            score = signal.extract(ctx)

            step_out = small_model.generate_step(
                prompt,
                step_delimiters=step_delimiters,
            )
            per_step.append((step_idx, score))
            committed_text = step_out.text + (step_delimiters[0] if not step_out.finished else "")
            history += committed_text
            last_step_extra = dict(step_out.extra)
            if stop_on_repetition:
                stop_reason = update_strict_step_repetition(
                    repetition,
                    committed_text,
                    min_chars=repetition_min_chars,
                )
                if stop_reason is not None:
                    last_step_extra["stop_reason"] = stop_reason
                    break
            if step_out.finished:
                stop_reason = str(step_out.extra.get("finish_reason") or "finished")
                break

        large_out = large_model.generate_full(large_model.render_prompt(q["question"], ""))
        large_reference = _extract_reference_answer(large_out.text)
        small_correct = int(check_answer(history, large_reference))
        embed = embed_fn(q["question"]) if embed_fn is not None else None
        if debug_path is not None:
            debug_rows.append(
                {
                    "question_id": q["id"],
                    "question_preview": _question_preview(q),
                    "question_meta": dict(q.get("meta", {})),
                    "n_steps": len(per_step),
                    "small_correct": small_correct,
                    "large_reference": large_reference,
                    "small_extracted": extract_answer(history),
                    "large_extracted": extract_answer(large_out.text),
                    "small_tail": history[-1000:],
                    "large_tail": large_out.text[-1000:],
                    "stop_reason": stop_reason,
                    "last_step_extra": last_step_extra,
                    "large_extra": dict(large_out.extra),
                }
            )

        for step_idx, score in per_step:
            out.append(
                CalibrationExample(
                    question_id=q["id"],
                    step_idx=step_idx,
                    score=score,
                    small_correct=small_correct,
                    embed=embed,
                )
            )
    _write_debug_rows(debug_path, debug_rows)
    return out


def _extract_reference_answer(text: str) -> str:
    answer_region = text.split("</think>", 1)[1] if "</think>" in text else text
    boxed = extract_boxed_answer(answer_region)
    if boxed is not None:
        return clean_latex_answer(boxed) or boxed
    choice = extract_choice_answer(answer_region)
    if choice is not None:
        return choice
    number = extract_final_number(answer_region)
    if number is not None:
        return number
    return (
        extract_answer(answer_region)
        or extract_answer(text)
        or text
    )


def _question_preview(q: dict, max_chars: int = 800) -> str:
    return str(q.get("question", ""))[:max_chars]


def _write_debug_rows(debug_path: str | Path | None, rows: list[dict]) -> None:
    if debug_path is None:
        return
    path = Path(debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_fit_inputs(examples: Iterable[CalibrationExample]) -> FitInputs:
    examples = list(examples)
    scores = np.array([e.score for e in examples], dtype=np.float64)
    correct = np.array([e.small_correct for e in examples], dtype=np.int64)
    qids = np.array([e.question_id for e in examples])
    if examples and examples[0].embed is not None:
        embeds = np.stack([e.embed for e in examples], axis=0).astype(np.float32)
    else:
        embeds = None
    return FitInputs(
        scores=scores,
        small_correct=correct,
        question_embeds=embeds,
        question_ids=qids,
    )

