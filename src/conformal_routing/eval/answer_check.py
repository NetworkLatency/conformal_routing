"""Answer-checking utilities.

For math benchmarks, use `math-verify` (https://github.com/huggingface/Math-Verify)
which handles LaTeX equivalence (\\frac, \\boxed, etc.).

For multiple choice (GPQA), parse the predicted letter choice.
For code (LiveCodeBench), execute against test cases — out of scope of this prototype;
we'll use exact-match on extracted answer for v0 and upgrade later.
"""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract content inside \\boxed{...}. Handles nested braces minimally."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1] if matches else None


def extract_final_number(text: str) -> str | None:
    """Last integer in the text — fallback for AIME (answers are 0-999 ints)."""
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def extract_choice_answer(text: str) -> str | None:
    """Extract a final A/B/C/D multiple-choice answer."""
    boxed = extract_boxed_answer(text)
    if boxed is not None and re.fullmatch(r"\s*[A-Da-d]\s*", boxed):
        return boxed.strip().upper()

    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Da-d])\)?",
        r"\(([A-Da-d])\)\s*$",
        r"\b([A-Da-d])\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return None


def _normalize_int(text: str) -> str | None:
    text = text.strip()
    if re.fullmatch(r"-?\d+", text):
        return str(int(text))
    return None


def check_answer(predicted_text: str, gold: str) -> bool:
    """Best-effort comparison.

    Order of preference:
      1. math_verify if available
      2. \\boxed{} content equality (whitespace-stripped)
      3. last integer match (for AIME)
    """
    try:
        # If math-verify is installed
        from math_verify import parse, verify  # type: ignore

        pred_set = parse(predicted_text)
        gold_set = parse(gold)
        return bool(verify(gold_set, pred_set))
    except Exception:
        pass

    pred_box = extract_boxed_answer(predicted_text)
    if pred_box is not None:
        if pred_box.strip() == gold.strip():
            return True
        pred_int = _normalize_int(pred_box)
        gold_int = _normalize_int(gold)
        if pred_int is not None and gold_int is not None and pred_int == gold_int:
            return True

    gold_choice = extract_choice_answer(gold)
    pred_choice = extract_choice_answer(predicted_text)
    if gold_choice is not None and pred_choice is not None:
        return pred_choice == gold_choice

    # Fallback: integer compare (for AIME)
    pred_int = extract_final_number(predicted_text)
    gold_int = _normalize_int(gold)
    if pred_int is not None and gold_int is not None and _normalize_int(pred_int) == gold_int:
        return True

    return False
