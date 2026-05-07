"""Answer-checking utilities.

For math benchmarks, use `math-verify` (https://github.com/huggingface/Math-Verify)
which handles LaTeX equivalence (\\frac, \\boxed, etc.).

For multiple choice (GPQA), parse the predicted letter choice.
For code (LiveCodeBench), execute against test cases - out of scope of this prototype;
we'll use exact-match on extracted answer for v0 and upgrade later.
"""

from __future__ import annotations

import re


CLOSE_THINK_TAG = "</think>"


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed answer, including nested braces."""
    positions = []
    start = 0
    while True:
        idx = text.find(r"\boxed", start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(r"\boxed")
    if not positions:
        return None

    idx = positions[-1] + len(r"\boxed")
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return None
    if text[idx] == "{":
        depth = 0
        content_start = idx + 1
        for j in range(idx, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:j]
        return None

    end = idx
    while end < len(text) and not text[end].isspace():
        end += 1
    token = text[idx:end].strip()
    return token or None


def clean_latex_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    s = str(answer).strip()
    if not s:
        return None
    while True:
        if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
            s = s[2:-2].strip()
            continue
        if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
            s = s[1:-1].strip()
            continue
        break
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9])(?![A-Za-z0-9])", r"\\sqrt{\1}", s)
    return s.strip() or None


def extract_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return clean_latex_answer(boxed)
    if CLOSE_THINK_TAG in text:
        return clean_latex_answer(text.split(CLOSE_THINK_TAG, 1)[1])
    return clean_latex_answer(text)


def extract_final_number(text: str) -> str | None:
    """Last integer in the text - fallback for AIME (answers are 0-999 ints)."""
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
        pred_clean = clean_latex_answer(pred_box)
        gold_clean = clean_latex_answer(gold)
        if pred_clean == gold_clean:
            return True
        pred_int = _normalize_int(pred_clean or "")
        gold_int = _normalize_int(gold_clean or "")
        if pred_int is not None and gold_int is not None and pred_int == gold_int:
            return True

    pred_answer = extract_answer(predicted_text)
    gold_answer = extract_answer(gold)
    if pred_answer is not None and gold_answer is not None and pred_answer == gold_answer:
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
