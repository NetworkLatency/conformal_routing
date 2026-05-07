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
MAX_SHORT_ANSWER_CHARS = 240


EXPLICIT_ANSWER_PATTERNS = (
    re.compile(
        r"(?:final\s+answer|answer|result|solution|value)\s*(?:is|=|:)\s*"
        r"(?P<answer>[^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:therefore|thus|hence|so)[^\n.]{0,200}?\b(?:is|=)\s*"
        r"(?P<answer>[^\n.]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<answer>(?:\\\(|\\\[|\$)?\s*[A-Za-z]?\s*=?\s*-?"
        r"(?:\\frac|\\sqrt|\d|\[|\(|\\?[A-Za-z])[^.\n]*?)\s+"
        r"is\s+the\s+(?:answer|solution|result)",
        re.IGNORECASE,
    ),
)


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
        if len(s) >= 4 and s.startswith(r"\(") and s.endswith(r"\)"):
            s = s[2:-2].strip()
            continue
        if len(s) >= 4 and s.startswith(r"\[") and s.endswith(r"\]"):
            s = s[2:-2].strip()
            continue
        break
    if s.startswith((r"\(", r"\[")):
        s = s[2:].strip()
    if s.endswith((r"\)", r"\]")):
        s = s[:-2].strip()
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9])(?![A-Za-z0-9])", r"\\sqrt{\1}", s)
    return s.strip() or None


def _answer_search_regions(text: str) -> list[str]:
    regions = []
    if CLOSE_THINK_TAG in text:
        regions.append(text.rsplit(CLOSE_THINK_TAG, 1)[1])
    regions.append(text)
    return regions


def _strip_assignment(candidate: str) -> str:
    if "=" not in candidate:
        return candidate
    if any(op in candidate for op in ("<=", ">=", r"\le", r"\ge", r"\leq", r"\geq")):
        return candidate
    lhs, rhs = candidate.rsplit("=", 1)
    if re.fullmatch(r"\s*[A-Za-z][A-Za-z0-9_]*\s*", lhs) or len(rhs.strip()) < len(lhs.strip()):
        return rhs.strip()
    return candidate


def _clean_prose_candidate(candidate: str | None) -> str | None:
    s = clean_latex_answer(candidate)
    if s is None:
        return None
    s = re.split(r"(?i)\s+(?:where|because|since|which)\b", s, maxsplit=1)[0]
    s = re.sub(r"(?i)\s+(?:is|are)\s+the\s+(?:answer|solution|result)\b.*$", "", s)
    s = re.sub(
        r"(?i)\s+(?:degrees?|units?|square\s+units?|ways?|possibilities)\b.*$",
        "",
        s,
    )
    s = _strip_assignment(s)
    s = s.strip().strip("`").strip()
    s = s.strip(" .,:;")
    s = clean_latex_answer(s)
    if s is None:
        return None
    lower = s.lower().strip()
    if lower in {"the answer", "the solution", "the result", "answer", "solution", "result"}:
        return None
    if len(s) > MAX_SHORT_ANSWER_CHARS:
        return None
    words = re.findall(r"[A-Za-z]+", s.replace("\\", ""))
    if "\\" not in s and len(words) > 8:
        return None
    return s


def _extract_explicit_answer(text: str) -> str | None:
    search_text = text[-4000:]
    matches: list[tuple[int, str]] = []
    for pattern in EXPLICIT_ANSWER_PATTERNS:
        for match in pattern.finditer(search_text):
            matches.append((match.start(), match.group("answer")))
    for _, candidate in sorted(matches, reverse=True):
        cleaned = _clean_prose_candidate(candidate)
        if cleaned is not None:
            return cleaned
    return None


def _unique(candidates: list[str | None]) -> list[str]:
    out = []
    seen = set()
    for candidate in candidates:
        cleaned = clean_latex_answer(candidate)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _answer_candidates(text: str) -> list[str]:
    candidates: list[str | None] = []
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        candidates.append(clean_latex_answer(boxed))
    for region in _answer_search_regions(text):
        candidates.append(_extract_explicit_answer(region))
    fallback_region = text.rsplit(CLOSE_THINK_TAG, 1)[1] if CLOSE_THINK_TAG in text else text
    fallback = clean_latex_answer(fallback_region)
    if fallback is not None and len(fallback) <= MAX_SHORT_ANSWER_CHARS:
        candidates.append(fallback)
    return _unique(candidates)


def _compact_answer(answer: str | None) -> str | None:
    cleaned = clean_latex_answer(answer)
    if cleaned is None:
        return None
    return re.sub(r"\s+", "", cleaned)


def extract_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return clean_latex_answer(boxed)
    explicit = _extract_explicit_answer(text)
    if explicit is not None:
        return explicit
    if CLOSE_THINK_TAG in text:
        return clean_latex_answer(text.rsplit(CLOSE_THINK_TAG, 1)[1])
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
    pred_candidates = _answer_candidates(predicted_text)
    gold_candidates = _answer_candidates(gold) or [gold]
    math_pred_inputs = pred_candidates[:]
    if len(predicted_text) <= MAX_SHORT_ANSWER_CHARS:
        math_pred_inputs.append(predicted_text)
    math_gold_inputs = gold_candidates[:]
    if len(gold) <= MAX_SHORT_ANSWER_CHARS:
        math_gold_inputs.append(gold)

    try:
        # If math-verify is installed
        from math_verify import parse, verify  # type: ignore

        for pred in _unique(math_pred_inputs):
            for gold_candidate in _unique(math_gold_inputs):
                try:
                    pred_set = parse(pred)
                    gold_set = parse(gold_candidate)
                    if bool(verify(gold_set, pred_set)):
                        return True
                except Exception:
                    continue
    except Exception:
        pass

    pred_box = extract_boxed_answer(predicted_text)
    if pred_box is not None:
        pred_clean = clean_latex_answer(pred_box)
        gold_clean = clean_latex_answer(gold)
        if _compact_answer(pred_clean) == _compact_answer(gold_clean):
            return True
        pred_int = _normalize_int(pred_clean or "")
        gold_int = _normalize_int(gold_clean or "")
        if pred_int is not None and gold_int is not None and pred_int == gold_int:
            return True

    for pred_answer in pred_candidates or [extract_answer(predicted_text)]:
        for gold_answer in gold_candidates:
            if _compact_answer(pred_answer) == _compact_answer(gold_answer):
                return True
            pred_int = _normalize_int(pred_answer or "")
            gold_int = _normalize_int(gold_answer or "")
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
