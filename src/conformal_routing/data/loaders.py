"""Offline benchmark loaders.

The experiment server is expected to have no network access. Loaders therefore
read local files only, following the same pattern as the related GlimpRouter
codebase: configure ``dataset_paths`` in YAML and point each benchmark to a
local ``.jsonl``, ``.json``, ``.csv``, ``.tsv``, ``.parquet`` file, or a
HuggingFace ``Dataset.save_to_disk`` directory.

Each loader returns: ``list[{"id", "question", "answer", "meta"}]``.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


DEFAULT_DATASET_PATHS: dict[str, str] = {
    "aime2024": "data/aime24.jsonl",
    "aime24": "data/aime24.jsonl",
    "aime2025": "data/aime25.parquet",
    "aime25": "data/aime25.parquet",
    "math500": "data/math500.jsonl",
    "gpqa_diamond": "data/gpqa_diamond.jsonl",
    "gpqa": "data/gpqa_diamond.jsonl",
    "livecodebench": "data/livecodebench_lite.jsonl",
    "livecodebench_lite": "data/livecodebench_lite.jsonl",
    "lcb": "data/livecodebench_lite.jsonl",
    "lcb_v5": "data/livecodebench_v5.jsonl",
    "lcb_v6": "data/livecodebench_v6.jsonl",
}

DATASET_ALIASES: dict[str, tuple[str, ...]] = {
    "aime2024": ("aime2024", "aime24"),
    "aime24": ("aime2024", "aime24"),
    "aime2025": ("aime2025", "aime25"),
    "aime25": ("aime2025", "aime25"),
    "gpqa_diamond": ("gpqa_diamond", "gpqa"),
    "gpqa": ("gpqa_diamond", "gpqa"),
    "livecodebench": ("livecodebench", "livecodebench_lite", "lcb"),
    "livecodebench_lite": ("livecodebench", "livecodebench_lite", "lcb"),
    "lcb": ("livecodebench", "livecodebench_lite", "lcb"),
    "lcb_v5": ("lcb_v5", "livecodebench_v5"),
    "lcb_v6": ("lcb_v6", "livecodebench_v6"),
}

MATH_ID_KEYS = (
    "id",
    "ID",
    "problem_id",
    "question_id",
    "unique_id",
    "uid",
    "path",
    "file",
    "filename",
)
MATH_PROBLEM_KEYS = (
    "problem",
    "Problem",
    "question",
    "Question",
    "prompt",
    "Prompt",
    "input",
    "Input",
    "query",
    "Query",
    "problem_statement",
    "problem_text",
)
MATH_ANSWER_KEYS = (
    "answer",
    "Answer",
    "target",
    "final_answer",
    "final",
    "output",
    "solution",
)
MATH_NESTED_RECORD_KEYS = ("raw", "example", "data", "record", "row")
MISSING_TEXT_VALUES = {"", "none", "nan", "null"}


def _first_present(ex: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in ex and ex[key] is not None:
            return ex[key]
    return default


def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_TEXT_VALUES


def _available_keys(ex: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key in ex.keys())


def _coerce_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, Mapping):
                return parsed
    return None


def _candidate_mappings(ex: Mapping[str, Any]) -> list[tuple[str | None, Mapping[str, Any]]]:
    candidates: list[tuple[str | None, Mapping[str, Any]]] = [(None, ex)]
    for container_key in MATH_NESTED_RECORD_KEYS:
        nested = _coerce_mapping(ex.get(container_key))
        if nested is not None:
            candidates.append((container_key, nested))
    return candidates


def _field_label(container: str | None, key: str) -> str:
    return key if container is None else f"{container}.{key}"


def _first_field_match(
    ex: Mapping[str, Any],
    keys: tuple[str, ...],
) -> tuple[Any | None, str | None]:
    for container, candidate in _candidate_mappings(ex):
        for key in keys:
            if key in candidate and not _is_missing_text(candidate[key]):
                return candidate[key], _field_label(container, key)
    return None, None


def _candidate_to_text(value: Any) -> str:
    if isinstance(value, list):
        messages = []
        plain_parts = []
        for item in value:
            if isinstance(item, Mapping):
                role = str(item.get("role", "")).lower()
                content = item.get("content")
                if role in {"user", "human"} and not _is_missing_text(content):
                    messages.append(str(content).strip())
            elif not _is_missing_text(item):
                plain_parts.append(str(item).strip())
        if messages:
            return "\n\n".join(messages)
        return "\n\n".join(plain_parts)

    nested = _coerce_mapping(value)
    if nested is not None:
        nested_value, _ = _first_field_match(nested, MATH_PROBLEM_KEYS)
        if nested_value is not None:
            return _candidate_to_text(nested_value)
        content = nested.get("content")
        if not _is_missing_text(content):
            return str(content).strip()

    return "" if value is None else str(value).strip()


def _clean_required_text(
    value: Any,
    *,
    field_name: str,
    dataset: str,
    row_idx: int,
    source: Path,
    tried_keys: tuple[str, ...],
    raw: Mapping[str, Any],
) -> str:
    text = _candidate_to_text(value)
    if text.lower() in MISSING_TEXT_VALUES:
        keys = ", ".join(_available_keys(raw)) or "<none>"
        nested = ", ".join(MATH_NESTED_RECORD_KEYS)
        raise ValueError(
            f"{dataset} row {row_idx} in {source} is missing {field_name}. "
            f"Tried direct keys: {', '.join(tried_keys)}. "
            f"Also inspected nested records: {nested}. Available keys: {keys}"
        )
    return text


def _clean_optional_id(value: Any, default: str) -> str:
    text = _candidate_to_text(value)
    if text.lower() in MISSING_TEXT_VALUES:
        return default
    return text


def _math_prompt(problem: str) -> str:
    return (
        "Solve the following math problem and return ONLY the final answer.\n"
        "Please reason step by step, separate logical reasoning steps with two "
        "newline characters (\\n\\n), and put your final answer within \\boxed{}.\n\n"
        f"Problem: {problem.strip()}\n\n"
    )


def _math_item(
    ex: Mapping[str, Any],
    idx: int,
    *,
    path: Path,
    benchmark: str,
    default_prefix: str,
) -> dict:
    problem_value, problem_field = _first_field_match(ex, MATH_PROBLEM_KEYS)
    answer_value, answer_field = _first_field_match(ex, MATH_ANSWER_KEYS)
    problem = _clean_required_text(
        problem_value,
        field_name="problem text",
        dataset=benchmark,
        row_idx=idx,
        source=path,
        tried_keys=MATH_PROBLEM_KEYS,
        raw=ex,
    )
    answer = _clean_required_text(
        answer_value,
        field_name="answer",
        dataset=benchmark,
        row_idx=idx,
        source=path,
        tried_keys=MATH_ANSWER_KEYS,
        raw=ex,
    )
    id_value, id_field = _first_field_match(ex, MATH_ID_KEYS)
    meta: dict[str, Any] = {
        "benchmark": benchmark,
        "dataset_path": str(path),
        "raw_keys": _available_keys(ex),
        "problem_field": problem_field,
        "answer_field": answer_field,
    }
    subject, subject_field = _first_field_match(ex, ("subject", "Subject", "type", "Type"))
    level, level_field = _first_field_match(ex, ("level", "Level"))
    if subject is not None:
        meta["subject"] = subject
        meta["subject_field"] = subject_field
    if level is not None:
        meta["level"] = level
        meta["level_field"] = level_field
    if id_field is not None:
        meta["id_field"] = id_field
    return {
        "id": _clean_optional_id(
            id_value,
            default=f"{default_prefix}_{idx}",
        ),
        "question": _math_prompt(problem),
        "answer": answer,
        "meta": meta,
    }


def _resolve_dataset_path(
    name: str,
    dataset_paths: Mapping[str, str] | None = None,
) -> Path:
    search_keys = DATASET_ALIASES.get(name, (name,))
    configured = dataset_paths or {}
    for key in search_keys:
        value = configured.get(key)
        if value:
            path = Path(value).expanduser()
            if not path.exists():
                raise FileNotFoundError(
                    f"Configured local dataset path for {key!r} does not exist: {path}"
                )
            return path

    for key in search_keys:
        default = DEFAULT_DATASET_PATHS.get(key)
        if default and Path(default).exists():
            return Path(default)

    expected = [configured.get(k) or DEFAULT_DATASET_PATHS.get(k) for k in search_keys]
    expected = [str(p) for p in expected if p]
    raise FileNotFoundError(
        f"No local dataset file found for {name!r}. Set dataset_paths.{search_keys[0]} "
        f"in the config. Checked: {expected}"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object.")
            rows.append(value)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        for key in ("data", "train", "test", "examples", "rows"):
            if isinstance(value.get(key), list):
                rows = value[key]
                break
        else:
            dict_items = [(key, row) for key, row in value.items() if isinstance(row, dict)]
            if dict_items and len(dict_items) == len(value):
                rows = []
                for key, row in dict_items:
                    normalized = dict(row)
                    normalized.setdefault("id", key)
                    normalized.setdefault("question_id", key)
                    rows.append(normalized)
            else:
                rows = [value]
    else:
        raise ValueError(f"Unsupported JSON dataset shape in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSON dataset rows must be objects: {path}")
    return list(rows)


def _read_delimited(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required to load local parquet files.") from exc
    return pd.read_parquet(path).to_dict(orient="records")


def _read_hf_disk_dataset(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to load Dataset.save_to_disk directories."
        ) from exc

    ds = load_from_disk(str(path))
    if isinstance(ds, DatasetDict):
        for split_name in ("test", "train", "validation"):
            if split_name in ds:
                ds = ds[split_name]
                break
        else:
            ds = next(iter(ds.values()))
    return [dict(row) for row in ds]


def load_local_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser()
    if source.is_dir():
        return _read_hf_disk_dataset(source)

    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(source)
    if suffix == ".json":
        return _read_json(source)
    if suffix == ".csv":
        return _read_delimited(source, ",")
    if suffix == ".tsv":
        return _read_delimited(source, "\t")
    if suffix == ".parquet":
        return _read_parquet(source)
    raise ValueError(f"Unsupported local dataset format {suffix!r}: {source}")


def load_aime(
    year: int = 2024,
    dataset_paths: Mapping[str, str] | None = None,
) -> list[dict]:
    path = _resolve_dataset_path(f"aime{year}", dataset_paths)
    rows = load_local_rows(path)
    return [
        _math_item(
            ex,
            i,
            path=path,
            benchmark=f"aime{year}",
            default_prefix=f"aime{year}",
        )
        for i, ex in enumerate(rows)
    ]


def load_math500(dataset_paths: Mapping[str, str] | None = None) -> list[dict]:
    path = _resolve_dataset_path("math500", dataset_paths)
    rows = load_local_rows(path)
    return [
        _math_item(
            ex,
            i,
            path=path,
            benchmark="math500",
            default_prefix="math500",
        )
        for i, ex in enumerate(rows)
    ]


def _build_gpqa_prompt(ex: Mapping[str, Any], idx: int) -> tuple[str, str, dict[str, str]]:
    question = str(_first_present(ex, "Question", "question", "problem"))
    if "Correct Answer" in ex:
        choices = [
            (str(ex["Correct Answer"]), True),
            (str(ex["Incorrect Answer 1"]), False),
            (str(ex["Incorrect Answer 2"]), False),
            (str(ex["Incorrect Answer 3"]), False),
        ]
        rng = random.Random(42 + idx)
        rng.shuffle(choices)
        letters = "ABCD"
        formatted = [f"{letter}. {choice}" for letter, (choice, _) in zip(letters, choices)]
        answer = next(letter for letter, (_, correct) in zip(letters, choices) if correct)
        return f"{question.strip()}\n\n" + "\n".join(formatted), answer, {
            letter: choice for letter, (choice, _) in zip(letters, choices)
        }

    answer = str(_first_present(ex, "answer", "label", "target", "correct_answer"))
    choices: dict[str, str] = {}
    for letter in "ABCD":
        value = _first_present(ex, letter, letter.lower(), f"choice_{letter.lower()}")
        if value is not None:
            choices[letter] = str(value)
    if choices and not any(f"{letter}." in question for letter in choices):
        question = question.rstrip() + "\n\n" + "\n".join(
            f"{letter}. {text}" for letter, text in choices.items()
        )
    return question, answer, choices


def load_gpqa_diamond(dataset_paths: Mapping[str, str] | None = None) -> list[dict]:
    path = _resolve_dataset_path("gpqa_diamond", dataset_paths)
    rows = load_local_rows(path)
    out = []
    for i, ex in enumerate(rows):
        prompt, answer, choices = _build_gpqa_prompt(ex, i)
        out.append(
            {
                "id": str(_first_present(ex, "id", "ID", "question_id", default=f"gpqa_{i}")),
                "question": prompt,
                "answer": answer,
                "meta": {
                    "benchmark": "gpqa_diamond",
                    "dataset_path": str(path),
                    "choices": choices,
                    "correct_answer": _first_present(ex, "Correct Answer", default=None),
                },
            }
        )
    return out


def load_livecodebench(
    name: str = "livecodebench_lite",
    dataset_paths: Mapping[str, str] | None = None,
) -> list[dict]:
    path = _resolve_dataset_path(name, dataset_paths)
    rows = load_local_rows(path)
    out = []
    for i, ex in enumerate(rows):
        title = _first_present(ex, "question_title", "title", default="")
        content = _first_present(ex, "question_content", "content", "question", default="")
        starter = _first_present(ex, "starter_code", "starter", default="")
        parts = [str(part).strip() for part in (title, content) if str(part).strip()]
        if str(starter).strip():
            parts.append(f"Starter code:\n{starter}")
        public_tests = _first_present(ex, "public_test_cases", default=None)
        private_tests = _first_present(ex, "private_test_cases", default=None)
        out.append(
            {
                "id": str(_first_present(ex, "question_id", "id", default=f"lcb_{i}")),
                "question": "\n\n".join(parts),
                "answer": str(private_tests or public_tests or ""),
                "meta": {
                    "benchmark": "livecodebench",
                    "dataset_path": str(path),
                    "public_test_cases": public_tests,
                    "private_test_cases": private_tests,
                    "platform": _first_present(ex, "platform", default=None),
                },
            }
        )
    return out


def load_split(
    name: str,
    split: str = "test",
    limit: Optional[int] = None,
    dataset_paths: Mapping[str, str] | None = None,
) -> list[dict]:
    """Load a local benchmark and apply the project calibration/test split."""
    if name.startswith("aime"):
        year = int(name[4:])
        items = load_aime(year, dataset_paths=dataset_paths)
    elif name == "math500":
        items = load_math500(dataset_paths=dataset_paths)
    elif name in {"gpqa", "gpqa_diamond"}:
        items = load_gpqa_diamond(dataset_paths=dataset_paths)
    elif name in {"lcb", "livecodebench", "livecodebench_lite", "lcb_v5", "lcb_v6"}:
        items = load_livecodebench(name=name, dataset_paths=dataset_paths)
    else:
        raise ValueError(f"Unknown benchmark {name}")

    rng = random.Random(42)
    rng.shuffle(items)
    cutoff = int(len(items) * 0.2)  # 20% for calibration
    if split == "calibration":
        items = items[:cutoff]
    elif split == "test":
        items = items[cutoff:]
    elif split == "all":
        pass
    else:
        raise ValueError(f"Unknown split {split}")

    if limit is not None:
        items = items[:limit]
    return items
