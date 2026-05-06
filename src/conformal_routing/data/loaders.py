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


def _first_present(ex: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        if key in ex and ex[key] is not None:
            return ex[key]
    return default


def _math_prompt(problem: Any) -> str:
    return (
        "Solve the following math problem and return ONLY the final answer.\n"
        "Please reason step by step, separate logical reasoning steps with two "
        "newline characters (\\n\\n), and put your final answer within \\boxed{}.\n\n"
        f"Problem: {problem}\n\n"
    )


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
        {
            "id": str(_first_present(ex, "id", "ID", "problem_id", default=f"aime{year}_{i}")),
            "question": _math_prompt(_first_present(ex, "problem", "Problem", "question", "Question")),
            "answer": str(_first_present(ex, "answer", "Answer", "target", "final_answer")),
            "meta": {"benchmark": f"aime{year}", "dataset_path": str(path)},
        }
        for i, ex in enumerate(rows)
    ]


def load_math500(dataset_paths: Mapping[str, str] | None = None) -> list[dict]:
    path = _resolve_dataset_path("math500", dataset_paths)
    rows = load_local_rows(path)
    return [
        {
            "id": str(_first_present(ex, "id", "ID", "problem_id", default=f"math500_{i}")),
            "question": _math_prompt(_first_present(ex, "problem", "Problem", "question", "Question")),
            "answer": str(_first_present(ex, "answer", "Answer", "target", "solution")),
            "meta": {
                "benchmark": "math500",
                "dataset_path": str(path),
                "subject": _first_present(ex, "subject", "Subject", default=None),
                "level": _first_present(ex, "level", "Level", default=None),
            },
        }
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
