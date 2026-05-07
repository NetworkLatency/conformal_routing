from __future__ import annotations

import json

from conformal_routing.config_paths import load_experiment_config, resolve_local_paths_in_config
from conformal_routing.data.loaders import load_gpqa_diamond, load_split


def test_load_split_reads_local_aime_jsonl(tmp_path):
    path = tmp_path / "aime24.jsonl"
    rows = [
        {"id": f"p{i}", "problem": f"{i}+1?", "answer": str(i + 1)}
        for i in range(10)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    calib = load_split(
        "aime2024",
        split="calibration",
        dataset_paths={"aime2024": str(path)},
    )
    test = load_split(
        "aime2024",
        split="test",
        dataset_paths={"aime2024": str(path)},
    )

    assert len(calib) == 2
    assert len(test) == 8
    assert all(item["meta"]["dataset_path"] == str(path) for item in calib + test)


def test_gpqa_local_loader_builds_seeded_choices(tmp_path):
    path = tmp_path / "gpqa_diamond.jsonl"
    row = {
        "Question": "Which option is correct?",
        "Correct Answer": "right",
        "Incorrect Answer 1": "wrong1",
        "Incorrect Answer 2": "wrong2",
        "Incorrect Answer 3": "wrong3",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    first = load_gpqa_diamond(dataset_paths={"gpqa_diamond": str(path)})[0]
    second = load_gpqa_diamond(dataset_paths={"gpqa_diamond": str(path)})[0]

    assert first["question"] == second["question"]
    assert first["answer"] == second["answer"]
    assert first["answer"] in {"A", "B", "C", "D"}
    assert "A." in first["question"]


def test_config_paths_resolve_relative_to_project_root(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    data_dir = project / "data"
    model_dir = project / "models" / "small"
    config_dir.mkdir(parents=True)
    data_dir.mkdir()
    model_dir.mkdir(parents=True)
    (data_dir / "aime24.jsonl").write_text("", encoding="utf-8")
    config_path = config_dir / "run.yaml"
    config_path.write_text("", encoding="utf-8")

    cfg = {
        "dataset_paths": {"aime2024": "data/aime24.jsonl"},
        "small_model": {"model_name_or_path": "models/small"},
    }
    resolved = resolve_local_paths_in_config(cfg, config_path)

    assert resolved["dataset_paths"]["aime2024"] == str(data_dir / "aime24.jsonl")
    assert resolved["small_model"]["model_name_or_path"] == str(model_dir)


def test_global_config_merges_vllm_defaults(tmp_path):
    project = tmp_path / "project"
    config_dir = project / "configs"
    model_dir = project / "models" / "small"
    config_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    config_path = config_dir / "default.yaml"
    config_path.write_text(
        """
run_name: smoke
output_dirs:
  inference: outputs/inference
vllm_defaults:
  dtype: bfloat16
  offline: true
  require_local_model: true
small_model:
  model_name_or_path: models/small
  n_params_billion: 1.5
large_model:
  model_name_or_path: models/small
  n_params_billion: 1.5
""",
        encoding="utf-8",
    )

    cfg = load_experiment_config(config_path)

    assert cfg["small_model"]["dtype"] == "bfloat16"
    assert cfg["small_model"]["offline"] is True
    assert cfg["small_model"]["model_name_or_path"] == str(model_dir)
    assert cfg["output_dirs"]["inference"] == str(project / "outputs" / "inference")


def test_default_global_config_loads_from_project_root():
    cfg = load_experiment_config("configs/default.yaml")

    assert cfg["runtime"]["model_backend"] == "vllm"
    assert cfg["small_model"]["offline"] is True
    assert cfg["small_model"]["dtype"] == cfg["vllm_defaults"]["dtype"]


def test_remote_config_keeps_remote_model_id_and_resolves_tokenizer():
    cfg = load_experiment_config("configs/math_remote_qwen3.yaml")

    assert cfg["large_model"]["backend"] == "openai_compatible"
    assert cfg["large_model"]["model_name_or_path"] == "Qwen3-14B"
    assert cfg["large_model"]["tokenizer_name_or_path"].startswith("/home/")
    assert cfg["small_model"]["use_chat_template"] is True
