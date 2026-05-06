"""Check that a config points to existing local model and dataset assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from conformal_routing.config_paths import load_experiment_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    return parser.parse_args()


def _check_path(label: str, value: str) -> bool:
    path = Path(value)
    ok = path.exists()
    marker = "OK" if ok else "MISSING"
    print(f"[{marker}] {label}: {path}")
    return ok


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)

    ok = True
    for model_key in ("small_model", "large_model"):
        model_cfg = cfg.get(model_key, {})
        is_remote = bool(model_cfg.get("api_base_url")) or str(
            model_cfg.get("backend", "")
        ).lower() in {"openai", "openai_compatible", "remote_vllm"}
        for field in ("model_name_or_path", "tokenizer_name_or_path"):
            if is_remote and field == "model_name_or_path":
                print(f"[SKIP] {model_key}.{field}: remote model id {model_cfg.get(field)!r}")
                continue
            value = model_cfg.get(field)
            if value:
                ok = _check_path(f"{model_key}.{field}", value) and ok

    for key, value in (cfg.get("dataset_paths") or {}).items():
        ok = _check_path(f"dataset_paths.{key}", value) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
