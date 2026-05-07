"""Fit calibrators on collected calibration data.

Usage:
    python scripts/run_calibration.py

Workflow:
  1. If `cached_examples_path` exists, load it (skip collection).
     Else: run collect_with_outcome_propagation and cache it.
  2. For each (calibrator_name, hyperparams) in config, fit and save.
  3. Save a registry mapping config name -> serialized calibrator path.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from conformal_routing.calibration import (
    ConformalCalibrator,
    GMMCalibrator,
    QuestionConditionalCalibrator,
    build_calibrator,
)
from conformal_routing.calibration.collect import (
    collect_with_agreement,
    collect_with_outcome_propagation,
    to_fit_inputs,
)
from conformal_routing.config_paths import configured_output_dir, load_experiment_config
from conformal_routing.data.loaders import load_split
from conformal_routing.eval.answer_check import check_answer
from conformal_routing.models import build_model
from conformal_routing.signals import build_signal


def _collection_limits(cfg: dict) -> tuple[int | None, int, int | None]:
    pipeline_cfg = cfg.get("pipeline", {})
    max_steps = pipeline_cfg.get("max_steps", cfg.get("max_steps", 32))
    max_tokens_per_step = pipeline_cfg.get(
        "max_tokens_per_step",
        cfg.get("max_tokens_per_step", 1024),
    )
    max_total_tokens = pipeline_cfg.get(
        "max_total_tokens",
        cfg.get("max_total_tokens"),
    )
    return (
        None if max_steps is None else int(max_steps),
        int(max_tokens_per_step),
        None if max_total_tokens is None else int(max_total_tokens),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml", type=Path)
    p.add_argument("--out", default=None, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    out_dir = configured_output_dir(cfg, "calibrators", "outputs/calibrators", args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = cfg.get("calibration_strategy", "outcome_propagation")
    cache_path = out_dir / f"{cfg['run_name']}_{strategy}_examples.pkl"
    if cache_path.exists():
        print(f"[fit] loading cached examples from {cache_path}")
        with cache_path.open("rb") as f:
            examples = pickle.load(f)
    else:
        small = build_model(cfg["small_model"])
        large = (
            build_model(cfg["large_model"])
            if strategy in {"agreement", "b", "B"}
            else small
        )
        sig = build_signal(cfg["signal"]["name"], **cfg["signal"].get("kwargs", {}))
        questions = load_split(
            cfg["benchmark"],
            split="calibration",
            limit=cfg.get("limit"),
            dataset_paths=cfg.get("dataset_paths"),
        )
        pipeline_cfg = cfg.get("pipeline", {})
        max_steps, max_tokens_per_step, max_total_tokens = _collection_limits(cfg)
        if strategy in {"agreement", "b", "B"}:
            examples = collect_with_agreement(
                questions=questions,
                small_model=small,
                large_model=large,
                signal=sig,
                max_steps=max_steps,
                max_tokens_per_step=max_tokens_per_step,
                max_total_tokens=max_total_tokens,
                step_delimiters=tuple(cfg.get("step_delimiters", ["\n\n"])),
                stop_on_repetition=pipeline_cfg.get("stop_on_repetition", True),
                repetition_min_chars=pipeline_cfg.get("repetition_min_chars", 10),
            )
        else:
            examples = collect_with_outcome_propagation(
                questions=questions,
                small_model=small,
                large_model=large,
                signal=sig,
                answer_checker=check_answer,
                max_steps=max_steps,
                max_tokens_per_step=max_tokens_per_step,
                max_total_tokens=max_total_tokens,
                step_delimiters=tuple(cfg.get("step_delimiters", ["\n\n"])),
                stop_on_repetition=pipeline_cfg.get("stop_on_repetition", True),
                repetition_min_chars=pipeline_cfg.get("repetition_min_chars", 10),
            )
        with cache_path.open("wb") as f:
            pickle.dump(examples, f)
        print(f"[fit] cached {len(examples)} examples -> {cache_path}")

    fit_inputs = to_fit_inputs(examples)
    print(
        f"[fit] {len(fit_inputs.scores)} step examples, "
        f"{int(fit_inputs.small_correct.sum())} positive ({fit_inputs.small_correct.mean():.2%})"
    )

    for cal_cfg in cfg["calibrators"]:
        name = cal_cfg["name"]
        cal_kwargs = cal_cfg.get("kwargs", {})
        cal_id = cal_cfg.get("id", name)

        if name == "qcond":
            # qcond needs base_factory; build it from inner config.
            inner = cal_cfg["base"]
            cal = QuestionConditionalCalibrator(
                base_factory=lambda inner=inner: build_calibrator(
                    inner["name"], **inner.get("kwargs", {})
                ),
                **cal_kwargs,
            )
        else:
            cal = build_calibrator(name, **cal_kwargs)

        cal.fit(fit_inputs)

        with (out_dir / f"{cfg['run_name']}_{cal_id}.pkl").open("wb") as f:
            pickle.dump(cal, f)

        # Print summary.
        if isinstance(cal, ConformalCalibrator):
            print(f"  conformal[alpha={cal.alpha}]: tau = {cal.tau_:.4f}, "
                  f"n_calib_correct = {cal.n_calib_correct_}")
        elif isinstance(cal, GMMCalibrator):
            print(f"  gmm[K={cal.n_components}]: trust_component = {cal.trust_component_}, "
                  f"means = {cal.gmm.means_.ravel()}")
        else:
            print(f"  fitted: {cal_id}")


if __name__ == "__main__":
    main()
