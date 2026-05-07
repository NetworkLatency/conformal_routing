"""Preliminary signal-correlation study.

THIS IS THE FIRST THING YOU RUN. It answers the question:
    "Does any of our candidate signals predict step correctness well enough
     to be worth building on?"

Outputs:
  1. Spearman / Pearson correlation between each signal and small-step correctness
  2. ROC-AUC of each signal as a binary predictor of small_correct
  3. Visualization: histogram of signal values for {correct, wrong} steps
  4. Recommendation: which signal(s) to take forward

Decision rule:
  - If self_consistency Spearman > h_init Spearman + 0.05 -> recommend Route 1 (Conformal)
    on top of self_consistency.
  - If signals are within 0.05 of each other -> recommend Route 2 (QCond) on top of
    h_init or logit_confidence (cheaper, similar power).
  - If all signals AUC < 0.6 -> the routing problem may be ill-posed on this benchmark;
    try a different one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.conformal_routing.calibration.collect import (
    collect_with_agreement,
    collect_with_outcome_propagation,
)
from src.conformal_routing.config_paths import configured_output_dir, load_experiment_config
from src.conformal_routing.data.loaders import load_split
from src.conformal_routing.models import build_model
from src.conformal_routing.signals import build_signal


def _json_metric(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml", type=Path)
    p.add_argument("--out", default=None, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    out_dir = configured_output_dir(cfg, "preliminary", "outputs/preliminary", args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load models ---
    small = build_model(cfg["small_model"])
    large = build_model(cfg["large_model"])  # used by some signals; unused otherwise

    # --- load data: small calibration split ---
    questions = load_split(
        cfg["benchmark"],
        split="calibration",
        limit=cfg.get("limit"),
        dataset_paths=cfg.get("dataset_paths"),
    )
    print(f"[preliminary] loaded {len(questions)} questions from {cfg['benchmark']}")

    # --- answer checker (math-verify or simple regex; user fills) ---
    from src.conformal_routing.eval.answer_check import check_answer  # see file
    answer_checker = check_answer

    # --- iterate over signals ---
    signal_names = cfg["signals"]  # e.g. ["h_init", "logit_confidence", "self_consistency"]
    strategy = cfg.get("calibration_strategy", "outcome_propagation")
    results = {}

    for sname in signal_names:
        sig = build_signal(sname, **cfg.get("signal_kwargs", {}).get(sname, {}))
        print(f"[preliminary] collecting signal={sname} strategy={strategy}")
        if strategy == "agreement":
            examples = collect_with_agreement(
                questions=questions,
                small_model=small,
                large_model=large,
                signal=sig,
                max_steps=cfg.get("max_steps", 32),
                step_delimiters=tuple(cfg.get("step_delimiters", ["\n\n"])),
                debug_path=out_dir / f"debug_{sname}.jsonl",
            )
        elif strategy == "outcome_propagation":
            examples = collect_with_outcome_propagation(
                questions=questions,
                small_model=small,
                large_model=large,
                signal=sig,
                answer_checker=answer_checker,
                max_steps=cfg.get("max_steps", 32),
                step_delimiters=tuple(cfg.get("step_delimiters", ["\n\n"])),
                debug_path=out_dir / f"debug_{sname}.jsonl",
            )
        else:
            raise ValueError(
                "run_preliminary_study.py supports calibration_strategy="
                "'outcome_propagation' or 'agreement'."
            )
        scores = np.array([e.score for e in examples])
        labels = np.array([e.small_correct for e in examples])

        if len(np.unique(labels)) < 2:
            print(f"[warn] signal={sname}: only one label class. Skipping AUC.")
            auc = float("nan")
        else:
            auc = float(roc_auc_score(labels, scores))

        sp = float(spearmanr(scores, labels).statistic)
        pr = float(pearsonr(scores, labels).statistic)

        results[sname] = {
            "n_examples": len(examples),
            "n_pos": int(labels.sum()),
            "spearman": _json_metric(sp),
            "pearson": _json_metric(pr),
            "auc": _json_metric(auc),
            "score_mean": _json_metric(float(scores.mean())),
            "score_std": _json_metric(float(scores.std())),
        }
        # Save raw for later plotting.
        np.savez(out_dir / f"raw_{sname}.npz", scores=scores, labels=labels)

    # --- save summary ---
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== Preliminary Study Summary ===")
    for sname, r in results.items():
        auc_text = "nan" if r["auc"] is None else f"{r['auc']:.3f}"
        sp_text = "nan" if r["spearman"] is None else f"{r['spearman']:.3f}"
        print(f"  {sname:20s}  AUC={auc_text}  Spearman={sp_text}  "
              f"n={r['n_examples']}  pos%={r['n_pos']/r['n_examples']:.2%}")

    # --- recommendation ---
    aucs = {k: v["auc"] for k, v in results.items()}
    best = max(aucs, key=lambda k: aucs[k] if aucs[k] is not None else -1)
    best_auc = aucs[best]
    best_auc_text = "nan" if best_auc is None else f"{best_auc:.3f}"
    print(f"\nRecommendation: best signal = {best} (AUC={best_auc_text})")
    if best_auc is None:
        print("  WARNING: no usable AUC. Labels contain only one class.")
    elif best_auc < 0.6:
        print("  WARNING: best AUC < 0.6. Routing problem may be hard on this benchmark.")
    elif aucs.get("self_consistency") is not None and aucs["self_consistency"] >= max(
        aucs.get("h_init") or 0, aucs.get("logit_confidence") or 0
    ) + 0.05:
        print("  -> Take self_consistency forward; build Conformal on top (Route 1).")
    else:
        print("  -> Use h_init or logit_confidence; build QCond on top (Route 2).")


if __name__ == "__main__":
    main()
