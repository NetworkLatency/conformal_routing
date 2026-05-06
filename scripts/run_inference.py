"""Run routing inference with a fitted calibrator.

Usage:
    python scripts/run_inference.py \\
        --calibrator outputs/calibrators/conformal_aime_conformal_alpha010.pkl

Outputs JSONL per question with the full RoutingTrace + correctness label.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from tqdm import tqdm

from conformal_routing.config_paths import configured_output_dir, load_experiment_config
from conformal_routing.data.loaders import load_split
from conformal_routing.eval.answer_check import check_answer
from conformal_routing.eval.metrics import aggregate
from conformal_routing.models import build_model
from conformal_routing.routing import PipelineConfig, RoutingPipeline
from conformal_routing.signals import build_signal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml", type=Path)
    p.add_argument("--calibrator", required=True, type=Path,
                   help="Path to a fitted calibrator pickle.")
    p.add_argument("--out", default=None, type=Path)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    out_dir = configured_output_dir(cfg, "inference", "outputs/inference", args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load calibrator ---
    with args.calibrator.open("rb") as f:
        calibrator = pickle.load(f)
    print(f"[infer] loaded calibrator {args.calibrator.name}: {type(calibrator).__name__}")

    # --- build models / signal / pipeline ---
    small = build_model(cfg["small_model"])
    large = build_model(cfg["large_model"])
    signal = build_signal(cfg["signal"]["name"], **cfg["signal"].get("kwargs", {}))
    pipe = RoutingPipeline(
        small_model=small,
        large_model=large,
        signal=signal,
        calibrator=calibrator,
        config=PipelineConfig(**cfg.get("pipeline", {})),
    )

    # --- run on test split ---
    questions = load_split(
        cfg["benchmark"],
        split="test",
        limit=args.limit,
        dataset_paths=cfg.get("dataset_paths"),
    )
    print(f"[infer] running on {len(questions)} questions")

    out_path = out_dir / f"{cfg['run_name']}_{args.calibrator.stem}.jsonl"
    traces = []
    correctness = []
    with out_path.open("w") as fout:
        for q in tqdm(questions):
            trace = pipe.run(q["id"], q["question"])
            correct = int(check_answer(trace.final_answer_text, q["answer"]))
            traces.append(trace)
            correctness.append(correct)
            fout.write(json.dumps({
                "question_id": q["id"],
                "correct": correct,
                "summary": trace.summary(),
                "steps": [
                    {
                        "step_idx": s.step_idx,
                        "score": s.score,
                        "decision": s.decision,
                        "n_tokens": s.n_tokens,
                    }
                    for s in trace.steps
                ],
            }) + "\n")

    # --- aggregate ---
    agg = aggregate(traces, correctness)
    print("\n=== Inference Summary ===")
    for k, v in agg.as_dict().items():
        print(f"  {k:30s}  {v}")

    (out_dir / f"{cfg['run_name']}_{args.calibrator.stem}.summary.json").write_text(
        json.dumps(agg.as_dict(), indent=2)
    )


if __name__ == "__main__":
    main()
