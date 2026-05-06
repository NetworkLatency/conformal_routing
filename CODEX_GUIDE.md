# CODEX IMPLEMENTATION GUIDE

This file is now a completion checklist for the current implementation. The
original skeleton tasks have been filled in, and the project is configured for
offline server execution with local model and dataset paths.

## Project Status

| Module | Status | Notes |
|---|---:|---|
| `models/base.py` | complete | Adds default `sample_prefixes` fallback. |
| `models/vllm_wrapper.py` | complete | vLLM init, chat-template rendering, probing, step/full generation, batched prefix sampling, offline path validation. |
| `models/openai_compatible.py` | complete | Remote OpenAI-compatible vLLM completion backend. |
| `models/factory.py` | complete | Selects local vLLM vs remote OpenAI-compatible backend from config. |
| `signals/base.py` | complete | Shared signal interface. |
| `signals/h_init.py` | complete | Entropy-based signal. |
| `signals/logit_confidence.py` | complete | Max-prob / margin / negative-entropy signal. |
| `signals/self_consistency.py` | complete | Uses `ModelWrapper.sample_prefixes`; vLLM backend issues one `n=N` request. |
| `calibration/base.py` | complete | Calibrator API unchanged. |
| `calibration/gmm.py` | complete | STEER-style GMM calibrator. |
| `calibration/conformal.py` | complete | Split conformal calibrator. |
| `calibration/question_conditional.py` | complete | Cluster-conditioned conformal wrapper. |
| `calibration/collect.py` | complete | Strategy C and strategy B agreement collection implemented. |
| `routing/pipeline.py` | complete | Synchronous stepwise routing pipeline. |
| `data/loaders.py` | complete | Local/offline dataset loading only. |
| `eval/metrics.py` | complete | Aggregation utilities. |
| `eval/answer_check.py` | complete | Math, AIME integer, and GPQA choice checks. |
| `scripts/run_preliminary_study.py` | complete | Resolves config-local paths and uses local datasets. |
| `scripts/run_calibration.py` | complete | Supports outcome propagation and agreement strategy. |
| `scripts/run_inference.py` | complete | Requires a fitted calibrator path. |
| `tests/test_calibration.py` | complete | CPU tests pass. |
| `tests/test_loaders.py` | complete | Local loader and config-path tests pass. |
| `tests/test_signals.py` | complete | CPU tests pass. |

## Completed Task List

### Task 1: implement `VLLMWrapper`

Done in `src/conformal_routing/models/vllm_wrapper.py`.

- Instantiates `vllm.LLM` with prefix caching, tokenizer path, dtype, model length,
  tensor parallelism, GPU memory utilization, seed, and trust-remote-code settings.
- Defaults to offline mode and sets:
  - `HF_HUB_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_DATASETS_OFFLINE=1`
- Requires local model/tokenizer directories by default.
- Implements:
  - `probe_first_token`
  - `generate_step`
  - `generate_full`
  - `sample_prefixes`
- Supports forced prefix token ids for probe reuse.
- Supports tokenizer `apply_chat_template` continuation rendering, optional
  `chat_template_path`, and explicit reasoning prefill such as `<think>`.

### Remote Qwen3 LLM support

Done in `src/conformal_routing/models/openai_compatible.py`.

- Uses a remote OpenAI-compatible vLLM `/v1/completions` endpoint.
- Uses a local tokenizer path to render Qwen3 chat-template prompts.
- Select it in config with:

```yaml
large_model:
  backend: openai_compatible
  api_base_url: http://127.0.0.1:8000
  api_model: Qwen3-14B
  tokenizer_name_or_path: models/Qwen3-14B
```

### Task 2: fix data loading

Done in `src/conformal_routing/data/loaders.py`.

- All benchmark loaders read local files only.
- No remote dataset-loader calls remain in source, scripts, configs, or tests.
- Supported formats:
  - `.jsonl`
  - `.json`
  - `.csv`
  - `.tsv`
  - `.parquet`
  - HuggingFace `Dataset.save_to_disk` directories
- Configs use `dataset_paths` to point to local files.
- GPQA multiple-choice prompts are built deterministically with a seeded RNG.

### Task 3: optimize self-consistency

Done in `src/conformal_routing/signals/self_consistency.py` and
`src/conformal_routing/models/base.py`.

- `ModelWrapper.sample_prefixes` provides a loop fallback.
- `VLLMWrapper.sample_prefixes` uses one vLLM request with `SamplingParams(n=N)`.
- `SelfConsistencySignal.extract` calls the helper directly.

### Task 4: implement agreement collection

Done in `src/conformal_routing/calibration/collect.py`.

- Runs small model step-by-step and records per-step signal scores.
- Runs large model end-to-end.
- Extracts the large model reference answer.
- Labels every small-model step with small/large final-answer agreement.
- `scripts/run_calibration.py` can select it with:

```yaml
calibration_strategy: agreement
```

The default remains:

```yaml
calibration_strategy: outcome_propagation
```

## Offline Server Checklist

Default config layout:

```text
models/
  DeepSeek-R1-Distill-Qwen-1.5B/
  DeepSeek-R1-Distill-Qwen-32B/
data/
  aime24.jsonl
  aime25.parquet
  math500.jsonl
  gpqa_diamond.jsonl
  livecodebench_lite.jsonl
```

If your server uses a different layout, edit `dataset_paths`, `model_name_or_path`,
and `tokenizer_name_or_path` in `configs/default.yaml`. Absolute paths are supported.
`configs/preliminary.yaml` and `configs/conformal_aime.yaml` remain as specialized
examples, but the scripts now default to the global config.

For the DeepSeek-1.5B local SLM + remote Qwen3-14B LLM MATH smoke test, start from:

```text
configs/math_remote_qwen3.yaml
```

## Pre-flight Checks

Run these before a GPU experiment:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/
PYTHONPATH=src python scripts/check_offline_assets.py
```

If the server has no internet access, install dependencies from a prepared local
wheelhouse or prebuilt environment before running the commands above.

## Recommended First Run

```bash
# 1. Preliminary study.
python scripts/run_preliminary_study.py

# 2. Inspect outputs/preliminary/summary.json and choose the best signal.
#    Then update configs/default.yaml -> signal.name / signal.kwargs.

# 3. Fit calibrators on the calibration split.
python scripts/run_calibration.py

# 4. Run inference for each fitted calibrator.
for cal in outputs/calibrators/conformal_aime_*_*.pkl; do
    python scripts/run_inference.py \
        --calibrator "$cal"
done
```

PowerShell equivalent:

```powershell
Get-ChildItem outputs/calibrators/conformal_aime_*_*.pkl | ForEach-Object {
    python scripts/run_inference.py `
        --calibrator $_.FullName
}
```

Inference writes JSONL traces and summary JSON files under `outputs/inference/`.

## Verification Snapshot

Last local CPU verification:

```text
python -m compileall src scripts tests
PYTHONPATH=src pytest tests/
12 passed
```

Also checked with ripgrep: no remote dataset/model identifiers or remote
dataset-loader calls remain in source, scripts, configs, README, or tests.

## Things Codex Should Not Do

- Do not change the calibrator API. GMM, Conformal, and QCond all share
  `fit/decide/confidence`.
- Do not add training-based methods. This framework is intentionally
  training-free.
- Do not replace conformal calibration with a learned classifier.
- Do not refactor `routing/pipeline.py` to async/distributed unless explicitly
  requested.
