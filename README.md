# Conformal Routing for Cost-Efficient Reasoning

A pluggable framework for stepwise routing between small and large reasoning
models. It can reproduce STEER-style confidence routing and run conformal
calibration on top of step-level confidence signals.

The current code is designed for an offline GPU server: models and benchmark
data are loaded from local paths only.

## Design

```text
question + history
    |
    v
SignalExtractor
    -> scalar score per upcoming step
    |
    v
Calibrator
    -> route decision: small or large
    |
    v
Step generation
    -> append step to history and continue
```

Implemented signals:

- `h_init`
- `logit_confidence`
- `self_consistency`

Implemented calibrators:

- `gmm`
- `conformal`
- `qcond`

## Module Layout

```text
src/conformal_routing/
  models/       vLLM wrapper and model interfaces
  signals/      step-level confidence signals
  calibration/  GMM, conformal, qcond, and data collection
  routing/      stepwise routing pipeline
  data/         local/offline benchmark loaders
  eval/         answer checking and aggregate metrics
scripts/
  check_offline_assets.py
  run_preliminary_study.py
  run_calibration.py
  run_inference.py
configs/
  default.yaml
  preliminary.yaml
  conformal_aime.yaml
```

## Configuration

The main config is:

```text
configs/default.yaml
```

It is the recommended single place to edit paths and vLLM defaults. The scripts
use it automatically when `--config` is omitted. The older
`configs/preliminary.yaml` and `configs/conformal_aime.yaml` remain available as
specialized examples.

Fields you usually need to modify on the server:

- `dataset_paths.*`
- `small_model.model_name_or_path`
- `small_model.tokenizer_name_or_path`
- `large_model.model_name_or_path`
- `large_model.tokenizer_name_or_path`
- `small_model.gpu_memory_utilization`
- `large_model.gpu_memory_utilization`
- `large_model.tensor_parallel_size`
- `large_model.api_base_url` if using a remote OpenAI-compatible vLLM server

Global vLLM defaults live under `vllm_defaults` and are merged into both models:

```yaml
runtime:
  model_backend: vllm
  offline: true

vllm_defaults:
  dtype: bfloat16
  max_model_len: 32768
  tensor_parallel_size: 1
  enable_prefix_caching: true
  seed: 42
  probe_logprobs: 200
  trust_remote_code: true
  offline: true
  require_local_model: true
  use_chat_template: true
  assistant_prefix_start: "<think>"
  continue_final_message: true
  add_generation_prompt: true
```

Model-specific fields override these defaults.

For a local DeepSeek-R1-Distill-Qwen-1.5B SLM plus remote Qwen3-14B LLM smoke
test on MATH, use:

```text
configs/math_remote_qwen3.yaml
```

Edit `large_model.api_base_url` and `large_model.tokenizer_name_or_path` before
running. The remote LLM uses an OpenAI-compatible `/v1/completions` endpoint; the
local tokenizer is still needed on the experiment host so prompts can be rendered
with Qwen3's chat template and token counts can be estimated consistently.

## Offline Server Setup

Put local model snapshots and benchmark files on the server. The default config
expects this layout:

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

If your server uses another layout, edit `configs/default.yaml`.

Absolute paths are supported. Relative paths are resolved against the project
root even if you launch scripts from another working directory.

`VLLMWrapper` defaults to:

```yaml
offline: true
require_local_model: true
```

It also sets these environment variables before constructing vLLM:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Supported dataset formats:

- `.jsonl`
- `.json`
- `.csv`
- `.tsv`
- `.parquet`
- HuggingFace `Dataset.save_to_disk` directories

## Install

On a connected machine, prepare the Python environment, wheels, model snapshots,
and dataset files first. On the offline server, install from your local
environment or wheelhouse.

Typical online/dev install:

```bash
pip install -e ".[dev]"
```

Offline wheelhouse-style install example:

```bash
pip install --no-index --find-links /path/to/wheels -e ".[dev]"
```

## Verify

Run CPU tests before launching GPU jobs:

```bash
PYTHONPATH=src pytest tests/
```

Windows PowerShell equivalent:

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests
```

Expected local result at the time of this update:

```text
12 passed
```

Check that the config points to existing local model and dataset assets:

```bash
PYTHONPATH=src python scripts/check_offline_assets.py
```

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python scripts/check_offline_assets.py
```

## Run Preliminary Study

This measures which signal best predicts small-model correctness on the
calibration split.

```bash
python scripts/run_preliminary_study.py
```

Output:

```text
outputs/preliminary/summary.json
outputs/preliminary/raw_<signal>.npz
```

After this, inspect `outputs/preliminary/summary.json` and update
`configs/default.yaml`:

```yaml
signal:
  name: logit_confidence
  kwargs:
    mode: max_prob
```

or choose `self_consistency` / `h_init` depending on the result.

## Fit Calibrators

Default calibration strategy is outcome propagation:

```bash
python scripts/run_calibration.py
```

To use small/large agreement labels instead, add this to
`configs/default.yaml`:

```yaml
calibration_strategy: agreement
```

Calibration writes fitted `.pkl` files under:

```text
outputs/calibrators/
```

## Run Inference

`run_inference.py` requires one fitted calibrator path.

Single calibrator:

```bash
python scripts/run_inference.py \
    --calibrator outputs/calibrators/conformal_aime_conformal_alpha010.pkl
```

All fitted calibrators on Linux/bash:

```bash
for cal in outputs/calibrators/conformal_aime_*_*.pkl; do
    python scripts/run_inference.py \
        --calibrator "$cal"
done
```

PowerShell:

```powershell
Get-ChildItem outputs/calibrators/conformal_aime_*_*.pkl | ForEach-Object {
    python scripts/run_inference.py `
        --calibrator $_.FullName
}
```

Inference writes per-question JSONL traces and summary JSON files under:

```text
outputs/inference/
```

## Output Interpretation

Each inference summary includes:

- `pass_at_1`
- `mean_flops`
- `mean_latency_s`
- `mean_intervention_rate`

For a Pareto plot, use `mean_flops` on the x-axis and `pass_at_1` on the y-axis,
one point per fitted calibrator summary JSON.

## Notes

- Aggregation happens inside `scripts/run_inference.py`.
- The calibrator API is shared across GMM, Conformal, and QCond:
  `fit/decide/confidence`.
- The framework is intentionally training-free.
