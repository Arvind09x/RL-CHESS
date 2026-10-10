# Running on the college GPU machine

The first supported target is Linux x86-64 with one visible NVIDIA GPU. No CUDA result has been established by the local CPU verification. Ask the college administrator for the allocated machine, driver information, storage location, and scheduler instructions. Respect the scheduler's `CUDA_VISIBLE_DEVICES`; select one permitted GPU if using an unscheduled machine with several devices.

## 1. Environment and hardware checks

From a fresh checkout, with `uv` installed:

```sh
bash runtime/cuda/setup.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-cuda-cu126/bin/python scripts/gpu_preflight.py --device cuda:0
mkdir -p artifacts
.venv-cuda-cu126/bin/python scripts/check_cuda_support.py   --device cuda:0 --directory artifacts/gpu-smoke-001
.venv-cuda-cu126/bin/python -m pytest -m cuda tests/test_cuda_pipeline.py
```

Setup creates a user-space environment with pinned Python and hash-locked CUDA packages. It does not install or change the system driver. If preflight rejects the driver, visible-device count, or runtime, resolve that mismatch before training. Use a new smoke directory for each attempt.

Preflight runs the actual M06 network. The separate-process smoke covers supervised update/save/reload/inference; CUDA-marked tests cover mixed updates/resume and self-play. CPU continuation is checked exactly; CUDA comparisons use documented numerical tolerances under the same runtime. AMP, TF32, multi-GPU, and distributed training are outside this initial contract.

## 2. Portable data

Make an ignored `artifacts/local-source.json` following `configs/data.example.json`. Its six fields must point to the authoritative accepted label bundle, manifest, games, split audit, prepared samples/protected registries, and teacher evidence. These source files must already be present. Do not substitute only the labels or omit held-out protection files.

Stage the bytes on the machine holding the data:

```sh
uv run --locked --extra model python scripts/stage_cuda_data.py   --config artifacts/local-source.json --destination artifacts/data
```

The destination must be new. Staging verifies the dataset before and after copying, records per-file hashes, and writes `data-config.json` with relative paths. An optional `--checkpoint PATH` also copies weights. Copy the whole staged folder to the college machine through the college-approved transfer route, then run:

```sh
.venv-cuda-cu126/bin/python scripts/gpu_preflight.py   --device cuda:0 --data-config artifacts/data/data-config.json
```

Keep backups of the original prepared/protected registries and label bundle on institutional storage. They are separate from the Git checkout. Staging evidence includes source paths and belongs in private artifact storage.

## 3. Bounded supervised pilot

Start with a small run to measure performance and memory:

```sh
.venv-cuda-cu126/bin/python scripts/train_supervised.py   --device cuda:0 --data-config artifacts/data/data-config.json   --output artifacts/pilot-001 --steps 100 --batch-size 16 --validation-every 50
```

Resume at a completed update boundary into a fresh directory, keeping data, backend, seed, batch size, learning rate, and runtime unchanged:

```sh
.venv-cuda-cu126/bin/python scripts/train_supervised.py   --device cuda:0 --data-config artifacts/data/data-config.json   --resume artifacts/pilot-001/step-000100.pt --output artifacts/pilot-002   --steps 200 --batch-size 16 --validation-every 50
```

`--steps` is the target total count including restored progress. Metrics and summary contain training/validation results, elapsed time, and peak GPU allocation. Throughput can be calculated from samples processed and elapsed time. Validation checkpoints are immutable. The best checkpoint in a resumed directory means best within that segment, so compare validation records across segments before selecting P. The reserved test split is not opened.

The reported training samples per second includes validation and checkpoint time, making it a conservative whole-run measurement.

Checkpoint inference can move from GPU to CPU; optimizer continuation requires the original backend/runtime. A separate explicit library `runtime_policy="compatible"` allows OS descriptions and Python patch versions to differ, with a warning that bitwise continuation is unverified. Normal commands use exact restoration.

## 4. Search and reinforcement-learning pilot

`run_r01.py` accepts `--device cuda:0`. A creation plan's checkpoint path resolves relative to the plan file. CLI sessions write an immutable move journal plus a small boundary record; recovery verifies every referenced move file. Resume can supply `--checkpoint PATH` when the original weights moved, but their bytes and identity must match.

```sh
.venv-cuda-cu126/bin/python scripts/run_r01.py --help
```

The shared mixed updater and checkpoint schema support CUDA. A full P03 learning campaign, batched independent games, R-versus-P evaluation, and Gate P acceptance are still subsequent project work. First measure batch-of-one search on the allocated GPU; faster supervised training alone does not establish faster self-play.

Keep datasets, checkpoints, logs, and runs under ignored `artifacts/` or institutional storage. Commit code, configurations without private paths, tests, and compact experiment summaries only.
