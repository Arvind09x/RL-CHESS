# Explicit single-GPU CUDA preparation — 10 October 2026

Independent review pending. Isolated branch `codex/cuda-support` starts at accepted
combined T03/T04 main `ea0f8a1`. The primary main checkout and its uncommitted
planning edits are preserved. No applicable AGENTS.md was found in the source,
isolated checkout or inspected ancestors. The shared interface, accepted M05/M06
model and T01/T02/T04 contracts govern this change. No self-play execution or
game-record module is edited. This prepares a backend; it selects no P checkpoint.

## Audit and scope

The accepted network constructors, registered relative-index buffers, attention
operations, T01 losses and D07 `SupervisedBatch.to_torch(device=...)` already carry
device information. T01 takes its execution device/dtype from model parameters.
No backbone, head, encoding, target, trainer or search API rewrite is needed.

The actual blockers at the accepted base were:

| Blocker | Resolution |
|---|---|
| T02 `ModelConfig` and live model validation permit CPU only; reconstruction always uses CPU | Explicit `cuda:0` float32 configuration; construct under the CPU RNG fork, then copy the complete model to CUDA |
| T02 only records CPU Torch RNG and CPU runtime; validators require CPU saved tensors | Versioned CUDA extension with CPU snapshots, CUDA RNG and runtime metadata; reconstruct/validate serialized state on CPU before device placement |
| T04 creates CPU features/masks and calls NumPy directly on inference tensors | Both inputs follow the model device; normalization stays there; transfer float32 probabilities to CPU before the existing owned-array result constructors |
| Linux Torch is deliberately resolved from the CPU index in pyproject/uv.lock | Separate CUDA dependency input, hash lock and setup; original pins and environments remain unchanged |
| Frozen scripts/settings/evidence reference absolute `/private/tmp` artifact paths | A separate path configuration and byte-copy staging utility; D07 already accepts explicit paths and verifies content identities |

Default CPU construction, save schema and resume behavior remain v1. Existing
CPU checkpoints load through the same reader. CPU float64 remains supported.
CUDA is limited to one visible device, addressed as `cuda:0`, and float32.
Set `CUDA_VISIBLE_DEVICES` before process startup. Zero or multiple visible
devices fail explicitly; there is no fallback. Mixed devices, AMP, MPS,
schedulers, accumulation, worker prefetch, custom generators, fused/capturable/
differentiable Adam and distributed execution are outside this contract.

## Shared APIs and checkpoint extension

`ModelConfig(device="cuda:0").build()` builds the unchanged combined architecture.
`load_inference(path, device="cuda:0")` and
`RealEvaluator(path, device="cuda:0")` add a keyword with default `"cpu"`.
`evaluate(states)` is unchanged. `expected_config` compares the **stored**
configuration; returned inference `config.device` describes the **execution**
device. The default can read either supported checkpoint version on CPU.
CUDA inference from a CPU float64 checkpoint fails rather than converting its
precision silently. Transfer float32 CPU checkpoint weights for inference or
explicitly start a fresh CUDA optimizer/order/RNG boundary for training.
Cross-backend training restore is rejected; CPU RNG cannot stand in for CUDA RNG.

`m06-checkpoint-v1` keeps its exact schema and CPU backend. CUDA saves use
`m06-checkpoint-cuda-v2`; v2 requires stored `device="cuda:0"` and
`dtype="float32"`. All other top-level fields retain T02's schema/meaning.
The new training fields are:

| Field | Meaning |
|---|---|
| `training.rng.cuda` | Exactly `{device: "cuda:0", state: CPU uint8 vector}`, the default device generator state |
| `training.cuda_runtime` | CUDA/cuDNN versions, GPU name/capability/memory, visible device count, TF32 flags, cuDNN benchmark/determinism and cuBLAS workspace configuration |

Python, NumPy and CPU Torch RNG remain present. CUDA model weights and Adam
tensor state are detached, independently cloned to CPU storage before publication.
The shared safe `weights_only=True, map_location="cpu"` reader validates the
entire payload without requiring a GPU for inference. GPU RNG receives structural
validation on CPU; restoring CUDA also validates it with an independent CUDA
generator before installing caller state.

Save at a quiescent completed optimizer step, on the default execution stream,
after all updates, with a completed D07 batch cursor and no in-flight mutation.
CUDA save synchronizes the sole device. Standard Adam requires
`foreach=False, fused=False` explicitly, with noncapturable scalar `step`
tensors on CPU and moments/AMSGrad maxima on the model device. Live placement
is checked before snapshotting. Restore creates the model on CUDA before loading
Adam so moments follow parameters while scalar steps stay on CPU. Shared GAB
identity and unique optimizer membership retain the original checks.

Resume requires the exact recorded model, provenance, data membership/order,
progress, loss/optimizer configuration and CPU/CUDA runtime fingerprints.
The caller configures one thread, strict deterministic algorithms, float32
default dtype, highest matmul precision, both TF32 flags disabled,
cuDNN benchmark disabled/deterministic enabled, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launch. Settings are never changed
implicitly by loaders. Successful restore installs saved RNG last; failures
preserve caller RNG. No pending gradients are serialized because the shared
step clears them before the next backward.

This boundary covers the recorded device/runtime class and explicit
single-process ordering. It does not promise bitwise agreement across CPU and
CUDA, across machines/drivers/releases, or for arbitrary asynchronous work.
The hardware probe records the driver separately with `nvidia-smi`.
PyTorch itself documents reproducibility limits across platforms/releases:
[reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

## Evaluator behavior

Adjudication still precedes encoding. Empty/all-terminal calls bypass model work;
nonterminals share one forward, and results restore input order. Features and
boolean masks follow the execution device; features retain the model dtype.
Shared legal masking gives exact illegal zeros and one legal normalization.
WDL keeps root-side `[win,draw,loss]`; value is calculated from returned float32
`P(win)-P(loss)`. Each result owns independent CPU float32 arrays. State/history
and caller Python/NumPy/CPU/CUDA RNG are preserved. The evaluator remains serial;
no batching queue or concurrent traversal is added.

## Separate runtime and portable data

`runtime/cuda/setup.sh` targets Linux x86_64/glibc 2.28+, CPython 3.12.12 and
uv 0.12.10. It creates a fresh `.venv-cuda-cu126`, refuses an existing destination,
and hash-verifies the separate 33-package lock. Torch is pinned to
`2.8.0+cu126` from the official CUDA 12.6 index, with pinned CUDA dependencies
and chess 1.11.2. Source-build tooling is constrained separately. The lock was
resolved locally for the target platform; its Linux installation and CUDA driver
compatibility await an authorized host. No CPU environment was installed into.
The selected wheel backend follows the
[official PyTorch 2.8 instructions](https://pytorch.org/get-started/previous-versions/#v280).

The retained regeneration command is:

```sh
uv pip compile runtime/cuda/requirements.in --torch-backend cu126 \
  --python-version 3.12.12 --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --emit-index-url --emit-index-annotation --no-sources \
  --cache-dir /private/tmp/rl-chess-cuda-uv-cache \
  --output-file runtime/cuda/requirements.lock
```

The initial sandbox network attempt failed DNS; authorized resolution succeeded.
A wheel-only exploratory resolution selected chess 1.10.0; the final input
explicitly pins accepted chess 1.11.2 and allows its hashed source distribution.
Neither exploratory result was installed.

`scripts/stage_cuda_data.py` validates a source through D07, copies unchanged
bundle/manifest/game/audit/registry/teacher-evidence bytes, records SHA-256 and
size before/after, and reopens the copies through D07. It emits relative
`data-config.json` beside a separate `staging-evidence.json`. Original
serialized path strings remain historical provenance, never execution lookup
paths. No settings/manifest/labels or original frozen evidence are reserialized.
The SQLite candidate pool, archive, engine and NNUE executables are not required
for loading completed labels. An optional checkpoint is also copied unchanged.

Local staging was performed with:

```sh
PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python -B \
  scripts/stage_cuda_data.py --config runtime/cuda/local-frozen-data.json \
  --destination "/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/external-artifacts/cuda-support-portable" \
  --checkpoint "/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/external-artifacts/t03-tiny-overfit-20261009/final.pt"
```

Copy the entire staged directory to the CUDA host; its relative configuration
can move intact. This check uses the accepted tiny D05 **training** labels,
not held-out labels or a new annotation. D07 verifies protected registries for
separation; they supply no training targets.

## Bounded hardware commands and frozen tolerances

From this branch on an already authorized host, with compatible NVIDIA driver:

```sh
bash runtime/cuda/setup.sh
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > cuda-driver.txt
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Self-contained D07 synthetic fixture, full accepted model, dropout 0.1.
timeout 600s .venv-cuda-cu126/bin/python -B scripts/check_cuda_support.py \
  --device cuda:0 --directory ./checkpoints/cuda-synthetic-check

# Portable real train-only artifact and existing CPU checkpoint.
# Place the staged directory at ./cuda-data first; use a fresh output directory.
timeout 600s .venv-cuda-cu126/bin/python -B scripts/check_cuda_support.py \
  --device cuda:0 --directory ./checkpoints/cuda-real-check \
  --data-config ./cuda-data/data-config.json \
  --checkpoint ./cuda-data/checkpoint.pt
```

Each command performs one warmup update, immutable checkpoint save/reload,
one uninterrupted next update and the same update resumed in a **fresh process**.
It checks full-model forward/backward, all 215 gradients, model/Adam state,
placement, shared bank, RNG, batch membership, train mode, predictions and
mixed terminal/nonterminal batch-versus-single evaluation. The optional accepted
CPU checkpoint adds inference-only coverage. Each command is bounded to three
optimizer-step executions total, with no fitting loop or training criterion.
Outputs preserve CPU snapshots for per-tensor review. Runtime configuration and
tolerances are written before workers start.

| Comparison | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Save/reload weights/predictions at the same device and batch shape | 0 | 0 |
| CUDA next-update floating outputs/gradients/parameters/Adam state/metrics | 1e-6 | 1e-5 |
| CPU next-update continuation | 0 | 0 |
| Batch versus single priors/WDL/value | 2e-6 | 2e-5 |
| Independent float64 NumPy normalization oracle | 1e-6 | 1e-5 |
| Probability sums | 1e-6 | 0 |
| Illegal zeros, terminal handling, order/history/array ownership, RNG and integer counters | Exact | Exact |

No CPU-versus-CUDA numerical comparison gate or speed/strength claim is made.
Deterministic-kernel errors or a tolerance failure must be investigated and
recorded; these constants must not be widened after observing a failure.

## Local evidence and review boundary

The local host is macOS 14.5 arm64, Torch 2.8.0, with CUDA unavailable and zero
visible devices. No authorized CUDA host was provided or accessible in this
task. **CUDA hardware verification and Linux runtime installation are outstanding.**
CPU checks, fresh-process continuation, real-byte staging, source preservation,
command/output hashes and exact external artifact identities are retained under
`runs/cuda-support`. Large tensor snapshots/checkpoints/staged data remain in
the separate workspace `external-artifacts/cuda-support-*` directories.
Historical evidence remains unchanged.

| Locally executed check | Result |
|---|---|
| Full model regression, historical M05 diagnostic plugin | 1,421 passed |
| Full Torch-free adapter regression | 1,023 passed, 12 expected optional-Torch skips |
| Final focused checkpoint/evaluator/CUDA-schema/staging checks | 120 passed, including subsequently added device rejection/staging cases |
| Staging in the Torch-free environment | 1 passed |
| Historical accepted review probes | 84 passed in each environment |
| M06 numerical checker | Passed original tolerances |
| Frozen adapter replay | 13 fixtures byte-identical in both environments |
| Synthetic and portable real training-data continuation | Exact next-update tensors/metrics/RNG in fresh processes |
| Accepted T03 CPU checkpoint | Every model hash and final metric exact; zero updates |
| CUDA lock replay and setup syntax | Lock payload identical offline; shell syntax passed |
| Preservation | Every other accepted tracked blob, 76 historical external artifacts, main and original user patch unchanged |

The full-suite commands precede the final three added focused cases; all final
cases are covered by the separate 120-case command. Production behavior did not
change after the full regression. The final comparison utility was replayed
against retained continuation tensor reports with zero extra updates.

The final result is committed for independent review. Main is not integrated.
No hardware rental, P01, sustained training, held-out tuning, AMP, scheduler,
distributed work or self-play campaign was performed.
