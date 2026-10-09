# T02 checkpoints — 9 October 2026; independent review pending

Implementation is isolated in `t02-checkpoints`, branch `codex/t02-checkpoints`,
based on reviewed integrated main `b26cf38f04a09fd04283023795c0752753ce3106`.
The primary main checkout and its uncommitted planning edits are preserved.
No subagent or independent reviewer was used; automated verification is not
independent review acceptance. T02's historical checkbox remains unchecked.

## Shared format and API

`chess_rl.training.checkpoints` owns save, inference load and training restore.
The common validator rejects unknown/missing fields, unsupported versions and
incompatible configurations with `CheckpointError` (a `ValueError`). Deserialization
uses `torch.load(..., weights_only=True, map_location="cpu")`; it does not import
serialized model/optimizer objects. No second checkpoint reader is introduced.

`m06-checkpoint-v1` stores the accepted combined M06 architecture version,
independent policy/value head widths, dropout, relative terms enabled, dtype and
CPU backend. Encoding version, move vocabulary, WDL order and ending policy come
from shared `AdapterConfig`. The state includes both heads, all eight GAB/Q/K/V
blocks, caller's nonempty JSON provenance, checkpoint identity, model mode,
Adam defaults and each parameter group's effective options/named membership,
moments/steps, loss weights, step/epoch/sample counts, data ordering and RNG.

Model construction occurs inside `torch.random.fork_rng`, so save/load validation
and inference leave the caller's dropout stream untouched. Saving rejects mixed
model modes, modified module settings/geometry and inconsistent dropout. Loading
checks every state key, shape, dtype and finite value. Contradictory duplicated
state-dict paths for the shared GAB bank fail explicitly; normal copying into the
constructed network retains one bank object. Adam membership lists each unique
parameter exactly once, including when groups have distinct options or order.
Adam moments, AMSGrad maxima and step counts must agree with model/progress.

`save_checkpoint(path, model=..., optimizer=..., checkpoint_id=..., provenance=...,
weights=..., progress=..., data_order=...)` validates the entire payload before
publication. A temporary file is flushed/fsynced, published with an atomic
no-clobber hard link, then the directory is fsynced. Existing files are refused;
temporary files are removed on success/failure. The caller must use a quiescent,
completed optimizer-step boundary. Previous step gradients are not saved because
the shared `supervised_step` clears them before the next backward.

`load_inference(path, expected_config=None)` returns an evaluation-mode model,
construction config, checkpoint identity, contracts and provenance. It validates
the whole common format without restoring training RNG, mode, progress or sampler
to the caller. The inference API does not require a matching training runtime.
It is a checkpoint loader only; T04's terminal partitioning/batched evaluator is
not implemented here.

`restore_training(path, expected_config=..., expected_provenance=..., dataset=...,
source_id=..., expected_weights=None, expected_optimizer=None)` reconstructs the
model and Adam, verifies expectations/runtime/data, then installs Python, NumPy
and CPU Torch random states last. It returns training mode, weights, progress and
ordering. Failure leaves the caller's RNG and existing objects unchanged. An
explicit `ModelConfig` and exact provenance are required. Optional expected loss
weights and optimizer configuration reject caller-level configuration drift;
optimizer expectations have `kind`, `defaults`, and `groups` fields matching the
saved metadata, excluding tensor state. Callers must explicitly configure the
recorded runtime before restoration; thread or backend settings are not changed
implicitly.

## Data and random state boundary

`DataOrder` records a complete permutation of the training split, completed-batch
cursor, batch size, drop-last flag, epoch, source identity and SHA-256 of source-order
game/ply membership. `reference_hash(dataset)` derives membership through D07;
`validate_dataset` requires the D07 train dataset, matching source identity, size
and membership order. The source ID must identify the already verified source
bytes (the real check uses the label-file SHA-256). `next_indices()` returns the
next batch's indices, retaining partial batches or respecting drop-last. D07 still
owns batching/replay; checkpoint code does not implement an epoch/training loop.

The explicit permutation and cursor reproduce the current next batch even after
shuffling. Global Python and NumPy MT19937 state, including cached normal draws,
and Torch CPU state reproduce dropout and future random ordering. There is no
worker prefetch, asynchronous sampler, custom `Generator`/`numpy.random.Generator`,
scheduler, gradient accumulation, clipping or accelerator state in this supported
boundary. Callers needing those must extend the versioned contract before using
them; their state cannot be silently assumed resumable. Only standard CPU Adam
(including AMSGrad and multiple groups) is supported; SGD/AdamW/fused/capturable/
differentiable configurations are rejected. CPU float32/float64 model storage is
supported. Cross-runtime deterministic resumption is rejected; inference may load
the weights without matching the saved runtime.

## Verification

The bounded real check (`scripts/check_t02.py`) opens the accepted D05 tiny label
bundle through D07, chooses an explicit NumPy permutation with seed 20261009, uses
the accepted default M06 widths and dropout **0.1**, weights **0.7/1.3**, and Adam
with lr **0.0001**, betas **0.9/0.999**, eps **1e-8**, no weight decay/AMSGrad and
`foreach=False`, `fused=False`. One real warmup update establishes optimizer state;
one continuation update is the verification target. The resumed branch starts a
new Python process, scrambles all RNGs, reopens the audited data, and restores.

The uninterrupted and resumed reports are **byte-identical**, SHA-256
`9ea631482c2816e2045cb14d9cbb8a082df29225eba546e7cfd71a0fab3cba05`.
Every compared tensor is hashed from exact bytes with shape/dtype included, with
zero numerical tolerance. This covers the next batch/references, raw evaluation
policy/WDL logits and probabilities, train-mode forward outputs, all **215**
gradients/parameter tensors/Adam states, step 2, loss metrics, subsequent RNG,
post-update evaluation predictions and shared bank identity. **214** parameter
tensors change on the continuation step. The total loss is
`3.6107428073883057` (policy `2.8774099349975586`, WDL `1.228119969367981`).
The checkpoint's 60,871,811 bytes have SHA-256
`9f3ecddc1237f56aac1eb69c5c5fd87d442c937b23d591adcfed5df50853fefc`.

The external checkpoint and full per-tensor reports are retained at
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/external-artifacts/t02-cpu-continuation`.
They are correctness artifacts, not P or a selected playing checkpoint. No large
weights or data are committed. The tracked `runs/t02` evidence indexes exact
commands, environment overrides, source hashes, timings, exits and lossless
base64 output containers. The real-check summary/warmup and durable artifact
index are retained there for independent review.

The supported verification runtime is Python **3.12.12**, Torch **2.8.0**, NumPy
**2.5.3**, macOS 14.5 arm64 CPU float32, one intra-op and inter-op thread, with
deterministic algorithms enabled. No backend nondeterminism was observed in this
check. Accelerator continuation, other CPU runtimes and playing strength are
unverified. Float64 and nondefault model/Adam settings receive unit coverage.

Full regression results and preservation hashes are recorded in `runs/t02/summary.json`.
No T03, T04, overfit, held-out training, pilot training, P selection, self-play or
RL work was begun. Work stops at the implementation/evidence commit for independent
review; primary main is not advanced.
