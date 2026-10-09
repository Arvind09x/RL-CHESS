# T01 supervised losses and one optimizer step — 9 October 2026

T01 is implemented and automated checks pass; independent review is pending.
The isolated clone is `t01-supervised-losses` in the RL Project 2 workspace,
on `codex/t01-supervised-losses`, based on reviewed integrated main
`9565093abe40a7807e3472c1101f5f28c14b7afe`. The accepted M06 and D07 records,
`interface.md` and `coding-plan.md` govern this change. Main is preserved.
D08 is not a prerequisite: the real check uses the existing accepted D05
training labels, without any new annotation.

## API and ownership

`chess_rl.tensor_ops` owns legal normalization and numerical target validation.
`legal_policy_log_probs(logits, legal_mask)` masks before one row-wise
log-softmax, with `-inf` at illegal entries. `legal_policy_probs` exponentiates
that same result, yielding exact illegal zeros. These helpers accept nonempty
floating `[B,N]` logits and matching boolean support; chess losses require the
fixed 4,352 vocabulary. They are reusable by T04 and future distribution-target
losses without importing a trainer, data reader or search implementation.
No distribution-target RL loss or evaluator is implemented here.

`training.losses.policy_class_cross_entropy` takes raw `[B,4352]` logits,
boolean legal masks and integer `int64 [B]` classes. It gathers only the
teacher class and returns mean negative legal log probability. Classes never
become a stored/dense one-hot policy. Every illegal logit has exactly zero
gradient, including illegal scores larger than all legal scores.

`training.losses.soft_wdl_cross_entropy` takes raw `[B,3]` logits and floating
soft targets in shared root-side `[win,draw,loss]` order. It returns mean
`-sum(target * log_probability)` without argmax, perspective reversal,
clamping or target renormalization. Zero targets select zero log probability
before multiplication, preventing `0 * -inf` NaNs. The unit-sum check permits
absolute float32 rounding error up to `1e-6`, without changing target values.

Both losses reject empty batches, malformed shapes/dtypes/devices, nonfinite
raw inputs and invalid targets. Policy rejects empty legal support, illegal
classes and out-of-range IDs. WDL rejects negative/greater-than-one entries,
non-unit sums and target tensors requiring gradients. Finite extreme logits
through `1e30` pass float32/float64 numerical checks. If legal log probabilities
or a positive-mass loss exceed the dtype range, the call fails explicitly.
WDL `[3e38,-3e38,-3e38]` with target `[1,0,0]` produces finite zero loss and
zero gradients despite negative-infinite zero-mass log probabilities.

`training.trainer.supervised_step(model, optimizer, batch, weights=LossWeights())`
consumes a training-split `SupervisedBatch` from the accepted D07 loader.
D07 owns full replay, provenance and terminal rejection through the shared
adapter. Its existing focused tests reject checkmate, insufficient material,
actual repetition and reached fifty-move draws, including draws with legal
moves. A legal mask alone cannot establish nontermination; tensor-only helpers
and the step do not reconstruct a state from features or infer rule history.
Manually constructing a batch bypasses D07's provenance/terminal gate and is
appropriate only for controlled nonterminal tests.

The step validates the D07 shapes/dtypes, finite features, legal classes and
soft WDL before forward, requires the M06 combined network and exactly one
optimizer entry per unique model parameter, switches to training mode, clears
gradients, computes both losses, checks the weighted total, backpropagates,
checks all parameter gradients, and calls one optimizer step. It reports
pre-update losses and verifies finite resulting parameters. Input, loss or
gradient failure prevents the update; backward failure clears partial gradients.
An arbitrary optimizer failure is not transactional and has no rollback promise.

The conversion uses D07's `batch.to_torch`, which allocates independent storage.
Batch arrays, integer classes, soft targets and references remain unchanged;
fresh accesses reproduce the same adapter features and masks. No new encoding,
legality, batching, history or terminal implementation is added. Training code
does not select seeds, devices, threads or optimizer defaults implicitly.

## Recorded correctness run

[Configuration](../runs/t01/config.json) was saved before execution.
It fixes Adam with learning rate `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`,
weight decay 0, AMSGrad/foreach/fused disabled, policy/WDL weights `1/1`,
seed `20261009`, batch size 2 and exactly one optimizer step. There is no
scheduler or gradient clipping. CPU float32 uses one Torch compute thread,
one interop thread and deterministic algorithms; OMP/MKL/OpenBLAS thread
environment overrides are 1. Python is 3.12.12, Torch 2.8.0 and NumPy 2.5.3
on macOS 14.5 arm64. The fixed model retains eight layers, width 256, eight
heads, FFN 256, shared 8/32/32 GAB, every Q/K/V relative term, independent
head widths 128, 5,049,411 parameters and the accepted training dropout 0.1.

The D07 loader verifies all 12 `runs/d05/tiny-final` training records. Only its
first source-order batch is used for the real update: game
`g:000025602b7ecc9da7677d8baf42cad26dae7815ec1145a18af1c2792cbebb34`,
plies 5 and 9, teacher classes 405 and 1244. Their soft WDL targets are
`[.002,.745,.253]` and `[.001,.577,.422]`, retaining float32 rounding and
the root player's perspective. Held-out samples never enter the check.

| Loss | Step output | Independent NumPy float64 | Absolute difference |
|---|---:|---:|---:|
| Legal policy CE | 3.129796028137207 | 3.1297959231127015 | 1.0502451e-7 |
| Soft WDL CE | 1.1292858123779297 | 1.1292858515220288 | 3.9144099e-8 |

Weighted total is 4.259081840515137. Forward outputs, both losses, all 215
parameter gradients and updated parameters are finite. Exactly 214 parameter
tensors change; the promotion adjustment weight has zero gradient because
this batch has no legal promotions. All 24 relative tables, both heads and
the one shared bank receive gradients and update. The bank is included once
in the optimizer; every state records step 1. Every parameter entry agrees
with the independent first-step Adam formula within atol/rtol `1e-7/1e-6`.
[Full real-batch evidence](../runs/t01/real-batch.json) retains references,
soft targets, per-parameter gradient/update norms and unchanged-input checks.
No post-update fitting criterion is used; this proves update plumbing only.

## Numerical and regression evidence

The independent oracle uses explicit NumPy float64 row selection, exponentials,
log normalizers and analytic gradients. It calls no production helper or Torch
normalization/loss operation. Tests compare every policy/WDL gradient entry,
not just scalar losses. They cover asymmetric/noncontiguous policy inputs,
batches 1/3, float32/float64, extreme scores, illegal dominating logits,
single legal classes, zero soft-target probabilities and invalid/nonfinite
inputs. Loss/gradient comparisons use atol `2e-6`/`2e-7` and rtol `3e-6`.
Controlled adapter batches also verify weighted losses, an independent SGD
update, bank sharing, all relative-table gradients, unchanged arrays and
failure paths that never call the optimizer.

| Executed check | Result |
|---|---|
| Focused T01 and D07 tests | 219 passed (92 T01, 127 D07) |
| Full model regression, documented historical M05 diagnostic plugin | 1,273 passed |
| Full adapter regression with Torch absent | 1,012 passed, eight expected optional-Torch skips |
| Existing independent reviewer probes | 84 passed |
| Accepted M06 numerical checker | Passed |
| Frozen adapter replay in both environments | All 13 fixtures match the accepted report |
| Real D05/D07 batch | One finite combined-model Adam update; independent losses/gradients/update pass |

Exact argv, execution settings, UTC timestamps, exit codes, source hashes and
lossless stdout/stderr are under `runs/t01/*.command.json`. The source hashes
are unchanged during each recorded command. The initial development test had
91 passes and one incorrect assertion requiring every parameter gradient to
be nonzero; it was corrected to permit legitimate promotion-only zeros.
Production loss/step behavior did not require a fix for that test failure.
The initial Git whitespace check flagged the M06 raw stdout's EOF blank line.
Its exact captured bytes, including that line, are archived as a JSON string
by `runs/t01/archive_m06_stdout.py`; the command record identifies both decoded
bytes and container hashes. No output is trimmed or discarded.

The preservation audit compares every accepted Git blob with the working
bytes, permitting only appended README/coding-plan status text. It rehashes
the accepted external data, protected held-out registries, candidate database,
D06 benchmark and teacher bytes against the integration's saved identities,
and checks primary main's clean state and exact base commit. No accepted
implementation, historical evidence, manifest, dependency pin or interface
contract is changed.

T01 stops at this review boundary. No checkpoint, tiny overfit, P checkpoint,
T02, T04, RL, GPU run, new annotation or quality/performance campaign is begun.
CPU float32/float64 numerical correctness and one CPU update are the validated
scope; other devices and precision remain untested.
