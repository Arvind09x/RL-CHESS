# M05 intermediate GAB network automated acceptance — 8 October 2026

M05 only passes automated acceptance and stops for independent review. M04
commit `0fdbd1eaf5e4d1701a2627d55f87b2150a4f252a` has user-supplied
independent **AI review acceptance**, recorded in the appended
[M04 entry](m04-wdl.md) and [structured record](../runs/m05/m04-review-acceptance.json).
No human teammate review is claimed. M06 remains required before supervised/RL
training, the real evaluator and the combined-model pilot.

## Public operation and configuration

`ChessNetwork(*, d_policy=128, d_value=128, dropout=0.1, device="cpu",
dtype=torch.float32)` consumes `[B,64,128]` features and returns the tuple
`(policy_logits, wdl_logits)`, raw `[B,4352]` and `[B,3]` respectively.
`B >= 1`. The wrapper calls the existing `EncoderBackbone` once and passes
its exact encoder-final `[B,64,256]` tensor object to both existing heads.
It has no additional parameters, reset, token transform or copied components.
All three components receive the same factory device/dtype. Head widths remain
independent; positive-integer validation and dropout validation are delegated
to the accepted components. Invalid feature shapes retain backbone validation.

Pair ordering/scaling remains pair4352-v1; `wdl_order` and the head refer to
the same shared `WDL_ORDER = (win, draw, loss)`, from each input position's
side to move. The WDL pooled LayerNorm is a different module and parameter set
from the encoder-final token norm. The backbone retains eight independent
blocks/generators, dimensions 8/32/32, one shared bank, and all constructor
defaults documented in [M02](m02-backbone.md), [M03](m03-policy.md) and
[M04](m04-wdl.md). Those implementations/contracts are byte-identical to M04.
Their pinned attribution/notices remain in place; no new upstream port is added.

`.train()`/`.eval()` propagate to every descendant. Training dropout remains
.1 at all 24 explicit modules and eight attention-probability sites; heads add
no dropout. `dropout=0` is a numerical-check override. Build optimizers from
`network.parameters()` so the shared bank is deduplicated. Move the whole
network together; a float32→float64 move retains shared parameter identity.
Inputs must match the model's device/dtype. No checkpoint/schema is introduced.

An exploratory empty-batch test reproduced the accepted PyTorch MHA's ambiguous
zero-element mask reshape. M05 now rejects `[0,64,128]` clearly with ValueError
and leaves the backbone unchanged. [Exploratory command/result](../runs/m05/exploratory-check.json)
records the initial 34 passes and this failure. This tensor API's nonempty
requirement does not implement or revise the future evaluator's empty-state API.

## Independent joint checks

Six default-width cases cover batches 1/2/5 × CPU float32/float64, plus two
batch-2 cases with independent widths `d_policy=7, d_value=11`. Features are
asymmetric/noncontiguous; all parameters, positions and norm affine terms are
nontrivial. ReLU has active/inactive coordinates away from zero. Seeds, exact
objectives and configuration are saved in [network acceptance](../runs/m05/network-acceptance.json).

The accepted independent M02 reference supplies explicit affine operations,
per-head attention, manual normalization and erf GELU. The existing M03 reference
uses depth-wise scalar products and explicit policy slot lists. The existing
M04 NumPy float64 oracle supplies manual WDL forward and chain-rule gradients.
Its final-token cotangent joins the policy objective through the independent
backbone. No reference calls a production forward, MHA/SDPA, Linear or LayerNorm.
Outputs, input gradients and every element of all **191** unique parameter
gradient tensors agree for the combined objective. Inputs and parameters remain
unchanged; all gradient tensors are finite/nonzero (individual entries may be zero).

| Dtype | atol / rtol | Max policy error | Max WDL error | Max joint gradient error |
|---|---|---|---|---|
| float32 | 3e-5 / 5e-4 | 2.384185791015625e-6 | 7.450580596923828e-8 | 3.159046173095703e-6 |
| float64 | 2e-11 / 2e-10 | 6.161737786669619e-15 | 1.5265566588595902e-16 | 8.43769498715119e-15 |

These are the accepted backbone tolerances applied to the complete composition;
the head-specific tighter checks are independently rerun under M03/M04.
Hooks prove one backbone evaluation, exact final-norm output identity at both
head inputs, and no token mutation. Separate policy/WDL backward calculations
each contribute nonzero gradients through input, final tokens, every backbone
parameter and the shared bank; their sum agrees with joint backward. Each head
receives gradients from its own branch. Eight private generator sets have ten
tensors each and are pairwise disjoint. SGD construction sees 191 unique
parameters and exactly one bank entry; no optimizer step/training run occurs.

All 13 frozen real adapter fixtures pass as one batch in each dtype. Both outputs
and input/all-parameter gradients are finite/nonzero. Board identity, every
retained historical prefix, provenance and re-encoded NumPy features are unchanged.
Terminal fixtures are raw tensor probes only. Train outputs differ for both heads
under different dropout seeds, replay exactly with the same seed, and eval outputs
repeat exactly. Zero-dropout train/eval outputs agree exactly. Seeded construction
matches direct construction of all accepted components exactly for both dtypes
and default/independent widths, with no additional reset.

## Deduplicated intermediate parameter and memory accounting

Independently checked using explicit layer-shape arithmetic and the enclosing
network's deduplicated iterator, rather than relying only on component totals:

| Component | Elements | Unique tensors |
|---|---:|---:|
| Accepted backbone, shared bank once | 3,567,168 | 182 |
| Pair policy, width 128 | 66,048 | 3 |
| WDL, width 128 | 33,795 | 6 |
| M05 intermediate total | **3,667,011** | **191** |

General total: `3,567,168 + 516*d_policy + 515 + 260*d_value`.
The bank is 131,072 elements in one parameter, despite repeated registration
paths in the accepted backbone. Do not count repeated state-dictionary paths.

| Dtype | Parameter tensor bytes | Gradient tensor bytes | Combined payload bytes |
|---|---:|---:|---:|
| float32 | 14,668,044 | 14,668,044 | 29,336,088 |
| float64 | 29,336,088 | 29,336,088 | 58,672,176 |

Gradient payload assumes all 191 tensors are allocated with parameter dtype;
the adapter backward checks independently sum their actual allocated bytes.
These exact payload figures exclude input/output, activations, temporary tensors,
optimizer state, allocator metadata/cache, Python and process overhead.

Separately measured **checker process lifetime peak RSS: 819,396,608 bytes**,
via macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss` (native unit bytes).
This process includes imports, all cases, cloned parameters, independent references,
autograd graphs and allocator retention. It is not an isolated model forward/backward
peak, training memory estimate or performance benchmark. No GPU memory was measured.
This remains an intermediate GAB count; M06 must recount the combined architecture.

## Regressions, commands and preservation

[Commands](../runs/m05/commands.json) record argv, cwd, environment overrides,
UTC start times, durations and exit codes. [Runner](../runs/m05/verification-runner.py),
[final audit](../runs/m05/final-audit.py) and [execution context](../runs/m05/execution-context.json)
record the procedure. Local installed interpreters were used with
`PYTHONDONTWRITEBYTECODE=1`, explicit staged `PYTHONPATH`, and pytest
`-p no:cacheprovider`; no installation or network access was needed. Only the
focused changed/new files are archived into the working repository; prior evidence
is never overwritten. Model/checker code tested in staging is copied byte-identically.

| Check | Result |
|---|---|
| M05 checker | Eight numerical cases, joint/routing/sharing checks, two 13-fixture batches, constructors/count/memory/dropout passed |
| Model tests | **251 passed**, including 35 M05 tests |
| Adapter-only tests | **119 passed, five expected skips**, Torch absent |
| M01–M04 reruns with pinned Git audits | All 12/6/12/12 numerical cases and respective acceptance checks passed |
| Both adapter replays | All 13 fixtures matched per environment |
| Adapter import boundary | No installed/imported Torch |
| Dependencies | Compatible: 19 model / 9 adapter distributions |
| Source inventory | Ten full-file hashes, 18 operations, 43 references, two notices passed |
| Preservation | 153 baseline files are byte-identical; only README, model export and append-only M04 document change; all three original plans match saved hashes |

The repository `coding-plan.md`, `interface.md` and `plan1.md` are retained
byte-identically, as are the original planning directory, source pins/notices,
dependencies, adapter, accepted implementations/tests/scripts and all prior runs.
README and this dated focused document record current acceptance; historical
pending statuses in unchanged plans/rerun reports are superseded by the
user-supplied review acceptance. M05 has automated acceptance only.

## Remaining limitations

Python 3.12.12 / PyTorch 2.8.0 / macOS 14.5 arm64, local CPU, one Torch thread,
float32/float64 checks only. CUDA/MPS, reduced precision and other builds remain
unvalidated. No throughput, upstream RNG/framework equivalence, checkpoint
compatibility or playing-strength claim. M06 relative terms remain unimplemented
and required before supervised/RL training, the real evaluator and combined-model
pilot. No masking/softmax/value conversion, checkpoints, data, training, evaluator,
search, subagents or paid compute. Stop for review of the focused M05 commit.
