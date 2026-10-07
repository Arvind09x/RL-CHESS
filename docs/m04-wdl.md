# M04 standalone WDL automated acceptance — 8 October 2026

M04 only passes automated acceptance and stops for independent review.
M03 commit `83d8ac1bbaffe54bbf5cf3d02daf542982c03736` is accepted through
independent **AI review** supplied by the user on 8 October 2026; no human
teammate review is claimed. [Acceptance record](../runs/m04/m03-review-acceptance.json)
and the appended [M03 entry](m03-policy.md) record the reproduced checks and
no blocking findings. All historical run evidence remains byte-identical.

## Operation and contract

`WDLHead(*, d_value=128, device="cpu", dtype=torch.float32)` consumes
encoder-final `[B,64,256]` tokens. It averages the 64 squares, applies a
**separate** affine biased LayerNorm(256), biased Linear(256,d_value), ReLU,
and biased Linear(d_value,3). Returns raw `[B,3]` logits. `d_value` is a
validated positive integer, independently named from policy `d_policy`;
booleans, nonintegers and nonpositive widths raise ValueError. Input rank and
square/feature dimensions are validated; empty tensor batches return `[0,3]`.

`wdl_order` references the existing shared `WDL_ORDER`, `[win,draw,loss]`.
Rows 0/1/2 and future targets are assigned win/draw/loss for the input
position's side to move. Pinned Maia-3's UCI interpreter instead uses
`[loss,draw,win]`: only its computation is ported. Newly trained weights are
required; no checkpoint compatibility or implicit row-remapping loader is
provided. The head does not inspect board color or flip logits. Adapter
features already express the current mover's perspective.

Softmax, `P(win)-P(loss)`, terminal handling and outcome labels remain
consumer responsibilities. No dropout, ponder or auxiliary heads, assembled
network, relative terms, data, training, evaluator or search are introduced.
The diagnostic composition below creates no network/evaluator API.

## Pinned source and constructor settings

Adapted from CSSLab/maia3 commit
`1e13597c42d4858b7cfd7cfdae01e297263364b2`,
[models.py:323–326](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L323-L326)
and [396–397](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L396-L397).
Interpreter ordering: [uci.py:196–200](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/uci.py#L196-L200).
Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson /
UofT CSSLab, *Chessformer: A Unified Architecture for Chess Modeling*, ICLR
2026. [Source map](upstream-sources.md), [unchanged lock](upstream-lock.json),
[retained AGPL-3.0](notices/maia3-AGPL-3.0.txt). Local modifications: standalone
construction, fixed 256 input width, independent validated hidden width and
contract output semantics. No full upstream module is imported.

The checker hashes complete `models.py` and `uci.py` from pinned Git objects,
checks constructor lines, the verbatim two-line value body retained in
`tests/fixtures/wdl_reference.py`, and the upstream interpreter mismatch.
The value computation with the same parameters matches that body exactly.

Installed PyTorch **2.8.0**, build commit
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`, resolves the upstream default
`nn.LayerNorm(256)` to **eps=1e-5, elementwise_affine=True, bias=True**.
Population variance uses correction=0. Production explicitly records those
settings; normalization follows square pooling, in addition to the backbone's
encoder-final token norm. LayerNorm initializes gamma=1, beta=0. Both Linears
retain `kaiming_uniform_(a=sqrt(5))` and uniform weight/bias bounds
±1/sqrt(fan_in): hidden ±1/16, output ±0.08838834764831845 at d_value=128.
No model-wide reset is added. Seeded parity with fresh default constructors
is exact at widths 1/7/128 for float32 and float64. Installed constructor,
reset and LayerNorm-forward sources/hashes are saved in
[wdl-acceptance.json](../runs/m04/wdl-acceptance.json).
Upstream only specifies `torch`; original framework/RNG equivalence is unknown.

**Head only: 33,795 parameter elements in six tensors**:
512 norm + 32,768 hidden weight + 128 hidden bias + 384 output weight + 3 output
bias. Generally `515+260*d_value`. The accepted backbone stays at 3,567,168,
policy at 66,048. No final combined-model count or resource estimate is claimed.

## Independent checks

The known float64 case uses alternating channel values -1/+1, with
square offsets `(square-31.5)/8` that average to zero. Pooling yields the
alternating channels (channel mean 0, population variance 1). Set norm gamma
1.5 and beta .25; with `s=1/sqrt(1+1e-5)`, normalized channels are
`.25 ± 1.5*s`. Hidden preactivations are `1+3*s`, `-.25-1.5*s`, `.75`,
then zeros. The negative hidden coordinate has nonzero output weights to
ensure an omitted ReLU would fail. Hooks independently verify pooled inputs,
normalization, preactivations and output-layer inputs after ReLU. Output rows:

| Contract row | Closed-form raw logit | Observed |
|---|---|---|
| win | 2.6+3*s | 5.599985000112499 |
| draw | -.45-3*s | -3.449985000112499 |
| loss | -1.45+1.5*s | 0.0499925000562495 |

Twelve cases cover batches 1/2/5 × CPU float32/float64 × d_value 7/128.
Asymmetric noncontiguous inputs, nontrivial normalization weights/biases and
both linear weights/biases exercise active and inactive ReLU coordinates,
away from its nondifferentiable zero. An independent **NumPy float64 oracle**
uses explicit pooling/population-variance equations, affine matrix products
and a manual chain rule through ReLU, both Linears, LayerNorm and square mean.
It calls no Torch layer, production forward or autograd. Float32 operands
are promoted to float64 in the oracle before evaluation.

A nonuniform objective `sum(c*y + .03*y**2)`, with distinct c across rows and
batches, checks the output gradient, input gradient and every element of all
six parameter-gradient tensors. Every gradient tensor is finite and nonzero;
inactive hidden coordinates may correctly have zero entries. Inputs and
parameters remain exactly unchanged. A seeded random square permutation
preserves the pooled-head logits within the same reduction tolerances; this
claim applies to final tokens at the head boundary, not backbone invariance.

| Dtype | atol / rtol | Max output error | Max gradient error | Max permutation error |
|---|---|---|---|---|
| float32 | 2e-5 / 2e-4 | 5.960464477539062e-7 | 1.6689300537109375e-6 | 7.152557373046875e-7 |
| float64 | 2e-12 / 2e-11 | 1.3322676295501878e-15 | 6.217248937900877e-15 | 8.881784197001252e-16 |

Pinned value-body output error is zero in all cases. Constructor checks also
verify finite constant-input behavior (zero feature variance), train/eval
identity, biases, dtype/device, count and exact reset parity. Tests cover
invalid widths/shapes and empty tensor batches.

The diagnostic encodes all 13 frozen adapter fixtures, passes their features
through the accepted backbone in eval mode, then calls the separate WDL head.
Default backbone training dropout remains .1. Tokens `[13,64,256]` produce
finite `[13,3]` logits and finite nonzero gradients through features, tokens,
every backbone parameter tensor and all WDL tensors. Board identity, every
retained historical prefix, provenance, re-encoded features, input tensors and
tokens remain unchanged. Terminal fixtures serve as tensor probes only;
this is not terminal evaluation or a playing-strength test.

## Regressions and evidence

[Exact commands](../runs/m04/commands.json) record argv, cwd, environment,
UTC start times, durations and exit codes. [Runner](../runs/m04/verification-runner.py)
and [final audit](../runs/m04/final-audit.py) preserve the procedure. Outputs
were generated under `/private/tmp/rl-chess-m04-change/runs/m04` and archived
unchanged in `runs/m04/`; [execution context](../runs/m04/execution-context.json)
records paths and archive command. `PYTHONDONTWRITEBYTECODE=1` and
`-p no:cacheprovider` avoid repository cache writes. No dependency installation,
network access, subagents or paid compute was needed.

| Check | Result |
|---|---|
| M04 checker and pinned audit | 12 numerical cases, synthetic stages/order, permutation, constructors, 13-fixture diagnostic passed |
| Model tests | **216 passed**, including 29 M04 tests |
| Adapter-only tests | **119 passed, four expected skips** (GAB/backbone/policy/WDL; Torch absent) |
| M03 acceptance rerun | All 12 numerical cases, scalar/order/scaling/dependence, constructors, IDs and diagnostic passed |
| M02 acceptance rerun | Six cases, all input/182-parameter gradients, dropout/sharing/constructors and integration passed |
| M01 acceptance rerun | All 12 numerical cases, sharing and pinned source passed |
| Both adapter replays | All 13 fixtures matched per environment |
| Adapter import boundary | No installed or imported Torch |
| Dependencies | Compatible: 19 model / 9 adapter distributions |
| Source inventory | Ten full-file hashes, 18 operations, 43 references and two notices passed |
| Preservation | 118 baseline files and three original plans matched; historical M03 document retained as exact prefix |

[Preservation hashes](../runs/m04/preservation-check.json) protect every baseline
file except the five intentional export/current-status documentation edits.
All prior runs, adapter, M01–M03 implementations/tests/scripts, dependency pins,
source lock/notices, shared contract and interface/plan1 are byte-identical.
Repository coding-plan is updated as the execution record; the original
planning directory is untouched. Historical rerun scripts retain their former
pending-review metadata; the appended user-supplied AI review acceptance is
authoritative. M04 has automated acceptance only, pending independent review.

## Remaining limitations

Python 3.12.12 / PyTorch 2.8.0 / macOS 14.5 arm64, local CPU, one Torch thread,
float32/float64 and deterministic numerical checks only. CUDA/MPS, reduced
precision and other builds remain unvalidated. No checkpoint compatibility,
throughput or playing-strength claim. No assembled network or M05 onward work;
relative terms, data, training, evaluator, search and consumer WDL operations
remain future scope. Stop for review of the focused M04 commit.
