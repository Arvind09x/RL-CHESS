# M06 fixed combined architecture automated acceptance — 8 October 2026

M06 only passes automated acceptance and stops for independent review. M05
commit `2e6316dd0c49b0c22ff10ef72b55793856189a72` has user-supplied independent
**AI review acceptance**, appended to [M05](m05-network.md) and recorded in
[the structured review record](../runs/m06/m05-review-acceptance.json).
No human teammate review is claimed. Planning files retain their historical
statuses unchanged; these focused dated records describe current acceptance.

## Fixed operation and diagnostic paths

`ChessNetwork()` now enables all three Q/K/V relative terms by default.
Eight independent layers retain width 256, eight heads of depth 32, FFN width
256, packed biased Q/K/V and output projections, additive zero square embeddings,
Post-LayerNorm blocks, encoder-final norm, four accepted dropout sites per
layer and one shared GAB bank with d1/d2/d3=8/32/32. The policy and WDL modules,
contract semantics and their constructor initialization are byte-identical.
Both heads consume the same final token object, returning raw `[B,4352]` and
`[B,3]` logits in pair4352-v1 and shared win/draw/loss order. Default widths
remain independently named 128; default dropout remains .1.

Each layer owns three separate zero parameters `relative.q`, `.k`, `.v`, each
`[8,225,32]`. Logical tables preserve the pinned physical mapping:
`R[h,r,d] = upstream[d*8+h,r]`. The nonpersistent int64 `[64,64]` index buffer
uses query-minus-key displacement for both content operands:

```text
r(i,j) = 15*(rank_i-rank_j+7)+(file_i-file_j+7)
logits = (q_i·k_j + q_i·U_r(i,j) + k_j·T_r(i,j))/sqrt(32) + GAB[i,j]
alpha = softmax(logits, keys j)
output_i = sum_j dropped_alpha[i,j]*(v_j + V_r(i,j))
```

`RelativeQKV.logits` and `.values` remain separately inspectable. Indexed table
selection replaces the upstream dense one-hot factorizer. The content/relative
sum receives one scale, GAB follows scaling, and one attention-probability
dropout result feeds both ordinary and relative values. No position-only cross
term, reversed key displacement, extra relative-value scale or new dropout
site is added. Packed parameters are constructed by the accepted PyTorch MHA,
then read by the enabled path's explicit projections. No reset is added.
Zero allocations consume no RNG; seeded common weights and RNG state match M05
exactly in both dtypes, including the heads.

`network(features, relative_enabled=False)` bypasses all relative operations
on that same network without modifying weights; relative-table gradients are
absent for this call. It uses the accepted MHA/GAB path. Constructor
`ChessNetwork(relative_enabled=False)` omits tables, reproducing the accepted
M05 structure/count for historical checks. Standalone `EncoderBackbone` and
`GABSelfAttention` retain their accepted diagnostic defaults (no tables);
explicit `relative_enabled=True` constructs the combined backbone/attention.
These are correctness diagnostics, with no alternative training track.

## Pinned source inspected before porting

The existing F03 Git objects were read before implementation. Relative map,
zero initialization, depth-before-head reshape, q/k/v contractions, softmax,
scaling and head split were inspected at daniel-monroe/lczero-training commit
`7b3ee3fcdb49d8abeb034718935b099134c11882`, `tf/tfprocess.py:83–91,285–341,1960–2007`.
The complete file hash is
`487c998809d49e753d98f7d19ca09ac904dfdaf18b9dbac318bce6bac0733a24`.
[Acceptance report](../runs/m06/relative-acceptance.json) retains exact inspected
operation text and the full-file Git-object audit. Maia-3 projection/GAB/backbone
operations remain at `1e13597c42d4858b7cfd7cfdae01e297263364b2`.
[Unchanged source map](upstream-sources.md), [lock](upstream-lock.json),
[Leela GPL-3.0-or-later header](notices/lczero-tfprocess-header.txt) and
[Maia AGPL-3.0 text](notices/maia3-AGPL-3.0.txt) retain attribution.
Relative port attribution names Gian-Carlo Pascutto, Daniel Monroe and Leela
contributors. The combination is a project adaptation of the implemented
Q/K/V variant; no exact reproduction of the paper's omitted cross term is claimed.

## Independent acceptance and tolerances

Six cases each for isolated contractions, complete attention and the complete
network cover batches 1/2/5 × CPU float32/float64, asymmetric noncontiguous inputs,
nonzero independent tables, nontrivial weights/biases and normalization affine
parameters, with dropout=0. Two further network cases use independent head
widths 7/11. All input/operand and every parameter-gradient element are checked
with a nonuniform objective. All 215 unique network parameter-gradient tensors,
including all 24 relative tables, are finite and nonzero. Inputs and parameters
remain unchanged. Individual gradient entries may correctly be zero.

The independent relative reference builds a NumPy one-hot factorizer with four
coordinate loops, explicitly stacks physical rows in depth-before-head order,
and uses per-square matrix products. Isolated contraction cases expand via the
dense factorizer; complete references use its independently derived argmax
indices. A sentinel checks every head/depth/pair against the dense expansion;
all 4,096 indices, all 225 displacements and asymmetric opposite directions are
checked exactly. Deliberate head-first interpretation is detected.

Complete attention independently computes separate projections, the accepted
independent GAB reference, explicit stable softmax and ordinary/relative value
contractions. The combined reference extends accepted M02 explicit affine,
manual population-variance normalization and erf GELU math; it uses the M03
independent policy component reference and M04 NumPy float64 forward/manual
chain-rule WDL oracle. Relative/backbone derivatives use autograd on those
independent operations. References call no production attention/relative
operation, MHA/SDPA, Linear or LayerNorm. TensorFlow itself is not executed.

| Check maximum | float32 | float64 |
|---|---:|---:|
| Contraction output | 9.5367431640625e-7 | 1.1102230246251565e-15 |
| Contraction gradients | 1.4901161193847656e-8 | 5.551115123125783e-17 |
| Attention output | 7.152557373046875e-7 | 1.5543122344752192e-15 |
| Attention gradients | 4.7497451305389404e-8 | 1.1796119636642288e-16 |
| Joint policy/WDL output, including widths 7/11 | 2.6226043701171875e-6 | 5.773159728050814e-15 |
| Joint input/all-parameter gradients | 4.172325134277344e-6 | 1.0359768598533492e-14 |
| Zero-table versus M05 output | 6.198883056640625e-6 | 9.992007221626409e-15 |
| Zero-table versus M05 common gradients | 2.86102294921875e-6 | 3.552713678800501e-15 |

Atol/rtol are **3e-5/5e-4** for float32 and **2e-11/2e-10** for float64,
retaining accepted backbone tolerances. The zero-table enabled path uses explicit
attention rather than MHA/SDPA, so floating-point reduction differences are
expected. Disabled outputs and every common gradient match M05 **exactly**,
even with nonzero tables, for both dtypes and all three batches. Seeded default
dropout train outputs and eval outputs also recover M05 exactly when disabled. Zero-table
initialization already yields finite/nonzero gradients for every table.

Separate head objectives each reach inputs, final tokens, every backbone
parameter and bank; joint gradients match their sum. Heads receive only their
own branch gradients. All 24 table storage pointers are distinct, with direct
mutation isolation, and all eight generators share exactly one bank. Whole-model
dtype moves preserve sharing and int64 indices; deduplicated SGD construction
contains 215 entries and one bank (no new optimizer step).

A seeded attention dropout reference with nonzero tables matches the same mask
on both value terms in float32/float64. Seeds replay exactly; distinct seeds
change training output; eval repeats exactly. Probability dropout=1 removes
both value terms exactly, leaving only the output-projection bias. Full-network
checks with nonzero tables verify stochastic train outputs at both heads, seeded replay, propagated
train/eval flags and exact zero-dropout train/eval equality.

Both dtypes process all 13 frozen adapter fixtures together; policy/WDL outputs,
input gradients and every parameter gradient are finite/nonzero. Board identity,
all retained historical prefixes, provenance, features and tensor inputs remain
unchanged. Terminal fixtures are tensor probes only.

## Independent parameter and payload accounting

Explicit shape arithmetic counts the shared bank once, then separately checks
the deduplicated network iterator and actual allocated gradients.

| Component | Parameter elements |
|---|---:|
| Accepted backbone without relative tables | 3,567,168 |
| Accepted policy head | 66,048 |
| Accepted WDL head | 33,795 |
| Eight layers × three tables × 8 × 225 × 32 | 1,382,400 |
| Fixed combined total, 215 unique tensors | **5,049,411** |

Exactly `3,667,011 + 8*3*8*225*32 = 5,049,411`. Each layer adds 172,800
relative elements. General count is
`3,567,168 + 1,382,400 + 516*d_policy + 515 + 260*d_value`.
The provisional 6–8M range was an estimate, not shape-derived acceptance or a
requirement to increase model size. The locked width 256, FFN width 256, eight
layers/heads, compact heads and single shared bank produce this exact lower
count. The original estimate's derivation is not established; no extra layers,
wider FFN, duplicated bank or unused parameters are introduced to reach it.
This resolves the material discrepancy without changing the fixed architecture.

| Dtype | Parameter bytes | Allocated gradient bytes | Combined payload bytes |
|---|---:|---:|---:|
| float32 | 20,197,644 | 20,197,644 | 40,395,288 |
| float64 | 40,395,288 | 40,395,288 | 80,790,576 |

The 13-fixture backward independently sums actual parameter/gradient tensor
payloads. These exclude inputs/outputs, activations, temporary gathered tables,
optimizer state, allocator metadata/cache, Python and process overhead. The eight
int64 index buffers add 262,144 bytes outside parameter/gradient payloads.

Separately measured checker-process lifetime peak RSS is **1,210,515,456 bytes**,
from macOS `resource.getrusage(RUSAGE_SELF).ru_maxrss` in native bytes. This
includes imports, all cases, cloned parameters, reference/autograd graphs and
allocator retention. It is neither an isolated model forward/backward peak nor
a training-memory estimate. No GPU memory or throughput is measured.

## Commands, regression path and preservation

[Exact commands](../runs/m06/commands.json) record argv, cwd, environment,
UTC start time, duration, exit code and raw output path.
[Runner](../runs/m06/verification-runner.py) and [final audit](../runs/m06/final-audit.py)
record the complete procedure. Tests run in `/private/tmp/rl-chess-m06-change`
using installed model/adapter interpreters, `PYTHONDONTWRITEBYTECODE=1`, explicit
staged `PYTHONPATH` and pytest `-p no:cacheprovider`. No dependency installation
or network access was needed. Reviewed bytes are archived into the repository.

The unchanged M05 checker runs via `runs/m06/diagnostic.py scripts/check_m05.py`,
which adapts only its helper factory to constructor `relative_enabled=False`.
Full model pytest uses `-p runs.m06.diagnostic_plugin` for that same historical
M05 helper; enabled M06 tests and production defaults are unaffected. M01–M04
standalone diagnostics retain their accepted defaults. This explicit path keeps
all historical test/checker sources and prior run evidence byte-identical.
Historical rerun metadata is preserved by the unchanged checker; the dated
review acceptance above governs current status.

| Check | Result |
|---|---|
| M06 checker | All layout/contraction/attention/network/recovery/head/dropout/sharing/accounting/adapter checks passed |
| Full model suite | **287 passed**, including 36 new M06 tests |
| Adapter-only suite | **119 passed, six expected skips**, Torch absent |
| M01–M05 acceptance | All respective numerical, constructor, dropout, sharing, head and adapter checks passed through the documented diagnostic paths |
| Both adapter replays | All 13 fixtures matched in each environment |
| Adapter import boundary | No installed/imported Torch |
| Dependencies | Compatible: 19 model / 9 adapter distributions |
| Source inventory | Ten full-file hashes, 18 operations, 43 immutable references, two notices passed |
| Preservation | Exact baseline hashes, appended M05 prefix, three original plans and focused change manifest verified by final audit |

[Preservation report](../runs/m06/preservation-check.json) and
[change manifest](../runs/m06/change-manifest.json) identify intentional edits.
Existing model changes are confined to attention, backbone flag propagation and
network default/diagnostic integration. Heads, GAB, adapter/contracts, model
exports, dependency/source pins/notices, planning files, every prior run, and
historical tests/checkers remain unchanged. M05 documentation retains its exact
historical prefix with review acceptance appended. README provides current usage.

## Limitations and stop

Python 3.12.12 / PyTorch 2.8.0 / macOS 14.5 arm64, local CPU, one Torch thread,
float32/float64 only. CUDA/MPS, reduced precision and other builds remain
unvalidated. No upstream framework/RNG or checkpoint compatibility, performance,
playing-strength or relative-versus-GAB superiority claim. No data, training,
checkpoints, evaluator, search, GPU runs, paid compute or subagents were started.
Historical M01/M02 numerical SGD semantics checks were rerun unchanged; these
are diagnostic checks, not training runs. Stop after one focused M06 commit and
a clean-tree verification for independent review.


## Closeout log preservation

The initial staged whitespace check stopped before committing because two
checker stdout logs had an extra EOF blank line. Exact verification stdout is
retained in [checker-raw-output.json](../runs/m06/checker-raw-output.json);
readable logs normalize only trailing newlines. The failed check and command
remain in runs/m06. [Repair procedure](../runs/m06/closeout-repair.py) and
[record](../runs/m06/log-preservation-repair.json) document the continuation.
Model/checker/test sources and numerical results were unchanged. Final repaired
repository/staged whitespace, preservation, manifest and clean-tree checks passed.


## Independent AI review acceptance — 8 October 2026

The user supplied the planning agent's independent acceptance of M06 commit
`4765a8cb3d6ae9cf50619be283176ab9ac17cca6` for local CPU correctness in the
D01 request. That reviewer reproduced 287 model tests, 119 adapter tests with
six expected skips, the complete M06 numerical checker, both 13-fixture
adapter replays and 191 protected-file hashes. The working tree was clean.
This is AI review, not human review; GPU correctness and performance remain
unverified. This appended entry supersedes the preserved historical pending
review status. [Attribution record](../runs/d01/m06-review-acceptance.json).
