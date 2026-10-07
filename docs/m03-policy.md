# M03 pair policy automated acceptance — 8 October 2026

M03 only passes automated acceptance and stops for independent review. M02
commit `bc2066d57203459be18d60013eb874c88f4bc5c3` was accepted through independent
**AI review** supplied by the user on 7 October 2026; no human teammate review
is claimed. [Review record](../runs/m03/m02-review-acceptance.json) and the
appended [M02 entry](m02-backbone.md) record the reviewed checks and no blocking
findings. All evidence under earlier `runs/` directories remains byte-identical.

## Operation and configuration

`PairPolicyHead(*, d_policy=128, device="cpu", dtype=torch.float32)` consumes
contextual square tokens `[B,64,256]` and returns raw logits `[B,4352]`.
`d_policy` is a public policy-head configuration field; its constructor rejects
booleans, non-integers and integers <=0 with ValueError. Input rank/square/feature
dimensions are validated. An empty tensor batch returns `[0,4352]`.

Three bias-free Linears retain the pinned order: `proj_sq_from:256→d_policy`,
`proj_sq_to:256→d_policy`, `promo_bias_proj:d_policy→4`. For origin i,
destination j, from-file f, to-file g and promotion-piece index p:

```text
q = proj_sq_from(x); k = proj_sq_to(x)
S[i,j] = dot(q[i], k[j]) / sqrt(d_policy)
A[g,p] = sqrt(d_policy) * promo_bias_proj(k[56+g])[p]
promotion[f,g,p] = S[48+f,56+g] + A[g,p]
ordinary ID = 64*i+j
promotion ID = 4096+4*(8*f+g)+p
p = queen, rook, bishop, knight
```

Ordinary destination varies fastest; promotions append in from-file, to-file,
piece order. Production promotion assembly broadcasts `[B,8,8,1]` base scores
and `[B,1,8,4]` adjustments, then flattens `[B,8,8,4]`. The adjustment depends
on the projected destination key on rank 8, with the additional sqrt multiplier
and no further scaling after addition. Every slot remains, including self-pairs,
impossible promotions and ordinary rank7/rank8 pairs. Legality and move conversion
remain in the accepted adapter. The head has no mask, softmax, dropout or terminal
handling. No WDL head, assembled network or M04 onward operation is added.

## Source and constructor resolution

Adapted from CSSLab/maia3 commit
`1e13597c42d4858b7cfd7cfdae01e297263364b2`, `maia3/models.py:331–334,373–394`;
vocabulary `maia3/utils.py:23–46`. Daniel Monroe, George Eilender, Philip
Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab, *Chessformer: A Unified
Architecture for Chess Modeling*, ICLR 2026. [Source inventory](upstream-sources.md),
[unchanged lock](upstream-lock.json) and [retained AGPL-3.0](notices/maia3-AGPL-3.0.txt).
Local adaptations are standalone construction, validated policy width, fixed
256 input width and vectorized promotion assembly. No full upstream model imports.

The exact pinned forward body, lines 373–394, is retained with attribution in
`tests/fixtures/policy_reference.py`. The checker verifies its verbatim bytes
against the pinned Git object and complete `models.py` SHA-256
`9172bfd41e770baef734f472c6506e7615aa3f4b3f4227f41e9cf35e6ef0be0c`.

Installed PyTorch **2.8.0**, build commit
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`, resolves all three Linears to
`kaiming_uniform_(a=sqrt(5))`, equivalent to uniform ±1/sqrt(fan_in).
Origin/destination bounds are ±1/16; promotion bound at width 128 is
±0.08838834764831845. There are no bias parameters and no subsequent reset.
Seeded constructor comparisons match all three fresh Linears exactly for widths
1/7/128 and both dtypes. Installed constructor/reset source and hashes are saved
in [the report](../runs/m03/policy-acceptance.json). Upstream specifies only
`torch`; original upstream framework/RNG-stream equivalence is unavailable.

The head has **66,048 parameter elements in three tensors** at width 128:
32,768 origin + 32,768 destination + 512 promotion, generally `516*d_policy`.
The accepted backbone remains 3,567,168 elements. Neither is a final combined
architecture parameter count.

## Independent acceptance results

Twelve numerical cases: batches 1/2/5 × float32/float64 × policy width 7/128,
asymmetric noncontiguous inputs and nonzero weights. All 4,352 outputs match
an independent differentiable reference that projects with matrix products,
accumulates pair depth component by component, and builds explicit slot lists;
it calls no production forward/Linear/einsum/batched pair product. A separate
Python `math.fsum` scalar path independently calculates projections, eight
selected ordinary pairs and all 256 promotion slots, indexed through the existing
adapter vocabulary. The pinned source body also matches exactly.

A nonuniform objective checks finite, nonzero input and all three parameter
gradients against the independent component reference; inputs remain unchanged.

| Dtype | atol / rtol | Max full-output error | Max scalar error | Max gradient error |
|---|---|---|---|---|
| float32 | 2e-5 / 2e-4 | 3.814697265625e-6 | 4.76837158203125e-6 | 5.7220458984375e-6 |
| float64 | 2e-12 / 2e-11 | 3.552713678800501e-15 | 1.0658141036401503e-14 | 7.105427357601002e-15 |

Pinned-body error is zero in all cases. A width-4 float64 synthetic case checks
all 4,096 ordinary and 256 promotion slots exactly in each of two batches,
including both factors (divide by 2; multiply by 2) and distinct piece values.
With the origin projection zeroed, changing rank7 input cannot change adjustments;
changing unprojected extra destination features cannot change them; changing
one projected destination file changes exactly that file's four slots for all
origins by independently computed values. Zero inputs yield exact zero logits;
train/eval outputs match exactly. Invalid widths/shapes fail clearly.

Existing `move_to_id`, `id_to_move` and `legal_mask` APIs verify e2e4/e7e5=796,
White e7e8q/r/b/n and Black e2e1q/r/b/n=4240/4241/4242/4243 and their head
scalar values. Illegal ordinary promotion pairs and impossible geometry retain
finite raw logits. No adapter behavior changes.

The diagnostic feeds all 13 existing adapter fixtures through the accepted
backbone in eval mode (training default remains 0.1), then directly into the head.
Outputs are `[13,4352]`; inputs, contextual tokens, all backbone parameters and
all head parameters have finite nonzero gradients. Board identity, every history
prefix, provenance and encoded features are preserved. Terminal fixtures are
tensor probes only. This adds no assembled network or evaluator API.

## Regressions, exact commands and preservation

[Commands](../runs/m03/commands.json) include argv, cwd, UTC start times,
environment overrides, durations and exit codes. [Runner](../runs/m03/verification-runner.py)
records the exact commands and preservation procedure. Acceptance began before
midnight IST and the final documented verification ran on 8 October 2026.
Outputs were generated under `/private/tmp/rl-chess-m03-stage/runs/m03` and copied
unchanged into `runs/m03/`; [execution context](../runs/m03/execution-context.json)
records this. `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` avoid cache writes.
No dependency installation or network access was needed.

| Check | Result |
|---|---|
| M03 checker, pinned Git audit | 12 numerical cases, exact order/scaling/dependence, constructors, IDs, backbone diagnostic passed |
| Full model pytest | **187 passed**, including 30 M03 tests |
| Adapter-only pytest | **119 passed, three expected skips**: GAB, backbone, policy (no Torch installed) |
| M02 acceptance rerun | Six numerical cases, all 182 parameter/input gradients, dropout/sharing/constructors and 13-fixture integration passed |
| M01 acceptance rerun | All 12 numerical cases, sharing and pinned-byte check passed |
| Both adapter replays | 13 fixtures matched in each environment |
| Adapter import boundary | No installed/imported Torch |
| Both dependency checks | Compatible (19 model / 9 adapter distributions) |
| Source inventory | 10 full-file hashes, 18 entries, 43 references, two notices passed |
| Preservation | 90 protected repository files and all three original planning hashes matched reviewed baseline bc2066d; original M02 document retained as exact prefix |

[Preservation hashes](../runs/m03/preservation-check.json) cover all baseline
files except five intentional export/status edits. All prior runs, adapter,
M01/M02 implementations and tests, existing scripts, interface/plan1, source
lock/notices and dependency pins are byte-identical. Repository coding-plan is
updated as the execution record; the original planning directory is untouched.
M02 rerun report retains its historical script's pending-review/date metadata;
the new user-supplied AI acceptance record above is authoritative.

## Remaining limitations

Python 3.12.12 / PyTorch 2.8.0 / macOS 14.5 arm64, local CPU, one Torch thread,
float32/float64 and deterministic numerical checks only. CUDA/MPS, reduced
precision and other platform builds remain unvalidated. No checkpoint compatibility,
throughput benchmark or playing-strength claim. M03 awaits independent review;
M04 onward, WDL, assembled network, relative terms, data, training and search
remain unimplemented. No subagents or paid compute.
