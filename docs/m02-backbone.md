# M02 backbone automated acceptance — 7 October 2026

M02 only is implemented and passes the checks below. It stops for independent
review; no review acceptance of M02 is claimed. M01 commit
`66eea36ba74ba811298ce8347a5153f562c474be` was accepted through independent
AI review supplied by the user on this date. The reviewer inspected the
implementation and reran 139 model tests, 119 adapter-only tests with one
expected skip, all 12 numerical cases, sharing checks, both 13-fixture replays,
pinned-source and dependency checks, and verified original planning hashes and
the clean tree. No blocking findings were identified. This is AI review,
not human teammate review. [Structured acceptance](../runs/m02/m01-review-acceptance.json)
and the appended [M01 entry](m01-gab.md) preserve that distinction. Earlier
evidence under `runs/milestone-1/`, `runs/f03/` and `runs/m01/` is unchanged.

## Source and retained operations

CSSLab/maia3 at `1e13597c42d4858b7cfd7cfdae01e297263364b2`,
`maia3/models.py`: AbsolutePE 36–42; MHA 47–165; EncoderOnlyBlock 170–236;
CustomTransformerEncoder 239–283; projection/bank/PE/encoder construction
295–322 and projection/position/encoder forward 351–369. Authors: Daniel
Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson /
UofT CSSLab, *Chessformer: A Unified Architecture for Chess Modeling*, ICLR
2026. [Port map](upstream-sources.md), [immutable source lock](upstream-lock.json)
and retained [AGPL-3.0 text](notices/maia3-AGPL-3.0.txt) provide attribution.
The acceptance script checks the pinned full-file SHA-256
`9172bfd41e770baef734f472c6506e7615aa3f4b3f4227f41e9cf35e6ef0be0c`
against the existing Git object. No full upstream module is imported.

`EncoderBackbone` consumes all `[B,64,128]` features, applies biased
Linear(128,256), adds learned zero-initialized `[64,256]` square embeddings,
then eight independently constructed blocks and encoder-final LayerNorm.
It returns `[B,64,256]`. Each block has eight heads of depth 32 and a
256→256→256 FFN. The existing M01 CompactGAB generates `[B,8,64,64]` biases
from that block's input. A single SharedTemplateBank is shared across all
eight independent coefficient generators. Ordinary content logits are
`q @ k.T / sqrt(32) + GAB`, softmax over keys/destinations. Packed
`nn.MultiheadAttention` receives GAB through its floating additive mask,
with `need_weights=False`, following the pinned source.

```text
x = LN(x + dropout1(attention(x)))
x = LN(x + dropout2(linear2(dropout(GELU(linear1(x))))))
return encoder_final_LN(x)  # after all eight blocks
```

The local adaptation omits upstream slicing and Elo/time conditioning. It
adds no input/position dropout. Policy/WDL heads, relative terms and all M03
onward work are unimplemented. This intermediate backbone is not the final
combined architecture or a checkpoint-compatible Maia model.

## Resolved PyTorch 2.8.0 settings

Resolved from the pinned constructor calls and the **installed** framework
source, build commit `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`.
[Saved report](../runs/m02/backbone-acceptance.json) records the exact inspected
constructor/reset/attention/dropout source and its SHA-256 hashes, installed
distributions and build configuration. Upstream pins only `torch`; its
original framework defaults and random stream cannot be recovered from the pin.

| Component | Retained bias / normalization / reset |
|---|---|
| Input Linear(128,256), FFN Linears(256,256) | Bias enabled. Constructor `kaiming_uniform_(a=sqrt(5))`, equivalent to uniform ±1/sqrt(fan_in); bias uniform with the same bound. |
| Packed Q/K/V | One weight `[768,256]` and bias `[768]`. MHA initializes the whole packed weight with Xavier **uniform**, gain=1, fan_in=256, fan_out=768, bound sqrt(6/1024). Bias initialized to zero. Separate Q/K/V projection parameters are absent. |
| Attention output | Biased `[256,256]` Linear. Weight retains its Linear Kaiming-uniform constructor initialization (bound 1/16); MHA subsequently resets the output bias to zero. No extra key/value bias tokens or zero-attention token. |
| All LayerNorms | eps=1e-5, elementwise_affine=True, bias=True, biased variance (correction=0); initial affine weight=1, bias=0. GAB norms retain their M01 scopes (32 and jointly 256 coefficients); block/final norms normalize 256 token features. |
| GELU | Exact erf form, approximate='none'. |
| Square embeddings | `[64,256]`, all zero initially, additive after projection. |
| Shared GAB bank | Existing M01 Xavier **normal**, gain=1, `[4096,32]`, std sqrt(2/4128), one parameter. Generator constructor defaults unchanged. |
| Model reset / factory | No model-wide reset or cloning of initialized blocks. CPU float32 by default; explicit device/dtype constructors supported, float64 CPU additionally checked. |

Seeded tests compare all four MHA parameter tensors exactly to a fresh installed
PyTorch constructor and check packed shapes/bounds, zero attention biases,
all retained Linear biases, norm defaults, zero positions, independent initial
block weights and absence of extra Dropout modules.

All four dropout sites retain training default **0.1**:

| Site | Placement |
|---|---|
| Attention probability | After softmax and before weighted value contraction, delegated to MHA/SDPA. No dropout on raw logits. |
| Attention residual (`dropout1`) | Attention output before residual addition and norm1. |
| FFN hidden (`dropout`) | After GELU(linear1), before linear2. |
| FFN residual (`dropout2`) | linear2 output before residual addition and norm2. |

There are 24 explicit Dropout modules plus eight attention-probability sites.
`.eval()` disables them all; `dropout=0` is a diagnostic override. Tests check
stochastic training, exact seeded replay, exact repeated evaluation, zero-dropout
train/eval equality, and each site's independent effect with the other sites
disabled. No default change is implied by the zero-dropout numerical checks.

## Numerical, sharing and adapter results

Six independent-reference cases: batches 1/2/5 × float32/float64, asymmetric
noncontiguous inputs, nonzero parameters including projection/norm biases and
nonzero square embeddings, plus a nonuniform scalar objective. The independent
path uses explicit affine matmuls, per-square concatenation, per-head GAB bank
products, separate head projections, manual stable softmax, erf GELU and manual
biased-variance normalization. It calls no production forward/math helpers,
MHA or SDPA. Hooks compare projection, positional addition, each of eight
attention/norm1/norm2 outputs and final normalization. Input and all 182 unique
parameter tensors have finite gradients and match the independent path.

| Dtype | atol / rtol | Maximum output/stage error | Maximum gradient error |
|---|---|---|---|
| float32 | 3e-5 / 5e-4 | 5.7220458984375e-6 | 5.960464477539062e-7 |
| float64 | 2e-11 / 2e-10 | 9.769962616701378e-15 | 1.1586326516266343e-15 |

All eight generators retain the exact same bank module and parameter identity;
each generator has 10 private parameter tensors and each block has 22. The
eight private sets are pairwise disjoint. An optimizer built from
`backbone.parameters()` receives 182 unique tensors with exactly one bank
entry. Each layer contributes a nonzero bank gradient; combined backward
matches the sum of the eight isolated contributions. Maximum accumulation
errors: float32 `6.984919309616089e-10`, float64 `1.3010426069826053e-18`.
A single SGD semantics check matches the expected bank update (float32 maximum
error `3.725290298461914e-9`, float64 zero). This is not a training run.

All 13 real adapter fixtures are batched through the backbone, including White,
Black, promotions, en passant, incomplete prefixes and long repetition history.
Outputs/input/parameter gradients are finite. Features, board identity, every
retained prefix, counters/promoted bits, move stacks and provenance are unchanged.
Terminal fixtures are included only as tensor probes; this does not implement
or alter the terminal-first evaluator contract.

## Backbone parameter count

Counted through the deduplicating parameter iterator, verified as **3,567,168**
elements in 182 unique tensors. This is the **M02 backbone only**.

| Component | Elements |
|---|---|
| Input projection with bias | 33,024 |
| Square embeddings | 16,384 |
| Shared GAB bank (counted once) | 131,072 |
| Eight attention projections (263,168 each) | 2,105,344 |
| Eight FFNs (131,584 each) | 1,052,672 |
| Sixteen block norms | 8,192 |
| Eight private GAB generators including norms (27,496 each) | 219,968 |
| Encoder-final norm | 512 |

Heads and relative tables are absent, so this count must not be compared as the
final combined-model count. Construct optimizers from the enclosing backbone,
not concatenated per-layer parameter lists. Shared bank registration exposes
multiple state-dictionary paths; loading requires the same shared construction.
No checkpoint schema is implemented here.

## Commands, regressions and preservation

[Exact commands and raw outputs](../runs/m02/commands.json) are saved with cwd,
environment overrides, exit codes and durations. [Execution context](../runs/m02/execution-context.json)
records the actual temporary output paths and archive operation; the unchanged
output bytes are under `runs/m02/`. [Verification runner](../runs/m02/verification-runner.py)
also saves the preservation procedure. Tests used `-p no:cacheprovider` to avoid
writing cache outside the chat's writable workspace. Historical evidence was
never overwritten. The portable `scripts/check_m02.py` can run without a source
checkout or output path; the optional checkout audit reads pinned Git objects.

| Actual check | Result |
|---|---|
| `scripts/check_m02.py` with pinned Git audit | Six numerical cases, constructor checks, two sharing cases, dropout and adapter integration passed. |
| `scripts/check_m01.py` with pinned Git audit | All 12 numerical cases, two sharing cases, full-file checksum and verbatim method passed. |
| Full model-environment pytest | **157 passed**, zero failures/errors. |
| Adapter-only pytest | **119 passed, two expected module skips** (`test_gab`, `test_backbone`); no installed Torch. Historical M01 count remains one skip. |
| Adapter replay in both environments | All 13 frozen fixtures matched in each. |
| Adapter-only import check | Torch absent from installation/imports. |
| `uv pip check` in both environments | All installed packages compatible. |
| `scripts/check_upstream.py` using both pinned checkouts | 10 full-file hashes, 18 entries, 43 immutable references, two notices passed. |
| Original/historical preservation | Three original planning hashes match M01 evidence; 45 protected repository files match starting commit `0d0359a`, including adapter, original interface/plan1, all prior runs, M01 code/checks, source lock/notices and dependency pins. |

[Preservation report](../runs/m02/preservation-check.json) records exact hashes.
Repository `coding-plan.md` updates current execution status and appends evidence;
the original planning directory in Documents is untouched. Current status:
M01 AI review accepted; M02 automated acceptance complete, independent review
pending; M03 onward unstarted.

## Remaining limitations

Tested Python 3.12.12, PyTorch 2.8.0, macOS 14.5 arm64, local CPU, one Torch
thread and deterministic algorithms for the numerical report. CUDA, MPS,
reduced precision and other platform builds are unvalidated. There is no
upstream framework/RNG-stream equivalence, pretrained checkpoint compatibility,
performance benchmark or playing-strength claim. No subagents, paid compute,
heads, relative terms, data preparation, learning runs or search were used.
