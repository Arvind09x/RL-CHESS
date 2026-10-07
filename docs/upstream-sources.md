# F03: pinned source and port map

Recorded 7 October 2026. The original F03 audit covered the operations planned
in M01–M06, plus vocabulary/history reference behavior, without executing them.
M01 is ported and accepted through independent AI review; see [M01 evidence](m01-gab.md).
M02 backbone is ported and passes automated acceptance, awaiting review; see
[M02 evidence](m02-backbone.md). M03–M06 remain unimplemented. No full upstream module is imported.
`interface.md` remains the contract. The original planning files in
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2` are preserved.

## Repository identity and attribution

| Repository | Verified full commit | Identity / authorship evidence |
|---|---|---|
| CSSLab/maia3 | [1e13597c42d4858b7cfd7cfdae01e297263364b2](https://github.com/CSSLab/maia3/commit/1e13597c42d4858b7cfd7cfdae01e297263364b2) | Package author UofT CSSLab; paper credit Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton Anderson. Pinned merge author `ashtonanderson`, committer `GitHub`, 2026-05-25. |
| daniel-monroe/lczero-training | [7b3ee3fcdb49d8abeb034718935b099134c11882](https://github.com/daniel-monroe/lczero-training/commit/7b3ee3fcdb49d8abeb034718935b099134c11882) | Daniel Monroe's fork; pin author/committer Daniel Monroe, 2024-09-23. `tf/tfprocess.py` retains Gian-Carlo Pascutto's 2017–2018 copyright and Leela Zero notice. Inherited work also belongs to its contributors; the pin author is not asserted to be every line's author. |

`upstream-lock.json` is the compact machine-readable inventory: identity,
commit metadata, exact paths, immutable blob/raw URLs, full-file SHA-256 hashes,
and symbols/line ranges for every planned operation. The inspected Maia-3
implementation is in `maia3/`, not the `code/` compatibility wrappers at this pin.
The branch HEADs were used only to discover revisions; full commit objects and
their contents were then verified. Moving links in the preserved planning files
are background references; this lock is authoritative for ports.

Maia-3's [LICENSE](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/LICENSE)
contains the AGPL version 3 text. Its [package metadata](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/pyproject.toml#L11-L14)
points to that file; no project-specific copyright/or-later grant appears in
the selected source headers. The FSF copyright in the license is for the
license text, not an assertion of FSF authorship of Maia-3. A verbatim copy is
in `notices/maia3-AGPL-3.0.txt`. No separate NOTICE/COPYING file was found.

The earlier repository has **no standalone LICENSE/COPYING/NOTICE file** in
its pinned tree. Its [file header](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L1-L17)
explicitly grants GPL version 3 or later and carries the copyright and warranty
disclaimer; all 17 lines are retained verbatim in
`notices/lczero-tfprocess-header.txt`. No submodule license is substituted for
this file's notice. Ports must retain the source credits, grant/disclaimer and
applicable license texts and identify local modifications. The AGPL text
includes source/conveyance and remote-interaction provisions (sections 4–6,
13); this provenance commit does not relicense existing adapter code or claim
that a citation alone satisfies the licenses. Full GPL text is available at
the URL in the retained header and must accompany an eventual derivative.

The pinned [Maia-3 README citation](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/README.md#L17-L27)
requests: Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang and
Ashton Anderson, *Chessformer: A Unified Architecture for Chess Modeling*,
ICLR 2026, `monroe2026chessformer`,
https://openreview.net/forum?id=2ltBRzEHyd. Retain that attribution in model/report
documentation along with the earlier implementation's repository and notice.

## Complete immutable operation inventory

All links below contain the full pinned SHA. Hashes cover complete source files,
not normalized text or selected excerpts. Initializers delegated to framework
constructors are distinguished below from explicit upstream initializers.

| Operation | Planned task | Pinned path/symbol and lines |
|---|---|---|
| `compact_gab` | M01 | [MHA.__init__:73–90](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L73-L90); [MHA._sq_bias:95–118](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L95-L118) |
| `board_compression` | M01 | [MHA._sq_bias:100–114](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L100-L114) |
| `shared_template_bank` | M01 | [MAIA3Model.__init__:304–308](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L304-L308); [CustomTransformerEncoder.__init__:253–271](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L253-L271); [MHA._sq_bias:116–118](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L116-L118) |
| `input_projection` | M02 | [MAIA3Model.__init__:295–302](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L295-L302); [MAIA3Model.forward:351–369](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L351-L369) |
| `absolute_pe` | M02 | [AbsolutePE:36–42](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L36-L42) |
| `post_layernorm` | M02 | [EncoderOnlyBlock.__init__:193–210](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L193-L210); [EncoderOnlyBlock.forward:212–236](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L212-L236) |
| `dropout` | M02/M06 | [MHA.__init__:65–71](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L65-L71); [EncoderOnlyBlock.__init__:193–202](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L193-L202); [EncoderOnlyBlock.forward:212–236](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L212-L236) |
| `final_normalization` | M02/M04 | [CustomTransformerEncoder.__init__:267–271](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L267-L271); [CustomTransformerEncoder.forward:281–283](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L281-L283); [MAIA3Model.__init__:323–326](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L323-L326); [MAIA3Model.forward:396–398](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L396-L398) |
| `initialization` | M01/M02/M03/M04/M06 | [AbsolutePE.__init__:37–39](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L37-L39); [MAIA3Model.__init__:288–334](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L288-L334); [MHA.__init__:47–92](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L47-L92); [EncoderOnlyBlock.__init__:170–210](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L170-L210); [RPELogits.build:293–301](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L293-L301); [RPEValue.build:326–333](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L326-L333) |
| `pair_policy` | M03 | [MAIA3Model.__init__:331–334](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L331-L334); [MAIA3Model.forward:373–376](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L373-L376) |
| `promotion_adjustment` | M03 | [MAIA3Model.__init__:331–334](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L331-L334); [MAIA3Model.forward:378–394](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L378-L394) |
| `vocabulary_ordering` | M03 | [get_all_possible_moves:23–46](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/utils.py#L23-L46); [MAIA3Model.forward:375–394](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L375-L394) |
| `wdl_head` | M04 | [MAIA3Model.__init__:323–326](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L323-L326); [MAIA3Model.forward:396–400](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L396-L400); [wdl_from_value_logits:196–200](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/uci.py#L196-L200) |
| `make_rpe_map` | M06 | [make_rpe_map:83–91](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L83-L91) |
| `rpe_logits` | M06 | [RPELogits:285–317](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L285-L317) |
| `rpe_value` | M06 | [RPEValue:319–341](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L319-L341) |
| `qkv_contractions` | M06 | [TFProcess.split_heads:1956–1961](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L1956-L1961); [TFProcess.scaled_dot_product_attention:1963–2007](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L1963-L2007); [TFProcess.mha:2047–2063](https://github.com/daniel-monroe/lczero-training/blob/7b3ee3fcdb49d8abeb034718935b099134c11882/tf/tfprocess.py#L2047-L2063); [MHA.forward:143–165](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/models.py#L143-L165) |
| `history_padding` | reference only; A04 already implemented | [get_historical_tokens:44–66](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/dataset.py#L44-L66); [tokenize_board:16–30](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/dataset.py#L16-L30); [Maia3UCIEngine.cmd_position:436–450](https://github.com/CSSLab/maia3/blob/1e13597c42d4858b7cfd7cfdae01e297263364b2/maia3/uci.py#L436-L450) |

## Mapping to the locked interface

**Compact GAB (M01).** Let D=256, H=8, d1=8, d2=32, d3=32. Each layer consumes
its `[B,64,D]` input. Apply biased `sm1: D→d1` independently to squares, flatten
in square-then-feature order to `[B,64*d1]`, then `sm2: 512→32`, GELU,
LayerNorm(32), `sm3: 32→H*d3`, GELU, LayerNorm(H*d3), and reshape to
`[B,H,d3]`. There is no activation or norm between `sm1` and flattening.
`ln2` normalizes all H*d3 coefficients jointly before reshaping, not each head
separately. Multiply by one shared parameter bank `W[4096,d3]` with
`einsum('bhi,oi->bho', coefficients, W)` and reshape to `[B,H,64,64]`, with
destination fastest. The bank is shared across heads and all encoder layers;
coefficient generators are distinct per layer. Use Xavier **normal** for W,
with the upstream `[4096,32]` orientation; no per-head bank or transpose is
silently substituted. The `gab_per_square_dim=0` mean-pooling branch is an
upstream alternative, not our board compression. WDL mean pooling is separate.

**Projection/backbone (M02).** Upstream concatenates `12*history` piece channels,
optional three time channels and two Elo embeddings, then projects to D.
Our biased projection is exactly `128→256` on adapter `[B,64,128]`; discard
that token slicing, Elo interpolation/concatenation and all clock/ponder inputs.
Add zero-initialized `AbsolutePE.bias[64,256]`. Use eight Post-LayerNorm blocks,
eight heads (head depth 32), biased attention projections/output projection,
biased FFN `256→256→256`, GELU, ordinary affine LayerNorm (constructor default
epsilon), attention/residual/FFN dropout 0.1. Each block computes
`LN(x + dropout1(attention(x)))`, then
`LN(x + dropout2(linear2(dropout(GELU(linear1(x))))))`.
Retain attention-probability dropout from `nn.MultiheadAttention`, not dropout
on raw logits or an extra input/position dropout. Retain encoder-final
LayerNorm on tokens. Evaluation disables dropout; zero dropout for diagnostic
numerical checks is not a changed training default.

**Initialization.** Explicit upstream choices: zero absolute embeddings,
Xavier-normal shared GAB bank, and zero all three relative tables. Everything
else in the retained Maia-3 backbone/heads is initialized by its `nn.Linear`,
`nn.MultiheadAttention` and `nn.LayerNorm` constructors; no model-wide reset is
present. Preserve this constructor-default policy when porting (including
packed Q/K/V initialization rather than silently initializing three separate
matrices differently), then record the chosen PyTorch version and resolved
reset details in M02. Removed Elo embeddings have an explicit Xavier-normal
reset but are not parameters in our model. Earlier TensorFlow Q/K/V projection
and DeepNorm/quantization initializers are outside the relative-operation port.
**Limitation:** Maia-3's dependency metadata says only `torch`, so its exact
framework build and delegated initialization implementation cannot be recovered
from this source pin. F03 installs no model dependencies and does not assert
framework-default or random-stream equivalence across environments.

**Pair policy/promotions (M03).** Bias-free origin/destination projections
`256→d_policy=128`, `S[i,j]=dot(q_i,k_j)/sqrt(128)`. Bias-free
`promo_bias_proj:128→4` takes the **projected destination key** on rank 8, not
the origin or unprojected token. For destination file g and piece p, use
`A[g,p]=sqrt(128)*promo_bias_proj(k_(rank8,g))[p]`; promotion logit is
`S[(rank7,f),(rank8,g)]+A[g,p]`. The multiplication by sqrt(d_policy) is
intentional, in addition to the division in the base pair score. There is no
second scaling after addition. Flatten base `[64,64]` row-major, destination
fastest, followed by promotions with from_file, to_file, `[q,r,b,n]` order.
Keep all 4096+256 slots; IDs are exactly `64*i+j` and
`4096+4*(8*f+g)+p`. Impossible/self-pair/promotion geometry is masked by our
adapter. Ordinary promotion squares never implicitly encode queen promotion.
Black rank reflection matches `utils.mirror_square/mirror_move`; legality,
terminal partitioning and normalization remain our existing/later helpers.

**WDL (M04).** Use encoder-final tokens, mean across 64 squares, the separate
`last_ln:LayerNorm(256)`, biased `Linear(256,128)`, ReLU, biased `Linear(128,3)`.
Policy uses final tokens before pooled `last_ln`. Remove the parallel ponder
head. Source `head_hid_dim` couples policy and value-hidden widths; our initial
values are both 128, independently named in metadata. Return raw logits;
softmax/value conversion belongs to consumers.
**Mismatch:** pinned `uci.wdl_from_value_logits` explicitly interprets raw
upstream logits as `[loss,draw,win]`. Our contract is `[win,draw,loss]`. Port
the head computation, assign our output rows/targets in contract order and
never reuse upstream class interpretation unchecked. We train new weights;
this is not checkpoint compatibility or a reason to change the adapter.

**Relative map/tables (M06).** For query i and key j in a1=0 rank-major order,
`delta=(rank_i-rank_j,file_i-file_j)` and
`r=15*(rank_i-rank_j+7)+(file_i-file_j+7)` in 0..224. Upstream factorizer
`F[225,4096]` has `F[r,64*i+j]=1`, other entries zero. Physical relative
parameters are `[head_depth*head_count,225]`; after multiplication by F they
reshape to `[head_depth,head_count,64,64]`, **depth before head**. A logical
table `R[H,225,d]` maps as `R[h,r,d]=upstream[d*H+h,r]`; reshape alone in
head-first order is wrong. Each layer has independent q, k and v tables, all
zero initialized; do not share them or reverse delta for the k term.

`RPELogits(rpe_type='q')` contracts query content with its table U:
`einsum('bhqd,dhqk->bhqk',q,U_gathered)`.
`RPELogits(rpe_type='k')` contracts key content with its independent table T:
`einsum('bhkd,dhqk->bhqk',k,T_gathered)`.
`RPEValue` contracts attention probabilities with its V table:
`einsum('bhqk,dhqk->bhqd',alpha,V_gathered)`.
These code labels describe the content operand; do not exchange U and T using
the paper's aQ/aK labels. Head splitting is `[B,64,H,d]→[B,H,64,d]`, followed
by inverse transpose/reshape to concatenate the output heads.

Combined logits are `(q_i·k_j + q_i·U_delta + k_j·T_delta)/sqrt(d) + GAB[i,j]`.
Softmax is over keys j; output is `sum_j alpha[i,j]*(v_j+V_delta)`.
The whole content/relative-logit sum is scaled once; GAB is added afterward,
as in the earlier generated-bias path. `scaleDivisor=dk**0.25` in the source
has commented-out q/k divisions and is not an extra scaling. Relative values
have no sqrt multiplier. Preserve Maia attention-probability dropout during
training on the combined attention weights for both ordinary and relative
values; the earlier TensorFlow operation itself has no probability dropout.
Reference output/gradient comparisons therefore use dropout=0. Retain Maia
Q/K/V/output projections and backbone; do not port the earlier optional input
Gating, logit Gating, quantization or smolgen backbone.

The earlier implementation adds no `dot(aQ_delta,aK_delta)` position-only cross
term. Preserve the released Q/K/V variant; no rationale for the omission is
established. Maia's `RelativeBias`/`use_relative_bias` produces static scalar
relative bias and is an alternative branch to GAB at this pin. It cannot
replace these content-dependent tables. Our GAB+Q/K/V combination is an
explicit project adaptation, not a configuration available in pinned Maia-3.

**Vocabulary and history references.** `utils.get_all_possible_moves` and model
flattening agree exactly with `pair4352-v1`. `dataset.get_historical_tokens`
concatenates the deque oldest-first and prepends copies of `board_history[0]`;
its inference helper pads regardless of `cfg.use_padding`. The pinned UCI
producer appends snapshots after moves, and `tokenize_board` orients each
snapshot to **that snapshot's** mover. Our accepted `chess128-v2` instead uses
newest-first 13-channel blocks, one current-mover ownership/reflection across
the entire window, each real prefix's repetition flag, and copies the earliest
complete block for missing history. Only the earliest-position padding idea
is borrowed. Upstream has no repetition channel in these snapshots and is not
an encoder for our 112+15+1 feature hybrid. Retain the implemented adapter.

**Preset mismatch.** Pinned `model_registry.BASE_SIZE_CONFIG/MODEL_SPECS`
(inventoried with a checksum) use inference dropout 0, RMSNorm, no absolute PE,
omitted Q/K/V biases, FFN ratio 2, and, for 5M, GAB mean pooling with dimensions
0/64/64 and head width 256. Our 8/32/32 compact generator, LayerNorm, additive
PE, biases, FFN ratio 1, dropout 0.1 and head widths 128 are locked project
settings supported by retained source operations, not those release defaults.
Do not claim identical architecture, parameter count, or pretrained-weight
compatibility. Complete parameter count remains a later M06 measurement.

## Acceptance and remaining work

Run `scripts/check_upstream.py` as described in the repository README. It
retrieves only pinned files, verifies full-file hashes and source symbols/ranges,
checks all 18 required inventory entries, immutable URL construction, retained
notices and repository identity/commit objects in checkout mode. It never
imports/executes source operations. `runs/f03/source-check.json` records the
actual Git-object and independent HTTPS checks and original-plan preservation.
Source availability is network-dependent; any retrieval/hash/coverage failure
fails the check. These checks establish F03 provenance, not numerical ports.

Remaining: M01 GAB and M02 backbone initialization are resolved in the recorded
PyTorch 2.8.0 CPU environment; M03–M06 constructors require their milestone checks.
The GPL full-text omission upstream is documented;
WDL/history/preset adaptations must be respected by later ports. M03–M06
numerical operations/gradients and all learning/search work remain unstarted.

Review acceptance — 7 October 2026: the user's decisions accept A07/Gate A
through independent AI review of `868371c` (119 tests in both existing
environments, 13 matching replay fixtures), and independently accept F03
`5958fd638fce0b6cf98a6631bacf743ce89551b8`, including the source checker and
five corruption cases. No human teammate review is claimed. This dated entry
supersedes the prior pending-review status; historical evidence in
`runs/f03/source-check.json` and the completion log is preserved. M01 only,
compact GAB with its shared bank and numerical validation, is authorized next.


M01 acceptance — 7 October 2026: `src/chess_rl/model/gab.py` adapts only the
compact generator and shared-bank initializer. The verbatim `_sq_bias` test
method is in `tests/fixtures/gab_reference.py` with retained attribution; its
bytes were checked against the full-file checksum and pinned Git object.
Output/input/parameter gradients passed against that method and an independent
functional path. `runs/m01/numerical-acceptance.json` saves actual errors,
tolerances, environment and sharing results. See `m01-gab.md` for constructor
defaults, commands and remaining limitations. Existing source notices and the
F03 historical evidence remain intact.


M02 automated acceptance — 7 October 2026: the pinned MHA/AbsolutePE/
EncoderOnlyBlock/CustomTransformerEncoder operations are adapted in
`model/attention.py` and `model/backbone.py`. Existing M01 generator and bank
are unchanged. Packed biased PyTorch MHA receives GAB as additive square-pair
logits; independent functional math checks every backbone stage and all
gradients. Resolved PyTorch 2.8.0 constructor source/hashes and CPU limitations
are in `runs/m02/backbone-acceptance.json`; see `m02-backbone.md`. One shared
bank and eight independently constructed blocks are retained. No Elo/time
conditioning, heads, relative operations or later work are implemented.

The user-supplied M01 review acceptance covers commit
`66eea36ba74ba811298ce8347a5153f562c474be`, implementation inspection, 139
model tests, 119 adapter-only tests with one expected skip, all 12 numerical
cases, sharing, both 13-fixture replays, source/dependency checks, original
planning hashes and clean tree; no blocking findings. This is independent AI
review, not human teammate review. Historical lock/notices and `runs/f03/`,
`runs/milestone-1/`, `runs/m01/` remain unchanged. M02 stops for review.
