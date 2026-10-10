# R03 mixed update — 10 October 2026

R03 is implemented on isolated `codex/r03-mixed-update`, based on accepted R02
main `66913809a3fa79d6a7a41f8157b5dbc425684051`. Automated bounded CPU checks
pass; independent review is pending. R03's milestone checkbox remains unchecked.
The primary checkout and its uncommitted planning edits are unchanged. This
branch stops before R04, integration, CUDA, new annotation, pilot or matches.

## Settings frozen before execution

Commit `d223be2` records the [frozen plan](../runs/r03/frozen-plan.json) before
any mixed optimizer execution. The training source is the accepted audited D07
view of `runs/d05/tiny-final`, dataset `lichess-elite-2025-01-pilot-v1`, split
`train`, containing 12 labels. The replay source is the existing accepted R02
four-root archive, export SHA256
`4bdd6f656e159cfe8fb46db65264e876fdee5abe6abb1fc6f85bec50cf6eedf4`.
The plan binds every input file, protected registry and replay sample identity.

| Choice | Frozen value |
|---|---|
| Supervised fraction | 0.5, strictly between zero and one |
| Batch composition | Two supervised plus two self-play samples |
| Optimizer | Standard Adam, learning rate `1e-5`, betas `(0.9,0.999)`, epsilon `1e-8` |
| Other Adam settings | Weight decay zero; AMSGrad, foreach and fused disabled |
| Policy / WDL weights | `1 / 1` in each source |
| Sampling | Independent shuffled cyclic permutations; tails carry across epochs |
| Sampling RNG | Dedicated Python Random, seed `20261010` |
| Replay membership | Frozen for the cycle; no admission, eviction or replacement |
| Update schedule | One update per full four-sample batch; constant learning rate |
| Bounded verification | Three updates from scratch; independent one-update prefix and two-update resume |

There is no scheduler, clipping, accumulation or prefetch. The learning rate is
one tenth of recorded T01's `1e-4` and one thirtieth of T03's `3e-4`. There are
0.25 updates per consumed mixed sample, or 0.5 per consumed self-play sample.
The bounded three-update path reuses four distinct completed roots for six
self-play draws; this is an explicit fixture schedule, not a pilot setting.
Full-run learning rate and update-to-data choices still require pilot evidence.

The fixture model is seeded combined M06 with default head widths 128, float32,
dropout 0.1 and new Adam at step zero. It is a plumbing model, never pretrained
P. CPU execution fixes one compute thread, one interop thread and deterministic
algorithms, using the existing pinned Python 3.12.12 / Torch 2.8.0 / NumPy 2.5.3
environment without dependency installation.

## Source and target contracts

`SupervisedDataset.source_identity` exposes an owned copy of D07's verified
manifest, authoritative input, bundle, summary, teacher evidence and split
identities. Existing callers remain compatible; manually constructed datasets
have no audited identity. R03 requires the nonempty audited training split and
binds its complete reference, feature, mask, class and WDL membership separately.

`MixedSources` obtains replay only through R02's audited `ReplayBuffer.export`.
Replay identity covers its complete manifest, ordered receipts and targets;
sampling membership records the ordered eligible sample IDs. Incomplete receipts
remain unlabeled and never contribute a sample. Repeated games retain R02's
existing admission and identity rules.

Protected held-out game and position registries must match D07's audited file
identities before and after reading. Completed replay trajectories matching a
held-out canonical game, and roots matching protected rule positions, are
excluded before sampling. An empty eligible replay view is rejected. This is an
R03 training view; it never edits the accepted R02 buffer. Supervised validation
and test views are rejected. Reopened source or target membership drift prevents
resume before global RNG is installed; mutable supervised data is rechecked
before resume and every update.

Supervised policy targets remain integer classes; teacher WDL stays soft in
root-side `[win,draw,loss]` order. Self-play policy remains the full normalized
root visit distribution; final WDL must be exact one-hot in that same root-side
order. No source's policy is hardened, converted into the other contract or
renormalized. Actual accepted replay draws have two or seven nonzero visit
entries, so these checks exercise soft distributions as well as unit targets.

The new `policy_distribution_cross_entropy` reuses shared legal log-softmax.
It rejects illegal mass, empty legal support, malformed or nonfinite targets,
non-unit sums and targets requiring gradients. Zero-mass entries select zero
log probability before multiplication, avoiding `0 * -inf` at illegal entries.
Every illegal logit has exactly zero gradient. Unrepresentable legal log
probabilities fail explicitly, consistent with the existing shared helper.

`mixed_losses` computes four mean losses with separate target contracts:

`L = f * (wp * CE_class + ww * CE_teacher_WDL)`

`    + (1-f) * (wp * CE_visits + ww * CE_final_WDL)`

`MixedSettings` requires the fraction to equal the exact integer batch ratio,
with both sources nonempty. `mixed_step` validates source identity, M06, owned
standard Adam configuration and update progress before forward. It performs one
combined forward and backward, checks every parameter gradient is finite, and
updates once. Returned metrics retain source sample counts, policy/WDL losses,
source totals, weighted contributions and the mixed total. The returned sampling
state is committed only after optimizer success. Failed inputs, logits or
gradients cannot advance the immutable state or call the optimizer. An arbitrary
optimizer failure is not transactional; resume from the last immutable boundary.

## Shared persistence and resumption

The existing `save_checkpoint`, `load_inference` and `restore_training` APIs
remain the common persistence path. Legacy `m06-checkpoint-v1` still saves and
loads D07 `DataOrder`, including existing checkpoint tests. Mixed checkpoints
use explicit `m06-checkpoint-v2-mixed`; their training schema replaces
`data_order` with `mixed_state`. Supplying both order types is refused.

Mixed state preserves settings, supervised source and complete membership hash,
replay export identity and ordered membership, both complete permutations,
both cursors/epochs/sample counts, dedicated sampling RNG and mixed update count.
The common payload still preserves exact model, Adam groups/moments/steps,
weights, progress, global Python/NumPy/Torch RNG, contracts, provenance and runtime.
Progress must match both source counts and Adam steps. The legacy single epoch
field is zero for mixed progress; each source owns its independent epoch.

Starting a new mixed cycle uses a new Adam at step zero and explicit frozen
source membership. Existing model weights may be loaded through common
inference; a D07 cursor cannot be reinterpreted as mixed sampling history.
Resume requires reopened `MixedSources` and explicit expected `MixedSettings`,
model configuration and provenance. A legacy-only resume request cannot restore
a mixed checkpoint; a v1 checkpoint cannot satisfy a mixed resume request.
Inference validates both formats without installing either training RNG.
Atomic immutable publication and the established compatibility gates are shared.

## Bounded evidence

The [summary](../runs/r03/summary.json) and lossless command receipts retain
argv, runtime environment, timestamps, exit codes and tested source hashes.

| Check | Result |
|---|---|
| R03 focused CPU cases | 56 passed |
| Full bounded CPU fixture suite | 1,553 passed |
| Torch-free adapter fixtures | 1,023 passed; 14 expected optional-Torch skips |
| Actual accepted D07/R02 fixtures | Three finite mixed updates, six draws from each source |
| Fresh-process continuation | Exact update two and three, including final state |
| Preservation | 16 frozen inputs and 2,377 primary tracked files unchanged; user plan edits exact |

| Update | Mixed loss | Independent float64 loss | Supervised contribution | Self-play contribution |
|---|---:|---:|---:|---:|
| 1 | 4.180292130 | 4.180292794 | 2.109327316 | 2.070965052 |
| 2 | 2.918438911 | 2.918439035 | 1.176491499 | 1.741947532 |
| 3 | 4.108428478 | 4.108428263 | 2.345609665 | 1.762818813 |

Every update has 215 finite parameter gradient tensors. Independent scalar
math computes the four losses without production normalization or Torch CE;
all loss differences are below `2e-6`. Focused tests independently calculate
every policy/WDL logit gradient, including exact illegal zeros, non-unit loss
weights, float32/float64, extreme logits and safe zero-mass entries. Tests also
cover malformed targets, held-out roots, unfinished receipts, source/registry
drift, optimizer failures and incompatible checkpoint settings/cursors/RNG.

[Full](../runs/r03/bounded-full-final.json),
[prefix](../runs/r03/bounded-prefix.json) and
[resume](../runs/r03/bounded-resume.json) run in three distinct CPU processes.
The resumed process scrambles global RNG first, reopens both audited sources and
restores the one-update boundary. It matches exact sampled references/IDs,
losses, logits, output gradients, all parameter gradients, model tensors, Adam,
progress, both cursors, dedicated sampling RNG and global RNG. Tensor equality
is checked through SHA256 over full bytes, dtype and shape; scalar/state equality
uses no tolerance. The synthetic fresh-process case additionally crosses both
source epoch boundaries. Actual replay crosses its epoch on update three.
The saved checkpoint is outside Git at
`/private/tmp/r03-bounded-boundary-final.pt`; its file identity and preserved
field inventory are committed in the summary. The runner reproduces it.

The historical M05 diagnostic plugin is used only in the full model fixture
suite. An initial adapter command incorrectly loaded this Torch-only plugin in
the Torch-free environment; its lossless failure is retained. The corrected
adapter command omits it and passes without changing production code. A passing
development run precedes the runner's stricter frozen-input gate; the final
recorded run supersedes it. No failed output is removed or trimmed.

These checks establish bounded CPU update and persistence correctness. They do
not establish learning, chess strength or accelerator behavior. Independent
review and R03 integration remain pending; R04 has not begun.
