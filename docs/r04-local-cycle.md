# R04 local CPU cycle — 10 October 2026

R04 starts from independently accepted R03 main
`2a41e270f7c5d9f3f2f074fac07de31699194ac2` in isolated
`r04-local-cycle` on `codex/r04-local-cycle`. The primary checkout and its
uncommitted user planning edits remain separate. Independent Gate I review is
required before acceptance or pilot spending; the R04 checkbox stays unchecked.

The immutable [CPU plan](../runs/r04/frozen-plan.json) is committed before
execution. It binds the unchanged `T03-plumbing-step-26` checkpoint, SHA256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`,
all accepted D07 inputs and protected registries, openings, seeds, exploration,
search budgets, update count and short match protocol. R04 never selects P.

Two ordinary games use eight simulations per root and a total 12-ply cap,
with the accepted R01 temperature 1 and Dirichlet noise fraction 0.25/alpha 0.3.
The fixed seeds are 40401/40402; opening selection seed is 2026101004.
If both truncate, two separately declared setup games start at halfmove 99,
with White/Black counterparts. Each is valid and nonterminal, has no pawn or
capture, and every legal continuation reaches an actual fifty-move draw.
They use the real unchanged T03 evaluator and search, budget eight and cap one.
Their full game history begins at the explicitly declared setup, with no
invented prior moves. Truncations remain unlabeled. The generation cap is
four games, 22 generated roots and 176 simulations.

R02 exclusively creates a new buffer. R03 applies the accepted held-out
whole-game/root filters, then freezes the complete eligible membership before
any optimizer execution. The source is the 12 accepted D07 training labels,
with a 50% mixture, batch 2+2, policy/WDL weights 1/1 and standard Adam at
`1e-5`. Sampling is independent shuffled cycles with seed 2026101004.
There is one update per full batch and three logical updates. Verification
performs six optimizer executions: full three, separate prefix one and
fresh-process resume two. The unique-root reuse ratio is recorded once the
eligible membership exists. There is no scheduler, clipping, accumulation,
prefetch, replay admission or eviction during updates.

The new boundary loads only T03 inference weights, creates empty Adam and
`MixedState.initial` at zero, and saves through the shared checkpoint API.
It never restores the supervised optimizer or reinterprets its cursor.
Full and prefix runs restore that new mixed step-zero checkpoint; resume
restores the one-update mixed checkpoint in another process with scrambled
RNG. Exact per-step and final fingerprints bind samples, logits, gradients,
model, Adam, both source cursors, sampling RNG, global RNG and progress.
Every saved checkpoint is reloaded through shared inference with exact model
identity and unchanged caller RNG.

E01 compares actual starting and updated fixture checkpoints using identical
search settings: eight simulations, c_puct 1.5, zero unvisited value, fresh
roots, no noise, max visits/smallest ID. The seed is 40405. Two paired openings
(four games) use cap eight: one ordinary start and one separate near-terminal
evaluation setup. The unused protocol test opening is never played. The cap is
18 generated plies and 144 simulations. E01's existing P/R slot names are
protocol routing only: both checkpoints have kind `fixture`, and
`protocol_only=true`; there is no pretrained-P selection. Ordinary and fixture
outcomes will be reported separately, with all attrition. No strength claim is
permitted.

CPU uses the existing pinned Python 3.12.12/Torch 2.8.0/NumPy 2.5.3 environment,
float32, deterministic algorithms, one compute and one interop thread. Original
accepted artifacts are retained; new checkpoint files live separately under
`external-artifacts/r04-local-cycle-20261010` with committed hash receipts.

Run `runs/r04/verify.py` with the existing model Python and `PYTHONPATH=src:.`.
Its phases are `prepare`, `freeze`, `generate`, `admit`, `boundary`, `full`,
`prefix`, `resume`, `evaluate`, `audit`, `model`, `adapter`, `preserve`.
Commit `frozen-plan.json` before `generate` and `eligible-membership.json`
before `boundary`. Each command retains untrimmed stdout/stderr, argv,
environment, UTC timestamps, duration, exit code and code hashes. Files are
created exclusively; reproduction requires a fresh checkout/output namespace.

## Executed evidence and review boundary

Plan commit `cec350b` precedes the first generated search. Membership commit
`bbd2e40` precedes the new mixed boundary. No plan settings were revised after
execution. The [summary](../runs/r04/summary.json) binds all receipts and files;
the [chain audit](../runs/r04/chain-audit.json) connects every completed generated
root, raw visits, played action, final outcome, original replay target, consuming
mixed updates, updated checkpoint and E01 evaluation.

| Actual bounded result | Evidence |
|---|---|
| Ordinary R01 games | Both truncated at total ply 12; 20 roots stay unlabeled and excluded |
| Separate R01 setup games | Two completed fifty-move draws, one generated root each |
| Generation work | 22 generated roots, 176 completed simulations; no interruption |
| R02 admission | Two admitted completed games, two excluded truncations, zero duplicates |
| R03 filtering | Two eligible roots, zero held-out exclusions; 12 audited supervised train labels |
| Replay policy support | Three and four visited actions; original distributions remain intact |
| New boundary | Exact T03 model, empty Adam, zero mixed progress; no legacy optimizer/cursor restored |
| Updates | Three logical updates; six supervised and six replay draws; six total optimizer executions across verification paths |
| Gradients | All 215 parameter tensors finite per update; every illegal policy logit gradient exactly zero |
| Persistence | All four new checkpoints reload exactly through shared inference without changing RNG |
| Continuation | Updates two/three and complete final state exact in fresh-process resume; final step-three training restore also exact |
| E01 | Two paired openings/four games, 18 generated plies, 144 simulations; two ordinary truncations and two fixture draws |
| Model regressions | 1,553 passed |
| Torch-free regressions | 1,023 passed, 14 expected optional-Torch skips |

| Update | Supervised policy / WDL | Replay policy / WDL | Supervised / replay contribution | Mixed total |
|---|---|---|---|---|
| 1 | 5.932779 / 0.058838 | 1.525581 / 3.247109 | 2.995808 / 2.386345 | 5.382154 |
| 2 | 0.766914 / 1.889881 | 1.521565 / 3.152577 | 1.328398 / 2.337071 | 3.665468 |
| 3 | 0.950802 / 2.358533 | 1.516626 / 3.081140 | 1.654667 / 2.298883 | 3.953550 |

Each row has batch size two per source. Independent float64 scalar legal
normalization/CE agrees with all four source losses and mixed totals within
`2e-6`. The sampling schedule consumes both eligible roots in each update,
crossing two replay epoch boundaries. This is 1.5 updates per unique eligible
root, 0.5 per consumed replay draw, or 0.25 per consumed mixed sample. Repeated
sampling is explicit; it does not create duplicate admitted games. Across the
full/prefix/resume verification paths, 12 samples per source are consumed.

The updated checkpoint is `R04-plumbing-updated-step-3`, SHA256
`967dde2cd8c2d3213c3297ae5dc5871f0e4db8faa11cb8145b8e55f919d3376b`.
It differs from T03 and matches the uninterrupted model/Adam/state exactly.
All new checkpoint paths and byte identities are in the
[artifact index](../runs/r04/new-external-artifact-index.json); no original
checkpoint is overwritten.

The ordinary E01 pair has no completed outcomes and contributes no score.
Only the separate evaluation setup pair completes, with two draws; its
conditional paired score is 0.5, with no uncertainty interval because only one
pair completes. These fixture draws establish protocol execution. They provide
no evidence of improvement, learning, ordinary-game completion rate or playing
strength. The untouched protocol test opening is unused. Both real model
identities remain explicitly `fixture`/`protocol_only`; E01's P slot does not
select pretrained P. The replay targets in this actual cycle are draw WDL only;
accepted R02/R03 regressions retain independent win/loss and malformed-target
coverage.

The first final audit incorrectly compared the in-memory manifest's tuple WDL
order with its JSON list representation. The lossless failed command and
stderr remain as `chain-audit.*`. The corrected audit compares the entire
canonical JSON identity, passes as `chain-audit-final.*`, and independently
replays roots/outcomes, checks legal visits and unchanged targets, reopens held-out
registries, verifies source metrics and exact final training restoration, and
replays E01. No generation, update or match was rerun for this audit correction.

[Preservation evidence](../runs/r04/preservation-after.json) verifies all 2,522
primary tracked files, the exact user planning patch, staged state and raw index,
and all 5,633 existing files under the external-artifact namespace. All accepted
checkout files remain exact except scoped documentation appends to README and
the coding plan; accepted historical reports remain unchanged. Sixteen frozen
inputs were rehashed after execution.

R04's automated bounded CPU cycle is complete. Independent Gate I review is
pending. The branch is committed and stops here: no main integration, CUDA,
pilot spending, pretrained-P selection or strength claim. R04 remains unchecked
until independently accepted.
