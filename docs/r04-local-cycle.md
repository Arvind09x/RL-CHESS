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
