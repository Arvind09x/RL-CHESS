# Accepted R03 integration — 10 October 2026

The user supplied independent AI review acceptance of freeze `d223be2` and
implementation `9fefb55` against accepted R02 main `6691380`. The review found no
actionable defects and accepts bounded deterministic CPU update and persistence
correctness. The supplied [report](../runs/r03-acceptance/review-report.txt),
[review summary](../runs/r03-acceptance/review-summary.json) and
[acceptance attribution](../runs/r03-acceptance/acceptance.json) are separate
from fresh integration verification. No new independent review was commissioned.

The isolated checkout is `r03-integration` under RL Project 2, on
`codex/r03-integration`. Its merge retains both original commits and exactly
reproduces the reviewed tree. No production correction or dependency change was
needed. The freeze commit predates mixed execution: 13:30:29 IST versus the first
retained mixed command at 13:38:59 IST. Historical implementation and review
artifacts retain their original bytes and pending-review wording as dated records.

| Fresh check | Result |
|---|---|
| Full model regressions | 1,553 passed, including 56 R03 cases |
| Torch-free adapter regressions | 1,023 passed; 14 expected skips |
| Supplied independent reviewer probes rerun | 22 passed |
| Frozen adapter replay in both environments | All 13 records reproduce exactly |
| Legacy v1 real-source fresh-process continuation | Exact next update, model, Adam, gradients, D07 order and RNG |
| Mixed full/prefix/resume in distinct processes | Exact updates two and three and final state |
| Accepted source reopen | 12 audited train labels and four completed replay roots; no held-out overlap |
| Sampling audit | Both permutations, cursors, epochs, counts and dedicated RNG reproduced independently over 101 updates |

Mixed runs use the original frozen 50% supervised fraction, 2+2 batch, Adam at
`1e-5`, unit policy/WDL weights and one update per full batch. Supervised class
and soft teacher WDL targets remain separate from visit distributions and exact
final replay WDL. Each source contributes six consumed samples across three
updates. Mixed losses reproduce `4.1802921295`, `2.9184389114` and `4.1084284782`.
All 215 parameter gradients are finite; illegal logit gradients are exactly zero.

The fresh resume scrambles global RNG before reopening both sources. Complete
step/final fingerprints match the accepted historical run exactly, including
model, Adam, gradients, progress, both complete sampling states and global RNG.
The new mixed boundary checkpoint also matches the historical bytes exactly.
Replayed sampling crosses the replay epoch on update three. Synthetic regressions
and reviewer probes additionally exercise both epochs, held-out whole-game/root
exclusions, unfinished receipts, source drift and compatibility rejection.
The [source audit](../runs/r03-integration/source-audit.json) records protected
registry identities/counts and historical v1 checkpoints loaded through the
common inference API with caller RNG preserved. Sixteen frozen inputs were
rehashed. Every unfinished receipt stays unlabeled.

New mixed and legacy checkpoints are stored separately under
`external-artifacts/r03-integration-20261010`; original checkpoints are never
overwritten. Their paths and identities are recorded in the
[new artifact index](../runs/r03-integration/new-external-artifact-index.json).
Lossless command output and source hashes are in `runs/r03-integration`;
[the summary](../runs/r03-integration/summary.json) binds their results.

Before primary integration, the exact original `coding-plan.md` file, binary
patch, raw Git index, staged entries and empty staged patch were saved losslessly.
All primary tracked files, the reviewed checkout, supplied review and 93
historical external artifacts were rehashed. Primary application refuses drift,
saves and verifies a recovery stash, fast-forwards local main, and reapplies the
original patch. Only the original D06/D07/E01 checkbox edits remain unstaged;
their patch hunks and original-file projection are exact. Blob hash headers
necessarily change because accepted R03 content is added to the plan. The raw
pre-integration index stays exact until application and remains saved; the new
index must match the integrated HEAD while preserving the original empty staged
state. The application receipt records full-tree and artifact verification.

R03 is accepted. R04 remains unchecked and unstarted; CUDA is not integrated.
Rejected gradients may consume dropout RNG; arbitrary optimizer failures can
partly mutate parameters, so exact retry restores the last immutable checkpoint.
Acceptance makes no learning, strength or accelerator claim. Work stops here.
