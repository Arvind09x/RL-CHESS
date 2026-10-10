# Accepted R04 / Gate I integration — 10 October 2026

The user-supplied independent review accepts the complete original sequence
`cec350b → bbd2e40 → 01f3c3f` against accepted R03 `2a41e27`, with no actionable
findings. R04 and Gate I are accepted for bounded deterministic CPU plumbing.
The [supplied report](../runs/r04-acceptance/review-report.txt),
[review summary](../runs/r04-acceptance/summary.json) and
[acceptance attribution](../runs/r04-acceptance/acceptance.json) are separate
from [fresh integration verification](../runs/r04-integration/summary.json).
No new independent review was commissioned.

Integration was prepared in isolated `r04-integration`, on
`codex/r04-integration`. The merge retains all three original commit identities
and exactly reproduces the reviewed tree. Original freezes, historical reports,
checkpoints, failed audit receipt and supplied review failure receipt are kept
unchanged. Historical pending-review wording remains a dated record; this
acceptance entry supplies the current status.

| Fresh integration verification | Result |
|---|---|
| Model regressions | 1,553 passed |
| Torch-free regressions | 1,023 passed, 14 expected optional-Torch skips |
| Plan and membership freeze chronology | Original committed bytes exact; both freezes precede governed execution |
| Native chess/root/target audit | 22 roots; 20 ordinary truncated roots excluded, two completed setup roots |
| Fresh real-search generation and R02 admission | All configurations, visits, moves, history, RNG, receipts and exported replay exact |
| Held-out/source membership | 12 audited supervised train labels, two eligible replay roots, no held-out overlap |
| New mixed boundary | T03 model exact, empty Adam, zero mixed progress, no legacy cursor |
| Full / prefix / resume in distinct processes | Exact steps and complete checkpoint payloads; scrambled RNG restored |
| Original and fresh checkpoint reloads | All eight inference and full training-state restores exact |
| E01 reproduction | Entire four-game record exact; two ordinary truncations, two fixture draws, one complete pair |

Fresh updates preserve the original visit distributions and exact draw WDL.
Independent float64 cross-entropy checks agree within `2e-6`; features, losses,
logits, logit and parameter gradients, model, Adam, both source sampling states,
sampling RNG, global RNG and progress match every recorded update.
The saved payloads also preserve model mode and provenance. E01 uses identical
search settings and reversed player colors with no noise. Its completed-game
denominator is two; only the one completed fixture pair supplies score 0.5.
These diagnostic fixture draws provide no evidence of learning or strength.
Both checkpoint roles remain fixture/protocol-only; pretrained P is unselected.

Starting T03 retains SHA256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`.
Original updated R04 retains SHA256
`967dde2cd8c2d3213c3297ae5dc5871f0e4db8faa11cb8145b8e55f919d3376b`.
Original serialized checkpoint identities are rehashed; fresh files have their
own byte identities and exactly matching complete payloads. Fresh checkpoints
are in `external-artifacts/r04-integration-20261010`, with a separate
[artifact index](../runs/r04-integration/new-external-artifact-index.json).
All command output is lossless; receipts bind argv, environment, timestamps,
exit status, source identities and the tested isolated commit. The verification
harness adapts supplied review probes to this isolated tree and new namespaces;
the original review harness remains exact.

Before application, the original primary plan file, binary patch, raw index,
staged entries, empty staged patch and stash inventory are saved losslessly.
Fresh verification preserves every original source/review file and 5,652
historical external/input files. Primary application refuses drift, verifies a
recovery stash, fast-forwards local main and reapplies the exact original patch.
Only the original D06/D07/E01 checkbox edits remain unstaged. Their patch hunks
and original-file projection remain exact; diff blob hashes change with the new
accepted plan content. The original raw index remains exact until application
and is retained in the backup. The current index must track integrated HEAD
while preserving the user's original empty staged state.
The [application receipt](../runs/r04-integration/main-integration.json) records
final main verification.

Work stops after R04 / Gate I acceptance. CUDA integration, pilot execution and
pretrained-P selection have not begun.
