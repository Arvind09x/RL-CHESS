# Independent D04 and S04 review — 8 October 2026

Reviewed D04 `1042f1aa7f37298f2ef3a1b44eb2dceea4ed3f52` and S04
`4b7eee35f98f655296ced1263e10e5bd82541e0f`, each with exact parent
`37f286618642150b0ffa904abee0cfaac3ad75e8`. The coordinating Codex agent
performed independent AI review at the user's request; this is not human
teammate review. Both changes were combined without conflicts in
`/private/tmp/rl-chess-d04-s04-review` on `codex/d04-s04-review`.

## Findings

No blocking findings or production fixes were required. D04's four-field
actual-board identity conservatively removes shared pieces/turn/castling/legal
en-passant positions even when counters or network histories differ. Full
history remains authoritative for eligibility, encoding and future teacher
values. Material/ply phases and midpoint/farthest-progress sampling are explicit
heuristics, not strength or learning-adequacy claims. At most eight distinct
nonterminal positions are retained per game. Preparation legally checks every
source move; positions at and after the first environment terminal are excluded.

Test-first then validation-first ownership and deterministic refill establish
pairwise position separation. Same-split sharing across different games is
allowed; reference counts are not counts of globally unique positions.
Both held-out sample and game registries freeze before annotation. Expansion
revalidates original games/provenance/samples, preserves registry bytes, adds
only unprotected training references and quarantines new held-out games.

S04 counts successful nonempty backups as additional completed simulations,
preserves existing visits on reuse/failure, and returns snapshots independent
of mutable tree statistics. Training targets remain raw N/sum(N), independently
of action temperature. Root noise is explicit, uses supplied RNG, and restores
original root priors; descendant/evaluator priors remain unchanged. Terminal
roots return no action or training policy, including draws with legal moves.
Fresh zero-budget roots receive no fabricated policy. Ordinary failures expose
surviving statistics and exact new completions; interruptions propagate.

## Fresh checks

| Check | Result |
|---|---|
| Combined model regressions | 880 passed |
| Combined adapter regressions | 712 passed, six expected optional-model skips |
| Reviewer probes | 70 passed in each pinned environment |
| Frozen adapter replay | Both 13-fixture reports identical |
| Preservation | 698 of 700 accepted-base files unchanged; README/status are scoped changes |
| D04 resolved manifest metadata gate | Passed; no missing fields |
| Historical manifest metadata gate | Exit 1 as expected; original unresolved fields preserved |

The 17 new reviewer cases compare candidate eligibility/phase/identity against
independently replayed python-chess boards over 12 deterministic histories,
both colors and reached halfmove conditions. They verify cumulative versus new
simulation counts, policy snapshots, temperature/noise isolation, evaluator
failure after two new completions, repair/retry and terminal RNG isolation.
All 53 prior reviewer probes were also rerun. The model regression uses the
accepted M06 diagnostic plugin; reviewer files use importlib collection.

The full D04 audit reruns the separately spelled selector from the reviewed
implementation evidence, changing only its report destination and fresh replay
stride to every 991st ID-sorted game. It checks all game pools, selected
references, exclusions and owning splits, phase/coverage counts, registry
contents, pairwise separation and output hashes. Sampled roots are independently
replayed through the accepted adapter. This audit imports no production D04
selector or phase/identity helpers. It relies on preparation for unsampled
candidate completeness and terminal eligibility. Exact fresh replay counts are
in [structured acceptance](../runs/d04-s04-review/acceptance.json).

| Split | Initial references | Retained references | Covered games |
|---|---:|---:|---:|
| Train | 2,083,595 | 2,079,961 | 260,379 |
| Validation | 115,810 | 115,710 | 14,479 |
| Test | 115,416 | 115,416 | 14,430 |

There are 2,311,087 retained references, zero cross-split overlaps, 400,879
exclusions and 397,145 refills. The proposed 9,000/500/500 pilot allocation fits
reference capacity. All authoritative output identities match the existing
repeated selection. That repetition reverses candidate-cache insertion order;
it does not repeat full-corpus legal replay. Fixtures cover independent fresh
preparation, workers/order and append-only expansion.

Actual argv, cwd, UTC times, exit codes and lossless output hashes are in
`runs/d04-s04-review/`. Original implementation checkouts remain clean and their
code/evidence is preserved. The new D04 manifest differs from the preserved
original only in four preparation decisions; teacher settings and dependencies
are unchanged. Bulk artifacts remain external at `/private/tmp/rl-chess-d04-prepared`.

## Acceptance and next work

D04 and S04 are accepted for their scoped milestones through independent AI
review. This acceptance supersedes historical pending-review statements without
altering their evidence. The combined accepted change and review record are
prepared for fast-forward integration into local main; the chat reports the
verified integration commit. Nothing is pushed by this review.

D05 may now implement the Stockfish worker and run a tiny valid annotation
sample using `data/manifests/elite-2025-01-d04-v1.json`, with independent artifact
and teacher verification. The default historical manifest still intentionally
blocks annotation. D06's 1,000-position benchmark and D08's 10,000 successful
labels remain pending. Capacity does not establish learning adequacy, annotation
cost or feasible million-label training. No annotation or engine search occurred
during this review.

S05 may now implement a minimal game runner and dummy-search smoke checks.
Completed rule outcomes, truncation and interruption must remain distinct.
S05 is pending, so **Gate S is not accepted**. No real evaluator, training,
self-play learning or strength assessment is included.

Validation is limited to the existing CPU macOS arm64/Python 3.12.12 environments.
The review does not repeat full-corpus legal preparation or test real-corpus
expansion. Sequential ownership, no concurrent tree writes and no process-crash
durability remain S04 limitations.
