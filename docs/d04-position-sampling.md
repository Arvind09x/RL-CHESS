# D04 deterministic position sampling — 8 October 2026

D04 starts from exact accepted base `37f286618642150b0ffa904abee0cfaac3ad75e8`
on `codex/d04-position-sampling` in `/private/tmp/rl-chess-d04`. D03 and S03
are independently AI review accepted and already integrated into local main.
Historical statements about the primary checkout remaining unchanged describe
the earlier review boundary. D04 is implemented for independent review; no D04
review acceptance or merge is claimed. No engine search, annotation, benchmark,
training, teacher change or source-result targets are produced.

## Intake and compact references

Before replay, preparation checks the accepted split audit SHA-256
`029e3e78cf8114ec68796ef35644c67749184bf060c4a3321e3107b3f0534a36` against the
manifest, then verifies its completion/version/archive/split contract and
`games.jsonl` identity: 638,770,177 bytes, SHA-256
`f9f69d255b652f45276ee6163857b3a9c8054ac5ff037f36c94e9ecb039f9df5`.
These match the accepted D03/S03 review evidence. Input bytes are rehashed
while processing and again before completion. Every game ID and seed-based
assignment is recalculated; the sorted assignment checksum, split game/source/
ply counts must match D03. Inputs are opened read-only; a new output directory
is required. Failure leaves no completed audit.

Bulk files remain in `/private/tmp/rl-chess-d04-prepared/`. Canonical games and
all original intake/source/result provenance remain in their existing locations.
Unlabeled `position-reference-v1` samples store only `game_id`, `sample_ply`,
`split`, `phase` and `rule_position`, plus record version. No move array, network
tensor or teacher target is duplicated. Joining the original game and manifest
supplies full root/counter/history/version/source provenance. Replay the actual
prefix through `make_state(..., history_complete=True)`; available real snapshots
are `min(sample_ply+1,8)`. D05 must extend the label schema and validate teacher
outputs separately. PGN results never select phase, adjudicate roots or become
labels.

The external `candidate-pools.sqlite` contains compressed per-game candidate
triples and a selection table. It is a preparation cache, not authoritative
canonical games or trusted expansion evidence. JSONL sample/registry/exclusion
outputs are deterministically sorted; SQLite bytes are not a reproducibility
contract and are not committed.

## Phase and sampling decisions

`material-ply-v1` evaluates actual board material: both colors' nonpawn,
nonking pieces with N=3, B=3, R=5, Q=9. Combined value <=14 is **endgame**, with
priority over opening. Otherwise plies 0–19 from the declared PGN start are
**opening**, and ply >=20 is **middlegame**. Counters do not set phase. Thus
setup games can begin in endgame; short games need not contain all phases.
This is an explicit coverage heuristic, not a claim of uniquely correct chess
phase classification. Version changes require new reviewed dataset decisions.

`phase-spread-sha256-v1`, seed **20261009**, works independently per game:

1. Replay every move legally with the accepted adapter's `require_legal_move`
   and retain the full actual move stack. Query its `terminal_result` for every
   prefix until the first `actual-occurrence-v1` terminal. Root ply 0 and the
   final source ply can be sampled if nonterminal. Exclude the first terminal
   root and its entire suffix; legally validate historical suffix moves anyway.
   A next-move claim does not end the preceding root. Source resignation/results
   do not make the final board terminal.
2. Keep only the earliest eligible occurrence of each rule position within the
   game. This prevents repeated positions counting as distinct samples, even
   when histories/counters differ. Assign its recorded phase.
3. For missing available phases in opening/middlegame/endgame order, select the
   root nearest that phase's earliest/latest-ply midpoint (compare doubled
   integer distances). Then fill to at most eight by maximizing minimum ply
   distance from already selected roots.
4. Break ties with SHA-256 of UTF-8 compact sorted-key ASCII JSON containing
   exactly `game_id`, integer `sample_ply`, `sampling_version` and integer `seed`,
   then by ply. No stateful PRNG, float distance or corpus rank is used.

Selection includes each available phase when capacity permits, distributes
roots across actual progress and remains stable under input reordering and
worker count. Repeated same-input fixture preparations compare all authoritative
output bytes and audit bytes. Input reordering changes the intake hash in the
audit, while selections and output reference/registry bytes remain identical.

## Overlap identity, ownership and refill

`rule-position-disjoint-v1` is the exact **first four fields** of actual-board
`board.fen(en_passant="legal")`: pieces, actual side to move, castling rights
and legal en-passant availability. It is stored as a string, without a digest
collision fallback. Legal en-passant distinguishes positions; irrelevant or
pinned en-passant targets normalize to `-`.

Halfmove/fullmove counters and repetition occurrence/history are excluded from
overlap identity. They still remain in replay and can change terminal handling,
network encoding and future teacher values. In particular, histories differing
in repetition planes or root counters **must not hide cross-split overlap**.
Game identity includes six-field start/counters and the entire mainline;
network encoding includes oriented history/repetition features. Neither is the
position-overlap identity.

Finalize selected **test first, validation second, train last**. A higher
split owns its final selected rule identities. Keep all lower-split initial
survivors unchanged, record each excluded reference with its owning split, and
refill from that game's remaining eligible candidates using the same phase/
spread selector, blocking every higher split's final selected identities.
Refill can use a different ply/phase; no game is reassigned. If alternatives
are insufficient, retain fewer than eight or zero, and report it. Validation
refill completes before training sees the protected union. Same-split positions
can occur in different games; counts distinguish reference capacity from
unique rule identities. Positions within one game are distinct. Pairwise
cross-split identities must be disjoint after refill.

## Freeze, pilot capacity and expansion

`protected-heldout.jsonl` freezes every final validation/test sample, including
its game/ply/history reference and rule identity. `heldout-games.jsonl` freezes
all validation/test game assignments, including games with zero samples.
The manifest pins the combined protected sample registry SHA-256; the overlap
audit pins both registry files and the source assignment checksum. The registry
contains full selected heldouts before any labels, irrespective of the initial
label allocation. Keep validation and test separate, and exclude both from
supervised/RL training and replay.

The initial 10,000-successful-label plan can use 9,000 train, 500 validation and
500 test records from the audited capacity. D04 produces **zero labels**; failed
annotations cannot count toward that target. Later 100,000/several-hundred-
thousand/optional-million allocations must fit the measured per-split capacity
and subsequent D06 costs/learning evidence. A separate 1,000-position benchmark
still belongs to D06 and has no selected membership here.

`expand_d04.py` requires a new accepted canonical D03 table/audit and explicit
SHA-256 for both the new split audit and previous overlap audit. It verifies
previous game/sample/registry bytes, revalidates all existing identities and
assignments, requires every old source occurrence with multiplicity to remain,
and replays all new-table games through the same adapter procedure. Existing
samples are locked and must remain eligible and exactly represented in their
new replay. Existing games cannot disappear or change assignments. It copies
both heldout registries byte-for-byte. New validation/test games keep D03
assignments but are quarantined from sampling under this frozen registry.
It adds only training candidates that avoid every protected identity, at most
eight total per game. Existing training samples also remain unchanged; prior
holes can fill only with eligible unprotected positions. Expansion writes a new
sorted sample table and `position-expansion-audit-v1`, preserving the prior
files. Repeat expansion is idempotent; each accepted expansion must use its
own new manifest/audit identity while retaining the protected registry hash.
No automatic refill after annotation failure or registry mutation is allowed.

## Execution, evidence and limits

Use the existing `.venv-acceptance` and `.venv` in the primary repository,
without syncing dependencies. Actual commands, UTC times/durations, exit codes
and lossless stdout/stderr byte counts/SHA-256s are in `runs/d04/*.command.json`.
`run_command.py` sets `PYTHONPATH` to this isolated checkout and disables bytecode.
The model suite uses the accepted M06 diagnostic plugin for historical M05
behavior. Accepted code, tests, pins and all historical evidence remain unchanged.
The new `elite-2025-01-d04-v1.json` manifest resolves only four D04 fields;
the accepted original manifest remains byte-identical. Metadata completeness
in the new manifest is not D04 review acceptance or permission to start D05.

Reproduction commands (new output directories are mandatory):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/prepare_d04.py --manifest data/manifests/elite-2025-01-d04-v1.json --games /private/tmp/rl-chess-d03-prepared/games.jsonl --split-audit /private/tmp/rl-chess-d03-prepared/split-audit.json --split-audit-sha256 029e3e78cf8114ec68796ef35644c67749184bf060c4a3321e3107b3f0534a36 --output-dir /private/tmp/rl-chess-d04-new-run --workers 4
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest tests/test_position_sampling.py -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python runs/d04/audit_output.py /private/tmp/rl-chess-d04-prepared
```

Preparation replays the complete corpus. The separate full audit checks every
candidate pool, baseline selection, removal/refill choice and retained reference
with a separately spelled selector; it recomputes split/phase/coverage counts,
all pairwise overlaps, output/registry bytes and exclusion ownership. Every
997th ID-sorted game is freshly replayed through `make_state` at all retained
roots to verify FEN/phase/nonterminal history. The audit does not independently
rerun terminal checks at every unselected root; those are checked by the full
preparation. Full-corpus reordering, a second full replay and real corpus
expansion are not run; reproducibility/order/worker/expansion are fixture-tested.
CPU macOS arm64/Python 3.12.12 only; other platforms, cost, throughput,
annotation failures, GPU and learning adequacy remain unmeasured. Preparation
elapsed time is recorded solely as execution evidence, not an engine benchmark.

The first file-expansion fixture expected additions from a new game containing
only protected rule positions. The pipeline correctly added zero. The fixture
was extended with distinct legal positions to exercise both rejection and
successful append. The original failure and corrected passing run are preserved.

The first complete output-audit attempt was interrupted (exit 130) after a
redundant query repeatedly enumerated every same-split occurrence of common
protected positions. The audit already independently derived the complete
expected exclusion stream and its owning splits; removing the redundant query
preserved that full check. The full corrected audit was rerun. The interrupt
reached the output-capturing wrapper before it saved partial child logs/times;
this limitation and the exact argv/exit are preserved in
`runs/d04/output-audit-interrupted.command.json`. No preparation output was changed.

## Actual corpus results

| Split | Eligible distinct candidates | Initial samples | Removed | Refilled | Retained | Covered games | Unique rule identities |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 22,911,825 | 2,083,595 | 381,315 | 377,681 | 2,079,961 | 260,379 | 1,851,150 |
| validation | 1,268,218 | 115,810 | 19,564 | 19,464 | 115,710 | 14,479 | 105,729 |
| test | 1,264,674 | 115,416 | 0 | 0 | 115,416 | 14,430 | 98,179 |

| Split | Opening before / after | Middlegame before / after | Endgame before / after |
|---|---:|---:|---:|
| train | 599,612 / 448,892 | 1,212,005 / 1,352,841 | 271,978 / 278,228 |
| validation | 33,486 / 28,577 | 67,425 / 71,968 | 14,899 / 15,165 |
| test | 33,385 / 33,385 | 66,909 / 66,909 | 15,122 / 15,122 |

Full replay: **25,712,214 roots** across **289,427 games**. Excluded 70,437 first terminal roots, 7,506 suffix roots and 189,554 repeated nonterminal rule occurrences.

Initial selection pairwise distinct overlap identities: {"train-test": 8583, "train-validation": 8666, "validation-test": 3139}. Final overlaps are zero across every pair of splits. Exclusion files record initial-sample removals; blocked candidate counts also include candidates bypassed during refill.

Total available retained reference capacity: **2,311,087**. Fresh independent adapter replay: 2,320 retained roots from 291 games. Each reference remains unlabeled.

| Target total successful labels | Train / validation / test proposal | Fits current reference capacity |
|---:|---|---|
| 10,000 | 9,000 / 500 / 500 | yes |
| 100,000 | 90,000 / 5,000 / 5,000 | yes |
| 500,000 | 450,000 / 25,000 / 25,000 | yes |
| 1,000,000 | 900,000 / 50,000 / 50,000 | yes |

These are capacity checks, not successful-label or cost claims. Larger scales remain optional and require later measured annotation and learning evidence.

Protected sample registry SHA-256: `813d40ec52f3058996ddf6d7586dea23ee568c3bcff995cd92879e306a1f8ed3`. Overlap audit SHA-256: `feb20d4e38fa46ff6e5fcc77b4c4b8d1b36b985f3dcf6a3785a412a2fbb91f5e`.

Final checks: focused-final: 37 passed, final-model-regression: 767 passed, final-adapter-regression: 599 passed, 6 expected skips, reviewer-adapter: 53 passed, reviewer-model: 53 passed. Both accepted 13-fixture adapter reports match byte-for-byte; pinned upstream and M06 checks pass. All final success commands exit 0.

[Complete compact result](../runs/d04/completion.json), [output audit](../runs/d04/output-verification.json), [repeated selection](../runs/d04/reproduction.json), [actual commands](../runs/d04/commands.json) and [preservation](../runs/d04/preservation.json).
