# D01 dataset manifest preparation — 8 October 2026

D01 prepares configuration and provenance only, and stops for independent
review. It has no independent review acceptance yet. The selected
[manifest](../data/manifests/elite-2025-01-v1.json) is preparation-ready and
**not annotation-ready**. No archive body, PGN replay, splitting/sampling
algorithm, annotation worker, benchmark, labels, training or search is produced.

## Source, identity and storage decisions

Select **January 2025**, dataset ID `lichess-elite-2025-01-pilot-v1`, from the
[Lichess Elite index](https://database.nikonoel.fr/):
`https://database.nikonoel.fr/lichess_elite_2025-01.zip`. The index links this
exact monthly ZIP; HEAD returned HTTP 200, ZIP content type, Content-Length
82,724,777 and Last-Modified 4 February 2025. These are server observations,
not a downloaded-byte checksum or an archive coverage measurement. The index
identifies its source as standard Lichess games filtered to 2500+ versus 2300+
from December 2021, excluding bullet. D02 must still reject variants, invalid
boards and illegal/incomplete source replay, and record actual archive/member
identity, download time, compressed byte count and SHA-256 of the downloaded
ZIP bytes. [Source verification](../runs/d01/archive-provenance.json) records
index SHA-256, selected link, HEAD response and verification time. The manifest
has explicit nulls for all three pending archive retrieval fields.

D01 selects split seed **20261008** and sampling seed **20261009** (uint32).
It does not define a PRNG, split assignment function or sampling/phase algorithm.
D03 must record the deterministic seeded 90/5/5 assignment procedure by game
before any position sampling. The percentages are assignment targets; exact
small-corpus counts/rounding must be resolved and audited in D03. D04 records
sampling, phase boundaries and deterministic removal/refill/expansion rules.
Those manifest fields remain null, preventing accidental readiness claims.

`start-moves-sha256-v1` defines both canonical game ID and duplicate identity:

- Normalize the actual starting board to standard-chess six-field
  `board.fen(en_passant="legal")`, including its actual counters. Retain the
  source-declared start and full legal actual-coordinate UCI mainline, with
  complete-history provenance. Do not infer missing history from a root FEN.
- Hash UTF-8 JSON containing exactly `identity_version`, `start_fen` and
  `moves_uci`, with sorted keys, `separators=(",", ":")`, `ensure_ascii=True`
  and no trailing newline. `identity_version` is `start-moves-sha256-v1`;
  `moves_uci` is the entire normalized actual UCI mainline array.
  `game_id = "g:" + sha256(payload).hexdigest()` (lowercase).
- Identical replay payloads share an ID regardless of player/header/result
  differences. Retain all source archive/member/one-based PGN ordinal and
  optional `LichessURL`/Site identities as provenance, rather than making a
  source URL necessary for identity. Conflicting source results are retained
  as source discrepancies and never become teacher targets. D03 must compare
  replay payloads when consolidating IDs and reject hash/payload conflicts.

D03 implements and tests this convention; D01 only records it. Distinct
starting counters intentionally imply distinct replay IDs; D04's broader rule
position audit still prevents leakage between their samples.

`replay-jsonl-v1` uses separate UTF-8 JSONL tables, one object per line, with
shared manifest versions and dataset ID. `games.jsonl` stores each authoritative
replay once: game ID, split, starting FEN, full actual `moves_uci`, complete
history, ending-policy ID, source provenance and separate PGN result metadata.
`samples.jsonl` references `(game_id, sample_ply)` (plies applied, including 0
for the genuine start); it retains root FEN, real snapshot count/history
provenance, versions, best-move UCI/class ID, raw and soft root WDL, returned
search/score/mate information and teacher config ID when labeled. The joined
record supplies all named fields in `interface.md`; no sample repeats the full
move list. Features and legal masks are rebuilt through the accepted adapter
or explicitly versioned caches. No dense stored one-hot teacher policies.
D02/D05/D07 will implement replay/label/load schemas when needed; this task
introduces only manifest/configuration schema code.

## Sampling, audits and protected sets

D01 locks **90/5/5 train/validation/test by game before sampling**, at most
**eight distinct nonterminal positions per game**. Distinct sample plies and
rule position identities are both required, avoiding repeated occurrences as
nominally distinct samples. D04 must use the shared adapter to adjudicate roots,
retain full history, spread samples across game progress, and provide dataset
opening/middlegame/endgame coverage using recorded phase boundaries.

`rule-position-disjoint-v1` audits pieces, actual side to move, castling rights
and en-passant availability affecting legal moves. It excludes halfmove and
fullmove counters, PGN headers, network orientation and network history tensor
differences. D04 must remove overlapping-position samples across every pair
of splits before annotation and record pre/post split and phase counts,
duplicates, rejected games, removals and deterministic refill behavior.
The audit is deliberately broader than network input equivalence.

Freeze validation/test game assignments and selected sample registries before
annotation; they remain separate and never enter supervised/RL training or
replay. Expansion retains game assignments and protected held-out identities,
repeats overlap audits, records actual counts and explicit deterministic
refill/new-training-sample rules. How heldouts take precedence during removal
and how expansion protects them are D04 decisions, not implemented here.
Manifest preparation evidence must identify the actual split/overlap audit
artifact SHA-256s and protected held-out registry SHA-256 before annotation.

The initial production target is **10,000 successful labels total**, distinct
from the **separate 1,000-position benchmark**. D06 must keep a separate run
identity/count and report its audited sample membership, without assuming
benchmark labels satisfy the D08 target or releasing heldouts for training.
Failures/exclusions never count toward either success target. Expansion to
100,000, several hundred thousand or potentially one million follows measured
coverage, annotation/replay/encoding/storage costs, training throughput and
held-out learning curves. No archive coverage, throughput, learning adequacy
or cost is measured here; one million is optional.

## Teacher provenance and contract

Teacher: **Stockfish 17.1**, official release tag `sf_17.1`, source commit
`03e27488f3d21d8ff4dbf3065603afa21dbd0ef3`. GitHub's official macOS M1 binary
asset host repeatedly reset connections (curl exit 35); the same release was
obtained as its official source tarball:
`https://codeload.github.com/official-stockfish/Stockfish/tar.gz/refs/tags/sf_17.1`.
This is a **local build of an official release**, not the downloaded official
binary asset. Its source-package hash and tag metadata are in the manifest and
[provenance evidence](../runs/d01/stockfish-provenance.json).

Build: macOS 14.5 arm64, Apple clang 15.0.0 (clang-1500.3.9.4),
`make -j2 build ARCH=apple-silicon COMP=clang`, O3, full LTO, NEON dot product.
The exact compile/link commands and inherited include/library flags are saved
in [the build log](../runs/d01/stockfish-build.log). The `build` target skips PGO
and runs no benchmark/search. Dynamic libraries are only system libc++ and
libSystem. No system install was performed.

Runnable executable: `/private/tmp/Stockfish-sf_17.1/src/stockfish`.
**Executable SHA-256:**
`3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`.
Copying these exact bytes preserves the identity; rebuilding may produce a
different hash and needs a deliberate new provenance record and verification.
The temporary executable/source directory is a local artifact outside Git.

Both default NNUE assets were retrieved from the official Fishtest NN endpoint,
checked against filename hash prefixes and recorded with their complete hashes:

| Embedded asset | SHA-256 |
|---|---|
| `nn-1c0000000000.nnue` | `1c0000000000a67d629999d932d0c373f7450ce43cd12d0562868f4eaf9ae2ad` |
| `nn-37f18f62d772.nnue` | `37f18f62d772f3107e1d6aaca3898c130c3c86f2ab63e6555fbbca20635a899d` |

The build evidence finds each complete NNUE byte blob inside the executable
and records its offset/length. **No separate runtime engine assets are required**
for these default networks. Syzygy is disabled; no tablebases were obtained.
Keep the official source release's GPL-3.0-or-later notice and source with the
local build; no engine binary/source is vendored into this project.

Idle `uci`, settings, `Clear Hash`, `isready`, `compiler`, `quit` verification
passed from an empty temporary directory. Identity is `Stockfish 17.1`; required
Threads/Hash/MultiPV ranges, UCI_ShowWDL and Clear Hash capabilities are present.
[Handshake and manifest check](../runs/d01/manifest-check.json) saves exact
commands, outputs and checksum. This proves executable identity and idle UCI
capabilities, **not search correctness, NNUE evaluation, WDL output completeness,
node-budget behavior or annotation performance**. D05 must verify those with
actual annotations. No `go`, `bench`, `speedtest` or `eval` command was sent.

`stockfish-17.1-100kn-v1` locks 100,000 requested nodes, Threads=1, Hash=128 MB,
MultiPV=1, UCI_ShowWDL=true, standard chess, full-history replayed roots and
**Clear Hash before each independent annotation**. Keep other UCI defaults
from the pinned executable; explicitly disable pondering, strength limiting,
Chess960 and Syzygy. D05 records actual returned nodes/search info separately
from requested nodes. Store the root PV's first legal actual move and transformed
best-move ID; raw WDL counts and soft `[win,draw,loss]/1000` use the root side to
move, with finite/nonnegative/unit-sum validation. Missing/interrupted output,
missing WDL/PV, invalid move or malformed counts is a failed attempt, retried
or explicitly excluded, never a successful label. No centipawn conversion,
partial-output acceptance or PGN-result substitution.

## Validation and reproduction

`chess_rl.data_manifest` imports ENCODING_VERSION, MOVE_VOCAB_VERSION,
ENDING_POLICY and WDL_ORDER from the accepted shared contract. Frozen records
and strict JSON loading reject incompatible versions, malformed provenance,
placeholder text, unknown/missing/duplicate fields, wrong types (including
boolean/integer coercion), invalid settings and policy changes. Canonical
serialization round-trips the selected manifest byte-for-byte.

`annotation_blockers()` lists unresolved archive retrieval fields and D03/D04
algorithm/phase/audit/heldout fields. `require_annotation_ready()` fails closed;
a caller-provided `annotation_ready=true` field is rejected. The gate establishes
metadata completeness, not the truth of supplied audit attestations. Future
consumers must verify referenced artifact bytes/contents, replay legality,
root history and target validity; D01 does not implement those tasks.

From this worktree, using the installed pinned environments without syncing:

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:.
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest tests/test_data_manifest.py -p no:cacheprovider
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/check_d01.py --teacher-executable /private/tmp/Stockfish-sf_17.1/src/stockfish
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/check_d01.py --require-annotation-ready
/Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python -m pytest -p no:cacheprovider -p runs.m06.diagnostic_plugin
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest -p no:cacheprovider
```

The annotation-ready command must exit nonzero on this manifest. Actual commands,
exit codes, focused/full suite results, M06 numerical checker, both adapter
replays and protected-file audits are saved under `runs/d01/`. The accepted python-chess UCI transport also passed idle identity/configuration/
ping checks without analyse/play calls; python-chess manages MultiPV through
the future analyse(multipv=1) call. The historical
M06 diagnostic plugin remains required for historical M05 tests; production
model defaults and prior tests/runs are unchanged. CPU correctness only;
GPU correctness/performance and all data-scale/cost claims remain unverified.


Actual results: **83 focused D01 tests**, **370 passed in the model environment**
(287 historical + 83 D01, using the M06 diagnostic plugin), and **202 passed /
six expected skips in the adapter environment** (119 historical + 83 D01).
The full M06 checker, both 13-fixture adapter replays, pinned source audit,
Torch-free import boundary and both UCI probes passed. All success commands
exited 0; the required incomplete-manifest annotation gate exited 1 as expected.
[Actual commands](../runs/d01/commands.json),
[retrieval commands including failures](../runs/d01/retrieval-commands.json),
[preservation hashes](../runs/d01/preservation-check.json) and
[focused file inventory](../runs/d01/change-manifest.json) are retained.
