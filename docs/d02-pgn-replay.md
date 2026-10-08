# D02 PGN source replay — 8 October 2026

D02 prepares **pre-split source mainlines** from the exact selected January
2025 archive. Independent review of D02 remains pending. It starts from merged
base `0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`, containing the independently
accepted D01 and S01 work. Their separate branches remain available. The
user-supplied planning-agent AI review attribution is recorded in
[runs/d02/review-integration.json](../runs/d02/review-integration.json).
This attribution is not a new review performed by the D02 implementer.

## Download and byte identity

Requested and final URL:
`https://database.nikonoel.fr/lichess_elite_2025-01.zip`.
The actual download returned HTTP 200 with **zero redirects**. Its exact byte
count is **82,724,777** and downloaded-byte SHA-256 is
`f2fa14565bcdaba7ad6de6a4f8f2348d0f9f8f262935c46e61925c5acce6f7b7`.
UTC retrieval time is **2026-10-08T07:45:09.776038+00:00**, taken from the
completed curl output file's modification time; it is not server Last-Modified.
The manifest's three retrieval fields were resolved together, and its evidence
reference now links the downloaded-byte provenance while retaining D01's
original index/HEAD evidence. The sandboxed attempt exited 6 (DNS); the
user-approved network retry exited 0. Exact argv, observed curl fields and raw
response headers are under [runs/d02/](../runs/d02/archive-provenance.json).

There is one ZIP member, `lichess_elite_2025-01.pgn`, with **274,106,096**
uncompressed bytes, **82,724,577** compressed member bytes, CRC32 `5e3f587e`,
and uncompressed SHA-256
`44162cdc6bf4611cff64e3a0f741d5cf524cc50100e09c56bd566ba3dabce83d`.
ZIP member timestamp, compression method, central-directory ordinal and local
header offset are also recorded. The whole ZIP size includes its metadata;
it is not interchangeable with the compressed member size. Streaming the
complete member verifies ZIP CRC/decompression and hashes every byte,
including inter-game whitespace. No extraction to a repository path occurs.
The ZIP remains at `/private/tmp/lichess_elite_2025-01.zip`.

## Package boundary and source rules

`chess_rl.pgn_replay` supplies binary source framing, strict syntax checking,
`parse_mainline`, `replay_record`, `replay_state`, and `prepare_archive`.
`scripts/prepare_d02.py` only handles arguments, progress and the report.
The pinned python-chess parser constructs PGN mainlines; the public shared
adapter `make_state` validates all actual-coordinate UCI moves and constructs
the retained state. No adapter rules, encoders, legal masks or episode APIs
are reimplemented. The module imports neither model nor search code or Torch.

The source checker accounts for every movetext token. Unknown text, malformed
or duplicate headers, unbalanced/empty/orphan variations, orphan annotation
glyphs, missing termination markers, movetext after a result, and wrong
mainline move numbers/colors are rejected. It accepts SAN, move numbers,
check/mate suffixes, NAGs, brace/semicolon comments, column-one percent escape
lines and well-formed variations. These checks are intentionally stricter than
python-chess's forgiving intake. All `Game.errors` cause rejection, including
errors in variations; the parser's mainline move count must also match the
fully accounted source tokens. A partial parsed prefix is never retained as
an accepted source game. Null moves are rejected explicitly. Mainline UCI
replay is validated again by the adapter, including invalid/illegal actions.

Absent setup tags mean the genuine standard initial board. A setup requires
both `SetUp="1"` and a six-field standard FEN with valid board status and
standard castling notation. `SetUp="0"` plus FEN, missing/inconsistent tags,
malformed counters/fields and invalid boards are rejected. Standard Variant
names `Standard`, `Chess` and `Normal` are recognized case-insensitively;
other labels, including Chess960/Fischerandom and variant boards, are rejected.
A valid declared setup is the genuine PGN root, even with Black to move or a
noninitial fullmove counter. This follows the interface contract; basic board
validity does not prove reachability from the ordinary start.

`history_complete=true` attests the declared source root and **every recorded
mainline move from that root**, including genuine setup games. No earlier
moves are inferred from FEN counters. `start_fen` preserves the adapter's
six-field root using `en_passant="fen"`; original declared FEN is retained in
PGN headers. D03 may later normalize legal en-passant for its canonical
identity. The complete actual-coordinate stack is retained; no rank-reflected
moves or eight-snapshot approximation replace it.

Source replay uses chess legality. It may continue after a historical prefix
at which `actual-occurrence-v1` would stop a self-play episode. The fixture with
49 plies crosses threefold/fivefold repetition cutoffs and then makes a pawn
move; all 49 real moves survive. Consumers adjudicate future roots using the
accepted adapter policy. D02 does not call `apply_move` for source replay.

## Records and accounting

Bulk output consists of `replays.jsonl`, `rejections.jsonl` and `summary.json`.
Staging schema is **`pgn-mainline-presplit-v1`**, before D03's consolidated
`replay-jsonl-v1` games table. Each accepted source gets its own ordered record:

- Dataset/encoding/move-vocabulary/ending-policy versions and WDL order.
- Genuine starting FEN, full actual `moves_uci`, complete-history origin and
  declared-setup indicator.
- Archive SHA-256, member name and one-based member index, **one-based game
  ordinal counting rejected frames too**, uncompressed byte interval
  `[byte_start, byte_end)`, inclusive one-based line interval and source-byte
  SHA-256. `LichessURL`/Site and all other supplied headers remain available.
- Separate PGN result header and mainline termination marker, an unfinished
  indicator, and any header/marker discrepancy.

Legal sources ending `*` are accepted as explicitly **unfinished source
replays**. Missing termination markers are rejected because loss of source
text cannot be ruled out. Missing Result headers remain null; no default PGN
result is fabricated. Conflicting valid header/marker results are retained
as discrepancies. No result is inferred from the final board, missing moves
are never supplied, and PGN results never become WDL targets. No teacher
labels or outcome labels are written.

Rejections identify the entire source span, ordinal, stable reason category
and explanatory detail. Blank lines outside brace comments terminate source
movetext; a following header block also begins a new source. Malformed frames
with known boundaries are rejected whole and the next source is processed.
An unclosed brace comment consumes the remainder of that member as one
rejected span, marks remaining boundaries unknown and sets
`source_game_count_exact=false`; apparent headers inside it are never used to
invent subsequent games/ordinals. Strict UTF-8 errors reject their entire
framed source. Fatal archive, decompression or I/O failures abort with a nonzero
exit and no completed summary; partial bulk output cannot attest completion.
Every member, including non-PGN members, is byte-identified and counted.

No canonical game IDs, duplicate consolidation, splits, samples, phases,
annotation, benchmark, training or search algorithms are implemented. Repeated
sources are kept separately for D03. Counts describe source/replay intake,
**not distinct games after deduplication or available sampling coverage**.

## Reproduction and verification

Use installed pinned dependencies without syncing either existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/prepare_d02.py --manifest data/manifests/elite-2025-01-v1.json --archive /private/tmp/lichess_elite_2025-01.zip --output-dir /private/tmp/rl-chess-d02-reproduction
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest tests/test_pgn_replay.py tests/test_data_manifest.py -p no:cacheprovider
```

The output directory must be new; existing directories are rejected to avoid
mixing records from different runs. Preparation verifies the ZIP bytes against
the manifest before creating output. Fixed ZIP bytes/manifest yield ordered,
canonical compact JSONL and deterministic summaries with no wall-clock fields.
Offline fixtures verify byte-identical accepted/rejected/report outputs across
two new directories, multiple members, Unicode and invalid UTF-8, source
ordinal gaps, byte/line identities, provenance/JSON round trips, all retained
prefixes, special moves for both colors, setup roots, long repetitions,
unfinished sources, result discrepancies and fail-closed archive identity.

Exact actual commands, exits, UTC times and raw outputs are saved in
`runs/d02/*-command.json` and corresponding logs. Development runs are retained;
one initial fixture test exited 1 because the test encoded `ÿ` as valid UTF-8
instead of inserting invalid byte `ff`. The fixture was corrected, and the
focused checks then passed. Preliminary full passes started before additional
move-number/variation checks; the final strict pass and output audit provide
the completion evidence. No claim of independent D02 acceptance is made.

The exact teacher executable, official source package and both embedded NNUE
blobs match D01's pinned hashes. The original executable remains at
`/private/tmp/Stockfish-sf_17.1/src/stockfish`; no rebuild or substitution took
place. A fresh idle UCI check passed, with no `go`, `bench`, `eval` or annotation
request. Evidence: `teacher-bytes.log` and `teacher-idle.json`. These checks do
not establish annotation/search behavior.

Plan1/interface, dependencies, adapter/model/search implementation, D01/S01
focused documents and every previous run remain preserved. CPU macOS
arm64/Python 3.12.12 with python-chess 1.999/chess 1.11.2 is the tested intake
environment. Other parser versions/platforms are unvalidated. The source
syntax policy deliberately rejects unsupported syntax instead of salvaging it.

Annotation remains blocked by unresolved D03/D04 split and sampling
algorithms, phase boundaries, split audit, overlap audit and protected-heldout
registry hashes. The unchanged manifest gate rejects annotation with exit 1.

## Independent raw-member census

A separate full-member scan measured **289,776 Event headers**, **7,867,282
lines** and **274,106,096 bytes**, independently reproducing the uncompressed
member SHA-256. It found no SetUp/Variant headers or brace/variation characters
in movetext. This is a raw header census, not a substitute for the completed
parser/adapter replay count, and provides no post-deduplication or sampling
coverage claim. Actual command/exit evidence: `source-census-command.json` and
`source-census.log` under `runs/d02/`.

## Final focused and regression checks

| Actual check | Result |
|---|---|
| Offline D02 source/replay tests | 72 passed, covering 64 source fixtures plus archive/accounting checks |
| D02 + adjusted selected-manifest tests | 155 passed (72 D02 + 83 D01), exit 0 |
| Full model environment, M06 diagnostic plugin enabled | 475 passed (403 merged-base + 72 D02), exit 0 |
| Full adapter environment | 307 passed (235 merged-base + 72 D02), six expected optional-model skips, exit 0 |
| Adapter replay, separate processes in both environments | Both 13-fixture reports matched, exit 0 |
| Complete M06 numerical checker and pinned upstream source audit | Passed, exit 0 |
| Torch/model/search-free preparation import boundary | Passed, exit 0 |
| Pinned teacher executable/source/embedded-NNUE bytes and fresh idle UCI | Passed, exit 0 |
| Annotation gate with unresolved D03/D04 fields | Exit 1 as required |
| Baseline preservation | 305 protected files byte-identical |

Actual full-suite logs and JUnit reports use the `complete-*-regression` and
`pytest-complete-*` names. Earlier smaller test counts remain development
history. No GPU work, search/annotation experiment or label success is claimed.


## Completed full selected archive

The final strict preparation command exited **0**, processing one archive, one
member and one PGN member: **289,776 accepted source games, 0 rejected**,
**25,429,063 actual-coordinate mainline plies**. There were no unfinished
sources, result discrepancies, declared setups or unknown-boundary spans.
The independent Event-header census agrees with the exact source-frame count.
All source ordinals/byte/line intervals/source hashes and member/output
checksums match the raw archive in the final all-source audit.

Authoritative bulk output is `/private/tmp/rl-chess-d02-accepted-prepared/`:
`replays.jsonl` is **541,174,538 bytes**, SHA-256
`69c10231f4d569a6b3cb4b26bc22a9535346d68f0a4753bd83c0c92981ea950b`.
`rejections.jsonl` is empty (0 bytes), SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The ZIP and every bulk table remain outside Git. Three full passes produced
byte-identical output tables and summaries; the first two are preliminary
hardening history, while `accepted-full-archive-command.json` records the
final strict pass. Here 'accepted' refers to source replay acceptance, **not
independent review acceptance of D02**.

Committed completion artifacts are `runs/d02/replay-summary.json`,
`preparation-evidence.json`, `final-output-audit.log` and their actual command
records. Full source counts are **pre-deduplication/pre-split measurements**.
D03/D04 must still establish distinct-game identities, split assignment and
usable audited sampling coverage. D02 stops for independent review.

The first staged whitespace check exited 2 for extra blank lines at EOF in
two captured stdout logs. Their exact original stdout is retained losslessly
in `runs/d02/stdout-raw.json` (with byte lengths and SHA-256s); readable logs
normalize only redundant trailing blank lines. The corrected staged check is
recorded separately. No scientific or source-data bytes were changed.


## Independent AI review acceptance — 8 October 2026

D02 is accepted for its scoped milestone through fresh independent review
by the coordinating Codex agent at the user's request. No blocking findings
or production fixes were required. The reviewed D02 and S02 commits are
integrated together. [Review, fresh checks and limitations](d02-s02-review.md)
and [structured acceptance](../runs/d02-s02-review/acceptance.json) record
the evidence. This entry supersedes earlier pending-review statements without
changing historical evidence. Annotation remains blocked by D03/D04;
Gate S remains unaccepted.
