# D03 canonical games and deterministic splits — 8 October 2026

D03 is implemented from independently accepted base
`1301d033b81ca88e51c66ba27c1035a80c50fe54` on
`codex/d03-canonical-splits`, in
`/private/tmp/rl-chess-d03-canonical-splits`. D02/S02 AI acceptance is already
recorded in that base. **D03 stops for independent review; no independent D03
acceptance is claimed.** No D04 sampling, phase, overlap or protected-sample
decisions are made here.

## Verified intake and preservation

The completed D02 summary is pinned to SHA-256
`494407e29f64046585151b91698778bb6ce3a0cfe849da0a252c200f85c2444b`
(1,907 bytes). It is byte-identical to the accepted base's
`runs/d02/replay-summary.json`. Before processing, preparation verifies that
summary hash, completion/version/dataset/archive identity, and the replay
table's byte identity against that summary:

`/private/tmp/rl-chess-d02-accepted-prepared/replays.jsonl`: 541,174,538 bytes,
SHA-256 `69c10231f4d569a6b3cb4b26bc22a9535346d68f0a4753bd83c0c92981ea950b`.

Inputs are opened read-only. Replay bytes are hashed again while processing,
and accepted counts/plies/source metadata counters are checked against the
summary. The summary is checked again before completion. A new output
directory is required; identity/schema/count/I/O failures abort without a
completed audit. The separate all-source audit also checks the original empty
rejection file and the summary against the historical committed copy. D02 bulk
records, ZIP, accepted implementation, dependency pins, teacher and historical
evidence are preserved.

## Exact canonical identity

`chess_rl.canonical_games.canonical_payload` implements the accepted
`start-moves-sha256-v1` identity from D01. The actual root is validated through
the shared adapter and normalized with six-field
`board.fen(en_passant="legal")`, **including halfmove and fullmove counters**.
The full normalized actual-coordinate UCI array is retained. Identity JSON has
exactly `identity_version`, `start_fen`, `moves_uci`; serialization is
`json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)`,
UTF-8, without a trailing newline. The ID is `g:` plus the lowercase SHA-256
of those bytes. No headers, results, dataset IDs or source URLs enter identity.

Legal en-passant availability is retained; an irrelevant or pinned target
normalizes away. Source roots and headers retain their original target and
counters. Different root counters, side, pieces, castling rights, legally
available en-passant or full mainline yield distinct payloads. Repeated
positions or transpositions do not identify games. Every repeated ID is checked
against its complete canonical payload; any differing payload aborts, even if
the digest was forced to collide. There is no collision fallback or renaming.

This stage consumes byte-verified **already legally replayed D02 records**.
It validates root/UCI syntax, versions and source/result contracts; it does not
repeat the entire 25-million-ply D02 legal replay. `canonical_payload` alone is
not a mainline legality validator. Consumers reconstruct through the accepted
adapter. Neither source replay nor consolidation adjudicates historical
prefixes as self-play endings.

## Versioned assignment decision

Procedure: **`game-sha256-threshold-v1`**, seed **20261008**.

For each canonical ID, SHA-256 the UTF-8 JSON object with exactly `game_id`,
integer `seed`, and `split_version="game-sha256-threshold-v1"`, using sorted
keys, compact separators, `ensure_ascii=True`, and no newline. Interpret all
256 digest bits as an unsigned big-endian integer `x`:

- Train when `100*x < 90*2**256`.
- Validation otherwise when `100*x < 95*2**256`.
- Test otherwise.

This uses integer comparisons, no PRNG state, float thresholds, modulo buckets,
rank ordering, quotas or corpus-dependent rounding. The 90/5/5 percentages are
assignment targets; small corpora can have empty splits. Assignments depend
only on identity, seed and procedure version. Input reordering, new games and
extra sources for existing games cannot change existing assignments. Expansion
must keep these choices and IDs. Protected held-out sample registries, position
overlap precedence, removal/refill and sampling expansion remain D04 work.

## Authoritative output and audit

Bulk output remains outside Git at `/private/tmp/rl-chess-d03-prepared/`.
`games.jsonl` is **638,770,177 bytes**, SHA-256
`f9f69d255b652f45276ee6163857b3a9c8054ac5ff037f36c94e9ecb039f9df5`.
Each `replay-jsonl-v1` record stores one canonical ID/split/root/full mainline,
identity and shared versions, complete history from the declared PGN start,
and `sources`. Each source entry retains the complete D02 `source` dictionary
(archive/member/index/ordinal/byte/line/source hash), original replay root,
declared-setup and history-origin fields, every original header (including
Site/LichessURL when present), and separate `pgn_result` header/marker/
unfinished/discrepancy fields. All occurrences retain multiplicity, including
repeated identical source records. Output has no teacher or outcome targets.

`result_metadata` summarizes distinct observed non-`*`, non-null source results
and flags disagreement; it chooses no authoritative result. A missing header
or `*` stays explicit in its source entry. Cross-source contradictions coexist
with each source's own header/marker discrepancy flag.

Games sort by ID; source entries sort by their entire canonical JSON text.
UTF-8 compact JSONL uses sorted keys, ASCII escapes and one trailing newline
per record. SQLite in a disposable external directory bounds memory and
provides ordering; its bytes are not an output artifact.

The completed `canonical-split-audit-v1` records input identities, archive,
procedure/version/seed/targets, actual game/source/canonical-ply split counts,
deduplication/result counts, games-table identity, and an assignment checksum.
The latter hashes ID-sorted compact JSONL objects with exactly `game_id` and
`split`, independently of provenance changes. Audit bytes have no path/time
fields or dependency on the manifest's audit checksum. The committed copy is
[runs/d03/split-audit.json](../runs/d03/split-audit.json), SHA-256
`029e3e78cf8114ec68796ef35644c67749184bf060c4a3321e3107b3f0534a36`.
The selected manifest resolves only `preparation.split_algorithm` and
`preparation.split_audit_sha256`; every other manifest decision is unchanged.

| Measurement | Actual count |
|---|---:|
| Accepted D02 source records | 289,776 |
| Source plies | 25,429,063 |
| Canonical games | 289,427 |
| Canonical plies | 25,422,787 |
| Additional duplicate source occurrences | 349 |
| Groups with multiple sources | 112 |
| Groups with conflicting completed source results | 13 |
| D02 rejected source frames | 0 |

| Split | Canonical games | Source occurrences | Canonical plies |
|---|---:|---:|---:|
| Train | 260,517 | 260,836 | 22,891,358 |
| Validation | 14,480 | 14,500 | 1,267,765 |
| Test | 14,430 | 14,440 | 1,263,664 |

These are game assignment counts, before position sampling. They establish no
usable sample coverage or cross-split position separation. Different starting
counters deliberately retain different game IDs; shared rule positions across
those games still need the broader D04 overlap audit.

## Reproduction and verification

Use the existing pinned environment without syncing dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/prepare_d03.py --manifest data/manifests/elite-2025-01-v1.json --replays /private/tmp/rl-chess-d02-accepted-prepared/replays.jsonl --summary /private/tmp/rl-chess-d02-accepted-prepared/summary.json --summary-sha256 494407e29f64046585151b91698778bb6ce3a0cfe849da0a252c200f85c2444b --output-dir /private/tmp/rl-chess-d03-new-run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest tests/test_canonical_games.py tests/test_pgn_replay.py tests/test_data_manifest.py -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python runs/d03/audit_output.py /private/tmp/rl-chess-d03-prepared
```

Actual argv, cwd, environment overrides, UTC times/durations, exit codes and
lossless stdout/stderr byte counts/hashes are saved in `runs/d03/*.command.json`
and associated logs. Commands in this section are reproduction instructions;
the saved command records are execution evidence. The initial fixture failure
is preserved: the test helper inserted an extra blank line before movetext,
and D02 correctly rejected that fixture. The helper was corrected; D02 code
was not modified.

Final automated evidence:

| Check | Actual result |
|---|---|
| D03 fixtures | 41 passed |
| D03 + D02 + manifest checks | 196 passed, exit 0 |
| Complete model environment with accepted M06 diagnostic plugin | 611 passed, exit 0 |
| Complete adapter environment | 443 passed, six expected optional-model skips, exit 0 |
| Frozen adapter replay in both environments | 13 fixtures each; reports byte-identical, exit 0 |
| Complete preparation, two new external output directories | Both exit 0; games and audit bytes identical |
| Separately spelled identity/split/provenance audit | All 289,776 source occurrences and 289,427 games matched; 290 sampled adapter replays, exit 0 |
| Pinned upstream source/notice audit | Passed, exit 0 |
| Pinned teacher executable/package/embedded NNUE bytes and idle UCI | Passed, exit 0; no search |
| Annotation gate | Exit 1 as required; four D04 blockers |

[Output verification](../runs/d03/output-verification.json) and
[full-run byte comparison](../runs/d03/reproduction.json) provide the resulting
hashes. The all-source audit imports no production D03 identity/split helpers.
Repeated input preparation yields identical audit bytes; reordered inputs yield
identical games/assignments but different input hashes in the audit. Expansion
fixtures cover all three splits and verify existing held-out assignments remain
stable when new games and duplicate sources are added. The full corpus was
prepared twice in original order; corpus reordering/expansion are fixture tests.

The initial staged whitespace check exited 2 on pytest's preserved failure XML.
The evidence directory's attributes exclude generated stdout/stderr/XML from
trailing/EOF whitespace checks, retaining their exact original bytes. Code and
documentation keep normal whitespace checks. The corrected staged check is
saved separately.

CPU macOS arm64/Python 3.12.12 with python-chess 1.999/chess 1.11.2 is tested.
Other builds/parser versions are unvalidated. No annotation, benchmark,
training, GPU, paid compute or search experiment is performed. The pinned
teacher receives only an idle identity/configuration check. Annotation remains
blocked by four D04 fields: sampling algorithm, phase boundaries, overlap audit
SHA-256 and protected-heldout registry SHA-256. D03 completion is not dataset
readiness or independent review acceptance.


## Independent AI review acceptance — 8 October 2026

D03 is accepted for its scoped CPU milestone by the coordinating Codex
agent at the user's request. No blocking findings or production fixes were
required. [Review, fresh checks and limitations](d03-s03-review.md) supersede
historical pending-review statements. D04/S04 are next; annotation remains
blocked and Gate S remains unaccepted.
