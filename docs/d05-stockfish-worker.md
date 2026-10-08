# D05 Stockfish annotation worker — 8 October 2026

D05 is implemented for independent review on `codex/d05-stockfish-worker` in
`/private/tmp/rl-chess-d05`, from exact accepted base
`ad8438acfbd859f734daaebc6115afdab0367b65`. No D05 acceptance or main integration
is claimed. D06, D07, D08, training and search integration remain pending.

## API and trust boundary

`chess_rl.stockfish_worker` exposes:

- `verify_artifacts(manifest, manifest_path, games_path, split_audit_path, prepared)`:
  fail closed on unresolved metadata, noncanonical manifest bytes, split/overlap
  audit contracts or mismatched games/sample/exclusion/registry identities.
- `select_tiny(prepared, games_path, manifest, identities)`: select at most 12
  audited **training** references, join their canonical games, reject protected
  game/position membership, rehash the inputs and replay every selected root.
- `replay_root(reference, game, manifest)`: validate canonical ID/assignment,
  replay the actual full prefix through the shared adapter, require a nonterminal
  matching rule position and phase. `replay_identity(state)` includes the actual
  starting counters and every retained move; it is not D04's overlap key.
- `verify_teacher_bytes(manifest, executable, build_evidence)`: require pinned
  python-chess 1.999/chess 1.11.2, executable SHA-256 and both complete embedded
  NNUE blobs at the accepted build offsets.
- `await engine_attempt(state, manifest, executable, timeout, trace)`: one fresh
  engine process, identity/capability checks, configuration, explicit Clear Hash,
  readiness and full-history `analysis(..., Limit(nodes=100000), multipv=1)`.
- `AnnotationWorker(manifest, provenance, attempt, sink, max_attempts=2)` with
  `await annotate(reference, game) -> dict | None`: replay, bounded attempts,
  strict output validation and one successful label at most per reference in
  this run. `attempt` is an async callable `(state, trace) -> (info, bestmove)`;
  `sink` synchronously persists each attempt. Exhaustion returns `None` for an
  explicit exclusion. Interruption/cancellation is recorded and propagated.

The lower-level worker receives **already audited** references and verified
teacher provenance; the CLI performs all gates before any annotation. Injected
attempts exist for controlled tests. There is no result cache, resumption,
cross-run deduplication, automatic refill or parallel annotation. A duplicate
reference within a run is rejected even after prior exclusion/interruption.
The sequential diagnostic observer must not be shared across concurrent engines.

## Engine and targets

The exact executable remains `/private/tmp/Stockfish-sf_17.1/src/stockfish`,
SHA-256 `3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`.
Both accepted embedded NNUE SHA-256s and build offsets are freshly verified.
The CLI rehashes the executable before every attempt. All required UCI options,
defaults, effective settings, ID and raw transport commands/responses are saved.
Threads=1, Hash=128 MB, UCI_ShowWDL=true, MultiPV=1, SyzygyPath empty, full
strength and standard chess are retained. python-chess manages MultiPV via
`multipv=1` and manages Ponder/Chess960; they are not passed to `configure`.
Other pinned defaults are preserved. Clear Hash completes before every search,
including retries; engine processes also start fresh.

The last principal **individual** response must contain WDL, score/mate, PV,
depth/seldepth, nodes/nps/hashfull/tbhits and finite time. It cannot inherit an
old WDL/PV from an incomplete final row. The separate UCI bestmove must be legal
and match PV[0]. Every PV move is checked along its line; shared move IDs must
round-trip. WDL uses `.pov(root.turn)` and strict integer nonnegative counts
summing to 1000, with `[win,draw,loss] = counts/1000`. Score bounds are retained
as returned information; a naturally completed node-limited search can finish
during a bounded iteration. Requested nodes are separate from returned nodes;
node overshoot is recorded, not rewritten. No centipawn conversion or PGN result
enters a target.

The pinned parser sometimes logs a malformed field and continues, or stops a
PV at an unrecognized token. The worker captures parser diagnostics and validates
raw Stockfish PV suffixes, rejecting either behavior even if later output is
valid. A timeout covers handshake/configuration operations and the entire search
completion separately (30 seconds each in this run). The worker closes incomplete
searches and never accepts stopped partials. A completed search quits normally;
cancellation during teardown still propagates without a successful label.

`stockfish-label-v1` stores compact `game_id/sample_ply` references, actual full
root FEN, history completeness/real snapshot count, full-prefix replay checksum,
versions, legal actual best-move UCI/class ID, raw/soft root WDL, requested
settings, returned principal search info and target provenance. Provenance links
the manifest, canonical game table, audited sample table, overlap audit,
executable and embedded assets. Attempts retain raw rows, raw UCI, bestmove/
ponder and failure detail. Full games remain authoritative external artifacts;
no repeated histories, tensors or dense one-hot policies are stored.

## Actual sample and replay

`data/manifests/elite-2025-01-d04-v1.json` is used explicitly. The historical
manifest is byte-preserved and still fails its annotation metadata gate.
All authoritative D03/D04 input SHA-256/byte identities match their accepted
audits before and after annotation. Both held-out registries and every original
bulk artifact remain unchanged.

Membership is the first two training references per actual color and recorded
phase in the audited table's order. It is fixed before analysis, with no
replacement on failure. This is a plumbing subset, not a revised sampling
decision or a representative cost/strength sample.

| Actual phase | White | Black | Successful final labels |
|---|---:|---:|---:|
| Opening | 2 | 2 | 4 |
| Middlegame | 2 | 2 | 4 |
| Endgame | 2 | 2 | 4 |

Final run: **12 selected, 12 attempts, 12 successful labels, 0 failed attempts,
0 exclusions, 0 pending**. Actual membership, labels, attempt ledger, settings,
engine information and raw transport are in `runs/d05/tiny-final/`.
Membership SHA-256:
`50f725d3f898dceaff59d953fe0ddba05fe1b95558aef9cca15b26a1fba5c5e3`.
Label SHA-256:
`ceec9a44e56a5ae6521036219388a73801baaf7c32df16359faa75b650784a7f`.
The earlier `tiny-real/` run is preserved, also 12/12 with the **same 12
references**. It preceded the malformed-response/normal-shutdown safeguards;
the final run reverified the same membership with those safeguards. Thus 24
real attempts were executed across two plumbing runs, covering 12 distinct
audited roots. Neither run contributes a D06 or D08 allocation.

`runs/d05/audit_sample.py` imports no D05 worker helpers. Every saved successful
label in both runs is freshly joined and replayed through the shared adapter
and separately through python-chess. Root FEN/history/snapshot/phase identity,
legal full PV, independently spelled move IDs, WDL perspective/counts,
probabilities, versions and attempt linkage agree. It saves feature/legal-mask
fingerprints. Raw commands confirm reset-before-go, exact requested nodes and,
in the final run, the complete actual move-stack position command and normal
exit. Independent replay is not a second teacher search.

## Checks and evidence

| Check | Actual final result |
|---|---|
| D05 controlled cases | 58 passed |
| Adapter regressions | 770 passed; six expected optional-Torch skips |
| Model regressions, accepted M06 diagnostic plugin | 938 passed |
| Accepted D02/S02, D03/S03, D04/S04 reviewer probes | 70 passed in each pinned environment |
| Both frozen adapter replays | Identical accepted 13-fixture reports |
| Real label independent replay | All 12 final and all 12 initial records pass |
| Teacher bytes/idle capabilities | Passed; source and embedded NNUE identities preserved |
| Pinned upstream checks / M06 numerical checker | Passed |
| Historical incomplete-manifest gate | Exit 1 as expected |

Controlled cases cover both WDL perspectives/colors, malformed/missing WDL,
invalid counts/score/moves/PV suffixes, deadline/cancellation/keyboard interrupt,
shutdown interruption, reset order, stale final fields, retry/exhaustion,
duplicate-success prevention, root mismatch/terminal rejection, mutations,
artifact tampering and distinct histories sharing an overlap key. Saved
`failure-fixtures/` ledgers retain seven explicitly controlled scenarios;
these are not real Stockfish failures or real labels.

Actual argv/cwd/UTC times/durations, exits and lossless stdout/stderr identities
are in `runs/d05/*.command.json`, indexed by `commands.json`. The first new
NNUE fixture attempts failed because fake assets violated existing filename and
source-URL constraints. The fixtures were corrected; both failed runs and their
logs remain. The initial HTTPS pin check failed on sandbox DNS, then passed
with approved network access; the accepted local-object pin check also passed.
No dependency sync, rebuild or replacement occurred.
The full staged whitespace check reports original trailing spaces/blank lines
in lossless pytest/UCI outputs (exit 2); those bytes are deliberately preserved.
The staged source/document/structured-evidence check excluding generated text
logs and XML passes (exit 0).

Reproduce with the existing environment, this checkout's `PYTHONPATH=src:.`
and bytecode disabled. Use a **new** output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/annotate_d05.py --manifest data/manifests/elite-2025-01-d04-v1.json --games /private/tmp/rl-chess-d03-prepared/games.jsonl --split-audit /private/tmp/rl-chess-d03-prepared/split-audit.json --prepared /private/tmp/rl-chess-d04-prepared --executable /private/tmp/Stockfish-sf_17.1/src/stockfish --output-dir /private/tmp/rl-chess-d05-new-sample --timeout 30 --max-attempts 2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest tests/test_stockfish_worker.py -p no:cacheprovider
```

CPU macOS arm64/Python 3.12.12 only. These runs establish annotation/replay
plumbing, not throughput projections, cross-platform determinism, strength,
training adequacy or production reliability. Process-crash recovery, resume,
concurrent workers and scale behavior remain unimplemented. Attempts and labels
are separately flushed/fsynced; a hard crash between writes can leave an
incomplete ledger, so an incomplete run is not reusable as a completed dataset.
D06's 1,000-position benchmark, D08's 10,000-label production, training and
search integration were not run. Stop at this commit for independent review.
