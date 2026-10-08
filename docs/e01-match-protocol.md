# E01 paired match protocols — independent review pending

Implemented on 8 October 2026 from exact accepted base
`4184387c7e4b67559355d6b212a2a300c86c5ce6`, in the isolated local clone
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/e01-match-protocol`, branch
`codex/e01-match-protocol`. S05 and Gate S are accepted in that base.
The primary checkout/main, accepted implementations, manifests, dependency pins,
planning files and all historical evidence are preserved. E01 stops for review.
No applicable AGENTS.md was found in the repository or checked ancestor paths.
Read `interface.md`, `coding-plan.md`, `docs/s05-search-smoke.md`,
`docs/d05-s05-review.md` and accepted adapter/search/game APIs.

## Small API and later wiring

```python
import json
from chess_rl.adapter import make_state
from chess_rl.evaluation import (
    Opening, OpeningSets, SearchPlayer, SearchSettings, run_match,
)
from chess_rl.evaluation.replay import audit_match
from chess_rl.search.evaluator import evaluate

# Explicit dummy protocol fixture. These are roles, not trained identities.
settings = SearchSettings(simulation_budget=2)
p = SearchPlayer("fixture-P", "P", "s01-uniform-dummy-v1", evaluate, settings)
r = SearchPlayer("fixture-R", "R", "s01-uniform-dummy-v1", evaluate, settings)
book = OpeningSets("example-fixtures-v1",
    validation=(Opening.from_state("v-e4", make_state(moves_uci=["e2e4"]),
                                   source="hand-authored validation fixture"),),
    test=(Opening.from_state("t-d4", make_state(moves_uci=["d2d4"]),
                             source="hand-authored reserved test fixture"),))
result = run_match(p, r, book, split="validation", match_id="example",
                   comparison="P-versus-R", seed=20261008, ply_cap=4,
                   protocol_only=True)
record = result.to_dict()
assert audit_match(record)["passed"]
print(json.dumps(record, allow_nan=False))
```

For fixed comparisons, replace `r` with `RandomLegalPlayer()`, `GreedyPlayer()`
or `StockfishPlayer(explicit_pin, nodes=128)` and use
`comparison="fixed-opponent"`. The left player may have P or R identity.
Records include `e01-match-v1` configuration, both legs of every opening pair,
`e01-game-v1` histories/attempts and a completed-only score summary.
`scripts/replay_e01.py` reads a JSONL of those match snapshots.

`Player.metadata()` returns finite JSON with explicit `player_id`, `kind`,
`version`, `settings` and any checkpoint/evaluator/engine identity.
`Player.choose(state, *, seed) -> Decision(move_id, details)` is called only on
nonterminal owned copies. It must leave that state unchanged and return a legal
shared ID. This interface supports controlled players and future evaluators.
The accepted batched `Evaluator` signature is injected into `SearchPlayer`;
search has no access to model layers or checkpoint-loading internals. Each
player uses its own evaluator throughout its root search, including hypothetical
opponent-turn leaves. Actual root ownership routes by the game's side to move.
Search uses accepted `Node`/`search(exploration=None)` and S05's `SearchAttempt`
record type. The E01 move loop adds two-player routing to S05's accepted execution
semantics; it calls the same adapter rules/application APIs and preserves the
candidate-copy publication boundary without changing the accepted S05 runner.

`Checkpoint.from_file(id, path, kind=...)` hashes real bytes. Identity records
retain ID, absolute path, SHA-256 and kind (`fixture`, `pretrained-P`, `rl-R`).
`run_match` verifies supplied checkpoint bytes before execution and validates
role/kind. With default `protocol_only=False`, search players require trained
checkpoint references; absent/fixture references require explicit
`protocol_only=True`. Unavailable references remain null with status
`unavailable`. Fixtures that test metadata use actual temporary fixture bytes
labelled `fixture`; no fake P/R checkpoint is saved.

T02/T04 are not implemented here. A reference is a byte identity, not a model
loader or proof that an injected evaluator loaded those bytes, disabled dropout
or validated encoding/model compatibility. That association is caller attested
until the real loader/evaluator is available. Use the accepted model/evaluator
contract when wiring it later. No real P/R checkpoints or trained games exist in
this evidence, and no strength result is claimed.

## Pairing, openings and reproducibility

Every selected opening yields exactly two legs: left White/right Black, then
right White/left Black. Replay the identical actual starting FEN and entire
opening prefix in both legs; do not reflect the board or alter side to move.
This also works for odd-length and declared setup openings. Pair IDs combine
explicit match ID, split and opening ID. Both legs are retained and attempted
independently when one fails or catches KeyboardInterrupt/SystemExit. No retries,
replacement openings, engine resignations or failure forfeits create outcomes.
Soft interruption is recorded at the game boundary; a hard process kill cannot
return a snapshot. This bounded API does not resume matches or self-play.

Both validation and test sets must be explicitly supplied and nonempty.
Opening IDs are globally unique, and opening roots must have distinct public
four-field legal-EP FEN rule identities across and within splits. Counters and
history differences do not allow transposed opening roots into different sets.
Common historical prefixes, such as the normal starting position, are allowed;
this is root separation, not disjoint ancestry or a complete future-transposition
ban. Records freeze the full contents, source descriptions and SHA-256 of both
sets, normalized initial FENs, actual prefixes and history-completeness flags.
An opening cannot continue beyond an actual-occurrence ending. A root that is
already terminal is allowed for controlled boundary checks. Isolated-FEN probes
remain history-incomplete; ordinary game roots use full retained histories.

The hand-authored smoke book is a protocol fixture, not the final evaluation
book. Before later real campaigns, freeze versioned external opening lists and
source provenance before games. Use validation for selection; E02 must freeze
R and the playable checkpoint before reserved test results. Running the test
fixture here does not consume a future production test set or implement E02.

`seed` is a required nonnegative Python integer. For leg `l`, game seed is the
first 16 hex digits of SHA-256 of compact sorted JSON
`["e01-game-seed-v1", seed, pair_id, l]`, interpreted as an integer. Each actual
turn similarly hashes `["per-turn-sha256-v1", game_seed, retained_ply, player_id]`.
No global random state is used. Thus faults do not shift the next leg's seeds.
Player settings, all derived seeds and IDs are saved. Random legality uses a
fresh Python MT19937 `Random(turn_seed)` and uniform choice over sorted shared
IDs. Reproducibility is scoped to recorded Python/dependency/algorithm versions;
an engine's platform-dependent behavior/timing is not promised byte-identical.

P and R must both be `SearchPlayer`, have explicit P/R roles, different player
IDs and exactly equal `SearchSettings`: simulation budget, PUCT constant and
unvisited value. Defaults are 16 simulations, c_puct=1.5, unvisited value=0.
Every root is fresh. Exploration is always `None`: no Dirichlet noise,
temperature sampling or RNG-driven search selection. S04 chooses maximum
completed visits, breaking exact ties by smallest shared ID. Defaults are
protocol choices, not calibrated strength settings; smoke uses 1–2 simulations.

## Fixed opponents and teacher separation

| Opponent | Recorded behavior |
|---|---|
| random legal, `e01-random-v1` | Uniform over sorted legal shared IDs; explicit per-turn MT19937 seed |
| greedy, `e01-greedy-v1` | Enumerate one legal ply; score moving player's post-move own-minus-opponent material; P=1, N=3, B=3, R=5, Q=9, K=0; adapter terminal win/draw/loss scores +1000/0/-1000 before material; ties smallest shared ID; no RNG |
| Stockfish, `e01-stockfish-v1` | Explicit executable/release/source/name/checksum pin; 1–10,000 requested nodes, default 128; fresh process and Clear Hash each move; legal completed bestmove only; engine tie handling, no external RNG |

Stockfish evaluation options are Threads=1, Hash=16 MiB, MultiPV=1,
Skill Level=20, UCI_LimitStrength=false, UCI_ShowWDL=false, empty SyzygyPath,
Ponder=false and UCI_Chess960=false. Python-chess manages MultiPV/Ponder/Chess960;
capabilities are checked, and every decision saves all effective option defaults
plus overrides, engine UCI identity, hash-reset completion, requested nodes,
returned bestmove and measured nodes/depth/time/etc. Each engine runs in an
empty temporary working directory with an explicit timeout (default 10 seconds).
A missing/illegal move, missing positive node accounting, unexpected tablebase
hits, byte/name mismatch, transport exception or timeout fails the game. Node
limits may overshoot; actual nodes are saved rather than clamped to the request.
No engine outcome or WDL substitutes for the adapter's rule result.

The bounded smoke uses accepted D01 Stockfish 17.1 binary
`/private/tmp/Stockfish-sf_17.1/src/stockfish`, source
`03e27488f3d21d8ff4dbf3065603afa21dbd0ef3`, executable SHA-256
`3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`.
The checksum pins its embedded networks. Two evaluation rungs request 32 and
128 nodes. These deliberately small diagnostic rungs are not Elo-calibrated.
No annotation worker or manifest teacher settings are imported or changed.
The locked annotation teacher remains 100,000 nodes, Hash=128 MiB,
UCI_ShowWDL=true. Evaluation metadata explicitly names its separate namespace.
The transport uses the documented [python-chess engine API](https://python-chess.readthedocs.io/en/latest/engine.html)
and the locally pinned 1.999/chess 1.11.2 implementation.

## Endings, failure accounting and replay

The cap counts all retained plies, including the opening. At each boundary,
including after the final permitted move, call `terminal_result` before comparing
the cap or asking a player. Shared `actual-occurrence-v1` adjudication preserves
library precedence, then reached threefold, then reached fifty-move conditions.
An available intended move alone never ends a game. No new rules/encoding or
FEN-only history reconstruction is introduced.

| Status | Reason/outcome |
|---|---|
| completed | Exact adapter reason and winner; `1-0`, `0-1` or `1/2-1/2` |
| truncated | `ply_limit`, outcome/winner null |
| failed | `execution_failed`, original exception type/message, outcome/winner null |
| interrupted | `execution_interrupted`, KeyboardInterrupt/SystemExit, outcome/winner null |

Each status retains initial board, opening source/prefix, complete actual trace,
history completeness, final FEN/promoted bits, player metadata and attempts.
Each attempt records actual ply/color/player/seed, stage/status, selected ID/UCI,
details and whether application succeeded. A selected move may be unapplied.
Players receive owned copies; mutation is rejected. Application uses another
candidate copy, publishing only on success. Partial search traces retain completed
backups in parent-player perspective and failure stage. Caller histories and
paired opening records survive failures and interruptions.

For the left player, W/D/L and `(wins + 0.5*draws)/completed_games` include only
completed games; zero denominator yields null score. A completed lone leg still
enters this denominator, while incomplete-pair counts are explicit. For paired
uncertainty, retain only fully completed opening pairs and average their two
left-player scores. A descriptive 95% percentile bootstrap resamples those pair
means with recorded seed and 2,000 samples. Fewer than two complete pairs gives
null uncertainty. Tiny samples/attrition can bias this conditional estimate;
no statistical significance, Elo, improvement or causal advantage is inferred.
This is an E01 summary utility, not the final E03 analysis/campaign.

`audit_match` independently reconstructs all histories through the adapter,
checks source/set digests, split membership, pair/leg/color routing, seeds,
shared legal move IDs, search counts/edges/selection, effective engine settings,
final board/promoted bits, terminal-before-cap, unfinished outcomes and score
accounting. It requires no evaluator, checkpoint file or engine. It does not
reproduce neural predictions or externally prove search/engine computation;
saved timing, partial failures and checkpoint/evaluator association remain
original recorded evidence.

## Verification and review boundary

Final exact commands, outputs, durations, exit codes, JUnit, snapshots,
replay audits and baseline byte hashes are in
[`runs/e01/20261008T150821584550Z/`](../runs/e01/20261008T150821584550Z/summary.json)
and [`runs/e01/completion.json`](../runs/e01/completion.json).
The earlier full run and development attempts are retained separately.

| Final check | Adapter environment | Model environment |
|---|---|---|
| New controlled E01 cases | 62 passed | 62 passed |
| E01 + accepted S04/S05 focused checks | 221 passed | 221 passed |
| Full regressions | 878 passed; six expected optional-model skips | 1,046 passed |
| Accepted reviewer probes | 84 passed | 84 passed |
| Frozen adapter replay | 13 fixtures; reports match | 13 fixtures; reports match |
| Accepted S05 records | 15 replayed | 15 replayed |
| Saved E01 fixtures | 24 matches / 48 games | 24 matches / 48 games |
| E01 replay in both environments | All 48 pass in each | All 48 pass in each |

Each final E01 bundle has 12 completed, 34 truncated, one controlled failure
and one controlled interruption, with 58 generated plies and 58 completed pupil
search simulations. Only 12 completed games enter outcomes. Four actual
Stockfish decisions per producer request 320 nodes in total (two at 32, two at
128); each producer records 329 actual engine nodes. Special-move fixtures
include both castling sides/colors, both EP directions and all four promotions
for both colors, preserving promoted bits. Non-engine bundles match exactly.
All 1,174 accepted-base files match byte-for-byte, and primary main remains the
exact requested base. All final verification commands exit 0. Raw first-failure
pytest stdout/XML contain six trailing-whitespace diagnostic lines; the staged
check excluding exactly those two lossless evidence files passes.
Reproduce from this checkout with:

```sh
PYTHONPATH=src:. PYTHONDONTWRITEBYTECODE=1 /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -B runs/e01/verify.py
```

It uses both existing pinned CPU environments, full regressions with accepted
M06 diagnostic plugin in the model environment, 84 accepted reviewer probes,
both frozen 13-fixture adapter reports, accepted S05 replay, two separately
produced 24-match/48-game fixture bundles and four E01 replay audits. Non-engine
records must agree exactly between producers; measured engine timing/work may
differ. No dependency installs, training, T04, self-play, GPU or large evaluation
campaign is run. Production opening-book/checkpoint choices and actual future
game allocation await model-guided cost measurements; 200–400 final games is an
aim from the plan, not an E01 quota or an inference from tiny fixture cost.

Development failures are retained: the first scripted evaluator reused its
root-only move in deeper leaves, and candidate-application injection remained
active during replay. Both were fixture issues corrected before final checks.
No accepted production file was changed to fix them. Original failure stdout/XML
are preserved verbatim. CPU macOS arm64/Python 3.12.12 only; engine timing,
other platforms, batching/concurrency, process crash recovery and real playing
strength remain unvalidated. E01 awaits independent review; main integration is
not authorized by these checks.
