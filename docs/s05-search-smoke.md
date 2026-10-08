# S05: minimal game runner and dummy-search smoke checks

Implemented 8 October 2026 on `codex/s05-search-smoke` in the isolated checkout
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/s05-search-smoke`, with
exact parent `ad8438acfbd859f734daaebc6115afdab0367b65` from
`/Users/arvindsrinivass/Desktop/RL-Chess`. D04/S04 are independently AI review
accepted in that base. **S05 awaits independent review; Gate S is unaccepted.**

Read the applicable repository/ancestor instruction locations, `interface.md`,
`coding-plan.md`, `docs/s02-puct.md`, `docs/s03-value-backup.md`,
`docs/s04-root-policy.md` and `docs/d04-s04-review.md`, plus the accepted
adapter/evaluator/tree/search APIs. No applicable AGENTS.md was found.
All 925 accepted-base files remain byte-identical, including README,
coding-plan, dependency pins, adapter, search and historical evidence.

## API

```python
import json
from chess_rl.adapter import make_state
from chess_rl.search.game import run_game

opening = make_state(moves_uci=["e2e4", "e7e5"])
game = run_game(opening, 16, ply_cap=24)
print(game.status, game.outcome)
record_json = json.dumps(game.to_dict())
```

`run_game(starting_state, simulation_budget, *, ply_cap, config=SearchConfig(),
evaluator=evaluate, evaluator_id="s01-uniform-dummy-v1") -> GameRecord` owns a
full-history copy. The supplied state's retained moves are the explicit opening
prefix. The cap counts **all retained plies from the initial board, including
the opening prefix**, rather than additional plies or the FEN fullmove counter.
An incomplete starting FEN stays incomplete; declared setup completeness is
the caller's provenance attestation through the accepted adapter.

The search budget must be a positive Python integer. The cap must be a
nonnegative Python integer. Invalid arguments raise before execution. A custom
evaluator requires an explicit descriptive ID and must honor the accepted
read-only, batch-of-one evaluator contract. S05 supports one evaluator for both
colors; E01 owns opponent identities, paired openings and match protocols.

Each nonterminal move creates a fresh full-history Node and calls accepted
`chess_rl.search.runner.search`, with `exploration=None`. S04 selects maximum
completed visits, with smallest shared ID on exact ties. No RNG, temperature
schedule or root noise is introduced. Deliberate stochastic smoke configuration
is recorded as null. PUCT remains
`Q + c_puct*P*sqrt(max(1,sum outgoing N))/(1+N)`, with accepted defaults 1.5
and unvisited Q=0. All move applications use the shared adapter.

`GameRecord` saves the normalized initial FEN, explicit opening prefix, complete
actual UCI move trace, history completeness, final FEN/promoted bits, budget,
cap, PUCT settings, evaluator identity, selection/end-policy/vocabulary IDs,
search attempts and game status/reason. `to_dict()` is a JSON-ready independent
snapshot with schema `s05-game-v1`. There are no learning or WDL targets.

Each `SearchAttempt` records its actual retained ply, requested/completed counts,
status, selected shared ID/UCI and sparse visited root edges `(ID,N,W)` in
parent-player perspective. On a fresh root, sum outgoing N is precisely this
call's completed-search count, including surviving backups on interruption.
Root preparation and unresolved/failed backups count zero. Error records retain
the original exception type/message and S04 failure stage when available.
Attempt actions describe search selection; only `moves_uci` records actual
application. A successfully searched action can remain unapplied if application
fails or execution is interrupted afterward.

## Move-boundary and execution semantics

At each boundary the runner adjudicates through `terminal_result` before search
and before comparing the cap. After the final allowed move, it returns to that
same adjudication boundary. Checkmate, stalemate and reached draws therefore
take precedence over truncation, including outcomes with legal moves remaining.
Intended-move-only claims never end the game. The accepted ending policy remains
`actual-occurrence-v1`, with library outcomes first, then reached repetition,
then reached fifty-move conditions.

| Status | Outcome and reason |
|---|---|
| completed | Exact adapter reason; outcome `1-0`, `0-1` or `1/2-1/2`; winner white/black or null for a draw |
| truncated | `ply_limit`; outcome and winner null |
| interrupted | Caught KeyboardInterrupt/SystemExit; `execution_interrupted`; outcome and winner null |
| failed | Ordinary runtime error, including SearchFailure; `execution_failed`; outcome and winner null |

Every returned status retains full replayable actual history. Application occurs
on a candidate full-history copy and publishes it only on success; an injected
post-push failure/interruption cannot publish a partial game move. Caller boards
remain unchanged. S04 itself still propagates interruptions; this outer game
boundary records them. The runner does not retry failures or resume games.

## Recorded checks

From this checkout, using existing pinned environments without installing or
changing dependencies:

```sh
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -B runs/s05/verify.py
```

The verifier sets `PYTHONPATH=src:.`, disables bytecode and pytest cache, and
saves actual argv/cwd/environment, UTC start times, durations, exit codes,
stdout/stderr hashes and JUnit results. Full model regressions use the accepted
M06 diagnostic plugin. Accepted reviewer files use importlib collection.

Final evidence:
[commands](../runs/s05/20261008T132541185409Z/commands.json),
[summary](../runs/s05/20261008T132541185409Z/summary.json),
[preservation](../runs/s05/20261008T132541185409Z/preservation.json).
Every final recorded command exited 0.

| Check | Adapter environment | Model environment |
|---|---|---|
| Focused S05 + accepted S01–S04 | 406 passed, 7.14 s | 406 passed, 7.33 s |
| Full regressions | 758 passed, six expected optional-model skips, 11.52 s | 926 passed, 41.03 s |
| Accepted D02/S02, D03/S03 and D04/S04 reviewer probes | 70 passed | 70 passed |
| Frozen adapter replay | 13 fixtures matched | 13 fixtures matched; reports equal |
| Saved game replay | All 15 records from each producer replayed | All 15 records from each producer replayed |

The 46 new S05 checks cover both-color immediate terminal wins and backed-up
values, legal ID/move selection, terminal-before-search/cap precedence,
checkmate/stalemate, reached repetition/halfmove draws, intended-only claims,
last-allowed-move outcomes, truncation, complete/incomplete long opening history,
caller ownership, custom settings, JSON, injected evaluator failures and
KeyboardInterrupt/SystemExit during preparation/partial search and after an
actual move, and failed/interrupted candidate application. A subprocess blocks
Torch, model, training, data and engine imports while running a dummy game.

Both environments use Python 3.12.12, NumPy 2.5.3, python-chess 1.999 /
chess 1.11.2 and pytest 9.1.1 on macOS 14.5 arm64. The adapter environment has
no Torch. Existing accepted reviewer probes depend on the preserved external
D03 preparation artifact; no bulk data preparation/audit is repeated.

The first manual preflight omitted PYTHONPATH and resolved the pinned editable
primary checkout instead of S05, failing collection. The recorded development
run `runs/s05/20261008T132345578252Z` then had 397 passes and three fixture
failures: evaluator call 8 interrupts the second root after one completion,
not the intended two. Injection was corrected to call 9. Those failures and
the collection explanation are retained in `runs/s05/preparation.json` and the
original development outputs; no accepted implementation was changed.
The final staged inventory and whitespace checks are recorded in
`runs/s05/final-checks.json`. The preserved failed pytest stdout/XML contain
six trailing-whitespace diagnostic lines; the full check reports these raw
evidence lines. A check excluding exactly those two lossless evidence files
passes for all source, tests, documentation, scripts and final evidence.

## Bounded smoke evidence

[Configuration](../runs/s05/20261008T132541185409Z/games-adapter/configuration.json)
is written before executing the fixed suite. Budgets are 16 per move for
ordinary dummy games, 64 for uniform-prior immediate mate games, 4 for controlled
boundary/fault checks and 1 for intended-claim probes. Caps and opening histories
are explicit for every record. Setup-FEN tactical probes retain incomplete
history provenance. Records produced independently in both environments are
byte-identical:
[adapter records](../runs/s05/20261008T132541185409Z/games-adapter/games.jsonl),
[model records](../runs/s05/20261008T132541185409Z/games-model/games.jsonl).

| Suite | completed | truncated | failed | interrupted |
|---|---:|---:|---:|---:|
| Five actual dummy-search games | 4 | 1 | 0 | 0 |
| All 15 games/boundary/fault records | 11 | 2 | 1 | 1 |

The actual dummy games yield one White win, one Black win and two rule draws;
the remaining game truncates. The standard-start game reaches actual threefold
repetition after 12 plies; the 14-ply opening game truncates at total ply 22.
Across all records, only the 11 completed games contribute outcomes: one White
win, two Black wins and eight rule draws. The four unfinished statuses contribute
no outcome. There are 29 generated actual plies and 486 completed game-search
simulations, including surviving partial fault searches. The suite is bounded
by caps; early rule outcomes reduce actual work.

[Known positions](../runs/s05/20261008T132541185409Z/games-adapter/known-positions.json)
use uniform legal priors and zero dummy nonterminal values, without a supplied
winning move. At 64 completed simulations, White selects `g6g7` and Black selects
`g3g2`; each shared selected ID is 2998. Each winning edge has N=45, W=45, Q=+1
from its parent's perspective, with exact child terminal value -1. These two
separate diagnostic searches add 128 completions outside the 486 game searches.
Controlled one-hot evaluators separately exercise reached repetition and
stalemate on the final allowed ply; they are explicitly identified in records.

`runs/s05/replay_games.py` imports the adapter only and reads the saved JSONL.
It reconstructs the initial board/opening provenance, independently enumerates
legal actual moves and shared IDs, applies every generated move, verifies each
attempt's counts/legal edges, full final history/FEN/promoted bits, cap boundaries
and exact status/outcome semantics. Both producers' 15 records are replayed in
both pinned environments: four independent replay commands in total. The two
accepted 13-fixture adapter replay reports are also identical.

## Limitations and review boundary

Dummy and controlled games establish search plumbing, not playing strength.
Budgets are recorded smoke choices, not calibrated strength settings. CPU macOS
arm64/Python 3.12.12 is the tested platform; cost/memory, other platforms and
concurrent tree access remain unmeasured. Sequential owned copies prioritize
history correctness. Records are snapshots, not an incremental crash journal;
hard process termination cannot return an interruption record.

Full E01 opponent/paired-match protocols, R01 self-play schedules/resumable
learning episodes, real evaluators, checkpoint identities, teacher annotation,
training and GPU work are outside S05. No source/data-track status files or
accepted historical evidence are edited. Local main remains the exact accepted
base. Commit this scoped change and stop for independent review; do not infer
S05 acceptance or Gate S acceptance from these checks.
