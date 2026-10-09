# R01 sequential model-guided self-play — 10 October 2026

Only R01 is implemented on `codex/r01-self-play` in the isolated clone
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/r01-self-play`, from accepted
combined T03/T04 main `ea0f8a142ec3b1ec3266ffafef8853ea07ab0c2d`. Applicable repository
and ancestor AGENTS.md searches found none. The shared interface, coding plan,
accepted S05/T02/T03/T04/E01 methods and subsequent acceptance evidence were read.
Primary main, its exact unrelated planning patch and all accepted artifacts are
preserved. Automated verification is complete; independent review is pending.

## Shared dependencies and execution

`chess_rl.selfplay` adds only game orchestration and durable execution records.
It uses E01's `Opening` to validate initial position, source and complete prefix;
the adapter owns all replay, features, legality, move IDs and outcomes. Every
move uses a fresh full-history `Node` and the existing sequential `search` with
`SearchConfig`, `TrainingExploration` and one owned NumPy generator. Existing
search owns root Dirichlet noise, temperature selection, raw visits and the
visit distribution. `RealEvaluator` owns inference and calls the sole T02
checkpoint loader. No accepted production module or public dependency interface
was changed, including evaluator/runtime/checkpoint internals that parallel GPU
work may extend. No additional encoder, rules, policy normalizer or loader exists.

Execution is sequential CPU inference in evaluation/inference mode, with one
pending leaf and batch-of-one calls. There is no concurrent tree traversal or
multi-game inference queue. The public batched evaluator remains available for
later independent-game batching; this milestone does not add that scheduler.

`Settings` defaults to 16 simulations, a total 512-ply cap, c_puct=1.5,
unvisited Q=0, temperature 1 before total ply 30 and 0 thereafter, root noise
fraction .25 throughout and Dirichlet alpha .3. These explicit starting choices
are not calibrated strength settings. `Phase.start_ply` counts every retained
ply from the recorded initial board, including the opening, just as S05/E01's
cap does. A schedule must start at zero and strictly increase.

Every move boundary adjudicates the accepted `actual-occurrence-v1` terminal
result before applying the cap. This includes the position reached by the last
allowed move. Completed games have exact adapter reason, winner and PGN outcome;
truncated games have `ply_limit` and null winner/outcome; interrupted games have
`execution_interrupted` and null winner/outcome. An unfinished game is never
labeled a draw. There are no per-position WDL labels, buffer admission or training
updates. The saved visit distribution is the accepted search statistic, not R02
target finalization.

## Frozen records and durable boundary

`Session.create(directory, checkpoint_path, opening, *, settings, seed,
opening_selection_seed)` loads through `RealEvaluator`, requires complete opening
history, records exact checkpoint bytes/hash/ID/model configuration/contracts and
provenance, and freezes settings, opening source/selection seed/full prefix,
runtime and explicit RNG settings before any search. The directory must be new.
`configuration.json` is published without overwrite; `boundary.json` initially
holds the complete starting replay and adjudicated status. Both have canonical
JSON content checksums. An explicit setup FEN with `history_complete=True` is an
attestation that this is the genuine initial position of a declared setup game,
not an assertion that a FEN proves earlier history.

The supported recovery boundary is **one atomically committed move and its
search record, complete actual history, post-move RNG and adjudicated status**.
Files are flushed/fsynced, published atomically and the directory is fsynced.
The configuration freeze and initial boundary must finish before bounded work
starts; an interrupted creation with a missing initial file is refused on load.
`run()` holds a nonblocking POSIX advisory writer lock, rejects a stale session
and rechecks the recorded runtime. Consumers must serialize ownership and leave
the owned evaluator/model unchanged. Lock files and abandoned temporary files
are operational files, excluded from the evidence commit.

A nonfinal committed boundary is deliberately stored with `status=interrupted`,
including immediately after creation. It is resumable and explicitly unfinished.
Search, temperature/noise RNG consumption and candidate move application happen
against owned copies. Only a successful transaction publishes a new boundary.
KeyboardInterrupt/SystemExit or ordinary execution errors discard that partial
transaction, preserve the preceding RNG/history and append diagnostic metadata;
ordinary errors have `kind=execution-error` in the interruption ledger. Discarded
partial root visits are retained as diagnostics and never become played-position
records. `run(stop_after_moves=N)` cooperatively stops after N additional commits.
Terminal/cap outcomes reached by the Nth commit still take precedence over stopping.

A hard process kill leaves the last whole durable boundary. At most one search
is redone, from the exact pre-search RNG; partial trees need not survive. A
publication error propagates because publication might already have occurred:
reload the disk boundary instead of overwriting it with a guessed rollback.
Tests establish complete recovery from both old and new publication states.
Process interruption/crash is tested; hardware/power-loss and other filesystems
are outside the verified scope.

`Session.load(directory)` checks checksums, schema/contracts, replay and position
statistics, exact CPU runtime and checkpoint hash/identity/configuration. It uses
the same loader, without restoring training state. There are no resume overrides
for checkpoint, opening, cap, schedule or RNG. Truncated/completed records are
final; calling `run()` on them returns their snapshot without extending the cap.
Relocating checkpoint paths requires a future explicit migration workflow.

R02 can reconstruct every played root from the configuration's initial FEN and
the game's complete actual move sequence sliced at each position's `ply`. Each
committed position also stores root FEN/promoted bits/player, legal mask, dense
raw root visits `[4352]`, the shared search's visit policy `[4352]`, selected ID/UCI,
completed count, applied phase and before/after PCG64 state. These records survive
all statuses. `audit()` replays them through the adapter, checks legal visit support,
positive budget counts, search-policy storage consistency and terminal/cap outcome
precedence without inferring missing history or generating final labels.

The private NumPy PCG64 seed and full boundary state control exploration. Global
Python, NumPy and Torch CPU RNGs are explicitly unused for this inference path;
evaluation dropout is disabled. Loading does not alter them. Exact continuation
requires the same recorded runtime, including dependency/platform versions,
thread counts, strict deterministic algorithms, MKLDNN setting and matmul
precision. No cross-platform or accelerator continuation is claimed.

## Usage

The CLI sets one CPU compute/inter-op thread and strict deterministic algorithms.
Create a JSON plan with `checkpoint_path`, `seed`, `opening_selection_seed`,
`opening` (`opening_id`, initial FEN, actual `moves_uci`, `history_complete`, source)
and `settings` (simulation budget, cap, PUCT values and phase dictionaries):

```sh
PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python -B \
  scripts/run_r01.py create --plan /absolute/path/plan.json \
  --directory /absolute/path/new-game --stop-after-moves 2
PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python -B \
  scripts/run_r01.py resume --directory /absolute/path/new-game
```

The exact plan shapes and all fixture values are available in
[the frozen real suite](../runs/r01/real/frozen-plan.json). These fixtures use a
fixed hand-authored list with an explicit selection seed; no opening-selection
PRNG or external book is consumed. This is deliberately different from claiming
a sampled production opening book. Game execution seeds drive actual stochastic
exploration and differ for the two standard-start probes.

## Recorded evidence

The unchanged actual T03 plumbing checkpoint is `T03-plumbing-step-26`,
60,883,331 bytes, SHA-256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`,
at the preserved external T03 path. It is not selected pretrained P. The
checkpoint and all indexed accepted external bytes match their prior identities.

The suite freeze is SHA-256
`3275447269304ce2e8212761c907fd653d2aea02aee9988d2d18c6a281bb18b0`.
It precedes the execution-start receipt and freezes source hashes, runtime,
checkpoint identity, seeds, prefixes, budgets, phases, caps and interruption
locations. Neither caps nor settings changed after execution. Ordinary probes
use budgets 3–4 and short caps; the two mate probes use budget 64/cap 1 without
noise or temperature. No fitting or threshold selection ran.

| Frozen snapshot population | completed | truncated | interrupted | duplicate completed excess |
|---|---:|---:|---:|---:|
| All 20 snapshots | 8 (40%) | 11 (55%) | 1 (5%) | 1/8 (12.5%) |
| 15 snapshots with generated moves | 4 (26.67%) | 10 (66.67%) | 1 (6.67%) | 0/4 |

The four generated completions are both-color mates on the last allowed ply and
both-color reached fifty-move draws on the last allowed ply. Immediate terminal
fixtures test bypass/precedence, and the one deliberate repeated terminal-mate
fixture tests duplicate accounting. Duplicate excess is completed count minus
unique initial-board/full-actual-move-sequence keys. Unfinished games never enter
that denominator. These short caps deliberately cause truncation; the high rate
does not justify scaling. Production diversity and completion rates remain to
be measured under a separately frozen pilot configuration.

The counted snapshots retain 36 generated plies and 258 committed simulations.
The 19 ordinary fixture executions recorded 148 neural forwards, all exactly
`[1,64,128]`; terminal roots/leaves bypass inference. The additional continuation
comparison is reported separately to avoid counting reference/resume branches
as extra independent games.

The continuation fixture starts with 11 retained plies, stops after two new
moves, then is SIGKILLed on evaluator call 3 of the next search, after noise
consumption and one completed backup. A new process loads the unchanged prior
boundary and completes with deliberately scrambled global RNGs. Complete initial
history, opening prefix, checkpoint/search/phase settings, all played actions,
per-position visits/policies and private RNG match uninterrupted execution
exactly. Both core hashes are
`3e04ba91ae21b3fa895458e347eaa310ea754de971deed0b394d1378eaddbd58`.
The resumed game truncates at its original cap with no outcome. The interrupted
snapshot is retained separately from the final boundary, and global RNGs remain
unchanged. No unfinished snapshot was rewritten as a draw.

Verification uses existing pinned Python 3.12.12, NumPy 2.5.3, python-chess
1.999/chess 1.11.2, pytest 9.1.1 and Torch 2.8.0 on macOS arm64 CPU. No dependencies
were installed. The documented historical M06 diagnostic plugin is used only for
the full model regression; production inference retains combined GAB/Q/K/V.

| Check | Result |
|---|---|
| New R01 tests | 55 passed |
| R01 + search/game/evaluator/checkpoint focused tests | 308 passed |
| Full model regressions | 1,453 passed |
| Full Torch-free regressions | 1,023 passed; 12 expected optional-Torch skips |
| Accepted historical reviewer probes | 84 passed in each environment |
| Frozen adapter replay | All 13 fixtures match in both environments |
| M06 numerical checker | Passed |
| Real checkpoint suite and fresh-process crash/resume | Passed with exact continuation |

The first exploratory R01 test run had 52 passes/three fixture failures: the
historical long repetition trace continued past adjudication, and halfmove 149
mate roots were already fifty-move terminal. Fixtures were corrected to an exact
reached prefix and halfmove 99; production semantics did not change for those
failures. [Development receipt](../runs/r01/development.json) preserves this
distinction. Final commands store exact argv/environment/source hashes, timings,
exit codes, lossless output containers and JUnit results under `runs/r01/`.

[Real results](../runs/r01/real/summary.json),
[game snapshots](../runs/r01/real/games.json),
[summary/index](../runs/r01/summary.json) and preservation receipts provide the
review evidence. All historical tracked files remain exact except the two dated
documentation appends. Primary's exact head, branch, status, patch and tracked
bytes are compared before/after, along with 76 indexed external files.

Stop at the isolated R01 commit for independent review. No main integration,
R02/R03/R04, buffer finalization, RL update, new training, production book/P
selection, GPU work, performance campaign or playing-strength claim is included.
