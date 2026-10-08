# S04: completed budgets and root visit policy

Implemented 8 October 2026 on `codex/s04-root-policy`, isolated checkout
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/s04-root-policy`, from
exact accepted base `37f286618642150b0ffa904abee0cfaac3ad75e8` in
`/Users/arvindsrinivass/Desktop/RL-Chess`. D03 and S03 are independently AI
review accepted and integrated into local main. Earlier primary-main statements
in the preserved review documents describe their historical review boundary.

Read applicable repository/ancestor instruction locations, `interface.md`,
`coding-plan.md`, `docs/s02-puct.md`, `docs/s03-value-backup.md` and
`docs/d03-s03-review.md`, plus the accepted adapter/evaluator/search APIs.
No applicable AGENTS.md was found. No accepted-base file is changed, including
PUCT, evaluator, adapter, model, dependencies, shared README/coding-plan and
historical evidence. New code is confined to `chess_rl.search.runner`.
**S04 awaits independent review; Gate S remains unaccepted.**

## Public API

```python
import numpy as np
from chess_rl.adapter import make_state
from chess_rl.search.puct import Node, SearchConfig
from chess_rl.search.runner import search, TrainingExploration, SearchFailure

state = make_state(moves_uci=["e2e4", "e7e5"])
root = Node(state)  # Accepted full-history owned copy.
result = search(root, 16, config=SearchConfig())  # Evaluation, no noise/RNG.
assert result.completed_simulations == 16
assert result.root_visits.sum() == 16
assert result.training_policy.sum() == 1
print(result.move_id, result.move.uci())

training = search(
    Node(state), 16,
    exploration=TrainingExploration(temperature=1, noise_fraction=.25,
                                    dirichlet_alpha=.3),
    rng=np.random.default_rng(20261008),
)
```

- `search(root: Node, simulation_budget: int, *, config=SearchConfig(),
  evaluator=evaluate, exploration=None, rng=None) -> SearchResult` prepares
  the root idempotently, runs sequential batch-of-one traversal/backup, then
  selects a legal action. The default evaluator remains S01's synthetic dummy.
- `SearchResult` contains `requested_simulations`, `completed_simulations`,
  `root_visits` (independent CPU int64 `[4352]`), `training_policy` (independent
  CPU float64 `[4352]` or None), `move_id` (shared ID or None), `move` (decoded
  actual-coordinate `chess.Move` or None) and `terminal` (exact stored adapter
  result or None). Arrays are mutable snapshots, independent of tree statistics.
- `root_visits(root)` requires expansion and returns raw completed outgoing N,
  including legal zero-visit edges and zero on all illegal slots. It checks
  legal edge membership and nonnegative int64 counts. Terminal roots return
  an all-zero vector.
- `training_policy(root)` returns exactly `N/sum(N)` in float64, independent
  of action temperature and priors. A nonterminal root with zero completed
  visits raises ValueError. A terminal root returns None.
- `select_move(root, *, exploration=None, rng=None)` returns a shared legal
  ID, or None for terminal roots. A zero-visit nonterminal root raises ValueError.
  This helper selects only; root noise is applied during `search` traversal.
- `SearchFailure` stops at the first ordinary exception. `stage` identifies
  preparation, noise, simulation or selection; `partial` is a SearchResult
  with surviving visits and exact completions in this call; `__cause__` retains
  the original exception. Failure results never select an action. Positive
  surviving visits still yield their normalized target; zero visits yield None.

## Budget and failure semantics

A budget is a nonnegative Python integer (booleans and NumPy integer objects
are rejected). It requests **additional completed simulations in this call**,
not a desired cumulative tree visit count. Completion increments only after
accepted `backup` returns 1 for a nonempty, completed traverse-issued record.
Path depth is irrelevant. There is no attempt/evaluation/independent node
counter, virtual loss, wall-time budget or automatic failure retry.

Root preparation, pending paths, failed traversal/evaluation and failed backup
count zero. S03 retains atomic rollback of failed backup publication, so earlier
completed N/W survives. A later explicit search call can retry the retained
unexpanded child. Previously pending records are neither consumed nor counted
by the runner. Existing completed visits remain in all returned raw visits and
targets; `completed_simulations` describes only the current call.

A fresh zero-budget nonterminal root is prepared but has no action or target;
there is no prior/uniform fallback. A reused positive-visit root with budget
zero can return an action/target from its existing statistics, with zero new
completions. Terminal roots always return zero completions, no action and no
training target regardless of requested budget, including reached repetition
and halfmove draws with legal moves. Terminal evaluation, noise and sampling
are skipped. Zero budgets and terminal roots do not consume RNG state.

Invalid configuration arguments raise ValueError before preparation. Ordinary
runtime failures raise SearchFailure rather than reporting budget success.
KeyboardInterrupt/SystemExit propagate; accepted backup rollback and root-prior
restoration still run. The retained tree exposes surviving completed statistics
after an interruption; no interruption is reported as a successful return.

## Exploration decisions

These are explicit per-call plumbing choices, not tuned strength settings.
`exploration=None` means evaluation: no root noise, maximum raw visits, exact
ties resolved by smallest shared move ID. An optional RNG passed in evaluation
is unused. PUCT itself remains byte-identical:

```text
Q + c_puct * P * sqrt(max(1, sum outgoing N)) / (1 + N)
```

`SearchConfig` defaults remain c_puct=1.5 and unvisited Q=0, with parent-player
W/Q and smallest-ID exact score ties. Full-history ownership remains unchanged.

`TrainingExploration` is frozen and validates finite real fields:

| Field | Default | Behavior |
|---|---|---|
| temperature | 1.0 | Nonnegative; 0 gives deterministic maximum visits; positive values sample visited IDs with probability proportional to `N**(1/temperature)` |
| noise_fraction | 0.0 | In [0,1]; 0 disables noise; positive values mix root priors once per positive-budget search call |
| dirichlet_alpha | 0.3 | Positive concentration for every legal root ID, including zero-prior edges |

For epsilon=noise_fraction, temporary root P is
`(1-epsilon)*P + epsilon*Dirichlet(alpha,...,alpha)`, enumerated in sorted legal
ID order. No implicit epsilon=.25 is enabled; the example opts in explicitly.
Root edge priors are restored exactly in `finally`, including failure and
interruption. The stored evaluator prior array and all descendant priors remain
untouched. Noise changes which visits accumulate, while the target always uses
those raw counts. Action temperature never changes the training target or visits.

Positive temperature or positive noise_fraction requires an explicit
`numpy.random.Generator`, validated before any work (including zero budgets).
No global RNG or hidden seed is used. Positive-temperature action probabilities
use a stable log-count calculation; unvisited moves never gain action support.
Temperature zero keeps smallest-ID ties; positive temperature samples tied
visited moves. Repeated training calls draw fresh noise from the supplied RNG.
The caller owns configuration and RNG state; full self-play schedules and
resumption records belong to R01.

## Verification and limitations

Run from the isolated checkout with the existing pinned interpreter:

```sh
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -B runs/s04/verify.py
```

The runner reuses `.venv-acceptance` and `.venv` without installing dependencies,
sets `PYTHONPATH=src:.` and `PYTHONDONTWRITEBYTECODE=1`, disables pytest cache,
and saves actual argv/cwd/environment, UTC start times, durations, exit codes,
stdout/stderr and hashes, JUnit reports and preservation evidence. Reviewer
probes use `--import-mode=importlib` to avoid the accepted duplicate test filename
collection issue. Full model regressions use the existing M06 diagnostic plugin.

The first development run, `runs/s04/20261008T114804257477Z`, had 358 passes
and two failure-injection fixture failures. Its injected Edge setter intercepted
new edge construction before reaching backup publication. Restricting it to
owned root-edge writes produced the intended interruption after one new
completion. Product code was unchanged; failed command/output/XML is retained.

Final verification: [commands](../runs/s04/20261008T114835875222Z/commands.json),
[summary](../runs/s04/20261008T114835875222Z/summary.json) and
[preservation](../runs/s04/20261008T114835875222Z/preservation.json).
All recorded final checks exited 0.

| Check | Adapter environment | Model environment |
|---|---|---|
| Focused S04 + accepted S01/S02/S03 | 360 passed, 9.51 s | 360 passed, 8.99 s |
| Full regressions | 675 passed, six expected optional-model skips, 12.20 s | 843 passed, 54.14 s |
| Accepted D02/S02 and D03/S03 reviewer probes | 53 passed, 0.46 s | 53 passed, 0.56 s |
| Frozen adapter replay | 13 fixtures matched | 13 fixtures matched; reports equal |

The 113 new S04 cases pass in each environment. Both pinned environments use
Python 3.12.12, NumPy 2.5.3, python-chess 1.999 / chess 1.11.2 and pytest
9.1.1 on macOS 14.5 arm64; the adapter environment has no Torch. All **700
accepted-base files remain byte-identical**, including every historical run.
Primary main still points to the exact accepted base. The final staged
whitespace/inventory check is recorded in `runs/s04/final-checks.json`.

Checks cover exact budgets, multiple-depth paths, pending work, reuse, zero
budgets, terminal roots (including legal-move draws), independent scalar policy
normalization, deterministic ties, raw policy versus action temperature,
seeded exploration, independently generated noise and descendant/evaluator
isolation, failures during preparation/noise/evaluation/backup/selection,
KeyboardInterrupt rollback, repair/retry, legal move round trips, both castling
sides/colors, both colors' en passant and all forward/capture promotions.
Caller/root raw fields, stack identities, every historical prefix, complete and
incomplete provenance beyond eight snapshots, features and child replay are
checked. A subprocess blocks Torch/model/training/engine/data imports and encoding.

Sequential ownership of the accepted mutable tree is required. External tree
mutation, concurrent traversal, transposition caching, serialized pending
tickets and process-crash durability remain unsupported. Raw individual counts
must fit int64. CPU macOS arm64/Python 3.12.12 is the tested platform; costs,
memory, playing strength, other platforms and RNG equivalence across library
versions are unmeasured. Reviewer data replays depend on the preserved external
`/private/tmp/rl-chess-d03-prepared/games.jsonl` artifact. No full data preparation
or archive audit is rerun for this search-only change. No S05 matches, self-play,
real evaluator, training or GPU work is implemented. **Stop for independent
S04 review; Gate S remains unaccepted.**
