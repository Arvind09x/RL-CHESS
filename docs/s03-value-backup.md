# S03: parent-perspective value backup

Implemented 8 October 2026 on `codex/s03-value-backup` in
`/private/tmp/rl-chess-s03`, isolated from the exact accepted base
`1301d033b81ca88e51c66ba27c1035a80c50fe54` in
`/Users/arvindsrinivass/Desktop/RL-Chess`. S02 is independently AI review
accepted, as recorded in [the D02/S02 review](d02-s02-review.md).
Read the repository/ancestor instruction locations, `interface.md`,
`coding-plan.md`, `docs/s02-puct.md` and `docs/d02-s02-review.md`; no applicable
AGENTS.md was found. This deliverable stops for independent S03 review.
**Gate S remains unaccepted.**

## Public API and completion

The existing `Node`, `Edge`, `PathStep` and `TraversalResult` remain the search
records. The only added public operation is `backup(resolved) -> int` in
`chess_rl.search.puct`; `TraversalResult.completed` reports completion.

```python
from chess_rl.adapter import make_state
from chess_rl.search.puct import Node, backup, expand, traverse

root = Node(make_state())
expand(root)                         # Preparation: zero visits.
resolved = traverse(root)            # Resolved, still pending: zero visits.
assert not resolved.completed
completed_simulations = backup(resolved)  # 1 for this nonempty path.
assert resolved.completed
assert root.parent_visits == completed_simulations == 1
```

- Start from the resolved leaf result in its side-to-move perspective. Walk
  the path in reverse and negate before updating each parent edge. Each
  traversed edge receives exactly one N and the corresponding value in W;
  Q stays W/N. For leaf value +0.6, one-ply root W gains -0.6; two-ply
  root/child-parent edges gain +0.6/-0.6; three-ply gains -0.6/+0.6/-0.6.
- A successful nonempty backup returns **1**, regardless of path length,
  and sets `resolved.completed` only after every edge update succeeds.
  Callers count simulations from successful completion, not traversal attempts
  or leaf evaluations. There is no independent node/simulation counter.
- An expanded terminal-root empty path returns **0**, changes no statistics,
  and remains `completed=False`. Repeating this zero-work call still returns 0.
- Duplicate nonempty completion raises `ValueError` without writes. Shallow
  copies share a private completion ticket, so aliases cannot complete twice.
  Every fresh successful `traverse` issues a distinct ticket, including repeat
  visits to the same terminal child and distinct pending paths sharing edges.
- Backup accepts only records issued by `traverse`. Constructing a
  `TraversalResult` or using `dataclasses.replace` does not issue a new ticket;
  backup rejects those records. Keep the returned record rather than rebuilding
  it from its fields. Completion is an in-memory sequential ownership mechanism,
  not a serialization/resume format or a security boundary.
- Invalid resolution, disconnected/cyclic/detached path references, wrong
  parent/child perspective, nonfinite/out-of-range leaf values and invalid N/W
  raise before publication. All updates are staged first. A publication
  exception, including an injected interruption after partial writes, restores
  every original N/W and leaves completion false; a repaired failure can retry.
  Preexisting visits from other completed traversals remain intact.

Root preparation, pending resolved paths, failed traversal/evaluation and
failed backup add zero visits. An incoming edge can have more visits than the
child's outgoing sum; a newly evaluated leaf has no completed outgoing visits.
`Node.parent_visits` remains the sum of all outgoing edge N, including
zero-prior siblings. The accepted formula is unchanged:

```text
Q + c_puct * P * sqrt(max(1, sum outgoing N)) / (1 + N)
```

Defaults remain `c_puct=1.5` and unvisited Q=0; exact score ties choose the
smallest shared move ID. Selection, expansion, leaf values, terminal result
identity/WDL/provenance, evaluator calls and full-history traversal retain
accepted behavior. Backup invokes no evaluator, encoding or adapter mutation.

## Verification and saved evidence

Actual command, run from `/private/tmp/rl-chess-s03`:

```sh
/Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -B runs/s03/verify.py
```

The [runner](../runs/s03/verify.py) reuses both existing environments without
installing dependencies, sets `PYTHONPATH=src:.` and
`PYTHONDONTWRITEBYTECODE=1`, and disables the pytest cache. Actual argv, cwd,
environment, UTC start times, durations, exit codes and raw stdout/stderr are
saved in [commands.json](../runs/s03/20261008T111351092812Z/commands.json).
[summary.json](../runs/s03/20261008T111351092812Z/summary.json) indexes JUnit
counts, conventions, environments and preservation. Every check exited 0;
there were no failed development test runs for S03.

| Check | Adapter environment | Model environment |
|---|---|---|
| Focused S03 + accepted S02/S01 tests | 247 passed, 1.20 s | 247 passed, 1.23 s |
| Full regressions | 521 passed, 6 expected optional-model skips, 3.28 s | 689 passed, 35.06 s |
| Accepted D02/S02 reviewer probes | 26 passed, 0.19 s | 26 passed, 0.21 s |
| Frozen adapter replay | 13 fixtures matched | 13 fixtures matched; reports equal |

The model full regression uses the existing `runs.m06.diagnostic_plugin` for
historical M05 diagnostic behavior, just as the accepted base did. Both
environments are Python 3.12.12, NumPy 2.5.3, python-chess 1.999 / chess 1.11.2
and pytest 9.1.1 on macOS 14.5 arm64. The adapter environment has no Torch.

The 119 new S03 cases use hand-written nonzero sign tables for one, two,
three, four and seven plies, both colors and positive/negative leaf values.
Existing nonzero N/W accumulates correctly; all siblings remain unchanged.
Exact mate checks use an independent winner-versus-parent-color oracle,
including one-, two- and four-ply paths. Draws cover stalemate, insufficient
material, reached fifty-move and reached threefold repetition. Five fresh
visits to each terminal child reuse its exact result without evaluation; the
controlled losing-line repeat probes use c_puct=10 to keep that line selected.
Default-formula checks remain at 1.5. Terminal roots and pending/failing work
never count a simulation.

Failure checks cover invalid statistics at multiple depths, invalid values,
broken ownership/continuity/resolution, duplicate aliases, fabricated records,
and RuntimeError/KeyboardInterrupt during both an edge write and completion
publication. They check complete rollback and successful retry. Preservation
checks cover board/stack identities and raw fields, every retained prefix,
initial FEN, complete/incomplete provenance, encoded features beyond eight
snapshots, prior/result identity and array bytes, both castling sides/colors, en passant,
and every q/r/b/n forward/capture promotion. A separate process blocks Torch,
model/training/data/engine imports and forbids evaluator/adapter work in backup.

517 accepted-base files remain byte-identical, including README, coding-plan,
interface, plan1, accepted adapter/evaluator/model code, dependency pins, S02
tests/documentation and all historical evidence. The sole modified base file
is `src/chess_rl/search/puct.py`. New files are `tests/test_backup.py`, this
document and `runs/s03/` runner/evidence. Shared status edits remain owned by
the data track. Diff whitespace checks pass.

## Limitations and review boundary

Sequential access to search-owned nodes, states, edges, results and tickets
is required, as in S02. External mutation, deep-copy/reconstruction as a new
simulation, concurrent writers, process-crash durability and other platforms
are unsupported. Backup validates referenced path topology and statistics;
it relies on accepted traversal for legal full-history state construction and
does not replay or re-adjudicate boards. Copy/validation cost and playing
strength are unmeasured. Controlled/dummy values establish plumbing only.
S04 policy/move selection, exploration/noise/temperature, match runners,
self-play, real evaluation, training and GPU work remain outside this change.
S03 has passed automated checks and awaits independent review; **Gate S is
unaccepted**.
