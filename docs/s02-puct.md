# S02: PUCT selection, expansion and traversal

Implemented 8 October 2026 on `codex/s02-puct-tree`, worktree
`/private/tmp/rl-chess-s02`, from the accepted **common merged base**
`0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f` in
`/Users/arvindsrinivass/Desktop/RL-Chess`. The latest user instruction overrides
the earlier request to branch from S01 alone. D01 and S01 are independently
accepted and integrated in this base. D02 owns shared README/coding-plan
status synchronization; S02 leaves those files and dependencies unchanged.
S01 AI review attribution is appended to its focused document and recorded in
`runs/s02/s01-review-acceptance.json`; earlier documentation/evidence stays intact.

Read `plan1.md`, `coding-plan.md`, `interface.md`, `docs/s01-evaluator.md`, the
accepted evaluator/adapter code, and the applicable instruction locations.
No applicable AGENTS.md was found. This implementation is S02 only and stops
for independent review. **Gate S is not accepted.**

## Public usage

```python
from chess_rl.adapter import make_state
from chess_rl.search.puct import Node, SearchConfig, expand, select_edge, traverse

state = make_state(moves_uci=["e2e4", "e7e5"])
root = Node(state)                       # Independently owned full-history copy.
config = SearchConfig()                 # c_puct=1.5, unvisited_value=0.0.
root_evaluation = expand(root)           # evaluate([root.state]); preparation only.
edge = select_edge(root, config)
resolved = traverse(root, config)        # One root-to-leaf selection attempt.
assert resolved.path[0].edge is edge
assert resolved.path[0].parent is root
assert resolved.path[-1].child is resolved.leaf
assert resolved.result is resolved.leaf.result
assert root.parent_visits == 0           # No S03 backup has occurred.
print([step.edge.move_id for step in resolved.path], resolved.result.value)
```

`expand(node, evaluator=...)` and `traverse(root, config, evaluator=...)` accept
the shared `evaluate(states) -> list[Evaluation | TerminalResult]` callable;
the default is the accepted S01 dummy evaluator. Search calls it with one
nonterminal state. Custom controlled evaluators in tests use that same API.
`select_edge` requires an expanded nonterminal node. `traverse` requires prior
explicit root expansion. An expanded terminal root returns an empty path and
its exact result; selection raises ValueError.

## Exact PUCT decisions

These are project engineering choices for the initial plumbing milestone,
not tuned strength settings or an assertion of exact Leela/AlphaZero parity.

For parent s and its legal edge a:

```text
N_parent(s) = sum(edge.visit_count for every outgoing legal edge)
Q(s,a)      = edge.value_sum / edge.visit_count, if edge.visit_count > 0
              config.unvisited_value, otherwise
U(s,a)      = config.c_puct * edge.prior * sqrt(max(1, N_parent(s)))
              / (1 + edge.visit_count)
score(s,a)  = Q(s,a) + U(s,a)
```

| Decision | Exact behavior |
|---|---|
| c_puct | Constant positive finite real, default **1.5**; no visit-dependent schedule |
| Unvisited Q | Fixed finite real in [-1,1], default **0.0**; no parent-evaluation substitution or first-play reduction |
| Parent visits | Sum of **completed outgoing edge backups**, including zero-prior edges; no independent node visit counter |
| Zero-parent visits | Exploration numerator uses `sqrt(max(1,N_parent))`, so initially U=c_puct*P |
| First selection | Highest prior because all initial Q are equal; uniform priors choose smallest legal ID |
| Zero-prior edge | Retained; U=0, and its Q still participates in selection |
| Ties | Compare Python floating-point scores exactly; then smallest shared move ID; no epsilon tie tolerance or RNG |
| Simulation count | One completed simulation is a resolved path with at least one root edge **successfully backed up once by S03**, regardless of path length |
| Root preparation | Root evaluation/expansion consumes zero simulations and zero edge visits |
| Terminal root | Empty path; no action and no counted edge simulation |
| Unfinished work | No visit for selection attempts, child creation, evaluation failure/interruption, or a resolved leaf awaiting backup |

SearchConfig is immutable and validates both fields, rejecting booleans,
non-reals, NaN/infinity, nonpositive c_puct and out-of-range unvisited values.
No simulation budget or loop is introduced in S02: later callers must apply
the above counting rule. Edge constructors validate vocabulary range, prior,
nonnegative integer N, and finite W with abs(W)<=N. S03 must preserve those
invariants when mutating the statistics. There is no virtual visit/loss.

## Records and value perspectives

- `Node.state`: independently owned ChessState, initial FEN and full retained
  actual-coordinate move stack. `Node.result` is None until resolved; afterward
  it is the Evaluation or exact TerminalResult **from that node's side to move**.
- `Edge.move_id` and `Edge.prior`: the parent's shared network-view move ID and
  parent policy probability P. Prior is copied unchanged to a Python float,
  retaining the evaluator float32 value; search does not renormalize it.
- `Edge.visit_count` (N): successful completed backups through this edge.
- `Edge.value_sum` (W), and Q=W/N: scalar values **from the parent player's
  perspective**, never the child's or a fixed White perspective.
- `Edge.child`: a distinct path-owned node. No parent pointer, FEN dictionary,
  transposition merge or cross-node evaluation cache exists.
- `PathStep(parent, edge, child)`: actual object references ordered root to leaf,
  identifying the edge that later backup must update. `TraversalResult` carries
  an immutable tuple of those steps, leaf reference and leaf result.

S03 must start with `resolved.result.value` in the leaf player's perspective.
Iterate `reversed(resolved.path)`, **negating once before each parent edge**.
For each edge accumulate that parent-perspective value into W and increment N
once, only as part of a successfully completed backup. At one ply the parent
edge receives -v_leaf; at two plies the root edge receives +v_leaf and the
last edge receives -v_leaf. Exact terminal losses use v_leaf=-1 and draws 0;
terminal WDL/result/provenance must remain exact. Do not add root preparation,
leaf evaluation, incoming node visits, virtual visits or attempts to
N_parent. A newly evaluated nonterminal leaf has zero outgoing N until a
later simulation traverses one of its own edges; its incoming edge can have
more visits than its outgoing sum. Do not increment a separate node counter
and substitute it into this formula. Ensure a path is backed up at most once;
S02 itself has no backup or completion marker. Failed backup must not leave
partially completed counts. No value accumulation is implemented here.

## Expansion, history and failures

Expansion adjudicates through shared `terminal_result` before invoking the
evaluator or legal mask. This preserves checkmate precedence and reached
actual-occurrence draws with legal moves. Terminal nodes retain the exact
adapter result object and have no edges. Re-expansion returns the stored
result without new adjudication, evaluation or edge construction.

For nonterminal nodes, require exactly one Evaluation. Derive the complete
legal ID set through the shared legal mask; reject nonzero illegal priors.
Construct one edge per legal ID, including legal IDs with zero prior. Publish
edges and result together after validation. Evaluator errors or malformed
results leave that node unexpanded and retryable.

Traversal selects an edge, copies the parent's full-stack state, and applies
the ID using shared `apply_move`. It retains the initial FEN, counters,
promoted bits, every historical prefix and completeness metadata. Callers,
parents and earlier path nodes remain unchanged. States never come from a
current-FEN reconstruction. Existing children are reused only for their own
edge; a different path always creates another node, even with the identical
exact current FEN. A newly reached leaf is expanded; an already expanded
nonterminal child continues deeper; an existing terminal child stops again
without evaluation. Thus fresh trees/configs yield repeatable sequences,
while consecutive traversals on one tree may grow the same path before S03
changes its statistics.

Traversal success and failure both leave N/W unchanged. If leaf evaluation
fails after child creation, that child remains attached but unexpanded;
retry follows the same legal edge and resolves it without duplicate children.
The evaluator is expected to honor the accepted read-only state contract.
Node state/edges/results are search-owned and must not be externally changed
between calls; S03 owns only edge statistic mutations. These small mutable
records are for sequential traversal, not concurrent access.

## Checks and evidence

Run from `/private/tmp/rl-chess-s02`:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python runs/s02/verify.py
```

The runner reuses `.venv-acceptance` and `.venv` without modifying dependencies,
sets `PYTHONPATH=src:.`, disables bytecode/cache creation and saves fresh
UTC-named evidence. `commands.json` contains actual argv, cwd, environment,
start time, duration, exit code and raw-output paths; JUnit reports provide
counts. The full model regression uses the existing
`runs.m06.diagnostic_plugin` for historical M05 diagnostic semantics, as the
accepted base does. There is no new model diagnostic or model change.

Focused checks use independent nonzero score arithmetic (N_parent=16), Q/U
and first-play values, changed constants, uniform/nonuniform priors, exact and
near ties, reverse insertion order and invalid configurations/statistics.
Legal sets are independently enumerated from python-chess and mapped through
shared IDs. Tests cover idempotent expansion, terminal-before-evaluator/mask,
exact terminal identity, legal zero-prior edges, both-color terminal values,
repeatable traversals, path references and explicit root preparation.

History checks include raw board fields, stack identities, every prefix,
encoded features, complete and incomplete histories beyond eight snapshots,
caller mutation after node creation, and two different histories with the
same exact FEN but different input features/adjudication. A branching tree
also reaches the same FEN through separate paths without node/evaluation
merging. Both colors' kingside/queenside castling, en passant, all q/r/b/n
forward/capture promotions, and terminal mate/fifty/repetition children traverse
legally against an independent actual-coordinate push reference. Interruption
with nonzero existing N/W and malformed evaluation results stays retryable.
A subprocess blocks Torch/model/engine/training/data imports and encoding.

Actual final results and preservation totals are recorded in the saved run
summary and appended below after verification. `runs/s02/preparation.json`
records the initial 92-pass/one-failure fixture mistake, its correction and
subsequent 95-pass preflight. No production formula change was required.

## Limitations and review boundary

CPU macOS arm64/Python 3.12.12 only; other platforms, devices and concurrent
traversal are unvalidated. Full-history copies prioritize correctness; cost
and memory have not been profiled. Float32 evaluator priors retain ordinary
rounding error. Default constants are uncalibrated plumbing choices. Dummy
values establish no strength or performance result. S03 actual backup and
S04 root-visit policy/move selection remain unimplemented. No exploration,
noise, temperature, match runner, self-play, checkpoints, real evaluator,
training, GPU, subagents or paid compute is introduced or run. Independent S02
review remains required; the search gate is not accepted.


## Final verification — 8 October 2026

Evidence run: `runs/s02/20261008T075131593992Z/`. Every recorded check exited 0.

| Check | Actual result |
|---|---|
| Focused S02 in adapter environment | 95 passed, 0.34 s |
| Focused S02 in model environment | 95 passed, 0.36 s |
| Full adapter-environment regressions | 330 passed, six expected optional-model skips, 2.61 s |
| Full model-environment regressions, existing M06 diagnostic plugin | 498 passed, 37.31 s |
| Adapter replay in each environment | All 13 frozen records match; reports equal |
| Diff whitespace check | Passed |
| Common-base preservation | 308 files byte-identical; S01 document retains its original bytes as exact prefix |

The counts include the accepted D01/S01 common-base tests. Accepted adapter,
model, evaluator, dependencies, README, coding-plan, interface, plan1, all
previous runs and upstream provenance are unchanged. The adapter environment
has no Torch. Both environments use Python 3.12.12, NumPy 2.5.3, python-chess
1.999 / chess 1.11.2, pytest 9.1.1 on macOS 14.5 arm64.

Changed task files: `src/chess_rl/search/puct.py`, `tests/test_puct.py`,
`docs/s02-puct.md`, appended `docs/s01-evaluator.md`, and `runs/s02/` runner,
preparation/review-attribution records and actual verification evidence.
Automated checks pass; independent S02 review remains pending.


## Independent AI review acceptance — 8 October 2026

S02 is accepted for its scoped milestone through fresh independent review
by the coordinating Codex agent at the user's request. No blocking findings
or production fixes were required. The reviewed D02 and S02 commits are
integrated together. [Review, fresh checks and limitations](d02-s02-review.md)
and [structured acceptance](../runs/d02-s02-review/acceptance.json) record
the evidence. This entry supersedes earlier pending-review statements without
changing historical evidence. Annotation remains blocked by D03/D04;
Gate S remains unaccepted.
