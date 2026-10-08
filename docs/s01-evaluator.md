# S01: CPU dummy evaluator

Implemented 8 October 2026 on `codex/s01-dummy-evaluator`, in the separate
worktree `/private/tmp/rl-chess-s01`, based on
`4765a8cb3d6ae9cf50619be283176ab9ac17cca6`. Automated acceptance passes;
this focused change stops for independent review. Gate A acceptance and M06's
independent AI review for local CPU correctness are user-supplied prerequisites.
S01 uses no model. No human teammate sign-off or S01 review acceptance is claimed.

Read `plan1.md`, `coding-plan.md`, `interface.md` and `README.md` before
implementation. No applicable `AGENTS.md` was present in the repository,
its subdirectories, `/Users/arvindsrinivass/Desktop`, `/Users/arvindsrinivass`,
`/Users`, or `/`. The S01-specific instruction to preserve shared documents
supersedes the roadmap's general completion-log guidance. D01 can proceed
independently: this change adds only S01 files and preserves every baseline file.

## Public usage

```python
from chess_rl.adapter import Evaluation, TerminalResult, make_state
from chess_rl.search.evaluator import evaluate

assert evaluate([]) == []
state = make_state()
result, = evaluate([state])
assert isinstance(result, Evaluation)
assert result.priors.shape == (4352,)  # NumPy float32, 20 legal entries at 1/20
assert result.value == 0.0
assert result.wdl.tolist() == [0.25, 0.5, 0.25]

mate = make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
results = evaluate([state, mate, state])
assert isinstance(results[1], TerminalResult)
assert results[1].ending_reason == "CHECKMATE"
assert results[1].priors is None
```

The signature is
`evaluate(states: Sequence[ChessState]) -> list[Evaluation | TerminalResult]`.
The function returns one result per input in order. For each state it calls
the public adapter `terminal_result` first and returns that exact object when
present. This includes actual-occurrence draws with available legal moves.
Terminal states bypass the legal mask, policy normalization and nonterminal
`Evaluation` construction. S01 requires no feature encoding for any state.

For a nonterminal state, the public adapter `legal_mask` supplies the shared
move vocabulary. A fresh float32 vector is zero except at legal IDs, where
each entry is `1 / legal_move_count`. A zero legal count after nonterminal
adjudication raises a clear `ValueError` rather than dividing by zero.
The synthetic WDL is **[win, draw, loss] = [0.25, 0.5, 0.25]**, normalized
with equal win/loss probabilities and `value=P(win)-P(loss)=0.0` from either
side's perspective. This is a deterministic plumbing distribution, not an
estimate of the position or a terminal draw label.

Each nonterminal result and both arrays are independently owned, including
repeated occurrences of the same input and subsequent calls. Boards, retained
move stacks, all historical prefixes, initial FENs and history provenance are
preserved. No state reconstruction, encoder, move mapping or evaluation cache
is introduced. The evaluator imports only NumPy, shared contracts and public
adapter types/functions; it has no PyTorch, engine, model, training or data imports.

## Changes

- `src/chess_rl/search/__init__.py`: small package boundary.
- `src/chess_rl/search/evaluator.py`: the public dummy evaluator.
- `tests/test_evaluator.py`: 33 focused CPU acceptance cases.
- `docs/s01-evaluator.md`: this task's contract, usage, results and limitations.
- `runs/s01/verify.py`: evidence runner; every invocation creates a fresh directory.
- `runs/s01/preparation.json`: actual preparation commands and initial test result.
- `runs/s01/20261008T070959324954Z/`: commands, environment reports, raw stdout/stderr,
  JUnit XML, both replay reports, protected-file SHA-256 hashes and summary.

All 243 tracked files from the base commit are byte-identical, including
planning/status files, dependencies, adapter/model implementation and all
earlier evidence. S01 evidence resides only under `runs/s01/`.

## Verification

The actual evidence-run command, from `/private/tmp/rl-chess-s01`, was:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python runs/s01/verify.py
```

The runner reuses the two existing interpreters and directs imports to the S01
worktree using `PYTHONPATH=src:.` and `PYTHONDONTWRITEBYTECODE=1`; it does not
modify either environment. Exact argv, cwd, environment overrides, UTC start
times, exit codes, durations and raw outputs are in the run's `commands.json`.

| Actual check | Result |
|---|---|
| Focused `pytest tests/test_evaluator.py -p no:cacheprovider` in `.venv-acceptance` | 33 passed, 0.28 s |
| Same focused checks in `.venv` | 33 passed, 0.28 s |
| Full adapter-only `pytest -p no:cacheprovider` in `.venv-acceptance` | 152 passed, 6 expected optional-model skips, 2.04 s |
| `scripts/check_adapter.py`, separate process in each environment | All 13 frozen records match; both JSON outputs agree |
| `git diff --check` | Passed |
| Every baseline tracked file compared to its Git blob | 243 byte-identical files, SHA-256 evidence saved |

The 152 tests comprise the original 119 adapter checks plus 33 S01 cases.
The CPU environment confirms PyTorch is absent. Both environments use Python
3.12.12, NumPy 2.5.3, python-chess 1.999 / chess 1.11.2 and pytest 9.1.1 on
macOS 14.5 arm64. An initial focused run had 32 passes; one additional Black
actual-repetition case brought the final total to 33. No failed checks occurred.

Coverage includes empty, batch-of-one, repeated-input/repeated-call and mixed
batches; both sides to move; both castling flanks; legal en passant for both
colors; all four forward/capture promotions for both colors; both-color mates
and stalemates; actual repetition, including history beyond eight snapshots;
reached fifty-move draws; insufficient material; automatic fivefold/75-move
endings; and mate precedence over the halfmove ending. Intended-move-only
threefold/fifty claims produce ordinary nonterminal evaluations.

Support checks enumerate `board.generate_legal_moves()` independently of the
evaluator's mask, map those moves through public `move_to_id`, and compare
the complete set of positive prior IDs. They check every legal entry's
uniform weight, illegal zeros, float32 shape/finiteness, normalization within
1e-6, synthetic WDL and value. Terminal checks verify exact adapter object
identity and prohibit mask/Evaluation/encoding calls. Mixed batches record
adjudication-before-mask order and compare every output with its input.
Preservation checks include board and stack identities, raw board fields,
retained stack entries, every prefix's FEN/promoted bits, provenance and
encoded features before/after. Mutation checks confirm returned arrays never
alias other results. A fresh-process import guard blocks model, torch,
engine, training and data modules and rejects feature encoding.

## Limitations and review boundary

CPU macOS arm64/Python 3.12.12 is the tested environment; other builds and
platforms are unvalidated. Float32 priors have ordinary rounding error, so
normalization is checked with a 1e-6 tolerance. Input states are expected to
meet the accepted adapter contract; S01 adds no new validation framework.
The synthetic outputs establish plumbing only and carry no playing-strength
claim. No PUCT, search configuration, backup, root visits, exploration,
matches, self-play, real evaluator, dataset processing, training, engine,
GPU work, subagents or paid compute was implemented or run. Full model
regressions are outside S01; only its focused checks use the model environment.
Independent review of the resulting S01 commit remains required.


## S01 review acceptance — 8 October 2026

The user reports that the planning agent independently accepts S01 commit
`834eab9317d96547333b4958c863b503f02ecba5` for **CPU evaluator plumbing**.
The combined D01/S01 copy passed **403 model tests**, **235 adapter tests with
six expected skips**, and both **13-fixture replays**. All **243 S01 baseline
files** were preserved; no blocking findings were reported. These are
user-supplied independent **AI review** results, not human teammate sign-off
or a new reviewer run by the S02 implementer.

D01 and S01 are integrated into common base
`0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`. This dated attribution supersedes
the historical pending S01 review status above and preserves the original
evidence unchanged. S02 starts from this common base in a separate worktree;
D02 owns shared README/coding-plan acceptance and status synchronization.
Structured attribution: `runs/s02/s01-review-acceptance.json`.
