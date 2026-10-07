# RL-Chess

The first milestone implements the shared standard-chess CPU/NumPy adapter in
`interface.md`. `coding-plan.md` records task status and acceptance evidence.

## Setup and checks

Tested on macOS arm64 with uv 0.12.10 and Python 3.12.12. `.python-version` pins
Python; `pyproject.toml` pins direct dependencies and the build backend;
`uv.lock` pins the resolved dependency graph with artifact hashes. Runtime:
NumPy 2.5.3, python-chess 1.999 (the importable chess package is 1.11.2).
Tests use pytest 9.1.1. The supported Python range is 3.12.x; other OSes have
not been tested. Model/device dependencies are deferred to their milestone.

```sh
cd /Users/arvindsrinivass/Desktop/RL-Chess
uv sync --locked
uv run --locked python -c "import chess_rl"
uv run --locked pytest --junitxml=runs/milestone-1/pytest.xml
uv run --locked python scripts/check_adapter.py
uv pip check
```

The first setup downloads Python if unavailable and installs dependencies.
Tests require no GPU, teacher engine or dataset. A clean-environment check was
also run using these exact commands (the environment stays inside RL-Chess):

```sh
UV_PROJECT_ENVIRONMENT=.venv-acceptance uv sync --locked
UV_PROJECT_ENVIRONMENT=.venv-acceptance uv run --locked python -c 'import chess_rl; print(chess_rl.__file__)'
UV_PROJECT_ENVIRONMENT=.venv-acceptance uv run --locked pytest --junitxml=runs/milestone-1/fresh-environment-pytest.xml
UV_PROJECT_ENVIRONMENT=.venv-acceptance uv run --locked python scripts/check_adapter.py
```

Both suites passed 119 checks; all 13 frozen replay records matched across
separate processes and both environments. XML results, replay output, exact
installed versions and source-plan preservation hashes are in
`runs/milestone-1/`. The frozen fingerprints are regression checks; separate
hand-written tests establish channel values, move IDs and rule outcomes.

## Using the adapter

```python
from chess_rl.adapter import (
    make_state, encode_position, move_to_id, id_to_move,
    legal_mask, terminal_result, apply_move,
)
import chess

state = make_state()
move_id = move_to_id(state, chess.Move.from_uci("e2e4"))
assert move_id == 796
assert id_to_move(state, move_id).uci() == "e2e4"
apply_move(state, move_id)
encoded = encode_position(state)
assert encoded.features.shape == (64, 128)
assert encoded.available_snapshots == 2
assert encoded.history_complete
```

`ChessState` owns an actual-coordinate board, initial FEN and complete retained
move stack. `state.copy()` retains all of them. Use `apply_move` for episode
moves and `make_state` for source replay. Replay checks chess legality and may
continue through reached draw conditions; execution may not. Direct edits to
`state.board` bypass the adapter's guarantees and should be limited to controlled
push/pop on copies. Inspection uses full history and leaves the owned state intact.

An explicit starting FEN defaults to incomplete history, even if it resembles
the standard initial position. Setting `history_complete=True` on a declared
setup is the caller's attestation that the setup is the actual game start and
all moves were retained; the adapter cannot verify the source's provenance.
Completeness never becomes true merely by adding moves. Terminal results carry
that provenance too. Network padding repeats the earliest retained position
without creating rule occurrences. Black reflects ranks and preserves files.

Legality and episode endings are separate: `legal_mask`, `move_to_id` and
`id_to_move` describe chess-legal actions, including when an adjudicated draw
still has legal moves. `apply_move` rejects terminal states before decoding.
Null/illegal moves, Chess960, variants, invalid boards and bad IDs raise
`ValueError`. Exact generated legal moves exclude python-chess's king-to-rook
castling aliases; standard king-destination UCI is required.

## Review and next work

F01, F02 and A01–A06 have passed their acceptance checks. A07 implementation
and automated regression/replay checks pass; its Both-owner human review is
pending. A07 stays unchecked in the coding plan until that review occurs.
The reviewer can run the setup/check commands above and inspect the 13 replay
records, independent channel fixtures, draw precedence and preservation checks.
No second person's execution or approval is claimed.

There is no model, search, teacher annotation or training implementation yet.
Policy normalization and evaluator batching belong to S01/T04 and later model
helpers. Next is second-person A07/Gate A sign-off; the next implementation task
is F03, pinning and attributing upstream sources before model ports. F03 can
proceed independently of human sign-off. No paid compute has been used.
