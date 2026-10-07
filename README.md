# RL-Chess

The first milestone implements the shared standard-chess CPU/NumPy adapter in
`interface.md`. `coding-plan.md` records task status and acceptance evidence.

## Setup and checks

Tested on macOS arm64 with uv 0.12.10 and Python 3.12.12. `.python-version` pins
Python; `pyproject.toml` pins direct dependencies and the build backend;
`uv.lock` pins the resolved dependency graph with artifact hashes. Runtime:
NumPy 2.5.3, python-chess 1.999 (the importable chess package is 1.11.2).
Tests use pytest 9.1.1. The supported Python range is 3.12.x; other OSes have
not been tested. The optional model extra pins PyTorch 2.8.0; adapter setup omits it.

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

F01, F02, A01–A07/Gate A and F03 are accepted. On 7 October 2026, the
user supplied independent AI review acceptance of adapter commit `868371c`
(119 tests in both existing environments; all 13 replay fixtures matched) and
F03 commit `5958fd638fce0b6cf98a6631bacf743ce89551b8` (source checker and five
corruption cases). This is not a claim of human teammate review. The dated
completion-log entry supersedes the preserved historical pending statuses.

M01 commit `66eea36ba74ba811298ce8347a5153f562c474be` is accepted through
independent AI review supplied by the user on 7 October 2026: implementation
inspection, 139 model tests, 119 adapter-only tests with one expected skip,
12 numerical cases, sharing, both 13-fixture replays, source/dependency checks,
original planning hashes and clean working tree; no blocking findings.
[Historical M01 evidence and acceptance](docs/m01-gab.md) are preserved.

M02 backbone passes automated acceptance and stops for independent review.
Results: 157 full model tests, 119 adapter-only tests with two expected skips,
six independent backbone reference cases, dropout/sharing/real-adapter checks,
M01 rerun, source/dependency and preservation checks.
[M02 settings, results and limitations](docs/m02-backbone.md) and
[exact commands](runs/m02/commands.json) provide the evidence. M03 onward,
policy/WDL heads, relative terms, data, training and search remain unimplemented.
No human teammate review, subagents or paid compute is claimed.


## Pinned upstream sources (F03)

[Port map and attribution](docs/upstream-sources.md) explains the exact source
operations and their mapping to `interface.md`.
[Source lock](docs/upstream-lock.json) records full commit SHAs, immutable URLs,
10 file SHA-256 checksums and 18 operation/reference entries. Verbatim license
text/header copies are under `docs/notices/`. Use this lock for model ports;
the planning files' moving links are background references. The repository
completion log records F03's completion; the original task roadmap is preserved.

The standard-library acceptance command retrieves only pinned source text and
commit/tree metadata (no packages, weights or data):

```sh
cd /Users/arvindsrinivass/Desktop/RL-Chess
.venv/bin/python scripts/check_upstream.py
```

It requires GitHub network access. For an offline check against already
retrieved repositories, supply both checkout paths. It reads the pinned Git
objects even if the working trees have other contents:

```sh
.venv/bin/python scripts/check_upstream.py \
  --maia3-checkout /private/tmp/rl-chess-f03-maia3 \
  --lczero-checkout /private/tmp/rl-chess-f03-lczero
```

Those temporary checkouts were created for this audit and are not required
runtime dependencies. To recreate them without relying on moving branches:

```sh
git clone --no-checkout https://github.com/CSSLab/maia3.git /private/tmp/rl-chess-f03-maia3
git -C /private/tmp/rl-chess-f03-maia3 cat-file -t 1e13597c42d4858b7cfd7cfdae01e297263364b2
git clone --no-checkout https://github.com/daniel-monroe/lczero-training.git /private/tmp/rl-chess-f03-lczero
git -C /private/tmp/rl-chess-f03-lczero cat-file -t 7b3ee3fcdb49d8abeb034718935b099134c11882
```

Use unused directories when recreating checkouts. Both Git-object and
independent HTTPS verification passed: 10 files, 18 entries, 43 symbol/range
references and 2 retained notices. [Saved evidence](runs/f03/source-check.json)
contains the actual commands, results and original-plan preservation hashes.
The checker failed closed for five isolated corruptions (coverage, moving URL,
checksum, symbol and notice); it never imports or executes upstream models.

Documented differences: Maia-3 raw WDL is `[loss,draw,win]`; our contract is
`[win,draw,loss]`. Its oldest-first 12-channel history/per-snapshot orientation
and release presets differ from our locked input/backbone. The earlier source
has a GPL-3.0-or-later file notice but no standalone license file; Maia-3 has
AGPL-3.0 text. Upstream leaves PyTorch unpinned. M01 resolves its GAB
constructor defaults under the environment below; upstream framework/RNG-stream equivalence is not
claimed. M02 numerical acceptance is recorded below; M03–M06 remain pending.


## Compact GAB (M01)

Install the optional model environment with `uv sync --locked --extra model`.
Plain `uv sync --locked` preserves CPU/NumPy adapter-only installation. The lock
contains PyTorch 2.8.0 macOS arm64 wheel hashes and 2.8.0+cpu Linux/Windows
artifacts; only macOS 14.5 arm64, Python 3.12.12, CPU execution is tested here.
[Environment, defaults and acceptance](docs/m01-gab.md) records the dependencies,
constructor resets, reference checks and limitations.

```sh
uv sync --locked --extra model
uv run --locked --offline --extra model pytest
uv run --locked --offline --extra model python scripts/check_m01.py
uv run --locked --offline --extra model python scripts/check_adapter.py
```

Ordinary numerical tests and `check_m01.py` use a frozen pinned operation plus
an independent functional calculation, with no network or full Maia import.
For an optional source comparison against the F03 checkout, use:

```sh
.venv/bin/python scripts/check_m01.py \
  --maia3-checkout /private/tmp/rl-chess-f03-maia3 \
  --output runs/m01/numerical-acceptance.json
```

```python
import torch
from chess_rl.model import CompactGAB, SharedTemplateBank

bank = SharedTemplateBank()  # [4096,32], Xavier normal, CPU float32
layers = torch.nn.ModuleList([CompactGAB(bank), CompactGAB(bank)])
assert layers[0].bank.weight is layers[1].bank.weight
bias = layers[0](torch.randn(2, 64, 256))
assert bias.shape == (2, 8, 64, 64)
optimizer = torch.optim.SGD(layers.parameters(), lr=0.01)
```

Construct the optimizer from the enclosing module's `parameters()` to
deduplicate the shared bank. Generator projections/norms are separate. Move
the enclosing collection to a new device/dtype as a unit. Shared modules may
have multiple state-dictionary paths; reconstruct with the same supplied bank
when loading. M02 integrates this GAB into attention below; no checkpoint format is implemented.


## Encoder backbone (M02)

```python
import torch
from chess_rl.model import EncoderBackbone

backbone = EncoderBackbone()  # training dropout 0.1, CPU float32
backbone.eval()
with torch.no_grad():
    tokens = backbone(torch.randn(2, 64, 128))
assert tokens.shape == (2, 64, 256)
optimizer = torch.optim.SGD(backbone.parameters(), lr=0.01)
```

This intermediate backbone has **3,567,168 parameters**, including one shared
GAB bank. This is not the final combined-model count. It has no policy/WDL
heads or relative terms. Use `.eval()` for inference; set `dropout=0` for
numerical diagnostics. Adapter batching remains the consumer's responsibility.

Recheck without overwriting historical evidence:

```sh
.venv/bin/python scripts/check_m02.py --maia3-checkout /private/tmp/rl-chess-f03-maia3
.venv/bin/python -m pytest
.venv-acceptance/bin/python -m pytest
```

The checkout argument is optional; numerical checks run offline.
[Focused M02 documentation](docs/m02-backbone.md) resolves constructor resets,
epsilon, bias settings and dropout placement. CPU macOS arm64 float32/float64
only is validated; no GPU/reduced-precision or checkpoint compatibility claim.
