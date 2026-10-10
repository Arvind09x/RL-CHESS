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

M02 commit `bc2066d57203459be18d60013eb874c88f4bc5c3` is accepted through
independent AI review supplied by the user on 7 October 2026. The reviewer
reproduced 157 model tests, 119 adapter-only tests with two expected skips,
six backbone numerical cases, dropout/sharing, 12 M01 cases, both 13-fixture
replays, source/dependency and preservation checks; verified 3,567,168 backbone
parameters and a clean tree; no blocking findings. [M02 acceptance](docs/m02-backbone.md)
and [structured record](runs/m03/m02-review-acceptance.json) preserve the evidence.

M03 commit `83d8ac1bbaffe54bbf5cf3d02daf542982c03736` is accepted through
independent AI review supplied by the user on 8 October 2026: 187 model tests,
119 adapter-only tests/three expected skips, all 12 numerical cases, M01/M02
acceptance, both 13-fixture replays, source/dependency and preservation audits;
66,048 head elements and clean tree verified, no blocking findings.
[M03 acceptance](docs/m03-policy.md) and [record](runs/m04/m03-review-acceptance.json).

M04 commit `0fdbd1eaf5e4d1701a2627d55f87b2150a4f252a` is accepted through
independent AI review supplied by the user on 8 October 2026: 216 model tests,
119 adapter-only tests/four expected skips, all 12 WDL numerical cases,
synthetic/permutation/constructor checks, M01–M03 acceptance, both 13-fixture
replays, source/dependency and preservation audits. The 33,795 head elements,
WDL order, change manifest and clean tree were verified; no blocking findings.
[Appended acceptance](docs/m04-wdl.md) and [record](runs/m05/m04-review-acceptance.json).

M05 commit `2e6316dd0c49b0c22ff10ef72b55793856189a72` is accepted through
independent AI review supplied by the user on 8 October 2026; no blocking findings.
[Appended acceptance](docs/m05-network.md) records the reproduced checks.
M06 combined GAB + Q/K/V relative network has user-supplied independent AI review acceptance for local CPU correctness on 8 October 2026.
[Focused settings/results/limitations](docs/m05-network.md) and
[exact commands](runs/m05/commands.json) record the evidence. All prior run evidence and
plan1.md/interface.md remain unchanged; dated M05 documentation supersedes their
historical pending status. [M06 evidence](docs/m06-relative.md) records the
finalized fixed architecture; the appended dated entry records the M06 AI review. No human teammate review, subagents or paid compute is claimed.


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
claimed. M02–M05 historical numerical acceptance is recorded below; current M06 acceptance
is documented in [the focused M06 record](docs/m06-relative.md).


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


## Pair policy head (M03)

```python
import torch
from chess_rl.model import PairPolicyHead

head = PairPolicyHead(d_policy=128)  # validated positive integer; CPU float32
logits = head(torch.randn(2, 64, 256))  # contextual square tokens
assert logits.shape == (2, 4352)
```

The three bias-free projections have **66,048 parameters** at width 128.
Ordinary scores divide by sqrt(d_policy); promotion adjustments multiply by
sqrt(d_policy) and use projected destination keys on rank 8. IDs remain
`64*i+j` and `4096+4*(8*f+g)+p`, with queen/rook/bishop/knight order. All slots
are raw logits; legality and move conversion remain owned by the adapter.
The policy module remains standalone; the separate M04 WDL head is below.

Recheck without overwriting historical evidence:

```sh
.venv/bin/python scripts/check_m03.py
.venv/bin/python scripts/check_m03.py --maia3-checkout /private/tmp/rl-chess-f03-maia3
.venv/bin/python -m pytest -p no:cacheprovider
.venv-acceptance/bin/python -m pytest -p no:cacheprovider
```

The checkout argument is optional. [Focused M03 documentation](docs/m03-policy.md)
records independent references, tolerances, constructor resolution, exact results
and CPU limitations. M03 is accepted through independent AI review.


## Standalone WDL head (M04)

```python
import torch
from chess_rl.model import WDLHead
from chess_rl.contracts import WDL_ORDER

head = WDLHead(d_value=128)  # independent of d_policy; CPU float32
logits = head(torch.randn(2, 64, 256))  # encoder-final square tokens
assert logits.shape == (2, 3)
assert head.wdl_order is WDL_ORDER  # win, draw, loss; input side to move
```

Mean across squares → separate affine LayerNorm(256, eps=1e-5, bias=True) →
biased Linear(256,128) → ReLU → biased Linear(128,3). All six tensors retain
PyTorch 2.8.0 constructor initialization: **33,795 parameter elements**.
Returns raw logits; softmax/value conversion, terminal handling and outcome
labels remain consumers. Upstream's interpreter uses loss/draw/win; our newly
trained rows use the shared contract. No upstream checkpoint compatibility.

```sh
.venv/bin/python scripts/check_m04.py
.venv/bin/python scripts/check_m04.py --maia3-checkout /private/tmp/rl-chess-f03-maia3
```

[Focused M04 documentation](docs/m04-wdl.md) and [run evidence](runs/m04/commands.json)
cover the independent NumPy/manual-gradient reference, known synthetic stages,
square permutation, constructors and real adapter/backbone diagnostic.
M04 has independent AI review acceptance; its operations are reused unchanged in M05 below.


## Intermediate GAB network (M05)

```python
from chess_rl.model import ChessNetwork

network = ChessNetwork(d_policy=128, d_value=128, dropout=0.1, relative_enabled=False)
policy_logits, wdl_logits = network(features)  # [B,4352], [B,3]; B >= 1
```

`features` is a device/dtype-compatible `[B,64,128]` tensor. The accepted
backbone runs once; both heads receive the same encoder-final tokens. Returns
raw logits in pair4352-v1 and shared win/draw/loss order. The pooled WDL norm
remains separate. Pass `device`/`dtype` to construct all components together;
use `.eval()` to disable backbone dropout and `network.parameters()` for a
deduplicated optimizer iterator. There is one GAB bank and eight independent
generators. Accepted components and their initialization remain unchanged.

**3,667,011 intermediate parameters / 191 unique tensors**, independently
verified. Parameter and gradient payloads are each 14,668,044 bytes for float32
or 29,336,088 bytes for float64; these exclude activations/process overhead.
[Focused documentation](docs/m05-network.md) distinguishes the separate
checker-process peak RSS measurement. [Acceptance checker](scripts/check_m05.py)
and [results](runs/m05/network-acceptance.json) cover combined independent
outputs/gradients, routing, dropout, both branches, sharing and adapter preservation.

```sh
.venv/bin/python runs/m06/diagnostic.py scripts/check_m05.py
```

M05 has independent AI review acceptance. This explicit diagnostic retains
the intermediate count and common weights for regression checks. Empty
tensor batches are rejected clearly because the accepted backbone does not
support them. CUDA/MPS/reduced precision and other builds remain unvalidated.


## Fixed combined network (M06)

```python
network = ChessNetwork()  # relative terms enabled; dropout=.1, CPU float32
policy_logits, wdl_logits = network(features)
gab_only_logits = network(features, relative_enabled=False)  # same common weights
```

Eight layers each own independent zero-initialized Q/K/V tables `[8,225,32]`.
Packed projections, constructor initialization, norms/dropout and both heads
retain accepted settings. One GAB template bank is shared across all layers.
The finalized fixed network has **5,049,411 parameters in 215 unique tensors**.
[Focused M06 documentation](docs/m06-relative.md) explains the independent
references, tolerances, accounting, provisional-estimate difference and limitations.

Run current acceptance without overwriting historical evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python scripts/check_m06.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m pytest -p no:cacheprovider -p runs.m06.diagnostic_plugin
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv-acceptance/bin/python -m pytest -p no:cacheprovider
```

The explicit pytest plugin adapts only the historical M05 helper's constructor
to `relative_enabled=False`; M06 tests use the enabled default. M01–M04 standalone
components retain their accepted diagnostic defaults. Historical snippets above
describe their respective milestones; use these current suite commands after M06.
**287 model tests**, **119 adapter tests with six expected skips**, M01–M05
acceptance and both 13-fixture replays pass. The user-supplied planning-agent AI review reproduced these results, the complete M06 checker and 191 protected-file hashes; the reviewed tree was clean. GPU correctness and performance remain unverified.
No data, training, checkpoints, evaluator, search, GPU runs, paid compute or subagents.


## Dataset manifest preparation (D01)

D01 passes automated preparation checks and awaits independent review.
[Selected manifest](data/manifests/elite-2025-01-v1.json) and
[decisions, teacher provenance, readiness and limitations](docs/d01-data-manifest.md)
record the January 2025 Lichess Elite archive (exact URL verified by index/HEAD;
no archive body downloaded), split/sampling seeds 20261008/20261009, 90/5/5
splitting by game before sampling, eight-position cap, authoritative replay
JSONL plus game/ply sample references and protected heldouts. The initial target
is 10,000 successful labels and a separate 1,000-position benchmark. Coverage,
annotation costs and expansion scale are unmeasured.

Stockfish 17.1 is a runnable local Apple-silicon build of its pinned official
source release; the official binary download failed at the asset host. Full
executable/source-package/embedded-NNUE SHA-256s, build and idle UCI checks are
recorded under `runs/d01/`. No search was run. The fixed teacher settings remain
100,000 nodes, Threads=1, Hash=128 MB, MultiPV=1, UCI_ShowWDL=true, full-history
roots and per-annotation Clear Hash with root-perspective WDL/failure handling.

From the D01 worktree with installed dependencies and `PYTHONPATH=src:.`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/check_d01.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/check_d01.py --teacher-executable /private/tmp/Stockfish-sf_17.1/src/stockfish
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -m pytest tests/test_data_manifest.py -p no:cacheprovider
```

The manifest is **preparation-ready, not annotation-ready**. The downloaded
archive SHA-256/retrieval provenance is pending D02; algorithms, phases and
split/overlap/protected-heldout evidence are pending D03/D04. Calling
`DatasetManifest.require_annotation_ready()` or running the checker with
`--require-annotation-ready` rejects it. D02–D08, annotation, training and search
are unimplemented here; S01 remains unchecked. D01 has no independent review
acceptance. Historical M04/M05 acceptance and all earlier evidence are retained;
[the dated M06 review record](runs/d01/m06-review-acceptance.json) attributes
acceptance to the user's planning-agent report, not the D01 implementer's checks.


## D01/S01 AI review acceptance and integration — 2026-10-08

The user reports independent planning-agent **AI review acceptance** of D01
commit `df58f37de792cee248a89ff6644d77b144360a1f` for manifest preparation
and S01 commit `834eab9317d96547333b4958c863b503f02ecba5` for CPU evaluator
plumbing. S01 was accepted on its separate `codex/s01-dummy-evaluator` branch,
which remains available. The updated repository context confirms that **both
accepted tracks are now integrated** in `main` merge commit
`0347ceb7e6efd867e1f1253b3bf5a57d4fd6e48f`, the common D02 starting point.
The reported combined checks passed: 403 model tests, 235 adapter tests with
six expected skips and both 13-fixture replays; pinned teacher executable,
source-package and embedded-NNUE hashes plus a fresh idle UCI check passed.
There were no blocking findings. This is user-supplied independent AI review
attribution, not human teammate review or a new reviewer run by the D02
implementer. Historical evidence and pending-status entries remain intact;
this dated entry supersedes their D01/S01 status. Structured attribution:
`runs/d02/review-integration.json`. D02 has no independent review acceptance.

## PGN source preparation (D02)

D02 implements strict standard-PGN mainline intake and full-history replay
through python-chess and the accepted shared adapter. The selected January
2025 ZIP is now downloaded and byte-pinned in the manifest. Pre-split records
retain source member/ordinal/byte/line provenance, the genuine starting FEN,
all actual UCI moves and separate PGN result metadata. Canonical IDs,
duplicate consolidation and splitting remain D03 work. Annotation remains
blocked by the six unresolved D03/D04 algorithm/phase/audit/heldout fields.
[Focused behavior, reproduction and evidence](docs/d02-pgn-replay.md).
Independent D02 review remains pending.

The complete selected archive contains one PGN member: **289,776 source games
accepted, zero rejected**, **25,429,063 replayed plies**. These are intake
measurements before deduplication/split/sampling. Three full passes produced
identical tables; all source locations and hashes passed the output audit.
Automated checks passed: 155 focused tests, 475 model tests, 307 adapter tests
with six expected skips, both 13-fixture replays and the M06/source/teacher/
preservation checks. [Actual replay summary](runs/d02/replay-summary.json)
and [commands/evidence](runs/d02/preparation-evidence.json) are retained.
Bulk records remain at `/private/tmp/rl-chess-d02-accepted-prepared/`, outside
Git. D02 stops for independent review; D03/D04 still block annotation.


## Current integrated status — 8 October 2026

D02 PGN replay and S02 PUCT selection/expansion/traversal have fresh independent
AI review acceptance and are integrated together. No blocking findings or
production fixes were required. Fresh combined checks pass: 570 model tests,
402 adapter tests with six expected skips, both identical 13-fixture replays
and 26 reviewer-authored probes. [Review and limitations](docs/d02-s02-review.md)
and [structured evidence](runs/d02-s02-review/acceptance.json) supersede
historical pending-review statements above.

`chess_rl.search.puct` exposes `Node`, `SearchConfig`, `expand`, `select_edge`
and `traverse`; explicit root expansion consumes zero visits.
[Exact formula, signs and simulation-count conventions](docs/s02-puct.md)
remain binding for S03. D03 canonicalization/splitting and S03 backup are next.
Annotation remains blocked by D03/D04, and Gate S is not accepted.

### D03 canonical consolidation and splits — 8 October 2026

D03 automated checks are complete and **independent review is pending**. From
accepted base `1301d033b81ca88e51c66ba27c1035a80c50fe54`, the isolated
`codex/d03-canonical-splits` worktree at `/private/tmp/rl-chess-d03-canonical-splits`
implements the exact `start-moves-sha256-v1` identity, retains all source/result
metadata and rejects hash/payload conflicts. Versioned
`game-sha256-threshold-v1` assigns by identity with seed 20261008 and 90/5/5
targets, independently of ordering or corpus expansion.

All 289,776 accepted source records consolidate into **289,427 canonical games**:
**260,517 train / 14,480 validation / 14,430 test**. There are 349 additional
duplicate source occurrences in 112 groups; 13 groups have conflicting source
results, retained as metadata. Bulk `replay-jsonl-v1` games remain outside Git
at `/private/tmp/rl-chess-d03-prepared`. Two complete runs produced byte-identical
tables and audits. A separate all-source provenance/identity/split audit matched
every game and source occurrence and replayed 290 output games via the adapter.

Checks: 196 focused tests, 611 full model tests, 443 adapter tests with six
expected skips, both identical 13-fixture adapter reports, pinned upstream
sources, teacher bytes/idle UCI and preservation checks. Actual commands,
exits/output hashes and the small split audit are in `runs/d03/`;
[D03 decisions, reproduction and limits](docs/d03-canonical-splits.md) describe
the output schema and exact hashes. Only the D03 split procedure/audit manifest
fields are resolved. Four D04 blockers still prevent annotation. Sampling,
phases, position overlap and protected sample registries remain unresolved;
these game counts establish no usable sample coverage. No D03 independent
review acceptance is claimed.


## Independent D03/S03 acceptance — 8 October 2026

D03 canonical game splits and S03 parent-perspective backup are accepted
through fresh independent AI review in the isolated combined review checkout.
[Review and evidence](docs/d03-s03-review.md) supersede historical pending
statements. D04 sampling/overlap and S04 root policies/actions are next.
Annotation remains blocked and Gate S is not accepted.

### D04 position sampling — implemented, pending independent review

D04's deterministic phase/progress sampling, cross-split overlap removal/refill
and frozen held-out expansion are implemented in the isolated
`codex/d04-position-sampling` checkout at `/private/tmp/rl-chess-d04`, from
accepted base `37f286618642150b0ffa904abee0cfaac3ad75e8`. The
[D04 decision and execution record](docs/d04-position-sampling.md) documents
rule-position identity, full-history terminal exclusions, ownership, coverage
and capacity. Bulk outputs remain under `/private/tmp/rl-chess-d04-prepared/`;
[execution evidence](runs/d04/) stays compact in Git. The new
[D04 manifest](data/manifests/elite-2025-01-d04-v1.json) resolves only D04's
four preparation fields; the accepted original manifest remains unchanged.
D04 has no review acceptance yet. No annotations, benchmark, training, teacher
changes or merge into main are part of this task. D05 waits for D04 acceptance.


## Independent D04/S04 review acceptance — 8 October 2026

D04 sampling/overlap and S04 root policies/actions are independently AI review
accepted. [Review and evidence](docs/d04-s04-review.md) supersede historical
pending statements. Use `data/manifests/elite-2025-01-d04-v1.json` for D05; the
preserved original manifest retains its earlier metadata blockers. D05 worker/
tiny annotation and S05 minimal games are next. Gate S is not accepted.


## Current D05/S05 acceptance and integration — 8 October 2026

D05 Stockfish annotation and S05 bounded dummy-search games are independently
AI review accepted and combined with their original branch histories. Gate S
is accepted for tested CPU correctness/plumbing. No blockers or production
fixes were required. [Review and validation](docs/d05-s05-review.md) and
[acceptance record](runs/d05-s05-review/acceptance.json) supersede historical
pending statuses above. The user authorized integration into local main.

Fresh combined checks passed: 984 model checks; 816 adapter checks with six
expected skips; 70 prior reviewer probes and 14 new special-move probes per
environment. Both D05 bundles and all 15 saved games replayed successfully.
The two annotation bundles cover 12 distinct roots. Dummy games establish
plumbing, not strength. D06 benchmarking, D07 loading/batching and E01 match
protocols are next. D08, training, GPU work and model-guided games remain pending.


### D06 separate annotation benchmark — 2026-10-08; independent review pending

D06 automated completion is ready for review on `codex/d06-annotation-benchmark`
in `/private/tmp/rl-chess-d06`, from exact accepted base
`4184387c7e4b67559355d6b212a2a300c86c5ce6`. Primary main remains unchanged.
[Actual D06 results and limitations](docs/d06-annotation-benchmark.md) and
[runs/d06/results.json](runs/d06/results.json) record 1,000 distinct audited
training positions, excluding the 12 D05 references, fixed before annotation.
The accepted worker produced 1,000 labels in 1,000 attempts with zero failures,
retries, exclusions, interruptions or pending references. All labels passed
separate full-history/PV/WDL/move-ID/raw-UCI replay audit; full table scanning
independently proved membership and held-out exclusion.

Annotation loop: 432.023s, 2.314690 successful labels/s; mean/median/p95/p99
latency 431.953/427.340/496.882/540.623 ms. Setup and post-input audit are separate.
Engine-reported search totals 76.667s; startup/reset/shutdown, retries, returned
search fields, full replay/encoding/mask and storage costs are measured.
Raw transport preserves 498 score-bound flags omitted by the pinned parser
when placed after WDL; raw and parsed bounds are separately reported.
Conditional annotation projections for 10k/100k/300k/500k/1m successful labels
are 1.2/12/36/60/120 worker hours, with replay/encoding/audit/storage costs and
scaling assumptions recorded. No final expansion size is selected before
subsequent learning-curve evidence. Bulk benchmark artifacts remain external
and are excluded from D08 production.

Regressions: 992 model passes, 824 adapter passes/six expected skips, 84 prior
reviewer cases per environment, both identical 13-fixture adapter reports,
offline pinned-source and dependency checks. Exact commands, times, hashes,
failed audit/check commands and preservation evidence remain under runs/d06.
Accepted implementations, pins, teacher, historical evidence, original inputs
and protected registries are preserved. D06 remains unchecked pending independent
review. No D07, D08, training, search integration or GPU experiments ran.


## Independent D06/D07/E01 review acceptance — 8 October 2026

The user supplied independent review acceptance of D06, D07 and E01 with no
blocking findings. Their exact original commits and historical evidence are
merged together; the completion log retains both D06/D07 appended entries.
[Scoped acceptance](docs/d06-d07-e01-review.md) and
[verification record](runs/d06-d07-e01-integration/acceptance.json) record
1,181 combined model passes, 1,012 adapter passes/seven expected Torch skips,
84 accepted reviewer probes per environment, the frozen 13-fixture adapter
replay, all 1,000 D06 labels, all 12 D07 roots, 15 S05 records and both E01
48-game bundles. Accepted source bytes, manifests, teacher settings, historical
evidence and protected held-out artifacts are preserved; bulk stays outside Git.
Historical pending statuses above are superseded only within these milestone
scopes. No D08, T01, later tasks, training or playing-strength evaluation ran.


### D08 initial labels — 2026-10-09; independent review pending

D08 was implemented and executed in the isolated `d08-initial-labels` clone on
`codex/d08-initial-labels`, from reviewed integrated main
`9565093abe40a7807e3472c1101f5f28c14b7afe`. Exactly 10,000 new labels succeeded:
9,000 train, 500 validation and 500 test. Frozen deterministic reference membership,
ordering, quotas and bounded failure policy exclude all 12 D05 and 1,000 D06
references; prior labels do not count. All failures, retries, interruptions,
exclusions, pending references and shortfalls are zero. Canonical assignments,
protected registries, phase rules and the eight-position contract are preserved.

The accepted D05 teacher/search/label path is reused with an explicit D08 replay
hook that preserves genuine held-out splits; its default remains train-only.
Teacher binary/NNUE assets, full-history roots, fresh processes, Clear Hash and
100,000-node settings remain pinned. Raw UCI and score-bound flags are retained.
Every success passed independent board/history/provenance/legality/move-ID/PV/WDL
audits; all cross-split rule-position overlaps are zero. D07 loaded and batched
all three real splits, verifying every sample and three complete batch passes.

Annotation took 4,491.422s; the independent audit took 54.712s; full D07 verification
took 213.240s. External labels use 25.476 MB; the full production bundle uses
302.496 MB. [D08 methods, composition, cost and limits](docs/d08-initial-labels.md)
and [exact external paths/hashes/results](runs/d08/results.json) document the run.
Full regressions: 1,192 model passes; 1,023 adapter passes/seven expected skips.
Accepted inputs and primary main are preserved. Production artifacts remain in
`/private/tmp/rl-chess-d08-production/`, outside Git. Automated checks are complete;
independent milestone review and integration remain pending, so the historical
D08 checkbox remains unchecked. No training, held-out tuning or larger expansion
selection was performed. Stop for independent review before dependent work.

## T01 supervised losses — 9 October 2026; independent review pending

The isolated `codex/t01-supervised-losses` branch implements shared legal
normalization, integer teacher-class policy CE, soft root-side WDL CE and one
supervised optimizer step through the fixed combined model. [T01 API and
evidence](docs/t01-supervised-losses.md) records Adam/learning rate/loss weights,
seed, batch size and CPU settings. The real check uses one two-position D07
batch from the existing D05 training labels; no D08 data is required.

Checks pass: 219 focused tests, 1,273 full model tests, 1,012 adapter tests/eight
expected Torch skips, 84 prior reviewer probes, M06 and both frozen replays.
Independent losses/gradients and an actual optimizer update agree; batch
ownership, shared GAB and accepted implementation/data/manifest bytes are
preserved. Main remains at the reviewed base. Stop for independent review;
checkpoints, overfitting, P, T02/T04 and RL remain outside this change.


### D08/T01 AI review attribution and integration — 9 October 2026

The user supplied independent AI review acceptance of D08
`bcb208d73271451b05d1409d4bf340aaad82eabf` and T01
`f669fc8b4de397a31768321b9105e255b0ff0e0e` and authorized integration from
actual accepted main `9565093abe40a7807e3472c1101f5f28c14b7afe`. This chat
performed implementation inspection and fresh combined verification in isolation;
it did not commission a new independent reviewer, use subagents or claim human
review. Both original completion records and source histories are retained.

Combined checks pass: 1,284 model tests; 1,023 adapter tests/eight expected
Torch skips; 84 accepted reviewer probes in each environment; both exact frozen
13-fixture adapter replays; M06 and T01 numerical checks; the fixed T01 one-step
Adam check; all 10,000 D08 labels and three full D07 batch passes in every split.
Accepted Git/external bytes, protected registries, manifests and prior evidence
are preserved. The complete ten-file D08 bundle has a hash-verified durable copy
outside RL-Chess Git; frozen evidence is unchanged. Primary D06/D07/E01 checkbox
edits are preserved as uncommitted edits.

[Scoped integration, artifact location and limitations](docs/d08-t01-integration.md)
and [structured verification](runs/d08-t01-integration/summary.json) supersede
D08/T01 pending-review status. D08 and T01 are accepted within their original CPU
correctness boundaries. No checkpoint, tiny overfit, T02 or later implementation
was started. Stop after integration evidence.

## T02 checkpoints — 9 October 2026; independent review pending

T02 adds shared versioned save/load for the accepted combined M06 model in
`training/checkpoints.py`, with separate inference loading and training restore.
It preserves construction/contract/provenance metadata, standard CPU Adam groups
and moments, loss weights/progress, explicit D07 permutation/cursor and Python,
NumPy and Torch CPU RNG. Incompatible or malformed state is rejected; shared GAB
identity is retained. Checkpoint publication refuses overwrite.

The next real D05/D07 update matches exactly after a fresh-process restore with
0.1 dropout: all 215 gradients, parameter tensors and Adam states, next-batch
order, losses, RNG and evaluation predictions. Verification passes 60 checkpoint
cases, 1,344 full model tests, 1,023 adapter tests/nine expected Torch skips,
84 accepted reviewer probes per environment, M06 and the frozen adapter replay.

[API, evidence and limitations](docs/t02-checkpoints.md) and
[recorded verification](runs/t02/summary.json) describe the supported CPU boundary.
T02 is implemented but not independently review accepted; its checkbox remains
unchecked. Work stops after the isolated commit on `codex/t02-checkpoints`.
No T03, T04, overfit, pilot training, P or RL was begun; primary main is preserved.

## T02 AI review acceptance and integration — 9 October 2026

Exact T02 commit `27731e32d8aa64d852b439945f30e7238e87a824` is independently
AI review accepted on the user's supplied review: no blocking findings, 152
checkpoint/training tests and exact fresh-process continuation/hash reproduction.
No human review is claimed. Its original history and completion evidence are
retained; this dated acceptance supersedes the historical pending-review status.

Fresh isolated integration verification passes 1,344 model tests, 1,023 adapter
tests/nine expected Torch skips, 152 focused checkpoint/training tests, 84 accepted
review probes per environment, both frozen adapter replays, M06 and offline
dependency/upstream checks. CPU continuation exactly reproduces the original
checkpoint and report hashes. Accepted data, code and evidence are preserved,
including the unrelated uncommitted primary planning edits. New evidence is in
`runs/t02-integration`; fresh weights/reports use a distinct external directory.

[Integration, artifact location and limitations](docs/t02-integration.md) and
[structured results](runs/t02-integration/summary.json) document the supported
pinned CPU boundary. Work stops after T02 integration; no T03, T04, pilot or RL.


## T03 tiny fixed training fit — 9 October 2026; independent review pending

T03 was implemented and executed once in isolated `t03-tiny-overfit` on
`codex/t03-tiny-overfit`, from reviewed integrated main
`4fa48d4afef91a3bea125fb5f9f85b4d5159fc28`. Before execution, the frozen plan
recorded eight exact accepted D05 training references, source/tensor hashes,
seed 20261009, full M06 combined architecture with dropout zero, Adam 0.0003,
T01 1:1 losses, CPU settings and a 1,000-update cap. No held-out samples were
used. Gates were 100% legal best-move accuracy, mean teacher-WDL KL at most
0.01 nats and maximum KL at most 0.03 nats, with a separate numerical identity
tolerance. Neither thresholds nor membership changed after execution.

The run passed at its first qualifying update, step 26: 8/8 best moves, mean KL
0.009595421, maximum KL 0.021492420. WDL CE 0.364435166 is compared with the
teacher entropy floor 0.354839762; CE-minus-entropy agrees with KL within
1.76e-8 nats. D07 loading, accepted combined M06 construction, T01 losses/step
and T02 checkpoints are reused without changing their implementation.

All 27 curve rows, checkpoint identities and exact fresh-process inference/
training restore verification are saved. Weights remain outside Git in the
workspace `external-artifacts/t03-tiny-overfit-20261009/final.pt`; they are a
plumbing checkpoint, never immutable pretrained P. The runner retains a final
checkpoint and explicit failure status on cap failure; tests cover that path.
Checks pass: 299 focused tests, 1,364 full model tests, 1,023 adapter tests/10
expected Torch skips, 84 accepted review probes, M06 numerical and the frozen
13-fixture replay. Accepted inputs/evidence and primary main with its unrelated
uncommitted edits are preserved.

[Methods, exact settings, results and limits](docs/t03-tiny-overfit.md),
[frozen plan](runs/t03/frozen-plan.json), [summary](runs/t03/summary.json) and
[curves](runs/t03/learning-curves.svg) record the evidence. T03 automated checks
are complete; independent review is pending and its historical checkbox remains
unchecked. Stop after the isolated commit. No T04, pilot/pretraining, P selection,
model-guided games or RL was started.

## T04 real batched evaluator — 9 October 2026; independent review pending

`chess_rl.search.real_evaluator.RealEvaluator` loads through T02, partitions
terminal states before encoding/inference, batches nonterminal states through
the combined M06 model in evaluation/inference mode, uses T01 legal
normalization and restores input order. It returns legal priors, root-side WDL
and `v=P(win)-P(loss)` through the accepted adapter result types. Pass the
callable directly to the existing search's `evaluator` argument.

Verification passes 34 T04 checks (94 with T02 checkpoint tests), 1,378 full
model regressions and 1,023 adapter regressions with 10 expected optional-Torch
skips. Frozen adapter replay passes. A preserved real T02 fixture checkpoint
verifies terminal bypass, mixed batching, exact draws with legal moves, history
and RNG preservation, independently calculated priors/WDL/value and bounded
search compatibility. Largest observed prior batch error is `9.83e-7`, within
the pre-recorded `atol=2e-6, rtol=2e-5` tolerance.

[API, evidence and CPU limits](docs/t04-real-evaluator.md) and
[structured results](runs/t04/summary.json) record this isolated implementation
on `codex/t04-real-evaluator`. T04 remains pending independent review. Work
stops after committing T04; no R01 or performance campaign is begun.


## T03/T04 accepted integration — 9 October 2026

T03 and T04 are accepted within their documented CPU scope through the supplied
independent AI review and fresh combined integration verification. Both original
commits and completion records remain intact. The actual T03 checkpoint loads
through T04 with exact saved model hashes and final metrics, legal normalized
outputs and valid bounded search. Fresh model regressions pass 1,398 tests;
Torch-free regressions pass 1,023 with 11 expected optional-Torch skips.

See [integration and preservation evidence](docs/t03-t04-integration.md) and
[structured results](runs/t03-t04-integration/summary.json). The original T03
freeze and overfit artifacts remain unchanged; separate integration verification
performs no original fitting rerun. This is supplied AI review, without human or
newly commissioned review. Scope ends at primary main integration, before R01,
GPU work or training campaigns.


### R01 model-guided self-play — 10 October 2026; independent review pending

Only R01 is implemented in isolated `r01-self-play` on `codex/r01-self-play`,
from accepted combined T03/T04 main `ea0f8a142ec3b1ec3266ffafef8853ea07ab0c2d`.
The shared adapter, search, RealEvaluator and T02 loader remain unchanged.
Sequential CPU batch-of-one execution freezes checkpoint identity, budget,
exploration phases, opening source/selection seed/full prefix and owned PCG64
settings before search. Atomic move-boundary records retain complete history,
played-root legal masks/raw visits/search policies and execution metadata.
Terminal outcomes precede the cap; unfinished games never receive draw outcomes
or WDL labels. R02 finalization and training are outside this implementation.

All 55 new tests and 308 focused cases pass; full regressions pass 1,453 model
cases and 1,023 Torch-free cases with 12 expected optional-Torch skips. Both
environments pass 84 historical reviewer probes and identical frozen 13-fixture
adapter replays; M06 numerical checks pass. The unchanged actual T03 plumbing
checkpoint executes the pre-frozen 20-snapshot suite: eight completed (40%),
11 truncated (55%), one interrupted (5%), and one deliberate duplicate completed
excess (12.5% of completions). Among 15 snapshots with generated moves: four
completed, ten truncated, one interrupted and zero duplicate completions.
Short caps explain the high truncation rate; no scaling/strength inference follows.

A long-history fixture stops between moves, is SIGKILLed during the next search,
then resumes in a fresh process with scrambled global RNGs. Initial board/full
history, opening prefix, checkpoint/search/phase settings, all actions/root
visits/search policies and private RNG exactly match uninterrupted execution.
The original cap is preserved and the resumed truncation remains unlabeled.
Primary main, its unrelated exact planning patch, accepted tracked bytes and
76 indexed external files are preserved. No pretrained P is selected.

[API, durable boundary and limits](docs/r01-self-play.md),
[frozen real suite](runs/r01/real/frozen-plan.json),
[real results](runs/r01/real/summary.json) and
[checks/preservation index](runs/r01/summary.json) save the review evidence.
The R01 checkbox remains pending independent acceptance. Commit only this
isolated R01 work and stop; do not integrate main or start R02/R03/R04, RL
updates, GPU work, production training or playing-strength claims.


## R01 accepted CPU self-play integration — 10 October 2026

R01 `2e31236` is independently AI review accepted and freshly verified in an
isolated integration checkout. All 20 real T03-checkpoint snapshots reproduce
exactly; fresh-process crash/resume matches uninterrupted histories, actions,
visits, policies and RNG. Historical implementation evidence remains intact.
See [integration and acceptance](docs/r01-integration.md) and
[fresh evidence](runs/r01-integration/summary.json). This accepts sequential CPU
execution and durable move-boundary recovery; R02 and CUDA integration are pending.


## R02 replay implementation pending independent review — 10 October 2026

Completed-game finalization is implemented in the isolated `codex/r02-replay`
branch from accepted R01 main `48bce1e`. Shared R01, adapter and search modules
are unchanged. Full-history played roots retain masks/raw visits/search targets
and source identities with exact root-side WDL. Transactional admission uses
a recorded duplicate policy and stable sample IDs; unfinished records remain
excluded and unlabeled. All 44 R02 cases and 1,497 model regressions pass;
Torch-free regressions pass 1,023 with 13 expected skips. The preserved 20-source
R01 evidence yields four training roots and repeats exactly after restart.
See [R02 API and policy](docs/r02-replay.md) and [evidence](runs/r02/summary.json).
R02 awaits independent review. R03 has not begun.
