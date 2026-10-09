# T04 real batched evaluator — 9 October 2026; independent review pending

Only T04 is implemented in isolated `t04-real-evaluator` on
`codex/t04-real-evaluator`, based on primary main
`4fa48d4afef91a3bea125fb5f9f85b4d5159fc28`. T02's independent AI review acceptance and integration are recorded
in [T02 integration](t02-integration.md) and its main receipt. T03 is not a
prerequisite and is outside this change. T04's checkbox remains unchecked until
independent review acceptance; no human review is claimed.

## API and ownership

`chess_rl.search.real_evaluator.RealEvaluator(checkpoint_path,
expected_config=None)` calls only the shared T02 `load_inference`. The
`checkpoint` attribute retains the loaded model, configuration, checkpoint ID,
provenance and adapter contracts. Incompatible configurations and checkpoint
versions fail through T02's existing validation. No alternate loader, encoder,
rules implementation or policy normalizer is introduced.

The object exposes `evaluate(states: Sequence[ChessState]) ->
list[Evaluation | TerminalResult]` and is callable with the same signature.
Pass it directly to the accepted search's `evaluator` argument:

```python
from chess_rl.adapter import make_state
from chess_rl.search.puct import Node
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.search.runner import search

evaluator = RealEvaluator("path/to/t02-checkpoint.pt")
assert evaluator.evaluate([]) == []
result, = evaluator([make_state()])
search_result = search(Node(make_state()), 3, evaluator=evaluator)
```

Every state is adjudicated before any encoding or inference. Terminal results
pass through unchanged, including exact WDL/value, reason, winner and history
provenance. Empty and entirely terminal batches perform no encoding, legal
normalization, model mode change or forward call.

The accepted adapter owns legal masks and feature extraction for nonterminal
states. One stacked feature tensor and one forward call cover all nonterminal
inputs in their original relative order. The owned combined M06 model runs in
evaluation mode under `torch.inference_mode()`, with GAB and all relative Q/K/V
terms enabled. T01's shared `legal_policy_probs` masks before normalization and
returns exact zero illegal mass. WDL softmax preserves root-side
`[win, draw, loss]`; value is `P(win)-P(loss)`. Results are scattered into their
original positions and each adapter result owns its arrays.

T02 supports CPU float32/float64. Features follow the loaded model's dtype;
the established adapter result boundary remains float32. Value is computed
from that returned WDL. Finite logits, output widths/batch membership and
nonempty legal support are checked. Loading/evaluation leave caller RNG
unchanged. Serialize access to the owned evaluator; mutating its parameters or
architecture is outside the API. Callers bound input batch size; this change
does not implement a queue or concurrent search traversal. The optional-Torch
module is imported explicitly, so dummy evaluator and search imports remain
Torch-free.

## Correctness evidence

The checks are [test_real_evaluator.py](../tests/test_real_evaluator.py),
[check_t04.py](../scripts/check_t04.py) and the recorded commands under
[runs/t04](../runs/t04). Tolerances were fixed before the first execution:

| Check | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Batch versus individual public outputs | `2e-6` | `2e-5` |
| Independent float64 NumPy conversion oracle | `1e-6` | `1e-5` |
| Probability sums | `1e-6` | `0` |
| Terminal WDL/value, illegal mass and history preservation | Exact | Exact |

The existing T02 unit fixture is constructed through its shared save/load API
with a single bounded fixture update, dropout .27 and distinct head widths.
An additional initial float64 checkpoint checks dtype conversion. Neither is
an overfit run or a selected playing checkpoint.

The separate real-artifact audit reads the preserved T02 integration checkpoint
at `external-artifacts/t02-integration-cpu-continuation-20261009/step-1.pt`,
60,871,811 bytes, SHA-256
`9f3ecddc1237f56aac1eb69c5c5fd87d442c937b23d591adcfed5df50853fefc`.
It performs no training or checkpoint write. The 31-input mixed batch includes
16 nonterminal rows and executes exactly one `[16,64,128]` forward. Its maximum
absolute batch-versus-individual errors are:

| Output | Observed maximum |
|---|---:|
| Priors | `9.834766387939453e-7` |
| WDL | `8.940696716308594e-8` |
| Value | `1.1920928955078125e-7` |

Independent conversion errors are at most `2.049916711377442e-8` for priors,
`3.324357814027401e-8` for WDL and `2.251419586873915e-8` for value. The oracle
uses float64 NumPy exponentiation over independently enumerated legal moves,
without the production Torch normalizer. Known asymmetric WDL logits also
verify class ordering and value sign for both root players.

Coverage includes empty inputs, singleton inputs, repeated references,
per-result array ownership, mixed batches, both players, castling, en passant,
all promotion choices, long and incomplete retained histories, exact terminal
outcomes and terminal identity/bypass. Actual threefold repetition includes a
16-ply retained history; a reached fifty-move draw pushes a legal move from
clock 99 to 100. Both draws still have chess-legal moves. Intended-move claims
remain nonterminal. The preservation fingerprint includes board/stack object
identities, raw fields, every prefix, counters, promoted bits, moves, initial
FEN and completeness, before and after evaluation and bounded search.

Eight nonterminal roots receive exactly three search simulations each, spanning
both players and special moves. Visit totals, legal actions/targets and state
preservation are checked. All 12 terminal fixtures receive a bounded search
request and complete zero simulations without inference. These are API
compatibility checks; no games, R01 execution or performance campaign is run.

The initial focused run passed 93 checks and failed an array-ownership test
that incorrectly required bitwise batch-versus-singleton agreement. Its
observed `2.98e-8` difference was already within the recorded tolerance. The
test now compares ownership against exact array snapshots and compares
different batch sizes within the original tolerance. The failed command/output
is retained; the final focused result is recorded separately.

The pinned runtimes are macOS arm64 Python 3.12.12, NumPy 2.5.3,
python-chess 1.999/chess 1.11.2, pytest 9.1.1 and Torch 2.8.0 for model checks.
Single-thread numerical environment settings and exact source hashes accompany
the commands. Full model regressions use the accepted historical M05 diagnostic
plugin; production M06 inference remains combined. Full adapter regressions
and frozen adapter replay check the optional-Torch boundary. Final counts and
evidence hashes are in [summary.json](../runs/t04/summary.json).

Primary main, its unrelated uncommitted plan edits and the source checkpoint
are rehashed after each recorded command. Existing tracked code, documents and
historical evidence remain unchanged except dated README/plan appends. Stop
after committing T04 for independent review, before R01, pilot work, P
selection, self-play, RL or performance campaigns. Numerical agreement is
limited to this pinned CPU correctness boundary; no playing-strength claim is
made.
