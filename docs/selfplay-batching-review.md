# Self-play batching review handoff

Isolated checkout:
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/selfplay-batching`

Base commit: `c5636c19400dd03a97f64a00f612c0a330487ca9` (R2 training).
All implementation changes are **uncommitted**. No commits, pushes, merges,
rebases, Slurm submissions or GPU workloads were performed. This is a standalone
local clone; its origin is the Desktop checkout. The production checkout remains
at the same commit, with only its original modifications to `run_match.py`,
`match_campaign.py`, `test_match_campaign.py` and the original untracked depth-1
script/test. Those files were not copied into this scope or edited.

The implementation reuses the delivered `EvaluationCollector`, fixed-shape
padding, evaluator tests and numerical acceptance helpers. It adds a self-play
execution identity, bounded game threads within each worker, checkpoint/source/
runtime resume checks, and evaluator injection into the existing journaled
Session. Search and replay algorithms are unchanged. Existing serial creation
and execution remain the default; batched execution requires a new campaign.
No match hook patch was applied anywhere.

The benchmark includes the five-worker serial baseline, B8/B16/B32, configurable
active games per worker, full-history opening/middle/late searches, actual
hardware/runtime/allocation identity, occupancy/queue/flush/memory/failure reports,
small same-configuration resume checks, and optional measured whole-game runs.
The bounded two-hour job is `jobs/b200-selfplay-batching/benchmark.sbatch`.
Commands and acceptance criteria are in [selfplay-batching.md](selfplay-batching.md).

## Exact CPU verification

Commands below ran in the isolated checkout using the existing Python interpreter,
with imports forced to the isolated source. No packages were installed in
production. For each pytest command the environment was:

```bash
export PYTHONPATH=src:.
export PYTHONDONTWRITEBYTECODE=1
PY=/Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python
```

Initial collector and self-play integration check:

```bash
"$PY" -m pytest -q tests/test_selfplay_batching.py tests/test_batched_evaluator.py
```

Result: **26 passed in 42.83 seconds**.

Affected tests after adding the benchmark and campaign configuration guard:

```bash
"$PY" -m pytest -q \
  tests/test_selfplay_benchmark.py tests/test_selfplay_batching.py \
  tests/test_batched_evaluator.py tests/test_selfplay.py \
  tests/test_selfplay_campaign.py tests/test_real_evaluator.py \
  tests/test_evaluator.py tests/test_search_runner.py tests/test_replay.py
```

Result at that intermediate revision: **310 passed, 13 failed in 96.82 seconds**.
The new campaign opening comparison incorrectly compared JSON lists with Python
tuples. All 13 failures were in self-play campaign/integration cases. The guard
was corrected to compare the existing canonical digest. No numerical tolerance
or expected game result was relaxed. Log:
`artifacts/selfplay-batching-review/focused-tests-before-fix.txt`.

The changed/failed files were rerun:

```bash
"$PY" -m pytest -q tests/test_selfplay_benchmark.py \
  tests/test_selfplay_batching.py tests/test_selfplay_campaign.py
```

Result: **27 passed in 71.83 seconds**. After adding two targeted checkpoint-race
checks and final provenance/underfill corrections, the same command passed
**29 tests in 72.01 seconds**. Logs:
`artifacts/selfplay-batching-review/focused-tests-after-fix.txt` and
`artifacts/selfplay-batching-review/final-focused-tests.txt`.

Existing move-journal regressions, selected without unrelated training tests:

```bash
"$PY" -m pytest -q \
  tests/test_readiness.py::test_selfplay_does_not_replay_owned_root_history_per_move \
  tests/test_readiness.py::test_move_journal_crash_before_pointer_is_recoverable \
  tests/test_readiness.py::test_sigkill_after_durable_move_before_pointer_matches_control
```

Result: **3 passed in 6.27 seconds**. Log:
`artifacts/selfplay-batching-review/journal-tests.txt`.

Final benchmark-only changes recheck the explicit checkpoint pin between layouts
and after serial model loading, plus a CLI rejection test. Only the affected
benchmark tests were rerun:

```bash
"$PY" -m pytest -q tests/test_selfplay_benchmark.py
```

Result: **7 passed in 8.34 seconds**. Log:
`artifacts/selfplay-batching-review/final-benchmark-tests.txt`.

**329 distinct test cases have successful focused results across these runs.**
This is an aggregate, not a claim of a single clean 329-test invocation. The
296 unaffected passing cases from the broad focused run were not unnecessarily
rerun after campaign/benchmark-only corrections. The full repository suite,
unrelated training/match suites, large-checkpoint acceptance matrices and Mac
performance campaigns were not run.

Additional checks:

```bash
bash -n jobs/b200-selfplay-batching/benchmark.sbatch
git diff --check
```

Both passed. Python `ast.parse` syntax checks passed on all 14 new/changed Python
files (three modified application files, seven new implementation/utility files,
and four test/helper files). The offline job test executed the shell script with
fake Python/timeout binaries and failing Git/sbatch sentinels; no Git invocation
or submission occurred. The source-manifest test used synthetic fixture records;
it did not commit files or prepare a real manifest for this dirty checkout.

## Limits and pending checks

- No CUDA result or real-checkpoint B8/B16/B32 speedup is established. GPU numerical
  acceptance, strict same-runtime resume, memory headroom, actual MIG identity
  and five-worker throughput remain for the user after the R2 test match.
- Padding is not a proof of numerical independence. Exact claims are limited to
  a frozen configuration and the exercised corpus. Cross-B trajectories may differ.
- K game threads share a worker's GIL. Larger K may increase copying, encoding,
  memory and scheduling costs; occupancy and throughput must both be measured.
- Per-worker game-end and collector summaries arrive when that worker's assigned
  group returns. Hard-killed groups can lose diagnostic counters, while durable
  game boundaries and completed records remain authoritative. Do not use partial
  campaign event-rate projections as a controlled benchmark.
- Search reports do not claim games/hour or usable training positions/hour. Those
  rates appear only for actual complete campaign comparisons after replay filtering.
- Source-manifest preparation intentionally requires the user to commit/review
  first. There is no real committed-source manifest for the current implementation.
