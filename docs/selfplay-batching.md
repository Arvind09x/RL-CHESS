# Optional cross-game self-play inference

This is isolated, uncommitted work based on `c5636c1` (R2 training). Do not copy it
into Desktop production before the R2 test match finishes. No match hooks,
candidate selection, validation compatibility, historical records, or production
jobs are changed. The user reviews and commits this work. No source manifest has
been prepared for the current dirty checkout.

## Execution and recovery

Serial remains the default. Existing campaign files remain readable and run
through the original process-pool path. Batching is selected only at **creation
of a new campaign**. There is no run-time switch to convert a serial campaign.
Explicit execution settings freeze workers, games per worker (K), fixed batch
size (B), flush wait, request timeout, padding and implementation versions.
Actual source files (including untracked modules) and checkpoint bytes are hashed.
The first batched run durably freezes its strict runtime; every worker and resume
must match it. Each game also binds its configuration to the complete campaign
plan digest. Moving/copying a campaign directory is supported; editing its plan,
source, checkpoint, runtime or execution layout to continue it is not.

Each CPU worker owns one collector/model and up to K game threads. A game submits
one leaf and sleeps until that request completes. No within-tree parallelism or
virtual loss is added. Separate processes supply CPU parallelism; Python game
threads share a GIL and are useful for overlapping inference waits, not for
multiplying CPU cores. K can exceed the process count. Active futures, registered
games and pending requests are bounded by K per worker. Requests carry a game
registration token, monotonic per-registration request ID and checkpoint hash.

The reused collector flushes at B, when all registered games expecting the
checkpoint are waiting, or after the oldest request's 20 ms default deadline.
Completion/unregistration wakes the collector; failures/timeouts/cancellation
release waiters and discard the collector. Workers use a new collector on retry.
The parent retains the campaign's bounded broken-process retry and hard deadline
handling. A stuck native call is contained by the request timeout and disposable
worker, with the parent/job deadline as the final bound.

Search, legal/terminal handling, simulation counts, temperature/root noise,
visit targets, WDL and replay filtering retain their existing owners. Sessions
still own independent PCG64 streams derived from campaign seed + game index;
worker assignment and arrival order never enter a seed. The immutable move journal
and atomic boundary pointer are unchanged. A hard kill loses only an uncommitted
search. An orphan move is checked against its deterministic retry before advancing
the pointer. Finished games are not rewritten. Replay consumes the same R01
configuration/game schema and archive JSONL; truncations and duplicates are still
filtered by the existing replay implementation.

## Numerical scope

B is a numerical configuration. Every nonterminal model call is padded with the
standard initial position to exactly B. Fixed shape alone does **not** prove
slot/neighbour/padding independence. Do not expect B1/B8/B16/B32 outputs or entire
game trajectories to be identical.

CUDA campaign creation requires a successful numerical acceptance JSON tied to
checkpoint SHA, source hash, B and strict runtime. The supplied benchmark runs
slot/neighbour/padding/repeat byte comparisons and altered-arrival/flush checks,
plus an absolute 1e-5 comparison of policy/WDL/value against B1. Serial selected
move differences are reported diagnostically; they are not hidden or rounded
away. Each B also gets a small same-configuration interrupted/resumed Session
comparison and archive audit before timed benchmarks. Exact recovery remains
conditional on this tested fixed-B numerical behavior; this is evidence on a
finite corpus, not a mathematical guarantee for all states or future CUDA kernels.
A different layout is a new execution configuration even if B is unchanged.

CPU creation without an acceptance artifact is allowed for development and tiny
fault tests. This does not authorize CUDA use or establish acceptance for arbitrary
large CPU checkpoints. The checked-in tests use tiny models, B2/B4, short games and
small simulation budgets. CUDA and B8/B16/B32 real-checkpoint evidence is pending.

## Later B200 benchmark (user submits after the R2 test match)

Use the **isolated checkout**, never Desktop production. After reviewing and
committing all desired files, stage that committed checkout and the pinned runtime
and R1-mix checkpoint on the cluster through the existing workflow. Prepare the
source manifest on the submission host, with a clean checkout, before submission:

```bash
cd /path/to/isolated/RL-Chess
PYTHONPATH=src:. .venv-cuda-cu128/bin/python scripts/prepare_batching_source.py \
  --output artifacts/batching-source.json
mkdir -p artifacts/slurm
sbatch jobs/b200-selfplay-batching/benchmark.sbatch
```

The manifest command deliberately refuses uncommitted/untracked work. It is not
run as part of this delivery. The compute job verifies hashes and execution-source
inventory without Git. Remove/move an obsolete manifest deliberately before
preparing another; the command never overwrites one.

The job follows `jobs/b200/common.sh`: student/gpu-small, six allocated CPUs,
48 GiB host memory, injected GPU/MIG, two-hour Slurm limit and a 110-minute GNU
`timeout` bound (30-second kill grace). It submits no child jobs. It pins R1-mix:

- Path: `runs/train/r1-mix/update-002303.pt`
- SHA-256: `32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2`

The four layouts use five CPU worker processes on the same allocation:
serial B1/K1, B8/K8, B16/K16, B32/K32. K is per worker, so total active capacity is
5, 40, 80 and 160 games/searches. One frozen corpus of 160 independent positions
has complete histories at 0/20/60/120/180 plies; all layouts use the same seeds,
64 simulations, search constants and self-play phase settings. Two repeats rotate
layout order. Workers warm up, synchronize at a sleeping barrier, then start the
timed region. Startup-inclusive wall time is reported separately. Occupancy will
fall during startup/tails or when configured K < B; no saturation claim is made.

For different active counts, run the driver directly inside an appropriate user
allocation (or edit/review/commit the job before preparing its source manifest):

```bash
PYTHONPATH=src:. .venv-cuda-cu128/bin/python scripts/benchmark_selfplay_batching.py \
  --checkpoint runs/train/r1-mix/update-002303.pt \
  --checkpoint-sha256 32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2 \
  --source-record artifacts/batching-source.json --device cuda:0 \
  --output artifacts/selfplay-batching-custom --mode search \
  --batch-sizes 8 16 32 --active-games-per-worker 16 32 64 \
  --positions 320 --simulations 64 --repeats 2
```

Use `--mode accept` for numerical and small resume checks alone. Use `--mode games
--games 160` in a separately planned allocation for actual campaigns from the
standard start with the existing 512-ply cap. This can take much longer; it is not
the default submitted benchmark. Only completed game-run comparisons report
finished games/hour, actually terminal games/hour and usable training
positions/hour after existing archive/replay verification and duplicate/truncation
filtering. Unfinished runs withhold those rates. Resume incomplete campaigns with
the regular campaign runner; their mixed-run wall times are not a clean speedup
measurement. A fresh comparison needs a fresh output directory.

Every result records source/checkpoint/execution settings, actual PyTorch runtime,
GPU/MIG name/UUID/visible-device identity, driver information and CPU allocation.
Search reports aggregate simulations/second, positions/second, useful occupancy,
request/queue/flush counts, inference time, wall time, per-process RSS and CUDA
allocated/reserved peaks, and failures. Per-process peaks are not simultaneous job
peaks; the sbatch log also records the job-step cgroup memory peak. Hard-killed worker groups can lose diagnostic
counters; their durable move/game files remain authoritative. Results are
published after each layout; `complete.json` is written only after success.
A timeout may leave only earlier reports. Treat absence of `complete.json` as an
incomplete comparison. No output path is silently reused or overwritten.

Both historical files `benchmark-576.json` and `benchmark-3g-577.json` report
**NVIDIA B200 MIG 1g.23gb**. The latter's filename is not evidence of a 3g slice.
Do not extrapolate full-B200 specifications or search/evaluator results into
measured whole-game speedups.

## Acceptance criteria

1. All fixed-B slot/neighbour/padding/repeat and changed-arrival checks are exact
   for each proposed checkpoint/runtime/B. B1 comparison stays within absolute
   1e-5 for legal policy, WDL and value. Nonfinite/illegal outputs fail through the
   shared evaluator. Failures block that B; do not loosen tolerance to obtain a pass.
2. Each B's same-configuration interrupted/resumed probe has identical durable
   positions, visits, RNG, moves and outcome (apart from interruption diagnostics),
   and both archives pass the existing audit. No cross-B trajectory requirement.
3. All requested layout/repeat reports and `complete.json` exist; no worker,
   evaluator, publication, search, timeout or allocation failures. Check actual
   MIG identity, allocation, source/checkpoint/corpus/settings equality and memory
   headroom before comparing rates. Examine tail occupancy and flush statistics.
4. A performance recommendation requires a repeatable aggregate search-throughput
   improvement over the measured five-worker B1 baseline under the same resources,
   without exceeding memory limits. The script does not invent a speedup threshold
   or automatically enable batching. Search improvement alone does not establish
   games/hour or usable-training-positions/hour improvement; those require the
   optional game runs and replay counts.

## Creating a later batched campaign

Inside the same accepted runtime/source, use the chosen B's acceptance JSON:

```bash
PYTHONPATH=src:. .venv-cuda-cu128/bin/python scripts/run_selfplay_campaign.py create \
  --directory runs/selfplay/new-batched-campaign \
  --checkpoint runs/train/r1-mix/update-002303.pt --device cuda:0 \
  --games 160 --simulations 64 --seed 20261018 --purpose 'new opt-in batched campaign' \
  --execution batched --workers 5 --games-per-worker 16 --batch-size 16 \
  --acceptance artifacts/selfplay-batching-JOBID/acceptance-b16.json
PYTHONPATH=src:. .venv-cuda-cu128/bin/python scripts/run_selfplay_campaign.py run \
  --directory runs/selfplay/new-batched-campaign --workers 5
PYTHONPATH=src:. .venv-cuda-cu128/bin/python scripts/run_selfplay_campaign.py archive \
  --directory runs/selfplay/new-batched-campaign
```

A GPU acceptance file is necessary but finite evidence, not a substitute for
reviewing the resume reports and the above acceptance criteria. The benchmark
never changes production configuration or enables batching automatically.
