# Experimental match batching: isolated development only

Production stays serial through round-2 validation and the test match. Only the
user submits acceptance, after that test match has finished (currently expected
Monday 19 October around 17:00 IST). This is a planning estimate, not a scheduler.
A later report-only head-to-head is the only proposed use; acceptance does not
enable production or self-play batching.

The standalone match development tools are integrated on top of Desktop main
5bb1e10. The shared evaluator, numerical acceptance helpers, source-pinning script
and evaluator tests were already integrated by the self-play batching change;
their newer versions are preserved. Production `match_campaign.py` and
`run_match.py` have no batching hooks. Their existing head-to-head edits and
round-2 training/depth-1 work are outside this integration.

The separate `batching-match-hooks.patch` remains deferred. Round-2 candidate,
selection and final-test binding must land before its owner rebases those hooks.
See `match-batching-owner-note.md`. Availability on local main does not enable
batching in production matches or authorize a cluster deployment.

## Execution and guarantees

The unchanged RealEvaluator still owns encoding, legality/masks, finite checks,
legal policy probabilities, WDL softmax, W-L and float32 ownership. The wrapper
pads actual nonterminal network inputs to B with valid initial positions and
discards padding results. Terminal-only requests avoid the model.

K game threads keep the unchanged synchronous search and one outstanding leaf
per game. One collector thread constructs and invokes models, separately by
verified checkpoint SHA-256. Each search player registers its coming checkpoint
before CPU tree work. The collector flushes a queue when it has B requests, all
registered games currently expecting that checkpoint have pending requests, or
the oldest request reaches its deadline. The oldest ready queue wins. The new
fallback is 20 ms, allowing more refill time than 2 ms; it is a hypothesis to
measure. Games unregister on completion, failure and cancellation. Deadlines
still release partial batches when a game is slow, between turns or playing a
non-network opponent. Counts for full/group-ready/deadline flushes are recorded.

New plans freeze all execution settings, including version `match-batching-v2`.
Version-1 execution records and the old deadline policy remain readable without
rewriting them. Changed source, runtime, checkpoints or execution still refuse
resume. Publication uses the original validator, schema and INSERT-only SQLite
store, with one connection per publishing thread. Pending requests fail closed
on model/delivery/timeout failure. Committed rows are never replayed/replaced.

Fixed padding is not proof of independence under arbitrary scheduling. Acceptance
requires byte-identical returned priors/WDL/values under slot, neighbour, padding,
repeat, arrival and flush changes, plus exact bounded-game records across changed
scheduling and hard kill/resume. Serial comparisons use the fixed absolute 1e-5
bound and move fixtures. Passing fixtures is evidence, not a universal guarantee.

## Mac checks and standalone commands

Use Python 3.12.12 and the committed dependencies. Set `PYTHONPATH=src:.`.
The standalone runner imports the original match schema, validators and store;
it does not require the proposed hooks to be applied.

```sh
python scripts/benchmark_matches.py accept --left /staged/P.pt --right /staged/R.pt \
  --device cpu --batch-size 4 --output /scratch/new-numerical
python scripts/benchmark_matches.py check-games --left /staged/P.pt --right /staged/R.pt \
  --device cpu --batch-size 4 --output /scratch/new-game-checks
python scripts/benchmark_matches.py benchmark --left /staged/P.pt --right /staged/R.pt \
  --device cpu --layouts 1x1x1 1x4x2 2x2x1 --pairs 2 --simulations 2 --ply-cap 12 \
  --output /scratch/new-throughput
python scripts/benchmark_matches.py run --directory /scratch/new-throughput/1x4x2
```

`PxKxB` selects processes, games per process and network batch size explicitly.
`PxK` remains supported with `--batch-size`. B stays fixed within each layout.
Exact game comparisons apply only between layouts with the same B; different B
values are different numerical configurations, not a scheduling-only change.
Every B in the acceptance job receives its own numerical and resume checks.

The short study defaults to eight pairs (16 games), 128 simulations and a 512-ply
cap. Its layouts are 5x1x1 serial, 2x8x4 and 1x16x8. The batched layouts use B=K/2
and can both fill their advertised capacity. Eight pairs cannot fill 5x8, 3x32 or
3x64. These remain explicit larger-study options, for example `--layouts 3x32x16
3x64x32 --pairs 96`, with a separately planned time budget. Underfilled schedules
are labelled in the report. Do not extrapolate their throughput as saturated
performance. Python's GIL, model split and deadline behaviour require measurement.

Each worker warms both models before a process barrier. Reports include startup
and total wall time, warmed span, moves/second, actual overlap, per-layout
occupancy, queue waits, flush reasons, RSS and CUDA peaks. Profiling separately
covers encoding, legality, GPU operations, synchronization, transfers and
postprocessing. Its instrumentation overhead is not throughput.

## User-submitted B200 acceptance after the test match

Stage a separate clean checkout and the pinned artifacts. Do not pull or change
the running production checkout. On the submission host, record the revision and
tracked file hashes; this is the only step that requires Git:

```sh
export PYTHONPATH="$PWD/src:$PWD"
python scripts/prepare_batching_source.py
mkdir -p artifacts/slurm
# Only after the round-2 test match, and submitted by the user:
sbatch jobs/b200-batching/acceptance.sbatch
```

The guarded scheduler receives one script path and no `--export` arguments. The
script hard-codes and verifies these checkpoints before any model subprocess:

- P: `artifacts/p-1m/step-090000.pt`, SHA-256
  `cf7442ee28969d7d93f5ab37a4829400f2c73efb319b6b08a1717d7adfe26dd2`.
- R1-mix: `runs/train/r1-mix/update-002303.pt`, SHA-256
  `32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2`.

The compute job uses Python to verify tracked files and the execution-source
inventory against
`artifacts/batching-source.json` and records its revision, with no Git call.
A source or weight mismatch stops it. It runs B=4 and B=8 numerical checks,
bounded scheduling/kill-resume checks, profiling, then the eight-pair layouts.
Outputs are only under `artifacts/batching-acceptance-$SLURM_JOB_ID`.

Runtime estimate: the historical 3.9-hour serial 64-pair match scales to roughly
29 minutes for eight pairs if game lengths and contention are comparable. There
is no measured time for the new batched paths. Allow roughly 60–110 minutes for
planning only; that range is unverified and slower batching can exceed it. The
job has a two-hour hard limit, not a promise to finish. If it times out, committed
rows survive and acceptance is incomplete. Resume a layout with the exact original
source/runtime/weights, but use a new uninterrupted study for comparable timing.

This job has not been submitted, and cluster scheduler acceptance is unverified.
Local shell tests exercise the script with a failing Git stub and mocked GPU
commands; they are not a cluster execution result.
