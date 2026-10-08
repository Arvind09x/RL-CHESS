# D06 separate annotation benchmark — 8 October 2026

D06 automated completion is ready for independent review. Exact accepted base:
`4184387c7e4b67559355d6b212a2a300c86c5ce6`. Isolated local clone:
`/private/tmp/rl-chess-d06`, branch `codex/d06-annotation-benchmark`.
Primary main remains clean and unchanged. D05/S05/Gate S acceptance is retained;
no D06 reviewer acceptance or main integration is claimed. No D07, D08,
training, search integration or GPU experiments ran.

## Fixed membership and teacher

Before the first annotation, `selected.jsonl` and fsynced `settings.json` froze
1,000 distinct audited **training references and rule positions**, from 999
canonical D03 games. Selection scans the resolved D04 audited sample table,
removes the 12 exact D05 plumbing references (both accepted bundles have the
same membership), and ranks distinct rule positions by SHA-256 of compact,
sorted-key ASCII JSON containing `selection_version=d06-rule-sha256-v1`,
`seed=20261010`, and `rule_position`. Select the 1,000 lowest ranks. Choose the
smallest `(game_id,sample_ply)` eligible audited reference per retained identity;
annotate in rank order. This is uniform hash selection over distinct rule
identities, rather than reference-weighted sampling. It changes the phase mix
relative to the D04 training-reference population. No labels/results select
membership, and no failures cause replacement. Every root is freshly joined,
replayed and checked nonterminal before search. Every selected game/position
is checked against the protected held-out registries.

| Phase | White | Black | Total |
|---|---:|---:|---:|
| Opening | 50 | 66 | 116 |
| Middlegame | 377 | 361 | 738 |
| Endgame | 73 | 73 | 146 |
| Total | 500 | 500 | 1,000 |

Membership froze at **2026-10-08 14:56:59.226147 UTC** (20:26:59 IST),
before annotation started at **14:56:59.238097 UTC**. Membership SHA-256:
`a7e71037dd8302f52d399d69c50e1d7cd23065564675a76b7ff99cae7c238b81`.
Full membership and labels remain external in
`/private/tmp/rl-chess-d06-benchmark/`. This bundle is exclusively D06 evidence;
it does not contribute to D08 production or freeze any expansion size.

The accepted worker and dependency/teacher settings are byte-preserved. It uses
the pinned `/private/tmp/Stockfish-sf_17.1/src/stockfish` executable, SHA-256
`3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`, with both
accepted embedded NNUE assets verified at their recorded build offsets.
Settings remain 100,000 requested nodes, Threads=1, Hash=128 MB, MultiPV=1,
UCI_ShowWDL=true, full strength, standard chess, empty SyzygyPath and all other
pinned defaults. Each attempt uses a fresh process, executable byte recheck,
full-history position command and completed Clear Hash/readiness before go.
The worker enforces completed individual principal responses and separate legal
bestmove agreement, complete legal PVs, move-ID round trips and root-perspective
integer WDL counts summing to 1,000. No stale response fields, centipawn-derived
WDL or source-result labels are used.

The frozen policy allows **two attempts maximum** per reference, with 30-second
deadlines for handshake/configuration operations and search completion as in
D05. Exhaustion is an explicit exclusion; interruption is recorded and
propagated with pending references. Failed/interrupted attempts never count as
labels. No automatic refill, resume, concurrent engine workers or result cache
is introduced. Partial bundles cannot be used as completed datasets.

## Actual measurements

Final counts: **1,000 selected / 1,000 attempts / 1,000 successful labels**;
**zero retries, failed attempts, interruptions, exclusions or pending roots**.
Invocation: 14:56:42.194768–15:04:11.941410 UTC
(20:26:42–20:34:11 IST). Annotation itself took **432.023s**;
setup/selection/replay gates took **17.031s** and post-input audit
**0.677s**. Complete invocation took
**449.745s**. Small settings/file/summary overhead explains
the residual between these separately bounded intervals.

| Rate / latency | Measured result |
|---|---:|
| Annotation-loop successful labels/s, including replays/checks/processes/fsync/retries | 2.314690 |
| Invocation labels/s, including setup and post-input audit | 2.223481 |
| Invocation plus successful independent-audit labels/s | 2.108095 |
| Engine-reported search-only labels/s | 13.043422 |
| Observed go-to-bestmove completed searches/s | 12.757989 |
| Aggregate returned nodes / aggregate engine-reported seconds | 1,278,052.7 |
| Position mean / median | 431.953 / 427.340 ms |
| Position p90 / p95 / p99 | 474.792 / 496.882 / 540.623 ms |
| Position min / max | 355.580 / 755.550 ms |

Nearest-rank percentiles are used. Observational `perf_counter` transport timing
adds no UCI commands and changes no worker behavior. Mean attempt components:
executable rehash **32.273 ms**; process creation/handshake to uciok excluding
rehash **296.257 ms**; explicit Clear Hash to readyok **12.969 ms**;
readyok to go, including python-chess's additional ucinewgame/readiness,
**2.846 ms**; go to bestmove **78.382 ms**; bestmove to attempt return,
including shutdown, **5.264 ms**. The rehash-inclusive engine attempt averages
**428.627 ms**. Worker replay/validation/persistence residual totals
**3.326s**; its boundaries
are not an isolated replay microbenchmark. Raw event timestamps, per-position
latencies and distributions for every component remain available.

Aggregate engine-reported search time is **76.667s**, for **97,984,467 returned
nodes**. Mean engine-reported time is 76.667 ms; returned NPS mean is 1,607,061.977,
which differs from the aggregate rate because per-row averaging weights searches
differently. All requested limits remain 100,000 nodes. **34 searches** completed
naturally below the node request, all with mate scores; **966** returned at least
the request. Returned nodes range 979–100,186, median 100,057; positive overshoot
is retained. There are 42 mate scores, depth range 11–245, seldepth range 2–54,
hashfull range 0–8 and zero tablebase hits. Full parsed rows, principal fields,
raw UCI, PVs, score/mate and WDL are retained, with summary distributions in
[results.json](../runs/d06/results.json).

The independent auditor found **498 raw score-bound flags** (300 upperbound,
198 lowerbound), versus zero surfaced parsed flags. Stockfish can place a bound
after WDL; the pinned python-chess parser recognizes bounds immediately after
score. The original transport rows preserve every flag. The auditor records
raw bounds independently rather than inferring equality from missing parsed
fields. The accepted worker remains unchanged. This parser limitation must be
considered by future consumers interpreting bounded scores; targets here are
legal best-move IDs and raw/soft WDL, not converted centipawn values.

## Independent replay and encoding costs

`scripts/audit_d06.py` imports neither the worker nor benchmark selector. It
scans **every eligible audited training row** and proves no unselected identity
ranks below the selection boundary, checks the smallest eligible representative
and D05 exclusion, and independently checks protected-game/position membership.
All 1,000 successful labels are joined and freshly replayed through both the
shared adapter and a direct python-chess board. Canonical IDs/assignments,
full-prefix checksums, FEN/counters/history/snapshots/phase, legal best moves,
independently spelled reflected move IDs, WDL perspective/probabilities,
complete PVs, actual final rows, default/effective options and provenance agree.
It also audits attempt numbering/success/exclusion accounting, completed resets,
full-history UCI position commands, go nodes, normal exits and raw score bounds.
This is independent audit code, not independent milestone review or a second
teacher search.

Successful audit total: **24.617s**. Corpus membership,
hash verification and canonical-game join cost **18.705s**.
Measured in-memory per-label component totals/means: adapter full-prefix replay
**2.204344s / 2.204344 ms**, encoding
**0.251791s / 0.251791 ms**, legal mask
**0.495719s / 0.495719 ms**. Complete
independent label checks total **4.280553s** and include
those components. Feature/legal-mask fingerprints and timing for every label
are external in `independent-replay.json`. Game loading/JSON parsing is a fixed
setup component here, not concealed inside replay timing. The audit's final
log/summary/check overhead explains the remaining elapsed time.

## Conditional scale projections

These are **linear projections**, not measured large runs or an allocation.
They assume the same host, sequential fresh-process worker, settings,
comparable position/history/phase mix, observed successful yield/retry rate,
mean cost and uncompressed storage. No parallel speedup is assumed. Zero
observed failures does not establish zero future failures. Stockfish search
hours are a subset of annotation hours. Complete audit hours include the
replay/encoding/mask components, which must not be added twice.

| Successful labels | Annotation worker h | Engine search h | Additional replay h | Additional encoding h | Complete label audit h | Compact labels GB | Full evidence GB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.200 | 0.213 | 0.0061 | 0.0007 | 0.0119 | 0.025 | 0.311 |
| 100,000 | 12.001 | 2.130 | 0.0612 | 0.0070 | 0.1189 | 0.255 | 3.107 |
| 300,000 | 36.002 | 6.389 | 0.1837 | 0.0210 | 0.3567 | 0.764 | 9.322 |
| 500,000 | 60.003 | 10.648 | 0.3062 | 0.0350 | 0.5945 | 1.274 | 15.537 |
| 1,000,000 | 120.006 | 21.296 | 0.6123 | 0.0699 | 1.1890 | 2.548 | 31.073 |

Add measured annotation setup/post-input verification **17.709s** and independent
audit setup/join **18.705s** separately per analogous corpus run. These fixed
costs may change with input size/new-corpus preparation. GB uses decimal bytes.
Actual label storage is **2,547,897 bytes** (2,547.897 bytes/success); full retained
per-position evidence is **31,073,332 bytes** for this run (31,073.332/success),
including membership, attempts, latency, raw transport, exclusions and full
independent replay report. Settings/summary add 8,738 fixed bytes. Exact artifact
bytes/SHA-256s are indexed in `results.json` and `artifact-index.json`.
Canonical games (638,770,177 bytes) and D04 sample table (567,767,218 bytes) are
reused once; no histories, dense policy vectors, tensors or masks are serialized
inside labels. No optional cache or D07 loader was implemented. New corpus
preparation, training, learning quality and GPU costs are outside these costs.
No monetary price or measured CPU utilization is inferred from wall time.

**No final expansion size is selected.** Decisions for 100k/several hundred
thousand/optional million labels require subsequent held-out learning curves,
training throughput and the reserved evaluation/resource budget.

## Verification, evidence and reproduction

Actual timestamped commands, cwd/argv, environment overrides, exits, lossless
stdout/stderr identities and XML are in `runs/d06/`, indexed by `commands.json`.
Final regressions: **992 model passes**, **824 adapter passes/six expected optional
Torch skips**, **84 prior reviewer cases per environment**, both identical
accepted **13-fixture adapter reports**, offline pinned-source checks and both
installed dependency checks. No dependencies were synced or replaced.
Preservation verifies every accepted tracked file against the exact base, with
only append-only README/coding-plan status additions; primary main, historical
runs, original manifests, source/dependency pins and protected registries match.
Archive/D02 original intake/D03/D04 authoritative outputs and teacher bytes
match their saved identities. The candidate SQLite cache is not an authoritative
annotation input and was never opened or changed by D06.

Retained development failures: the first independent audit demanded parsed
bounds equal raw flags and failed (18.358s); the corrected complete audit passed.
The first combined reviewer commands hit pytest module-name collisions and
passed with `--import-mode=importlib`. The first uv dependency checks hit a
sandbox cache permission error; the initial hardware sysctl query was also
blocked. Read-only reruns with approved access passed.
All original failure logs/commands remain. These are audit/check failures,
not failed Stockfish attempts. Original setup/clone commands are documented in
`setup.json`; precise setup-command times were not captured before the command
recorder existed. The benchmark and all checks have exact UTC timing records.

Reproduce from this checkout with the existing pinned environment and a **new**
output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/benchmark_d06.py --manifest data/manifests/elite-2025-01-d04-v1.json --games /private/tmp/rl-chess-d03-prepared/games.jsonl --split-audit /private/tmp/rl-chess-d03-prepared/split-audit.json --prepared /private/tmp/rl-chess-d04-prepared --executable /private/tmp/Stockfish-sf_17.1/src/stockfish --output-dir /private/tmp/rl-chess-d06-new-benchmark --timeout 30
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/audit_d06.py /private/tmp/rl-chess-d06-new-benchmark
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python scripts/report_d06.py /private/tmp/rl-chess-d06-new-benchmark --report /private/tmp/rl-chess-d06-new-results.json
```

Apple M3, 16 GiB RAM, eight physical/eight logical CPUs; teacher Threads=1.
CPU macOS 14.5 arm64/Python 3.12.12 only; one 1,000-position run on an interactive
host. No hardware isolation, power/thermal accounting, sustained scale,
concurrency, cross-platform determinism, crash recovery or playing-strength
claims. Replay/encoding measurements use warm files and games already in
memory, and do not establish loader/batch/cache/training throughput. Stop at
the D06 commit for independent review.
