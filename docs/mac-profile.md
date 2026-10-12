# Early Mac profiling

Measured production revision `042e9746e2b9948449b2ae69da708b6757fcad70` on 11 October 2026 UTC / 11–12 October IST. Stage 1 acceptance was not repeated. Only `scripts/profile_mac.py` and this report/measurement file were added; production chess, teacher, loader, checkpoint and recovery code were unchanged. No P was selected and no learning campaign was run. At publication, checkout HEAD was `dd7540b73c75c0eb57635fbf4e8e7c5eedda4719`: an intervening artifact-restoration documentation/helper commit. Production `src/`, tests and dependency definitions remain identical to the measured revision.

Exact settings and compact results: [mac-profile-measurements.json](mac-profile-measurements.json). Raw jobs, logs, process samples, move journals, profiles and disposable outputs: ignored `artifacts/mac-profile-20261011-a/`. Reproduction runs `.venv/bin/python scripts/profile_mac.py MODE --output <fresh-artifacts-directory>` separately for MODE=`prepare`, `smoke`, `campaign`, then `supplement`, in that order. Input paths in the harness identify the local accepted artifacts and must exist. The original campaign plan is retained; supplemental settings and cache correction are recorded separately.

## Machine and measurement scope

Apple M3 Mac15,12, **4 performance cores plus 4 efficiency cores**, 8 physical/logical cores, 16 GiB unified RAM; macOS 14.5 (23F79), arm64. Python 3.12.12, Torch 2.8.0 (Accelerate BLAS/OpenMP), NumPy 2.5.3, python-chess 1.999/chess 1.11.2. CPU float32, highest matmul precision, strict deterministic algorithms, one interop thread. MPS/CUDA were not used. Selected thread environment and Torch build details are in the measurements.

This was an interactive Mac with background load and swap (~6.8 GiB used initially). The recorded mid-campaign power sample reported AC Power, battery 68% and discharging; low-power mode was off. Later power snapshots are retained in raw groups. Stable unlimited power cannot be inferred from that sample. `memory_pressure -Q` initially reported 37% system-wide free; this is not an exact available-RAM count. Initial VM pages corresponded to about 150 MiB free plus speculative, 2.74 GiB inactive, and 7.09 GiB occupied by the compressor; these categories are not additive usable-memory guarantees. Raw VM/swap/pressure snapshots are retained before/after each group. One intensive configuration ran at a time, with warm-up, fixed seeds and repeated windows. Timed process-tree RSS was sampled every 0.2 seconds; supplemental groups also sampled setup every 0.5 seconds. Original label/self-play setup peaks were not sampled. RSS excludes the small orchestrator and compressed/swapped memory; sums may count shared pages repeatedly and miss transient peaks. The guard was 5 GiB. Primary training peaked at 1.54 GiB RSS and inference at 473 MiB, including sampled setup; all eight-game waves stayed below the guard. `ps %cpu` is a rolling estimate; 100% means one logical CPU. Profilers ran separately from throughput windows. The original timed-wave bound was 1,200 seconds; remaining 32-simulation waves used 2,400 seconds, with a 600-second setup bound. One interrupted 116-ply wave was excluded and rerun; its raw journal is preserved. A read-only record verification overlapped about 20 seconds (estimate) of the 16-simulation eight-worker second wave, so that window includes its extra host load.

Current storage: uncompressed JSONL canonical games/references/labels/attempts, immutable per-move JSON journals with dense masks/visits/policy, and `.pt` checkpoints. No packed storage was adopted. Repeat affected loader/publication measurements if storage changes.

## Fresh-process Stockfish labels

Stockfish 17.1 accepted executable SHA-256 `3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`, verified embedded NNUE and accepted teacher `stockfish-17.1-100kn-v1`: 100,000 nodes, Threads=1 per independent worker, Hash=128 MiB, MultiPV=1, UCI_ShowWDL=true, full strength, standard chess, empty Syzygy, full replay history, Clear Hash each annotation. Timeout 30 seconds, at most two attempts. Output fsync and executable identity verification remain in the production worker path.

A frozen training-only sample of 64 positions (14 opening, 42 middlegame, 8 endgame; 32 per side) was hash-ranked within phase/colour, seed 20261011. Every worker configuration labelled the same sample in three shuffled windows; one separate fresh-process warm-up per worker/window. Setup is excluded; group wall time includes completion polling/teardown. Labels/second is successful labels divided by total measured group wall time.

| Workers | Labels/s | Window range | Latency p50/p95/p99 (ms) | Failures | Peak RSS (MiB) | CPU mean (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.09 | 2.03–2.13 | 461/551/818 | 0 | 421 | 44 |
| 2 | 4.13 | 4.02–4.24 | 468/580/674 | 0 | 836 | 86 |
| 4 | 6.92 | 6.74–7.10 | 563/663/771 | 0 | 1683 | 197 |
| 8 | 7.96 | 7.27–8.50 | 944/1203/1394 | 0 | 3269 | 435 |

768 timed successful labels; 64 unique positions and 704 repeated field comparisons. All persisted fields matched across scheduling/worker count except the explicitly excluded timing telemetry `returned_search.time` and `.nps`. Accepted labels were not overwritten; their SHA-256 was checked afterward. The persistent worker was not tested or approved: its planned 500-position fresh-process label-field equivalence check remains required before production adoption.

## Neural training and inference

Real M06 architecture `m06-gab-relative-qkv-v1`, relative Q/K/V enabled, policy/value widths 128, dropout 0.1. Disposable freshly seeded training models used the production D08 train loader (9,000 positions), supervised loss/update, Adam lr=1e-4, foreach=false/fused=false, and production DataOrder cursor validation. Each thread/batch combination reset model, optimizer and seed, used three warm-up updates and three windows of six timed updates. Timing includes batch decoding, supervised update and cursor advancement; separate gradient-norm diagnostic computation is excluded. Production non-finite guards remain enabled; a failure would abort the run.

D08 opening/integrity validation: **38.49 s**; freezing: **24.66 s**. These are separate from steady-state updates. The first 64 MiB decoded-cache matrix mixed cache misses in early configurations with hits in later ones; its raw results remain exploratory. The primary matrix below uses the same production frozen loader with decoded cache disabled (`cache_bytes=0`) so thread comparisons see the same decoding work. OS file caches are still warm.

| Torch threads | Batch | Steps/s | Samples/s | Window samples/s range | Gradient norm min–max | Non-finite |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 6.47 | 6.47 | 6.40–6.55 | 10.6–18.2 | 0 |
| 1 | 4 | 4.94 | 19.74 | 19.62–19.82 | 5.6–67 | 0 |
| 1 | 16 | 2.86 | 45.79 | 45.47–46.03 | 3.49–12.4 | 0 |
| 1 | 32 | 1.84 | 58.87 | 58.69–59.08 | 2.36–6.55 | 0 |
| 2 | 1 | 6.98 | 6.98 | 6.78–7.11 | 10.6–18.2 | 0 |
| 2 | 4 | 5.40 | 21.59 | 21.32–21.83 | 5.6–67 | 0 |
| 2 | 16 | 3.18 | 50.82 | 50.46–51.07 | 3.49–12.4 | 0 |
| 2 | 32 | 2.05 | 65.62 | 62.20–67.93 | 2.36–6.55 | 0 |
| 4 | 1 | 6.39 | 6.39 | 6.04–6.61 | 10.6–18.2 | 0 |
| 4 | 4 | 5.20 | 20.82 | 20.71–20.92 | 5.6–67 | 0 |
| 4 | 16 | 3.27 | 52.35 | 52.10–52.74 | 3.49–12.4 | 0 |
| 4 | 32 | 2.12 | 67.94 | 66.44–68.85 | 2.36–6.55 | 0 |
| 8 | 1 | 6.34 | 6.34 | 6.23–6.43 | 10.6–18.2 | 0 |
| 8 | 4 | 4.68 | 18.73 | 17.68–19.36 | 5.6–67 | 0 |
| 8 | 16 | 3.11 | 49.73 | 49.65–49.79 | 3.49–12.4 | 0 |
| 8 | 32 | 2.09 | 67.01 | 66.16–68.04 | 2.36–6.55 | 0 |

Each configuration also crossed a production DataOrder epoch boundary: the actual final partial batch, exhausted cursor, NumPy reshuffle of all 9,000 indices, and first batch of the next epoch. This is a boundary probe, not a trained full epoch. Per-configuration tail/head/reshuffle timings are in the JSON. These outputs are disposable; no P02 checkpoint was saved.

Inference used the production checkpoint-backed RealEvaluator on the same representative 64-position fixture, three warm-ups and three windows of ten calls/configuration. The primary matrix rotates through 30 deterministic offsets spanning the fixture, with evenly spaced members within larger batches and identical memberships across thread counts. The initial fixed-cohort matrix is retained as exploratory raw data: its batch-one cohort was a single middlegame position. Checkpoint: **CUDA-preparation-step-1**, SHA-256 `ba19e500968f271941002b3e54f1441955126f276212a9bb7d9398a29f45f45f`, copied from `gpu-readiness/artifacts/cpu-portable-smoke-001/step-1.pt`. It has one diagnostic update and is **not accepted trained P**. Eval mode disables dropout.

| Torch threads | Batch | Latency p50/p95 (ms) | Samples/s | Window samples/s range |
| --- | --- | --- | --- | --- |
| 1 | 1 | 40.0/46.4 | 24.8 | 24.4–25.2 |
| 1 | 4 | 53.6/56.1 | 74.6 | 73.8–75.8 |
| 1 | 16 | 102.6/107.8 | 155.7 | 153.7–159.3 |
| 1 | 32 | 171.7/175.4 | 186.9 | 185.1–189.2 |
| 2 | 1 | 31.7/37.0 | 30.5 | 29.8–31.2 |
| 2 | 4 | 47.2/49.1 | 84.7 | 84.1–85.4 |
| 2 | 16 | 89.9/92.0 | 177.8 | 176.3–179.7 |
| 2 | 32 | 154.3/164.5 | 204.7 | 201.2–206.6 |
| 4 | 1 | 29.6/37.6 | 32.9 | 31.1–34.0 |
| 4 | 4 | 44.7/46.9 | 89.3 | 88.3–89.8 |
| 4 | 16 | 86.2/91.0 | 185.2 | 180.4–187.9 |
| 4 | 32 | 137.3/142.6 | 232.0 | 226.9–234.9 |
| 8 | 1 | 28.7/30.7 | 34.7 | 34.4–35.0 |
| 8 | 4 | 46.7/48.4 | 85.7 | 84.7–86.2 |
| 8 | 16 | 87.5/89.1 | 183.0 | 182.0–183.8 |
| 8 | 32 | 137.8/145.1 | 230.8 | 226.8–232.9 |

Larger batches improve evaluator samples/second in this microbenchmark. **Actual end-to-end batched self-play was not implemented or measured**; no batching gain in games/hour is established.

## Checkpoint-backed MCTS/self-play

Production Session, RealEvaluator and immutable move publication; standard initial position, 16 or 32 simulations/move, 512-ply cap, journal=true, c_puct=1.5, unvisited_q=0, temperature=1 before ply 30 then 0, root noise fraction=0.25 and alpha=0.3. One independent process/game, Torch intraop=1 and interop=1 to control oversubscription. One evaluator warm-up before each timed game. Seeds per wave are 20261011+i and 20261111+i, i=0…workers−1; two waves/configuration. Timing includes move publication and final snapshot copy, excludes process/checkpoint setup.

Smaller-worker waves are prefix subsets of larger waves, so aggregate games/hour also reflects game membership, length and idle tails after short games finish. This is a fixed-wave campaign, not a continuously refilled pool. Matched-seed trajectories were checked; early active-wave simulations/second gives a second scaling view while all workers are active. “Finished/h” includes terminal completed games and games truncated at the cap; completed-only throughput is shown separately.

| Sims | Workers | Completed/truncated | Mean plies | Finished/h | Completed/h | Sims/s | Peak RSS MiB | CPU mean % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1 | 1/1 | 287 | 16.26 | 8.13 | 20.7 | 458 | 98 |
| 16 | 2 | 2/2 | 377 | 16.17 | 8.08 | 27.1 | 901 | 152 |
| 16 | 4 | 5/3 | 360 | 25.15 | 15.72 | 40.3 | 1575 | 297 |
| 16 | 8 | 12/4 | 344 | 30.78 | 23.09 | 47.0 | 3267 | 549 |
| 32 | 1 | 2/0 | 311 | 8.69 | 8.69 | 24.0 | 522 | 99 |
| 32 | 2 | 4/0 | 256 | 14.77 | 14.77 | 33.6 | 911 | 142 |
| 32 | 4 | 8/0 | 187 | 22.90 | 22.90 | 38.1 | 1497 | 226 |
| 32 | 8 | 16/0 | 186 | 27.06 | 27.06 | 44.7 | 3035 | 438 |

| Sims | Workers | Wave finished/h range | Early active-wave sims/s range | Median move ms early/middle/late |
| --- | --- | --- | --- | --- |
| 16 | 1 | 9.1–77.2 | 22.1–22.5 | 690/686/713 |
| 16 | 2 | 14.7–17.9 | 29.4–35.3 | 997/988/918 |
| 16 | 4 | 22.9–27.9 | 43.9–64.3 | 1040/1337/1073 |
| 16 | 8 | 26.0–37.8 | 46.4–67.7 | 2461/2342/1845 |
| 32 | 1 | 5.3–24.2 | 24.2–25.5 | 1278/1292/1362 |
| 32 | 2 | 10.9–23.0 | 44.5–47.2 | 1405/1401/1290 |
| 32 | 4 | 16.1–39.8 | 57.7–67.4 | 1993/2106/1369 |
| 32 | 8 | 23.3–32.2 | 53.8–74.9 | 3942/4176/1399 |

Phase bins: plies 1–32, 65–128, 257–512. Aggregate phase differences also reflect board/material, game selection and workers leaving a wave; they are not a controlled history-length comparison.

## Bottlenecks, history costs and variability

Fresh-process labelling is dominated by startup. The separate 16-position transport profile averaged 421 ms per engine attempt: startup through `uciok` excluding the executable rehash was **302 ms (72%)**, rehash 33 ms (8%), and `go` to `bestmove` 69 ms (16%). Reset/configuration and return/quit account for the remainder. Transport intervals include scheduling and queued protocol work; they are not isolated kernel timings. This supports testing persistent-process reuse, but establishes neither its throughput nor correctness.

At batch 32 / four Torch threads, ordinary training time was **82.0% supervised update, 17.8% loader and 0.25% cursor validation**. The separate three-update cProfile showed native backward as the largest individual entry (0.546 s), plus dropout, einsum/relative attention, linear operations and finite guards; loader history replay/legality validation is a secondary cost. That profile includes extra gradient-norm diagnostics and inflates Python-heavy loader work, so its proportions must not replace ordinary timings. The 9,000-index epoch reshuffle took 1.38–1.54 ms; the tail/head update costs are in the JSON. Periodic checkpoint serialization was not measured.

Checkpoint-backed MCTS is dominated by neural forward execution: **85.4%** of the short 16-simulation profile, **85.0%** of the short 32-simulation profile, and **80.2%** of the 505-ply 32-simulation profile. Einsum and relative-logit/value tensor operations were the leading forward hot spots. Exclusive timers separate encoding, forward, evaluator preparation/postprocessing and record publication. The residual includes tree/board/history, record assembly and loop bookkeeping; it is not a pure tree timer. Final copy/run overhead was 4.25 s in the long profile and is outside the phase rows below.

Long-game instrumented breakdown (seed 20261111, 32 simulations, 505 terminal plies; excluded from throughput):

| Phase | Median move ms | Neural forward % | Encoding % | Evaluator other % | Publication % | Tree/board/history + assembly residual % |
| --- | --- | --- | --- | --- | --- | --- |
| early | 1469 | 88.83 | 1.31 | 5.23 | 0.13 | 4.50 |
| middle | 1480 | 85.91 | 1.68 | 5.31 | 0.13 | 6.96 |
| late | 1586 | 76.94 | 2.71 | 5.31 | 0.12 | 14.93 |

**Some hot paths still grow with game length.** Board move-stack copying remains linear in stored history, and immutable record assembly retains a small full move header. In the 32-simulation controlled prefix profiles, 200 `Board.copy` calls cost 1.76/50.23/122.59/265.98 ms cumulatively at 0/96/256/480 history plies. Python profiling amplifies copy overhead; these are instrumented costs, not ordinary search-time contributions. Repetition queries depend on the reversible history; neural features remain bounded to eight frames. The full final game snapshot is also copied once at termination. Long-profile residual grows from 4.5% early to 14.9% late, while publication stays around 0.12–0.13% and its median around 1.8–1.9 ms. Ordinary searches and games below show much smaller and non-monotonic total-cost changes because board/material and host conditions vary. Optimising history ownership/copying is worth a targeted correctness-preserving experiment, but neural execution remains the larger bottleneck.

| Simulations | History plies | Median search ms | Range ms | Neural forwards/search | Public admission ms |
| --- | --- | --- | --- | --- | --- |
| 16 | 0 | 672.2 | 669.9–727.5 | 17 | 0.13 |
| 16 | 96 | 699.9 | 690.9–702.7 | 17 | 1.12 |
| 16 | 256 | 713.5 | 705.2–722.9 | 17 | 3.21 |
| 16 | 480 | 725.6 | 724.8–759.9 | 17 | 6.58 |
| 32 | 0 | 1291.3 | 1289.7–1297.1 | 33 | 0.16 |
| 32 | 96 | 1546.0 | 1367.0–1836.3 | 33 | 1.09 |
| 32 | 256 | 1484.4 | 1471.0–1572.5 | 33 | 3.26 |
| 32 | 480 | 1478.5 | 1455.4–1618.4 | 33 | 6.48 |

The separate history probe uses a legal quiet 512-ply trajectory (fixture seed 761), production trusted search-owned root copying, one warm-up and three ordinary timings at each prefix; separate cProfile searches count neural calls. Boards/material differ across prefixes, so total search ratios are not pure history effects. Public Node admission is measured separately rather than repeated inside the per-move hot-path timing. These are search probes, not games/hour measurements.

Window ranges are reported rather than confidence intervals: three label/neural windows and two self-play waves are too few for precise sustained-rate claims. Label eight-worker throughput fell about 15% between its first and last window; the later one-worker anchor recovered to **2.48 labels/s** with zero failures and 64 exact non-timing field matches. Thus the campaign shows variability, not a confirmed monotonic thermal slowdown. Group pressure snapshots ranged **35–54% system-wide free**; initial compression/swap and varying background load limit extrapolation. Every saved `pmset -g therm` snapshot reported no recorded thermal/performance warning or CPU power status. Temperature, core residency and frequency were not measured, so absence of warnings does not rule out throttling. The four performance and four efficiency cores were identified, but no affinity experiment isolated either class.

## Estimates and next experiment

The following are **linear projections**, not measured larger annotation campaigns or measured throughput for 100–300 completed future games. Annotation assumes the fixed phase mix, unchanged teacher/settings and sustained observed rates. Self-play assumes this plumbing checkpoint, these seed/game-length distributions, settings and fixed-wave scheduling; learned P can materially change lengths and throughput. Startup costs, other workloads and thermal drift can change sustained rates.

| Label workers | 10k labels h | 100k h | 300k h | 1M h |
| --- | --- | --- | --- | --- |
| 1 | 1.33 | 13.27 | 39.80 | 132.66 |
| 2 | 0.67 | 6.73 | 20.18 | 67.26 |
| 4 | 0.40 | 4.01 | 12.04 | 40.13 |
| 8 | 0.35 | 3.49 | 10.47 | 34.90 |

| Sims | Workers | 100 finished games h | 300 finished games h | 100-game range from two waves h |
| --- | --- | --- | --- | --- |
| 16 | 1 | 6.15 | 18.45 | 1.30–11.01 |
| 16 | 2 | 6.19 | 18.56 | 5.57–6.80 |
| 16 | 4 | 3.98 | 11.93 | 3.58–4.37 |
| 16 | 8 | 3.25 | 9.75 | 2.65–3.85 |
| 32 | 1 | 11.51 | 34.52 | 4.14–18.88 |
| 32 | 2 | 6.77 | 20.31 | 4.34–9.20 |
| 32 | 4 | 4.37 | 13.10 | 2.51–6.22 |
| 32 | 8 | 3.70 | 11.09 | 3.11–4.28 |

For 100–300 terminal completed games, assuming the observed completed/truncated mixture persists and cap-truncated games do not count toward the target:

| Sims | Workers | 100 terminal completed h | 300 terminal completed h |
| --- | --- | --- | --- |
| 16 | 1 | 12.30 | 36.91 |
| 16 | 2 | 12.37 | 37.11 |
| 16 | 4 | 6.36 | 19.09 |
| 16 | 8 | 4.33 | 12.99 |
| 32 | 1 | 11.51 | 34.52 |
| 32 | 2 | 6.77 | 20.31 |
| 32 | 4 | 4.37 | 13.10 |
| 32 | 8 | 3.70 | 11.09 |

**Recommendation: optimise first, while continuing bounded local work.** Four Stockfish workers are a practical local starting point (6.92 labels/s); eight reached 7.96 labels/s, only 15% more throughput with substantially higher latency and memory. First run the planned 500-position persistent-versus-fresh label-field equivalence experiment before considering production reuse; this campaign provides no approval for it.

For local neural work, batch 32 with two to four Torch threads is a useful measured starting point: four delivered 67.94 training samples/s and eight 67.01, with overlapping window ranges. Four-thread batch-32 inference delivered 231.95 samples/s, versus 230.79 at eight; batch-one latency benefited modestly from more threads in this microbenchmark. These settings are measured options, not changes to production defaults.

Independent games help aggregate simulations/second, but scaling is modest after four workers: four to eight improved aggregate simulation rate by about 17% at either budget. Finished-game rates are additionally confounded by game membership and wave tails. More homogeneous performance CPU cores might help, but this mixed-core Mac cannot predict scaling on a different CPU. Prioritise a reviewable experiment on relative-attention tensor operations and history copying, with production semantic checks. Measure actual cross-game batched inference in end-to-end self-play before claiming a games/hour benefit. After that, a matched GPU comparison is more informative than choosing paid hardware from CPU timings alone.

Mac CPU timings establish local costs and scaling only. They do not establish CUDA or MPS throughput, GPU memory limits, strict deterministic kernel availability, host/device transfer cost, or end-to-end batched self-play speed. A useful next GPU comparison must run the same real model, D08 inputs, checkpoint and search settings; compare CPU and CUDA on the same GPU host to separate accelerator gains from host-CPU differences. No paid hardware choice follows from these results alone.

## Verification and limits

Accepted-label/input SHA-256 and both original/copied checkpoint SHA-256 checks passed. All 16,701 durable configuration/boundary/move record files passed production digest checks; 28 matched durable-record comparisons were byte-exact. Profile trajectories also matched ordinary runs. 28 matched-seed trajectory comparisons were exact across worker counts. Benchmark-only source passed Python compilation; no full pytest/Stage 1 campaign was repeated. All profile runs are excluded from throughput tables. Raw diagnostics retain setup failures and exploratory cache-mixed training rather than replacing their history.

