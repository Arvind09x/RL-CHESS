# CPU P02 pilot: prepared development playing-strength check

This is **supervised-candidate versus untrained**, not P-versus-R. The candidate
is `artifacts/p02-cpu-001/step-001410.pt`, selected by the completed pilot's
validation loss; the comparator is its actual `step-000000.pt`. This check does
not select or accept project P, start P03, or access training/validation/test
labels. No pilot matches have been launched during preparation.

## Completed-run preservation

Before changing evaluation code, `artifacts/p02-cpu-001-preserved/` captured 97
files: the exact current training runner, all production Python sources and
scripts, dependency definitions/lock, portable path configuration, and the
completed pilot's configuration, baseline, summary, metrics and learning curves.
Each copied file has a SHA-256 and size in `manifest.json`; all 11 original
checkpoints have frozen SHA-256 identities and sizes there. Original checkpoint
bytes remain in the pilot directory. Git HEAD, dirty status and tracked diff
are retained, including the pre-existing uncommitted pilot-runner edits.

| Role | Checkpoint | SHA-256 |
| --- | --- | --- |
| supervised-candidate | step-001410.pt | `115ba5be9cb9c4689bd70d8dd548b94bb140b0b839ad0b695d70695cdaf8e26a` |
| untrained | step-000000.pt | `e0efbf441a00ef8d801cff6f0d18e940ba0b93c1bd9ce53e9d0c113082ca2f8c` |
| final training checkpoint, not the evaluated candidate | step-002820.pt | `6d80efd72d9834900155c7dc59ff8a68135e552a3bb7de9cac5d3de99fe9581a` |

The run recorded 2,820 completed updates, batch size 32, learning rate 0.0001,
seed 20261012, CPU float32, four training Torch threads, one interop thread,
strict determinism and validation every 282 updates. `config.json` and
`summary.json` are preserved verbatim. The original runner did **not** record a
source digest while training: this is an exact post-run source snapshot taken
before these edits, not independent proof that every captured source byte was
the one executed historically. No original command line is invented from the
configuration. The preserved portable path file is captured context; its use
by that historical command cannot be proved from the run's saved metadata.

## Fixed design chosen before execution

**64 opening pairs, 128 scheduled games, four independent processes.** There is
no early stopping on score, no replacement opening for failures/truncations,
and no extension to obtain a desired number of decisive or completed games.

`configs/p02-cpu-strength-dev.json` freezes the full settings, 64 opening
histories, all 128 game IDs/colour assignments/seeds, checkpoint identities,
runtime versions, preservation digest and evaluation source hashes. The plan
is checksummed and cannot be overwritten with `prepare`. Execution and restart
reject source, runtime, checkpoint, opening or plan changes.

The suite uses 16 independently authored opening families with four branches
each: Italian, Ruy Lopez, Scotch, Petrov, Sicilian, French, Caro Kann, Pirc,
Queen's Gambit Declined, Slav, Queen's Gambit Accepted, Nimzo Indian, English,
Reti, King's Indian and Dutch. Prefixes are 5–11 plies, legal, nonterminal and
distinct by rule-position root. Full replay starts from the standard board.
These are development diagnostics, authored without reading a games corpus,
labels, or reserved test openings. Related family branches are not guaranteed
independent samples of chess positions.

Each opening is replayed identically twice: candidate White in leg 0, candidate
Black in leg 1. The position is not mirrored. Master seed 2026101202, explicit
game seeds and the production per-turn seed derivation are retained, even
though deterministic move selection uses no stochastic sampling. Scheduling
and worker timing cannot select openings or change a leg's seed.

Both checkpoints are loaded separately through production `RealEvaluator` /
`load_inference`, with expected M06 configuration, matching provenance and
embedded checkpoint IDs. They use exactly the same `SearchPlayer` settings:
32 completed simulations per move, c_puct=1.5, unvisited value=0, a fresh root,
maximum visits with smallest shared ID as tie-break, no exploration noise,
evaluation mode with dropout disabled, CPU float32, strict deterministic Torch
algorithms, highest matmul precision, one Torch intraop thread and one interop
thread per worker. Each process owns two models; there is no shared evaluator
or cross-game batching. Common BLAS thread limits are set to one before spawn.

The **512-ply cap includes the opening prefix**. The production adapter checks
actual rule termination before the cap, including actual-occurrence repetition
and fifty-move endings. A game still nonterminal at the cap is `truncated`, with
null winner and outcome. Material, neural values or a teacher never adjudicate
it; it never contributes a draw or any other scored outcome.

## Sensitivity and cost

The target is a **large improvement**, for example candidate score about 0.70
against untrained versus neutral 0.50. An opening-pair score is the mean of its
two leg scores (win=1, rule draw=0.5, loss=0), so it is bounded in [0,1]. At 64
independent completed pairs, its worst-case standard error is 0.5/sqrt(64) =
0.0625. A normal-approximation 95% margin can be about 0.123. The conservative
bounded-variable 95% margin is sqrt(log(40)/(2*64)) = 0.170, so a 0.70 estimate
could have a lower bound only around 0.530 under that independence assumption.

The illustrative two-sided normal planning approximation for 80% power uses
`n ≈ ((1.96 + 0.84) * 0.5 / gain)^2` independent **pairs**, not games:

| Score gain above 0.50 | Approximate pairs needed | Games | Hours at 22.9 games/h |
| --- | ---: | ---: | ---: |
| 0.20 | 49 | 98 | 4.3 |
| 0.10 | 196 | 392 | 17.1 |
| 0.05 | 784 | 1,568 | 68.5 |
| 0.02 | 4,900 | 9,800 | 428 |

These are approximate sensitivity illustrations, not a promised power or a
future execution plan. The 64-pair design has roughly 89% power for a 0.20
gain, about 36% for a 0.10 gain and about 12% for 0.05 under that worst-case
normal model. It cannot reliably detect small gains. Missing complete pairs
reduce effective sample size; related family branches and a fixed authored
suite weaken independence/generalization assumptions. Results support a
development comparison on these openings, not a population strength rating,
an Elo claim, an acceptance decision, or an untouched test-set conclusion.

The previous Mac profile measured 22.9 finished games/hour with four workers,
one thread each and 32 simulations, with two wave rates of 16.1–39.8/h. Linear
projection gives **about 5.6 hours for 128 games**, with that observed wave range
corresponding to **3.2–8.0 hours**. This is a planning estimate, not a time limit
or guaranteed range. Those games used a different diagnostic checkpoint and
noisy self-play; this check uses two different models and deterministic
evaluation. Game lengths, cap attrition, memory pressure, thermals and host load
can change cost substantially. Near-cap games can take longer. No extra
performance games have been run to refine this estimate.

## Durable execution and reporting

Workers commit whole game records directly to `results.sqlite3` before telling
the coordinator they finished. SQLite WAL with `synchronous=FULL` and unique
game IDs provides atomic durable publication. The DB embeds the exact frozen
plan; stored records also have plan identity and content digests. Restart
audits colours, checkpoints, seed derivation, applied move/search traces,
history and actual rule outcomes before scheduling missing IDs. Invalid or
corrupt results stop the run; they are not silently replaced.

All durably recorded statuses—completed, truncated, failed and interrupted—are
retained once, with no automatic replay or replacement. A hard crash before
commit leaves that game pending and restart replays it from its frozen opening;
there is no per-ply recovery. A crash after commit retains the game even if the
coordinator never received notification. Output locks prevent overlapping
coordinators and protect against still-running orphan workers. A process
failure before publication is a coordinator event and leaves the game pending,
not a fabricated game outcome. If restarted while a worker remains alive, the
command refuses to run until that worker exits.

`summary.json` is atomically replaced after execution/stop; the authoritative
results are the database and its WAL. Keep the output directory together,
including WAL/SHM files while active; do not copy just the main DB during a run.
`report` rebuilds a summary from committed results without playing. Reporting
while execution is active is a snapshot; reporting after exit gives the final
committed state. A force-kill can leave `summary.json` stale until report/restart.

The report contains completed/truncated/failed/interrupted counts and rates
among recorded games, fractions of all planned games, pending count, candidate
W/D/L and score over **completed games only**, completed-single-leg denominator,
complete/incomplete/missing opening-pair counts and complete-pair mean score.
Uncertainty resamples **whole complete opening pairs** with a 10,000-sample
95% percentile bootstrap, fixed seed 2026101203. It excludes pairs with any
unscored or missing leg, discloses that attrition, and is conditional on fully
completed pairs. No interval is returned with fewer than two complete pairs.
A separately labelled Hoeffding interval is also reported when at least one
pair is complete; it stays nonzero when bootstrap scores are identical. Both
intervals require caution about family dependence and the authored suite.

## Verification without pilot matches

Fixture tests cover legal opening diversity, frozen seeds, both colour routes,
four spawn workers, one-thread initialization, W/D/L and rule-draw scores,
truncation/failure exclusion, whole-pair bootstrap, partial pairs, restart
dispatch, unique publication, source/checkpoint tampering, output locking,
four concurrent writers, and crashes before/after commit. Terminal fixtures
make no search or evaluator calls; coordinator/worker execution tests mock the
game call. Actual pilot checkpoints are verified with production-loader
forward calls on the standard initial position, checking embedded identities,
distinct policies and byte-identical repeated outputs. There are no pilot
searches, matches or training steps in this verification.

The checkpoint verification is saved as
`artifacts/p02-cpu-001-preserved/strength-routing-verification.json`.
The five targeted test modules passed **275 tests in 17.25 seconds**; their
command, test-source hash and frozen-plan identity are saved in
`artifacts/p02-cpu-001-preserved/strength-test-verification.json`. The final
plan SHA-256 is
`1c2b22dec9dc1bf08c9573b05ac6886a088ed4fbc9f4b2b2a7d99bb77f03cdae`.
Preparation uses `prepare` once; `verify` loads and forwards checkpoints only.
Only `run` dispatches games.

## User launch command

Run this exact command from any working directory. Running it again with the
same output restarts only uncommitted games:

```sh
/Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python /Users/arvindsrinivass/Desktop/RL-Chess/scripts/check_p02_strength.py run --plan /Users/arvindsrinivass/Desktop/RL-Chess/configs/p02-cpu-strength-dev.json --output /Users/arvindsrinivass/Desktop/RL-Chess/artifacts/p02-cpu-001-strength-dev
```

Rebuild the report without playing:

```sh
/Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python /Users/arvindsrinivass/Desktop/RL-Chess/scripts/check_p02_strength.py report --plan /Users/arvindsrinivass/Desktop/RL-Chess/configs/p02-cpu-strength-dev.json --output /Users/arvindsrinivass/Desktop/RL-Chess/artifacts/p02-cpu-001-strength-dev
```
