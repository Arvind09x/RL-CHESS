# Stage 1 readiness review

Review input: `e0c6ec577408514d72126d5c17233d8054041fc6`, with clean
readiness and main checkouts. No applicable `AGENTS.md` was present. Changes
preserve unrelated work; GPU verification, packed storage, P02 training and the
broad Mac hardware benchmark are outside this review.

The compact numerical evidence is in [stage1-measurements.json](stage1-measurements.json).
The local fresh-clone/merge receipt is `artifacts/stage1/final-integration.json`
in the readiness checkout. It records the final tested revision, exact commands,
environment, all skips, result and merged main commit; detailed local logs remain
under ignored `artifacts/stage1/`.

## Acceptance checks

1. **M05/M06:** `tests/network_checks.py` explicitly constructs historical M05
   with `relative_enabled=False`. The independent reference equations remain
   unchanged. Ordinary pytest needs no injected plugin, configuration or
   environment workaround. Production relative-attention checks pass separately.
2. **Search ownership:** public `ChessState` has no trusted-copy method. Checked
   `Node` admission owns its board, moves and mutable history frames. Public node
   snapshots and evaluator inputs have independent storage. Trusted copying takes
   a search owner only; child adoption makes one full-stack copy. Regressions
   cover caller mutation, evaluator mutation, history-frame aliases, repetition,
   special moves and independent histories.
3. **Frozen datasets:** source opening fully audits artifacts/labels. Freezing
   rejects changes after admission, validates decoded targets and owns recursively
   immutable metadata. Decoded cache storage is private; samples and batches own
   their arrays/boards. Frozen supervised and replay metadata reject writes and
   deletion. Resume reopens sources and rejects changed bytes or identities.
   Mixed updates do not scan membership or validate the full permutation again;
   epoch reshuffles remain explicit work at epoch boundaries.
4. **Journal/recovery:** an actual subprocess is killed by SIGKILL after move-file
   publication and directory fsync, before publishing the boundary pointer.
   Fresh-process recovery exactly matches an uninterrupted four-move checkmate:
   every record, PCG64 state, outcome, configuration/checkpoint/source/game/sample
   identity and committed move. A 512-ply valid, nonterminal trajectory reaches
   the cap and reopens exactly. Per-move record-list copying, interruption deep
   copying, repeated UCI history formatting and owned root-history replay are
   removed. Explicit returned snapshots still own a full copy.
5. **Checkpoints:** exact restoration remains the default. Explicit
   `runtime_policy="compatible"` permits only OS description/Python patch changes.
   Subsequent provenance records even a same-runtime opt-in, and saves preserve
   it when callers pass their original provenance. Tests reject Python major/minor,
   numerical-library, machine, thread, precision, determinism and backend drift.
   CUDA runtime requirements remain exact.
6. **Historical recovery:** R01 was restored from `62af7ac` into an isolated
   location. All **91 files / 14,462,196 bytes** matched the external per-file
   SHA-256 manifest and Git bytes. The external compressed archive matched its
   recorded hash/size. Recovery instructions now include isolated extraction,
   full membership/hash verification and a Git-only fallback.

## Targeted measurements

Environment: macOS 14.5 arm64, Python 3.12.12, NumPy 2.5.3, Torch 2.8.0,
python-chess 1.999/chess 1.11.2; one Torch thread, strict determinism.

The criterion was fixed before measuring: 10k/1k median update time **<=1.25**
and bootstrap 95% upper bound **<=1.35**. Both sources contain valid synthetic
D07 samples, admitted through the production loader and MixedSources. Batch size
4 (two samples per source), zero decoded cache budget, five warm-up updates,
30 measured updates per size in alternating blocks, identical M06/Adam settings.
Model tensors match exactly after every paired block. An initial harness attempt
was discarded because unequal replay selections failed this equivalence check;
the replay RNG was aligned without changing the criterion.

| Samples | Open validation | Freeze/membership | Median update | Standard deviation |
|---|---:|---:|---:|---:|
| 1,000 | 0.756 s | 1.360 s | 174.714 ms | 4.742 ms |
| 10,000 | 7.712 s | 13.742 s | 174.370 ms | 3.865 ms |

Dataset setup is separate from training steps. Fixture generation took 1.038 s
and 11.189 s respectively. Steady-state ratio **0.998**, bootstrap upper bound
**1.014**: pass. This measures a fixed window without a supervised epoch wrap.

Self-play uses controlled legal search to isolate record/history costs, not
neural throughput. Untraced publication medians are **1.280 ms** at plies 1–32
and **1.310 ms** at 481–512. Peak temporary publication allocations stay around
**107–109 KB**. Complete move medians increase from **4.692 ms** to **14.471 ms**
as retained history grows; rule/history queries and independent board copies
remain necessary. At 512 plies, records occupy about **108 MB**, one owned board
history about **286 KB**, and a trusted snapshot copy takes **0.739 ms**.
The historical full-record deepcopy costs about **2.09 s / 58 MB** at 512 plies;
it is no longer per-move work. In-memory records still grow linearly and remain
bounded by the configured cap. Packed storage is not adopted here.

Allocation probes incur tracing overhead and are excluded from the timing
windows. Before/after the owned-root optimization, the complete 512-ply records,
RNG and identities match exactly. Source revisions, repeated timings, variability,
memory checkpoints and recovery comparisons are retained in the JSON evidence.

## Integration procedure

After final changes, clone the candidate into a new empty location, run
`uv sync --locked --extra model`, activate its new `.venv`, and run plain
`pytest`. The fresh checkout contains no `runs/`, training artifacts or hidden
test plugin. Report every skip and require the CPU/model suite to pass before
merging that exact candidate into clean local main with `git merge --ff-only`.
No push or publication is part of this procedure.
