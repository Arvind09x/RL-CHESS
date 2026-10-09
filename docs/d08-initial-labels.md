# D08 initial production labels — 9 October 2026

D08 starts from reviewed integrated main
`9565093abe40a7807e3472c1101f5f28c14b7afe`, in the independent local clone
`/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/d08-initial-labels`, on
`codex/d08-initial-labels`. The accepted D04–D07 implementation, review and
integration documents, `interface.md` and `coding-plan.md` define the inputs.
Main is preserved. Independent D08 milestone review and integration remain
pending. Automated D08 execution and all verification gates are complete;
no milestone-review acceptance or main integration is claimed.

## Membership frozen before annotation

`produce_d08.py freeze` verifies the accepted D03/D04 artifacts, both protected
registries and pinned teacher executable/embedded NNUE assets. It removes the
12 exact D05 plumbing references and 1,000 exact D06 benchmark references;
prior membership bytes are verified against accepted evidence. No prior label
is copied into this production bundle or counted toward its quota.

Selection is `d08-reference-sha256-v1`, seed **20261011**. For every remaining
audited D04 reference, hash compact sorted-key ASCII JSON containing exactly
`selection_version`, `seed`, `split`, `game_id` and `sample_ply`. Independently
within each split, retain the smallest ranks: **9,000 train, 500 validation,
500 test**. Break digest ties by `(game_id, sample_ply)`. Annotation order is
train, validation, test, each in ascending rank order. Selection is weighted
over references, not distinct rule identities. No result or teacher target
selects membership or ordering. Fixture tests compare against a complete sort
and reversed input order.

The fsynced `selected.jsonl` and `settings.json` freeze membership, ordering,
split quotas, teacher configuration, source identities, deadlines, retry limit
and failure accounting. Execution requires a pristine frozen directory,
rechecks source/input/teacher bytes and recomputes the exact
selection before calling the engine. Existing output prevents another execution.

Membership froze at **2026-10-09T07:22:40.075228+00:00** (12:52:40 IST),
SHA-256 `e9bb4ba052f1210773a89228a98fab692d4990f8d4b961a9567c8a770aebb36a`.
The 10,000 references cover **9,861 canonical games**, with at most **three**
references from any game. They are a subset of the existing D04 selection;
no positions, phases, canonical game assignments or registry membership are
changed. The existing eight-position limit therefore remains intact, including
the union with the excluded prior references.

| Split | Opening White / Black | Middlegame White / Black | Endgame White / Black | Total |
|---|---:|---:|---:|---:|
| train | 928 / 1,085 | 2,872 / 2,927 | 587 / 601 | 9,000 |
| validation | 61 / 56 | 151 / 161 | 28 / 43 | 500 |
| test | 81 / 61 | 138 / 149 | 34 / 37 | 500 |

D04's `material-ply-v1` phase rules remain unchanged: combined nonpawn/nonking
material <=14 is endgame first; otherwise plies <20 are opening, and remaining
plies are middlegame. Current counters and source results do not set phase.
The selected distinct rule-identity counts are **8,861 / 495 / 461**, respectively.
Same-split rule positions may occur in different games with different histories;
each reference receives a fresh independent search. Counts distinguish 10,000
references from 9,817 distinct rule identities. Cross-split overlap uses exact
four-field legal-en-passant FEN, excluding counters/history, and must be zero.
Every selected held-out reference must exactly match the protected sample
registry and its game registry; train must avoid every protected game/identity.

## Accepted worker and explicit split replay

D08 reuses `AnnotationWorker`, `engine_attempt`, response validation and label
schema from D05. The only change to the accepted worker is an optional
`root_replayer` callable. Its default is the unchanged train-only D05 replayer;
D08 explicitly supplies `replay_production_root`, which validates canonical
identity/assignment and replays the actual recorded prefix for all three splits.
Held-out games retain their real split throughout; nothing is relabeled as train.
Controlled tests prove the default still rejects held-out roots and the explicit
production path preserves held-out assignment and history.

The teacher remains `/private/tmp/Stockfish-sf_17.1/src/stockfish`, SHA-256
`3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4`, with
both accepted embedded NNUE assets verified at the recorded build offsets.
Requested settings stay **100,000 nodes**, Threads=1, Hash=128 MB, MultiPV=1,
UCI_ShowWDL=true, full strength, standard chess, no tablebases and pinned defaults.
Every attempt rehashes the binary and starts a fresh process, retains the full
history in its UCI root, completes Clear Hash/readiness before go, validates the
final individual principal row and separate bestmove, and exits normally before
success. Returned node overshoot or natural early completion is retained.

The frozen policy allows **two attempts maximum** per reference, with D05's
**30-second** operation/search deadlines. Attempts, failures, retries and
interruptions are recorded separately from successful labels. Exhaustion writes
an explicit exclusion. Interruption propagates and records pending references.
There is no refill, replacement, label reuse, result cache, concurrent engine
worker or resume. A shortfall produces `status=shortfall` and an error exit;
it cannot pass D07 or complete D08.

Raw transport and parsed principal rows are retained. Stockfish can put score
bounds after WDL, where the pinned parser does not surface them. The auditor
preserves both raw flags separately from parsed flags. Targets remain the legal
best-move class and raw root-side WDL counts divided by 1,000; source results
and centipawn conversions never become labels.

## Independent replay and D07 verification

`audit_d08.py` imports neither the worker nor production selector. It scans every
eligible D04 reference and proves no unselected reference ranks below its split
boundary. It verifies exact prior-reference exclusion, protected membership,
canonical identities/assignments, all pairwise rule-position overlaps and attempt
accounting. Every successful root is replayed through both a direct python-chess
board and the shared adapter, checking full FEN/counters, promoted bits, actual
move stack, full-prefix checksum, nonterminal state, phase and real snapshots.
It independently calculates every legal move ID and legal mask, verifies class
round trips, replays the complete PV legally, checks root-perspective soft WDL,
teacher/options/provenance, and matches raw reset/root/go/score/WDL/PV evidence.
Per-record feature/mask fingerprints and raw bounds stay outside Git.

`check_d08_loader.py` uses the unchanged accepted D07 `open_dataset` with explicit
train, validation and test selections. Each opening validates the full bundle.
Every individual sample must match the independent audit's feature/mask hashes
and its saved history/class/soft-WDL target. All samples also pass through batches
of 1 and 64 in saved order and 127 in reverse order, retaining partial batches.
Shapes, dtypes, references, legality and target/fingerprint equality are checked
for every batch row. No model or loss is called, and Torch is not imported.

## Actual completion and cost

Exactly **10,000 selected / 10,000 attempts / 10,000 new successful labels**:
**9,000 train, 500 validation, 500 test**. Failed attempts, retries, interruptions,
exclusions, pending references and per-split shortfalls are all **zero**.
All rows in the composition table became labels without membership changes.
All 10,000 successful records passed independent replay and target/provenance
checks; cross-split rule-position overlaps are **zero for all three pairs**.

Annotation ran **07:23:58.551204–08:38:50.756597 UTC**, or approximately
**12:53:59–14:08:51 IST**. The measured annotation loop took **4,491.422s /
74.857 minutes**, including fresh processes, full replay, validation, binary
rehashing, fsync writes and logging. The complete execute invocation took
**4,509.573s**; freeze took **43.494s**, and post-input revalidation took
**0.740s**. Execution setup/command overhead was **17.411s**. The annotation
loop produced **2.226466 successful labels/s**, with median position latency
**445.410 ms**, p95 **495.518 ms** and p99 **673.597 ms**. Regression checks
shared the host for approximately 54 seconds and are included in this actual cost.

Engine-reported search time totals **791.543s**, with **974,774,193 returned
nodes**. All 10,000 searches requested 100,000 nodes; **365** naturally completed
below that request. Actual node counts and overshoot are retained. The full raw
principal evidence contains **5,118 score-bound flags**: **1,931 lowerbound**
and **3,187 upperbound**. The pinned parser surfaced zero of these flags;
original lines and independent raw-bound fields remain retained for every label.
No score-bound flag is interpreted as an exact centipawn target.

The independent audit took **54.712s** (whole command **55.888s**). In-memory
full-prefix adapter replay totaled **17.469s**, encoding **1.922s** and legal
masks **4.363s**; the complete per-label checks totaled **40.692s**, including
these components. Setup/membership/join checks took **13.462s**. These are actual
component timings, not additional times to add twice to the complete audit.

D07 verification took **213.240s** (whole command **213.491s**). Every one of
10,000 records was checked individually and in three complete batch passes.
Each split open eagerly validated all 10,000 labels before filtering:

| Split | Open and full-bundle validation | Individual samples | Batch 1 | Batch 64 | Reverse batch 127 |
|---|---:|---:|---:|---:|---:|
| train (9,000) | 36.684s | 22.121s | 22.523s | 22.862s | 23.067s |
| validation (500) | 36.700s | 1.266s | 1.231s | 1.220s | 1.209s |
| test (500) | 39.416s | 1.128s | 1.119s | 1.126s | 1.160s |

Training batch throughput was **393.662 samples/s at batch 64**, including full
root replay, encoding/masks, batching and fingerprint/target verification.
No cached tensors, model inference or learning are included. Final partial
batches were retained; reverse-order batches preserve the explicit permutation.

Labels occupy **25,476,244 bytes / 25.476 MB**. The full external production
bundle occupies **302,495,956 bytes / 302.496 MB**, including membership,
settings, attempts, labels, raw transport, latencies, exclusions, summary,
per-record independent replay evidence and loader report. MB uses decimal bytes.
Canonical games and D04 capacity are reused, not included again in bundle size.
The label file SHA-256 is
`e653e2f5b098562eef2482478ef2b48009516a81bec2fbf0d53dbc92f5a30369`.
[Exact paths, every artifact hash and measurements](../runs/d08/results.json)
identify all external files; no bulk production artifacts are committed.

Full regressions passed: **1,192 model tests**, **1,023 adapter tests** with
**seven expected optional-Torch skips**. New checks cover selection ordering,
capacity/overlap/eight-root rejection, explicit held-out replay with genuine
assignments, unchanged D05 default restrictions, exhaustion and interruption
accounting. Frozen production sources remained unchanged throughout annotation.
All source, data, provenance and main-preservation checks are recorded under
`runs/d08/`; only the explicit replay hook and append-only status documents
change accepted tracked files.

The first preservation command failed its assumption that primary main would
remain clean. The main commit remains unchanged, but the primary coding plan
received three uncommitted D06/D07/E01 checkbox edits outside this isolated
checkout during annotation. The final checker proves these are the only primary
working-tree changes, records their bytes and preserves them without writing.
The initial failure and corrected check remain in the command evidence; they
are verification events, not annotation failures.

## Evidence, reproduction and scope

Production output is outside Git in `/private/tmp/rl-chess-d08-production/`.
Compact path/size/SHA-256 indexes, actual command timestamps/exits, lossless logs,
regressions and preservation checks belong under `runs/d08/`. No labels, selected
membership, raw UCI stream, per-position latency or full per-label audit is
committed. Canonical games and audited D04 capacity are reused in place.

Run from this checkout with bytecode disabled and `PYTHONPATH=src:.`, using the
existing pinned `.venv-acceptance` environment. A fresh output directory is
mandatory for production. `freeze` takes explicit manifest/games/split-audit/
prepared/executable/D06-bundle paths; `execute` takes only the frozen output.
See the recorded freeze and production command JSON for exact arguments.

Independent reviewers can rerun audits without changing production evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -B scripts/audit_d08.py /private/tmp/rl-chess-d08-production --report /private/tmp/d08-review-replay.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv-acceptance/bin/python -B scripts/check_d08_loader.py /private/tmp/rl-chess-d08-production --report /private/tmp/d08-review-loader.json
```

New report paths are required. CPU macOS arm64/Python 3.12.12 only; one sequential
run on an interactive host. Loader checks use warm files and validate all labels
on every split opening. Regression suites briefly shared the host during annotation;
actual wall time includes that contention. No cache, GPU, learning, strength or
cross-platform claim is made. No training, held-out tuning or larger expansion
selection is authorized or performed. Stop at the D08 commit for independent review.
