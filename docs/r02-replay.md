# R02 completed-game replay — 10 October 2026

Implemented only R02 in the isolated `r02-replay` clone on `codex/r02-replay`,
from accepted R01 main base `48bce1ecb469d07d9d5a57aff7d9cb4896a7e281`.
R01 implementation `2e31236` and its independent review/integration evidence
are retained. No applicable ancestor or repository AGENTS.md was found.
The interface and coding plan remain authoritative. R02 awaits independent
review; the R02 checkbox remains unchecked. R03 has not begun.

## Audit and targets

`chess_rl.replay.finalize(snapshot)` accepts an owned R01 configuration/game
snapshot. It validates schemas, finite primitive targets, strict boolean masks
and integer statistics, checkpoint metadata/contracts and complete-history
provenance, then calls the unchanged R01 `audit`. That audit replays every
actual move through the existing adapter, checks root FEN/promoted bits/player,
legal masks, positive simulation counts, legal visit support, the original
normalized visit policy, selected moves, RNG continuity and final adjudication.
Unknown position fields, including pre-attached outcome labels, are rejected.

Each admitted played root is additionally reconstructed with
`make_state(initial_fen, full_actual_moves[:ply], history_complete=True)`.
The adapter owns rule history, Black reflection, features, padding and terminal
checks. The replay location contains the complete actual prefix, even beyond
the eight feature snapshots. `available_snapshots` comes from the adapter.
No board is reconstructed from the recorded current root FEN alone.

Only completed games generate samples. WDL order is `[win, draw, loss]`:
the root player receives `[1,0,0]` if that player won, `[0,0,1]` if that player
lost, or `[0,1,0]` for a draw. The final adapter outcome determines every label.
Opening-prefix positions and the final terminal board are not played search
roots and are not sampled. Immediate terminal games with no generated roots
have zero samples, including completed games whose retained opening moves
already reach a repetition draw. Truncated/interrupted records generate no
samples and retain null winner/outcome. Interrupted search diagnostics cannot
become targets. Resuming an interrupted game can later produce an admissible
completed snapshot without relabeling the earlier unfinished receipt.

Each sample preserves the original `position` dictionary, including boolean
`legal_mask[4352]`, integer `root_visits[4352]` and normalized
`visit_policy[4352]`, plus played ID/UCI, exploration and RNG metadata. The
stored visit policy is preserved exactly; temperature-selected actions never
replace it. Search targets are distributions and WDL targets are exact labels.
Samples also retain full checkpoint ID/SHA256/model configuration/contracts/
provenance, configuration hash, opening identity/source/prefix and search settings.
Weights are not loaded, searched or updated during finalization. The R01
checkpoint hash is a preserved source identity; R02 does not require its path to
remain available or authenticate arbitrary source claims against weight bytes.
`read_source(directory)` also verifies the R01 configuration/boundary envelopes.

## Duplicate admission and IDs

Policy `first-admitted-completed-trajectory-v1` is frozen in the buffer manifest.
The game key is the existing R01 SHA256 digest of `[initial_fen, moves_uci]`,
including the full opening and actual played continuation. The first audited
completed snapshot committed for that key supplies all training samples.
Later completed duplicates contribute zero additional samples, even with
different checkpoint/search/opening identities or different played-root coverage.
Their complete original snapshots remain provenance receipts. This deliberately
avoids overweighting repeated games; it does not average targets or union roots.
If the first record has zero played roots, later duplicates still contribute zero.
Different trajectories can share positions and are admitted independently.
The representative depends on transaction admission order, not filesystem paths
or a claim that separately ordered collections must choose the same source.

An audited source ID is SHA256 of the canonical configuration/game snapshot.
Identical ingestion is a no-op, including for excluded sources. A resumed
completion has a different source ID. Stable sample IDs are SHA256 of
`["r02-sample-v1", game_id, configuration_sha256, ply]`. They depend on the
admitted source configuration and replay location; retries and restart never
generate new IDs or overwrite existing targets. All IDs use the unchanged
canonical JSON `digest` from the existing match contract. Versioned contracts
include `chess128-v2`, `pair4352-v1`, `actual-occurrence-v1` and WDL ordering.

## Persistence and API

`ReplayBuffer.create(path)` exclusively creates a SQLite buffer. `ReplayBuffer(path)`
opens and audits it; missing or partially created/corrupt buffers are refused.
`ingest(snapshot)` returns source ID, disposition, repeat flag and added count.
`export()` returns audited samples, provenance receipts, manifest and counts.
Only the `samples` table is a training population. The `sources` table is the
audit ledger and may hold unfinished snapshots without outcome labels.

Every admission uses `BEGIN IMMEDIATE` with foreign keys and `synchronous=FULL`.
One transaction commits the source receipt, canonical game and all its samples.
Failure or process death before commit leaves no partial admission; death after
commit leaves the entire admission. A retry inspects persistent identities under
the writer transaction. No in-memory cursor or external checkpoint is needed.
Opening and read exports use one consistent transaction. Opening, export and ingestion
replay every stored source and independently regenerate expected admissions and
sample payloads, rejecting target corruption even if edited JSON is well formed.
An interleaved WAL writer fixture verifies that an opening audit retains one
consistent snapshot across a concurrent commit. Creation interruption may leave
an invalid file that is refused on reopen;
creation is exclusive and never automatically replaces an existing artifact.

The implementation is intentionally bounded: validation revisits all receipts
and roots on each operation and stores dense vectors plus source records. No
capacity/eviction/sampling schedule, optimized large replay loader, GPU execution,
power-loss/filesystem guarantee or learning update is claimed. Dependencies are
the accepted R01 model environment plus Python's bundled SQLite; no new package.

## Commands and evidence

From this checkout, create an explicitly new buffer:

```sh
PYTHONPATH=src:. /Users/arvindsrinivass/Desktop/RL-Chess/.venv/bin/python -B \
  scripts/run_r02.py --records runs/r01-integration/real/games.json \
  --buffer /absolute/path/new-replay.sqlite --create \
  --report /absolute/path/new-admission.json --samples /absolute/path/new-targets.jsonl
```

Restart with the same buffer and omit `--create`; output report/JSONL paths must
be new. Each archive entry is its own transaction; a later corrupt entry does
not undo earlier committed valid sources. Re-ingesting the whole archive resumes
safely from those receipts. Archives may have only the R01 evidence `fixture`
annotation in addition to configuration/game.

All **44 bounded R02 tests pass**. Independent expected chess outcomes cover
both winners and losers for White/Black played roots, both-color draws, opening
prefix/repetition encoding, soft visit distributions, normalized-but-inconsistent
policies, 26 source corruption cases, changed checkpoint duplicate provenance,
unfinished-to-completed admission, database/envelope corruption and no-overwrite
creation. Fresh subprocess retries preserve targets. SIGKILL after the first
sample insert and after complete commit both recover the expected whole state.
Controlled fixtures use the existing shared search and explicit dummy evaluator;
they are correctness evidence, not neural playing-strength evidence.

The full CPU model regression passes **1,497 tests**. The Torch-free adapter
regression passes **1,023 tests**, with **13 expected optional-Torch skips**.
No accepted adapter, search, evaluator, checkpoint or training module changed.

The preserved actual T03-checkpoint R01 archive audits all 20 sources:
7 canonical completed games, 1 completed duplicate, 11 excluded truncations and
1 excluded interruption. Exactly **4 training positions** result: winning roots
for White and Black plus draw roots for White and Black. Immediate terminal
completions add none. The original and accepted-main integration archives have
identical snapshots; every accepted-main re-ingestion is a repeat. A fresh-process
restart preserves the audited export SHA256
`4bdd6f656e159cfe8fb46db65264e876fdee5abe6abb1fc6f85bec50cf6eedf4`.

[Summary](../runs/r02/summary.json), [admission](../runs/r02/admission.json),
[restart](../runs/r02/restart.json), [accepted-main audit](../runs/r02/accepted-main-audit.json),
[sample targets](../runs/r02/samples.jsonl), the SQLite buffer and JUnit results
are committed for independent review. Preservation checks compare all accepted
R01/main tracked bytes, primary's exact current user patch/index, and the 76
indexed external artifacts. Existing documentation receives only this milestone
append; accepted artifacts remain byte-identical. Stop here for independent
review before R03.
