# D07 supervised record loading and batching — 8 October 2026

Implemented on `codex/d07-record-loader` in the isolated local clone
`/private/tmp/rl-chess-d07`, from exact accepted base
`4184387c7e4b67559355d6b212a2a300c86c5ce6`. Independent review is pending;
main integration and D07 acceptance are not claimed. D05 is accepted and D07
requires no D06 benchmark. No new teacher annotation, losses, training or D08
work ran.

## Public API

`chess_rl.supervised` exposes `open_dataset`, `SupervisedDataset`,
`SupervisedSample` and `SupervisedBatch`. Use `open_dataset` as the factory:

```python
from pathlib import Path
from chess_rl.supervised import open_dataset

samples = open_dataset(
    bundle=Path("runs/d05/tiny-final"),
    manifest_path=Path("data/manifests/elite-2025-01-d04-v1.json"),
    games_path=Path("/private/tmp/rl-chess-d03-prepared/games.jsonl"),
    split_audit_path=Path("/private/tmp/rl-chess-d03-prepared/split-audit.json"),
    prepared=Path("/private/tmp/rl-chess-d04-prepared"),
    teacher_evidence=Path("runs/d01/stockfish-provenance.json"),
    split="train",
)

sample = samples[0]  # fresh full-prefix state, features, mask, class and soft WDL
for batch in samples.batches(5):  # default order; sizes 5, 5, 2 for this bundle
    features = batch.features
    targets = batch.policy_classes, batch.wdl
    # Explicit optional conversion, only in a Torch-equipped environment:
    # tensors = batch.to_torch(device="cpu")
```

All paths are explicit `Path` objects. `split` is required and accepts exactly
`train`, `validation` or `test`; there is no implicit split, combined split or
fallback. Training consumers must request `train`. Validation/test consumers
must deliberately select those splits. Every sample and batch retains its split
and `(game_id, sample_ply)` identities. The accepted D05 bundle has only training
labels; selecting either held-out split produces an empty dataset and no batches.
Controlled tests exercise actual separate nonempty split selections with sealed
synthetic fixtures; these are not new annotated data.

`len(samples)` returns the successful label count in that split. Integer indexing
supports Python's ordinary negative indices and raises `IndexError` out of range.
Each access replays a newly owned state from its canonical initial FEN and every
move in the prefix. Arrays and states can be mutated by consumers without changing
later accesses. The full retained history survives beyond the eight input snapshots.
`sample.policy_class` is a Python integer; `sample.wdl` is soft NumPy float32 `[3]`.

`batches(batch_size, *, order=None, drop_last=False)` yields batches. Default
order follows the label file within the selected split. An explicit `order` must
be a complete permutation of that dataset's integer indices; duplicate, missing,
foreign and boolean indices fail. No implicit shuffle or RNG is used. Final partial
batches are retained unless the caller explicitly sets `drop_last=True`. A batch
of one works; an empty dataset yields no batches. Arguments are checked when the
generator is iterated.

| Batch field | Shape | NumPy dtype | Meaning |
|---|---|---|---|
| `features` | `[B,64,128]` | `float32` | Shared `chess128-v2` adapter features |
| `legal_mask` | `[B,4352]` | `bool` | Shared `pair4352-v1` adapter legality |
| `policy_classes` | `[B]` | `int64` | Integer teacher move classes |
| `wdl` | `[B,3]` | `float32` | Soft root-side `[win,draw,loss]` targets |

`batch.to_torch(*, device="cpu")` explicitly imports Torch and returns a dictionary
with those four names. It preserves float32/bool/int64/float32 dtypes and creates
independent tensor storage. No device selection, one-hot conversion, model call
or loss calculation happens implicitly. Basic loading and batching use only the
pinned adapter environment; neither Torch nor a teacher executable is required.

## Validation and authoritative inputs

The caller supplies the accepted manifest, its completed split/overlap audits,
protected registries, canonical games and the accepted teacher build attestation.
Those accepted files are the trust roots. This reader checks their identity and
consistency; it does not recreate the archive/split audit or independently rebuild
the historical teacher binary. The executable and NNUE assets are not opened or
run. Their recorded SHA-256s, embedded asset metadata and build-attestation identity
must agree with the manifest and completed bundle. Relocation is possible because
identities, rather than the historical absolute paths in settings, bind artifacts.

Before exposing any sample, opening verifies:

- D05's annotation-ready manifest, canonical manifest bytes, pinned adapter
  versions, complete D03/D04 audit contracts, exact input/output hashes and
  protected registry identity, reusing `verify_artifacts`.
- Completed bundle status, empty pending list and exact identities for settings,
  selected membership, labels, attempts and exclusions. Selected references must
  be unique, match audited samples exactly, join unique canonical games and agree
  with game assignments and held-out game/position registries.
- Exact `stockfish-label-v1` fields, dataset/encoding/vocabulary/ending-policy/WDL
  versions, teacher settings/configuration, manifest/table/audit/teacher provenance.
- Canonical game identity and complete history; root FEN including counters,
  full-prefix replay checksum, real snapshot count, actual rule position and phase.
  Shared `terminal_result` rejects checkmates and adjudicated draws, including
  draws with legal moves. Current FEN alone is never used to reconstruct history.
- Legal best UCI and integer class ID, shared vocabulary round-trip, legal full
  teacher PV, strict nonnegative integer WDL counts totaling 1000, exact soft
  counts/1000 targets in root-side order, and completed principal search fields,
  reusing D05's `validate_response` after restoring serialized INFO types.
- Contiguous bounded attempt numbering, completed timestamps, one successful
  terminal attempt per label, matching reference/settings/principal response and
  both bestmove fields, completed analysis/hash-reset flags, final individual
  principal row, pinned UCI identity/effective options and no parser errors.
  When raw UCI capture exists, the full-history position command, reset-before-go,
  exact node request, quit and successful exit must agree. The earlier accepted
  D05 format lacks raw capture, so it uses the recorded completion/reset linkage.
- Exact counts and complete membership accounting. Exhausted references require
  their bounded failed ledger and explicit exclusion; exclusions never become
  targets. Interrupted, pending, orphaned, duplicated or mismatched records fail.

Every label in the bundle is checked before filtering for the requested split;
a malformed held-out label cannot be silently ignored by a training selection.
Inputs, bundle bytes, summary and teacher evidence are checked again after reading.
Failures raise explanatory `ValueError`s with bundle context and label line/key
where label replay/target validation fails. Duplicate JSON keys and nonfinite
values in relevant metadata/targets are rejected. PGN result metadata never
substitutes for the teacher target; a source game's result is not the bundle's
annotation completion status.

## Verification and limits

[Structured completion](../runs/d07/completion.json), command records and lossless
outputs under `runs/d07/` preserve actual checks and source hashes. Final results
are:

| Check | Result |
|---|---|
| Final D07 cases, model environment | 127 passed |
| Final D07 cases, adapter environment | 126 passed; one expected Torch skip |
| Full model regressions with accepted M06 diagnostic plugin | 1,111 passed |
| Full adapter regressions | 942 passed; seven expected optional-Torch skips |
| Accepted reviewer probes | 84 passed in each environment |
| Frozen adapter replays | Both identical accepted 13-fixture reports |
| Existing D05 final bundle | All 12 state/history/feature/mask/class/WDL checks agree |

The final `*-release` commands record the exact tested D07 source hashes and
confirm those sources remained unchanged during execution. Earlier development
and pre-final-guard checks are retained as historical evidence. The focused
D07 suite covers both colors, all four promotions,
both castling sides and en passant for both colors, long prefixes and repetition,
malformed shapes/types, incompatible versions, teacher/artifact provenance,
missing/duplicate joins, protected membership, successful-attempt linkage,
retry/exclusion accounting, input ownership, all batching choices and optional
Torch conversion.

`runs/d07/verify_sample.py` imports no D05 worker replay/target helpers. All 12
accepted final roots are independently replayed through direct python-chess and
fresh shared-adapter calls. Full FEN, promoted bits, retained moves, history hash,
features and masks agree. Feature/mask SHA-256s match the already accepted D05
independent replay. Independent test-only move-ID calculations enumerate masks
and check teacher classes and WDL perspective/counts. Batch sizes 1/5/12/20,
reverse order, drop-last and empty held-out selections are verified. There remain
12 distinct roots; no new teacher search or additional labeled membership exists.

No tensor/replay cache was added. Opening scans authoritative tables and validates
all label roots eagerly, retaining compact label records and only the selected
canonical games in memory; features/masks are derived on access. This is a small
CPU API, not a claim of scalable streaming, crash recovery, concurrency, GPU
performance or training adequacy. Large-run indexing/cache design and cost
measurement remain later work. Any future cache requires explicit versioning and
compatibility checks. Only D07 production code, tests, documentation and execution
evidence are added; accepted implementation/artifact bytes, both manifests,
protected held-out membership and primary main are preserved. Stop for independent
review before integration or dependent work.
