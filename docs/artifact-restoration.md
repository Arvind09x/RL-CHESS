# Backup, restoration and CPU artifact verification

Stage 1 is accepted at `042e9746e2b9948449b2ae69da708b6757fcad70`.
The prepared Stage 2 tar and source annex are repository-external **laptop**
storage, not a verified off-laptop backup. A second laptop directory is also only
a local copy. External backup destination, college access and persistent college
storage remain unconfirmed. No external backup, college restoration or Stage 2
completion is established by the local rehearsal described here.

Use a versioned release on an approved institutional service or removable medium
retained off-laptop. Confirm the destination and transfer authorization before
executing the backup steps. Preserve the existing tar, manifests, frozen reports
and Stage 1 history. Record new evidence separately under ignored `artifacts/`.

## 1. Identify the complete release

The prepared payload is `rl-chess-stage2-artifacts-20261011-v1.tar` (3,237,867,520
bytes). Extraction creates `transfer-bundle-v1/`. Its `inventory.json` identifies
2,762 payload files totaling 3,231,738,449 bytes; the inventory and `SHA256SUMS`
are additional files. These accepted anchors identify v1:

| Object | SHA-256 |
| --- | --- |
| Original v1 tar | `f4ce6e3c0df19b0a445e50d365d688ca7bd18d0ff4abd32c2a26fae8dfac3783` |
| Extracted `inventory.json` | `f092f04f760d01bb7bce141914b076a4f81ca547a109f4efd8b7406dd55bfd42` |
| Stage 1 annex `manifest.json` | `df53bc0c4bd5af617508171fc272325f09ac26076c4e859957578dd9d86f44a6` |
| Annex `source/rl-chess-source-042e974.bundle` | `ae94d3f1778b2d917979fb8d6c46cffd04becaffe3060f1d500bdc3fc94da3bf` |

Retain these hashes, and the new documentation source receipt described below,
independently of the backup (for example in a separately retained institutional
record). A checksum file stored beside the tar is useful, but is not an independent
anchor. Obtain hashes from that independent record when checking the destination
and restoration; do not recompute and trust a replacement hash from that destination.

The entire payload is required. Its layout identifies:

- `accepted-d08-original/`: the complete accepted D08 bundle from
  `bcb208d73271451b05d1409d4bf340aaad82eabf`: `attempts.jsonl`, `exclusions.jsonl`,
  `independent-replay.json`, `labels.jsonl`, `latencies.jsonl`,
  `loader-verification.json`, `selected.jsonl`, `settings.json`, `summary.json`,
  and `uci-transport.log`. Accepted quotas are 9,000 train, 500 validation and
  500 reserved test; no new held-out performance result is implied.
- `data/d08/`: relative `data-config.json`; `bundle/` containing attempts,
  exclusions, labels, selected references, settings and summary; `manifest.json`,
  `games.jsonl`, `split-audit.json`, `teacher-evidence.json`, and `prepared/`
  containing `samples.jsonl`, `overlap-audit.json`, `protected-heldout.jsonl`,
  `heldout-games.jsonl`, and `exclusions.jsonl`. These are the six production-loader
  inputs and their protected registries/teacher build attestation. Preserve
  `staging-evidence.json` too. CPU verification needs the attestation, not a
  Stockfish executable or a new teacher process.
- `data/d05/data-config.json`: relative wrapper for the original D05 tiny bundle
  in `historical-recovery/runs/d05/tiny-final/`, with the accepted shared
  manifest, canonical games, split/preparation registries and teacher evidence.
  Legacy T02/T03 checkpoints bind **12 D05 training positions**, not the 9,000
  D08 references. Inference on a D08 batch does not change that training identity.
- `checkpoints/`: retain each directory in full. T02 original and integrated
  `step-1.pt` have `warmup.json`, `resumed.json`, `uninterrupted.json`, and
  `summary.json`; T03 has `final.pt`, `execution-start.json`,
  `expected-reload.json`, `learning-curves.jsonl`, `reload-verification.json`,
  and `result.json`. R03 original/integrated `mixed-boundary.pt` and integrated
  `legacy-continuation/` are retained. R04 original `step-0.pt`, `step-1.pt`,
  `full-step-3.pt`, `updated-step-3.pt` and integrated `fresh-step-0.pt`,
  `fresh-step-1.pt`, `fresh-full-step-3.pt`, `fresh-resume-step-3.pt` are retained.
  Their remaining plans, receipts and sidecars live in the matching campaigns
  under `historical-recovery/runs/`.
- `data/r03-sources.json` and `data/r04-sources.json`: relative mixed-source
  wrappers. Mixed checkpoints additionally need their matching accepted replay
  membership/database and settings, not an arbitrary newly created database.
  R03 uses `historical-recovery/runs/r02-integration/fresh-replay.sqlite` and R04 uses
  `historical-recovery/runs/r04/replay.sqlite`; integrated/regenerated database
  copies and acceptance receipts remain in their corresponding campaigns.
  `verification/checkpoint-identities.json` catalogs checkpoint IDs, hashes,
  model configuration, original source configuration and replay identities.
- `historical/` and `historical-recovery/`: compressed historical archive,
  per-file manifest and recovered campaigns; preserve the corrected
  [Stage 1 Git recovery instructions](evidence-archive.md).
- `backend-diagnostics/` and `verification/`: keep unchanged for provenance.
  The catalog distinguishes 16 common-format checkpoints from four diagnostic
  tensor reports; the latter are not resumable checkpoints. Frozen inventory
  fields refer to earlier preparation revisions and pending documentation.
  New source/results belong in separate receipts, not edits to these records.

Frozen settings, manifests, checkpoint provenance and historical reports can
contain original absolute laptop paths. Preserve their bytes. Runtime wrappers
resolve paths relative to their configuration file and within the restored
payload. Do not rewrite frozen metadata or recreate symlinks to laptop artifacts.

## 2. Obtain exact source and prepare a source supplement

Install Git and `uv` on the restore host. Use Python 3.12.12 and the README's
locked model dependencies: `uv sync --locked --extra model`. Linux/Windows model
dependencies use the CPU wheel index in `uv.lock`; CUDA setup is separate.
Internet access or a populated locked dependency cache is required for setup;
the artifact/source backup alone does not include Python packages.

The prepared offline bundle in `stage1-final-annex-042e974/source/` is
self-contained and includes the exact accepted Stage 1 revision and ancestors:

```sh
git clone /path/to/verified/stage1-final-annex-042e974/source/rl-chess-source-042e974.bundle RL-Chess-stage1
git -C RL-Chess-stage1 checkout --detach 042e9746e2b9948449b2ae69da708b6757fcad70
git -C RL-Chess-stage1 rev-parse HEAD
git -C RL-Chess-stage1 bundle verify /path/to/verified/stage1-final-annex-042e974/source/rl-chess-source-042e974.bundle
```

That bundle predates this documentation fix and does not contain
`scripts/verify_restoration.py`. Retain the original annex and add a **new**
source supplement from the accepted documentation-fix checkout. The receipt
records its exact revision; do not assume a moving `main` or GitHub has it.
From that clean checkout, using a new local release directory:

```sh
release=/path/to/new/local/rl-chess-stage2-docs-release-v1
mkdir "$release"
git bundle create "$release/rl-chess-source-docs-fix.bundle" HEAD
git bundle verify "$release/rl-chess-source-docs-fix.bundle"
python3 - "$release" "$(git rev-parse HEAD)" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
bundle = root / 'rl-chess-source-docs-fix.bundle'
receipt = dict(source_revision=sys.argv[2], bundle=bundle.name,
               bytes=bundle.stat().st_size,
               sha256=hashlib.sha256(bundle.read_bytes()).hexdigest())
(root / 'source-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
PY
```

Retain the receipt independently. Copy the original v1 tar, its `.sha256`,
`inventory-anchor-final.json`, and the complete `stage1-final-annex-042e974/`
into this release without changing them. The new supplement belongs alongside
them, not inside the sealed tar/annex. This is still local preparation.

## 3. Future off-laptop backup and destination verification

After destination and transfer authorization are confirmed, copy the **whole
release** to a new version directory through the approved route. For an approved
mounted medium, an example is:

```sh
backup=/approved/off-laptop/storage/rl-chess-stage2-docs-release-v1
test ! -e "$backup"
cp -R "$release" "$backup"
```

For institutional services use their approved upload/download mechanism, with
the same version and no overwrite. Re-download to a fresh directory for
verification if the service does not expose file bytes locally. Check the tar
hash, source supplement hash/revision, annex manifest hash and every annex file
against independently retained anchors/receipt. Use this Python 3.12 command
on either a mounted destination or its independent download:

```sh
checked_release="$backup"
independent_source_receipt=/path/to/independently/retained/source-receipt.json
python3 - "$checked_release" "$independent_source_receipt" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def identity(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return dict(bytes=p.stat().st_size, sha256=h.hexdigest())
assert identity(root/'rl-chess-stage2-artifacts-20261011-v1.tar') == dict(
    bytes=3237867520, sha256='f4ce6e3c0df19b0a445e50d365d688ca7bd18d0ff4abd32c2a26fae8dfac3783')
receipt = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert json.loads((root/'source-receipt.json').read_text()) == receipt
assert identity(root/receipt['bundle']) == {k: receipt[k] for k in ('bytes','sha256')}
annex = root/'stage1-final-annex-042e974'
assert identity(annex/'manifest.json')['sha256'] == 'df53bc0c4bd5af617508171fc272325f09ac26076c4e859957578dd9d86f44a6'
for name, expected in json.loads((annex/'manifest.json').read_text())['files'].items():
    path = annex/name
    path.resolve().relative_to(annex.resolve())
    assert identity(path) == expected, name
print('release tar, source supplement and all Stage 1 annex files verified')
PY
```

Also extract the destination tar into a new verification directory and run the
hash checker in section 4 with `--location backup-destination`. Record storage
service/medium, version, host, source and destination paths, independently
retained anchors, commands, environment, identities and results. Only after
destination verification succeeds can that version be called a verified
off-laptop backup. Leave the original local files intact.

## 4. Restore solely from the verified backup and check on CPU

On the college host, obtain the versioned release **only from the verified
off-laptop backup**, into new persistent storage. Repeat section 3's release
verification with `checked_release` pointing to that restored release. Obtain
the independent source receipt separately. Restore exact source from the new
supplement (or fetch that exact receipt revision from an authorized source
repository if available):

```sh
source_revision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_revision"])' "$independent_source_receipt")
git clone "$checked_release/rl-chess-source-docs-fix.bundle" RL-Chess-restored
git -C RL-Chess-restored checkout --detach "$source_revision"
test "$(git -C RL-Chess-restored rev-parse HEAD)" = "$source_revision"
git -C RL-Chess-restored bundle verify "$checked_release/rl-chess-source-docs-fix.bundle"
cd RL-Chess-restored
```

For a **local restoration rehearsal**, set `checked_release` to the prepared
local release instead and label every result `local-rehearsal`. This tests the
same bytes and commands but does not satisfy either external/college gate.
Never reuse the laptop's original unpacked data; extract solely from the tar.

From the restored source checkout, run these commands in Bash. `evidence_dir`
must be new and outside the sealed payload. Record acquisition/backup commands
and section 3's output in that same evidence collection as well.

```bash
set -euo pipefail
evidence_dir="$PWD/artifacts/stage2-restore-001"
mkdir -p artifacts
mkdir "$evidence_dir"
exec > >(tee "$evidence_dir/commands.log") 2>&1
set -x
git rev-parse HEAD
git status --short
uv --version
uv sync --locked --extra model
uv run --locked --extra model python --version
uv pip freeze > "$evidence_dir/packages.txt"
restore_parent="$PWD/artifacts/restored-v1"
mkdir "$restore_parent"
tar -xf "$checked_release/rl-chess-stage2-artifacts-20261011-v1.tar" -C "$restore_parent"
root="$restore_parent/transfer-bundle-v1"
# Obtain this value from the independently retained v1 anchor above.
inventory_sha256=f092f04f760d01bb7bce141914b076a4f81ca547a109f4efd8b7406dd55bfd42
location=local-rehearsal
# Use college-restoration only on the college host after backup-only restoration.
# Use backup-destination for destination verification; omit --cpu if hash-only.
uv run --locked --extra model python scripts/verify_restoration.py "$root" \
  --inventory-sha256 "$inventory_sha256" --location "$location" \
  --report "$evidence_dir/restored-hashes.json"
uv run --locked --extra model python scripts/verify_restoration.py "$root" \
  --inventory-sha256 "$inventory_sha256" --location "$location" --cpu \
  --report "$evidence_dir/restored-cpu.json"
```

The checker verifies every inventoried restored file and exact file membership
before opening data or weights. It opens only `split="train"` through production
`open_dataset`, compares all 9,000 ordered unique references against the accepted
membership SHA-256
`e11bcc4f4941f1cb43a88eb72d65c67299a3001829661bcf526d1bf0b80b99fc`,
and constructs a batch of two (features `[2,64,128]`, legal mask `[2,4352]`).
The production loader internally validates the complete sealed bundle, including
other splits and protected registries, as required for integrity. The checker
does not directly inspect held-out targets or open a held-out dataset.

It loads restored `checkpoints/t03-tiny-overfit-20261009/final.pt` (SHA-256
`d4454a65793b1c21aaacbe7445f7682cd88220e5e1a46af5e11ea3d390beb13d`),
ID **T03-plumbing-step-26**, into its matching M06 model through shared
`load_inference`, expecting float32, relative attention, head widths 128 and
dropout 0.0. It performs finite CPU inference with policy `[2,4352]` and WDL
`[2,3]`, checking normalized legal policy/WDL probabilities. This checkpoint is
a correctness/plumbing artifact, **not selected P**.

`load_inference` reads and validates the common checkpoint format, including
stored training fields, and returns an eval model with inference weights. It
does **not** install optimizer, cursor or saved RNG state or establish training
continuation. Full training-state restoration requires the matching D05 source,
mixed replay where applicable, settings and runtime through `restore_training`;
that operation is outside this verification. No training, P selection, CUDA
checks, continuation or benchmarks are part of these commands.

Each JSON result records argv, source revision/status, lock and utility hashes,
host/environment, inventory identity and results. The CPU report also records
data configuration, exact training membership, batch, checkpoint identity,
matching model and inference results. Keep the shell transcript, package list,
independent anchors and source receipt together with these new reports. Report
local reproducibility separately from off-laptop backup and college restoration;
Stage 2 remains incomplete until the latter two gates have their own evidence.
