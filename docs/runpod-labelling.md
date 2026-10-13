# D09 label expansion on a rented CPU pod

Stockfish labelling is CPU-bound and scales linearly with cores at one thread
per worker. This runbook labels the frozen D09 membership on a 32-vCPU Linux
x86-64 pod and brings nested prefix bundles back to the Mac. A pod build of
Stockfish is a different binary from the manifest's reference build; it is
accepted through an attestation of the same release, source commit, source
archive and embedded networks, and its labels are checked for equivalence
against the accepted D08 labels on the 10,000 overlapping references.

## 0. On the Mac, before renting

```sh
cd ~/Desktop/RL-Chess
mkdir -p artifacts/data
ln -s "/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/external-artifacts/stage2-20261011-preparation/transfer-bundle-v1/data/d08" artifacts/data/d08
uv run --locked python scripts/expand_labels.py freeze \
  --data-config configs/data-d08.json --d08-bundle artifacts/data/d08/bundle \
  --output artifacts/data/d09 --train 1000000 --validation 10000 --test 10000 --per-game-cap 4
```

Freeze reads the 2.3M audited references, hashes every input (~2 GB) and
writes `artifacts/data/d09/selected.jsonl` + `plan.json`. It refuses a quota
above the post-cap capacity (1,040,950 train at cap 4).

Persistent-versus-fresh check on the Mac, 500 positions, before any
production run:

```sh
for policy in fresh-process-v1 persistent-clear-hash-v1; do
  uv run --locked python scripts/expand_labels.py freeze --data-config configs/data-d08.json \
    --output artifacts/data/equiv-$policy --train 500 --validation 0 --test 0 --without-d08
  uv run --locked python scripts/expand_labels.py run --output artifacts/data/equiv-$policy \
    --executable /private/tmp/Stockfish-sf_17.1/src/stockfish \
    --teacher-evidence artifacts/data/d08/teacher-evidence.json --workers 4 --engine-policy $policy
  uv run --locked python scripts/expand_labels.py finalize --output artifacts/data/equiv-$policy \
    --bundle artifacts/data/equiv-$policy/bundle
done
uv run --locked python scripts/expand_labels.py compare \
  --left artifacts/data/equiv-fresh-process-v1/bundle --right artifacts/data/equiv-persistent-clear-hash-v1/bundle \
  --output artifacts/data/equiv-compare.json
```

`compare` exits nonzero on any difference outside `returned_search.time`,
`returned_search.nps`, attempt timestamps and provenance. Adopt the persistent
policy only on a clean pass; the throughput printed by `run` is the measured
speed-up.

## 1. Pod setup (Ubuntu image, 32 vCPU)

```sh
apt-get update && apt-get install -y build-essential git curl rsync
curl -LsSf https://astral.sh/uv/install.sh | sh && . ~/.local/bin/env
mkdir -p /work && cd /work
```

Copy the repository (a git bundle or clone) and the data (~1.3 GB):

```sh
# from the Mac
cd ~/Desktop/RL-Chess && git bundle create /tmp/rl-chess.bundle main
rsync -av /tmp/rl-chess.bundle artifacts/data/d09/plan.json artifacts/data/d09/selected.jsonl POD:/work/
rsync -avL --exclude bundle/attempts.jsonl --exclude bundle/uci-transport.log \
  artifacts/data/d08/ POD:/work/data/d08/
# on the pod
cd /work && git clone rl-chess.bundle RL-Chess && cd RL-Chess
uv python install 3.12 && uv sync --locked
```

Labelling needs no Torch; `uv sync --locked` without `--extra model` is enough.
`-L` follows the Mac symlink. The D08 attempts ledger and transport log are
not needed on the pod (the overlap check reads `labels.jsonl`).

## 2. Build and attest Stockfish 17.1

```sh
cd /work
curl -L -o sf_17.1.tar.gz https://codeload.github.com/official-stockfish/Stockfish/tar.gz/refs/tags/sf_17.1
sha256sum sf_17.1.tar.gz   # must equal the manifest package_sha256: 0cfd9396...bec1f1
tar xzf sf_17.1.tar.gz && cd Stockfish-sf_17.1/src
# pick the ARCH for the pod CPU: `make help` lists them; x86-64-avx2 is safe on
# any modern x86; x86-64-vnni512 / x86-64-avx512 only if lscpu shows the flags.
make -j$(nproc) build ARCH=x86-64-avx2
./stockfish bench 2>&1 | tail -3      # records the bench signature; should match any 17.1 build
cd /work/RL-Chess
uv run python scripts/attest_stockfish.py --manifest /work/data/d08/manifest.json \
  --executable /work/Stockfish-sf_17.1/src/stockfish --source-archive /work/sf_17.1.tar.gz \
  --net-dir /work/Stockfish-sf_17.1/src --build-command "make -j$(nproc) build ARCH=x86-64-avx2" \
  --output /work/teacher-evidence-pod.json
```

The build downloads the two embedded networks into `src/`; the attestation
verifies their hashes against the manifest and locates them in the binary.

## 3. Run

The plan stores the Mac's absolute input paths; `--data-config` re-resolves
them on the pod and re-hashes the inputs against the frozen identities:

```sh
cd /work/RL-Chess && mkdir -p artifacts/data/d09 && cp /work/plan.json /work/selected.jsonl artifacts/data/d09/
cat > configs/data-pod.json <<'JSON'
{"version": "cuda-portable-data-v1", "paths": {"bundle": "/work/data/d08/bundle",
 "manifest_path": "/work/data/d08/manifest.json", "games_path": "/work/data/d08/games.jsonl",
 "split_audit_path": "/work/data/d08/split-audit.json", "prepared": "/work/data/d08/prepared",
 "teacher_evidence": "/work/teacher-evidence-pod.json"}}
JSON
nproc; lscpu | grep -E "Thread|Core|Model name"
nohup uv run python scripts/expand_labels.py run --output artifacts/data/d09 --data-config configs/data-pod.json \
  --executable /work/Stockfish-sf_17.1/src/stockfish --teacher-evidence /work/teacher-evidence-pod.json \
  --workers 16 --engine-policy persistent-clear-hash-v1 > /work/run.log 2>&1 &
tail -f /work/run.log
```

Workers = physical cores (vCPUs are usually hyperthread pairs; 32 vCPU is
typically 16 cores). Measure the first 10 minutes; if `labels_per_second`
scales, try 24. Memory per worker is ~250 MB (its games) plus the 128 MB
engine hash. The run is resumable: rerun the same command after any
interruption; it starts a new segment and skips completed references.

Order of work: the 10,000 D08 references, then validation to 10,000, test to
10,000, then train by rank. Held-out labels finish in the first minutes.

## 4. Prefix bundles while the run continues

```sh
P=/work/data/d08/bundle
uv run python scripts/expand_labels.py finalize --output artifacts/data/d09 --prior-bundle $P --bundle artifacts/data/d09/bundle-100k --train-prefix 100000
uv run python scripts/expand_labels.py finalize --output artifacts/data/d09 --prior-bundle $P --bundle artifacts/data/d09/bundle-300k --train-prefix 300000
uv run python scripts/expand_labels.py finalize --output artifacts/data/d09 --prior-bundle $P --bundle artifacts/data/d09/bundle-1m
```

Each finalize merges completed records, checks the 10k D08 overlap for label
equivalence (hard failure on mismatch), opens all three splits through the
production loader and writes `summary.json`. A bundle is five files plus
`teacher-evidence.json`; the 100k bundle is ~250 MB (labels ~2.5 KB, compact
attempts ~1.5 KB each).

Copy back and train:

```sh
# from the Mac
rsync -av POD:/work/RL-Chess/artifacts/data/d09/bundle-100k/ ~/Desktop/RL-Chess/artifacts/data/d09/bundle-100k/
cd ~/Desktop/RL-Chess && uv run --locked --extra model python scripts/train_supervised.py \
  --data-config configs/data-d09-100k.json --output artifacts/p-100k --device cpu --threads 4 \
  --steps 15625 --batch-size 32 --validation-every 500 --warmup-steps 500 --schedule-total-steps 15625
```

The data config points `teacher_evidence` inside the bundle, so the loader
verifies the pod build's attestation on the Mac without the pod binary.

## 5. Keep

Keep `artifacts/data/d09/plan.json`, `run.json`, `segments/` (the full ledger)
and the pod's `teacher-evidence-pod.json` with the bundles; they are the
provenance for every label. Loading the 1M bundle needs roughly 16 GB of RAM
(the loader holds the ledger in memory); do that on the GPU machine.
