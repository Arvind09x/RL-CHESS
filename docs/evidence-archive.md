# Historical evidence and artifact storage

Historical run evidence was copied into an external compressed archive and every archived file was checked against its original SHA-256. The compact inventory in [evidence-archive.json](evidence-archive.json) records the archive hash, size, source revisions, and file count. The archive and its full per-file manifest are held outside the repository in the project workspace's `external-artifacts/` folder.

The active `runs/` tree is removed from tracking. Its original files remain in the preserved local `artifacts-historical-runs/` directory, the verified external archive, and Git ancestors. This is a source-tree cleanup, not a history rewrite. Previously committed identifiers and paths remain in Git history.

The archive is named `gpu-readiness-run-evidence-20261010.tar.gz`; its detailed manifest is `gpu-readiness-run-evidence-20261010.tar.manifest.json`. Back up both alongside institutional project artifacts.

## Recovery from Git history

Restore into a fresh isolated directory first, from the repository root:

```sh
restore_dir=$(mktemp -d)
git archive 62af7ac runs/r01 | tar -x -C "$restore_dir"
# For the CUDA preparation campaign instead:
# git archive a97cf31 runs/cuda-support | tar -x -C "$restore_dir"
```

Both revisions are retained as ancestors. Select another campaign by changing
the subdirectory. Verify every restored file against the external per-file
manifest before using it; the manifest maps repository-relative paths to
`{"bytes": N, "sha256": "..."}`. For example (Python 3.12):

```sh
python - "$restore_dir" /path/to/gpu-readiness-run-evidence-20261010.tar.manifest.json <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
expected = {k: v for k, v in manifest.items() if k.startswith("runs/r01/")}
actual = {str(p.relative_to(root)): {"bytes": p.stat().st_size,
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in (root / "runs/r01").rglob("*") if p.is_file()}
assert actual == expected, "restored contents or membership mismatch"
print(f"verified {len(actual)} files")
PY
```

If the external manifest is unavailable, compare restored bytes with
`git show 62af7ac:<repository-relative-path>` for each file and compare membership
with `git ls-tree -r --name-only 62af7ac runs/r01`. The compressed archive itself
must match the SHA-256 and byte count in `evidence-archive.json` before extraction.
When ready, historical tools may need the restored campaign under ignored
`runs/`; use a separate historical checkout and preserve the isolated restore.

Original README, coding-plan, and CUDA preparation documentation are preserved
under `docs/history/`. Their links use their original locations and may need the
restored `runs/` tree or original Git revision when reading historical evidence.

## Future runs

Keep bulk datasets, weights, raw teacher traces, test XML, repeated command transcripts, and metrics under ignored `artifacts/` or external institutional storage. Keep one concise experiment record in `docs/` when a decision is accepted: revision, settings, input/checkpoint identities, environment, outcome, and pointers to the protected evidence.

Normal tests use small fixtures under `tests/fixtures/`; they do not require the historical run tree. The preserved provenance fixtures are exact copies of the accepted Stockfish and split identities, without raw machine logs.
