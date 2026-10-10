# Historical evidence and artifact storage

Historical run evidence was copied into an external compressed archive and every archived file was checked against its original SHA-256. The compact inventory in [evidence-archive.json](evidence-archive.json) records the archive hash, size, source revisions, and file count. The archive and its full per-file manifest are held outside the repository in the project workspace's `external-artifacts/` folder.

The active `runs/` tree is removed from tracking. Its original files remain in the preserved local `artifacts-historical-runs/` directory, the verified external archive, and Git ancestors. This is a source-tree cleanup, not a history rewrite. Previously committed identifiers and paths remain in Git history.

The archive is named `gpu-readiness-run-evidence-20261010.tar.gz`; its detailed manifest is `gpu-readiness-run-evidence-20261010.tar.manifest.json`. Back up both alongside institutional project artifacts.

## Recovery from Git history

The original main evidence can be recovered into the ignored `runs/` directory:

```sh
git archive 62af7ac runs | tar -x
git archive a97cf31 runs/cuda-support | tar -x
```

The second revision is retained as a merge parent. For a specific accepted campaign, select its subdirectory in the archive command. Original README, coding-plan, and CUDA preparation documentation are preserved under `docs/history/`. Their links use their original locations and may need the restored `runs/` tree or original Git revision when reading historical evidence.

## Future runs

Keep bulk datasets, weights, raw teacher traces, test XML, repeated command transcripts, and metrics under ignored `artifacts/` or external institutional storage. Keep one concise experiment record in `docs/` when a decision is accepted: revision, settings, input/checkpoint identities, environment, outcome, and pointers to the protected evidence.

Normal tests use small fixtures under `tests/fixtures/`; they do not require the historical run tree. The preserved provenance fixtures are exact copies of the accepted Stockfish and split identities, without raw machine logs.
