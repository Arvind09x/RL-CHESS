"""Run D05 only: at most 12 preselected audited training references."""
import argparse
import asyncio
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path

from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.stockfish_worker import (AnnotationWorker, engine_attempt,
    select_tiny, verify_artifacts, verify_teacher_bytes, PersistentEngine)

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    manifest = DatasetManifest.load(args.manifest)
    identities = verify_artifacts(manifest, args.manifest, args.games, args.split_audit, args.prepared)
    teacher = verify_teacher_bytes(manifest, args.executable, args.teacher_evidence)
    refs, games = select_tiny(args.prepared, args.games, manifest, identities)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    logging.basicConfig(filename=out / "uci-transport.log", level=logging.DEBUG,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    (out / "selected.jsonl").write_text("".join(json_line(r) for r in refs))
    settings = dict(manifest=str(args.manifest.resolve()), artifact_identities=identities,
        teacher=teacher, requested_settings=asdict(manifest.teacher), timeout_seconds=args.timeout,
        max_attempts=args.max_attempts, process_policy=args.engine_policy,
        selection="first two train references per material-ply-v1 phase/actual color in audited table order",
        no_refill=True, scope="D05 tiny sample; not D06 benchmark or D08 production")
    (out / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    pool = PersistentEngine(manifest, args.executable, args.timeout) if args.engine_policy == PersistentEngine.policy else None
    attempts, labels, exclusions = [], [], []
    status = "interrupted"
    with (out / "attempts.jsonl").open("x") as attempt_file, (out / "labels.jsonl").open("x") as label_file, (out / "exclusions.jsonl").open("x") as exclusion_file:
        def persist(handle, record):
            handle.write(json_line(record))
            handle.flush()
            os.fsync(handle.fileno())

        def sink(record):
            persist(attempt_file, record)
            attempts.append(record)

        async def attempt(state, trace):
            if pool is not None:
                return await pool.attempt(state, trace)
            # Fresh processes recheck bytes each attempt; the persistent owner
            # hashes on startup and checks file identity before each search.
            if file_identity(args.executable) != teacher["executable_identity"]:
                raise ValueError("Executable changed since verification")
            trace["process_policy"] = "fresh-process-v1"
            return await engine_attempt(state, manifest, args.executable, args.timeout, trace)

        worker = AnnotationWorker(manifest, dict(manifest_sha256=identities["manifest"]["sha256"],
            games_sha256=identities["games.jsonl"]["sha256"],
            samples_sha256=identities["samples.jsonl"]["sha256"],
            overlap_audit_sha256=identities["overlap-audit.json"]["sha256"],
            executable_sha256=teacher["executable_identity"]["sha256"],
            embedded_nnue=teacher["embedded_nnue"], process_policy=args.engine_policy), attempt, sink, max_attempts=args.max_attempts)
        try:
            for ref in refs:
                label = await worker.annotate(ref, games[ref["game_id"]])
                if label is not None:
                    persist(label_file, label)
                    labels.append(label)
                else:
                    exclusion = dict(reference=ref, reason="attempts_exhausted", attempts=args.max_attempts)
                    persist(exclusion_file, exclusion)
                    exclusions.append(exclusion)
            # Reverify all authoritative identities; no altered input becomes a completed run.
            if verify_artifacts(manifest, args.manifest, args.games, args.split_audit, args.prepared) != identities:
                raise ValueError("Input artifacts changed during annotation")
            status = "complete"
        finally:
            if pool is not None:
                await pool.close()
            finished = {(r["game_id"], r["sample_ply"]) for r in labels}
            finished.update((r["reference"]["game_id"], r["reference"]["sample_ply"]) for r in exclusions)
            summary = dict(status=status, selected=len(refs), successful_labels=len(labels),
                attempts=len(attempts), failed_attempts=sum(r["status"] != "success" for r in attempts),
                excluded=len(exclusions), pending_references=[r for r in refs if (r["game_id"], r["sample_ply"]) not in finished],
                output_identities={name: file_identity(out / name) for name in
                    ("selected.jsonl", "labels.jsonl", "attempts.jsonl", "exclusions.jsonl", "settings.json")})
            (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-evidence", type=Path, default=ROOT / "tests/fixtures/provenance/stockfish.json")
    parser.add_argument("--engine-policy", choices=("fresh-process-v1", "persistent-clear-hash-v1"), default="fresh-process-v1")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("timeout must be in (0,60]")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
