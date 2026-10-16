"""Audit and ingest self-play game archives into the replay buffer; restart without rewriting.

A .json archive is a list loaded whole (small fixture archives). A .jsonl
archive has one {configuration, game} per line and is ingested one game at a
time, so memory stays near one game; its report is built from receipts and
one streaming pass over the stored samples instead of a full export.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from chess_rl.evaluation.matches import digest
from chess_rl.replay import MANIFEST, ReplayBuffer


def _label(s):
    return dict(sample_id=s["sample_id"], ply=s["position"]["ply"], side_to_move=s["position"]["side_to_move"],
                wdl=s["wdl"], outcome=s["outcome"], checkpoint_id=s["checkpoint"]["checkpoint_id"])


def _publish(path, lines):
    """Write to a pending file, fsync, then link into place: a stop never leaves a partial file
    under the final name, and an existing file is never overwritten."""
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    try:
        with pending.open("x") as stream:
            for line in lines:
                stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(pending, path)
    finally:
        if pending.exists():
            pending.unlink()


def _file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def whole(args, buffer):
    records = json.loads(args.records.read_bytes())
    if not isinstance(records, list):
        raise ValueError("R01 archive must be a list")
    receipts = []
    for record in records:
        # fixture is an R01 evidence-archive label, not part of a source snapshot.
        if set(record) not in ({"configuration", "game"}, {"fixture", "configuration", "game"}):
            raise ValueError("unsupported R01 archive entry")
        receipts.append(dict(fixture=record.get("fixture"), **buffer.ingest(
            {k: record[k] for k in ("configuration", "game")})))
    exported = buffer.export()
    report = dict(records_sha256=hashlib.sha256(args.records.read_bytes()).hexdigest(),
        manifest=exported["manifest"], receipts=receipts, summary=exported["summary"],
        audited_export_sha256=digest(exported), samples_sha256=digest(exported["samples"]),
        sample_labels=[_label(s) for s in exported["samples"]])
    if args.samples:
        _publish(args.samples, (json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                                for sample in exported["samples"]))
    return report, report


def streaming(args, buffer):
    receipts = []
    with args.records.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if set(record) != {"configuration", "game"}:
                raise ValueError(f"unsupported archive entry on line {number}")
            receipts.append(dict(fixture=None, **buffer.ingest(record)))
    labels, hasher = [], hashlib.sha256()
    if args.samples and args.samples.exists():
        raise FileExistsError(args.samples)

    def sample_lines():
        for sample in buffer.iter_samples():
            labels.append(_label(sample))
            line = json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
            hasher.update(line.encode())
            yield line

    if args.samples:
        _publish(args.samples, sample_lines())
    else:
        for _ in sample_lines():
            pass
    report = dict(records_sha256=_file_sha256(args.records), manifest=copy.deepcopy(MANIFEST),
        receipts=receipts, summary=buffer.summary(), sample_labels=labels,
        samples_jsonl_sha256=hasher.hexdigest() if args.samples else None,
        streaming="one game per line; no full export, so export digests are omitted")
    printed = dict(records_sha256=report["records_sha256"], summary=report["summary"],
                   receipts=len(receipts), samples_jsonl_sha256=report["samples_jsonl_sha256"])
    return report, printed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True, help="games.json list or games.jsonl archive")
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--create", action="store_true", help="exclusively create a new buffer")
    parser.add_argument("--report", type=Path, help="write a new receipt; never overwrite")
    parser.add_argument("--samples", type=Path, help="export audited targets as new JSONL")
    args = parser.parse_args()
    buffer = ReplayBuffer.create(args.buffer) if args.create else ReplayBuffer(args.buffer)
    report, printed = (streaming if args.records.suffix == ".jsonl" else whole)(args, buffer)
    if args.report:
        _publish(args.report, [json.dumps(report, indent=2, sort_keys=True) + "\n"])
    print(json.dumps(printed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
