"""Audit and ingest bounded R01 snapshot archives; restart without rewriting."""
import argparse
import hashlib
import json
from pathlib import Path

from chess_rl.evaluation.matches import digest
from chess_rl.replay import ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, help="R01 games.json snapshot archive")
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--create", action="store_true", help="exclusively create a new buffer")
    parser.add_argument("--report", type=Path, help="write a new receipt; never overwrite")
    parser.add_argument("--samples", type=Path, help="export audited targets as new JSONL")
    args = parser.parse_args()
    buffer = ReplayBuffer.create(args.buffer) if args.create else ReplayBuffer(args.buffer)
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
        sample_labels=[dict(sample_id=s["sample_id"], ply=s["position"]["ply"],
            side_to_move=s["position"]["side_to_move"], wdl=s["wdl"], outcome=s["outcome"],
            checkpoint_id=s["checkpoint"]["checkpoint_id"]) for s in exported["samples"]])
    if args.report:
        with args.report.open("x") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.samples:
        with args.samples.open("x") as stream:
            for sample in exported["samples"]:
                stream.write(json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
