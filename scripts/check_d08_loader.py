"""Read/batch every D08 success through unchanged D07; no model or training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from chess_rl.pgn_replay import file_identity
from chess_rl.supervised import open_dataset

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(features, mask):
    return (hashlib.sha256(features.astype("<f4").tobytes()).hexdigest(),
            hashlib.sha256(mask.tobytes()).hexdigest())


def run(out, destination=None):
    started, full_clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    settings = json.loads((out / "settings.json").read_text())
    audit = json.loads((out / "independent-replay.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["successful_labels"] == 10000
    assert audit["successful_labels_replayed"] == 10000 and audit["quota_reached"]
    frozen = {(r["game_id"], r["sample_ply"]): r for r in audit["record_checks"]}
    labels = [json.loads(r) for r in (out / "labels.jsonl").read_text().splitlines()]
    args = dict(bundle=out, manifest_path=Path(settings["manifest"]), games_path=Path(settings["games_path"]),
                split_audit_path=Path(settings["split_audit_path"]), prepared=Path(settings["prepared_path"]),
                teacher_evidence=ROOT / "runs/d01/stockfish-provenance.json")
    reports = {}
    for split, quota in settings["split_quotas"].items():
        print(json.dumps(dict(split=split, phase="opening_and_validating_entire_bundle")), flush=True)
        clock = time.perf_counter()
        data = open_dataset(**args, split=split)
        open_seconds = time.perf_counter()-clock
        expected_labels = [r for r in labels if r["split"] == split]
        assert data.split == split and len(data) == len(expected_labels) == quota
        expected_refs = [(r["game_id"], r["sample_ply"]) for r in expected_labels]
        expected_targets = {k: r for k, r in zip(expected_refs, expected_labels)}
        clock = time.perf_counter()
        coverage = Counter()
        for i, label in enumerate(expected_labels):
            sample = data[i]
            assert sample.reference == expected_refs[i] and sample.split == split
            assert sample.state.history_complete and len(sample.state.board.move_stack) == label["sample_ply"]
            assert sample.state.board.fen(en_passant="fen") == label["root_fen"]
            assert sample.policy_class == label["best_move_id"] and sample.legal_mask[sample.policy_class]
            assert np.array_equal(sample.wdl, np.array(label["teacher_wdl"], dtype=np.float32))
            verified = frozen[sample.reference]
            assert verified["split"] == split
            assert fingerprint(sample.features, sample.legal_mask) == (verified["features_sha256"], verified["legal_mask_sha256"])
            coverage[label["phase"]+":"+("white" if sample.state.board.turn else "black")] += 1
        individual_seconds = time.perf_counter()-clock
        print(json.dumps(dict(split=split, individual_samples_verified=quota, open_seconds=open_seconds)), flush=True)
        batch_reports = []
        for size, reverse in ((1, False), (64, False), (127, True)):
            clock = time.perf_counter()
            seen, sizes = [], []
            order = list(reversed(range(quota))) if reverse else None
            for batch in data.batches(size, order=order):
                n = len(batch.references)
                assert batch.split == split
                assert batch.features.shape == (n, 64, 128) and batch.features.dtype == np.float32
                assert batch.legal_mask.shape == (n, 4352) and batch.legal_mask.dtype == np.bool_
                assert batch.policy_classes.shape == (n,) and batch.policy_classes.dtype == np.int64
                assert batch.wdl.shape == (n, 3) and batch.wdl.dtype == np.float32
                assert np.isfinite(batch.features).all() and np.isfinite(batch.wdl).all()
                for j, k in enumerate(batch.references):
                    label, verified = expected_targets[k], frozen[k]
                    assert int(batch.policy_classes[j]) == label["best_move_id"]
                    assert batch.legal_mask[j, batch.policy_classes[j]]
                    assert np.array_equal(batch.wdl[j], np.array(label["teacher_wdl"], dtype=np.float32))
                    assert fingerprint(batch.features[j], batch.legal_mask[j]) == (verified["features_sha256"], verified["legal_mask_sha256"])
                seen.extend(batch.references)
                sizes.append(n)
            assert seen == (expected_refs[::-1] if reverse else expected_refs)
            seconds = time.perf_counter()-clock
            batch_reports.append(dict(batch_size=size, reverse_order=reverse, samples=len(seen),
                batches=len(sizes), final_batch_size=sizes[-1], seconds=seconds, samples_per_second=quota/seconds))
            print(json.dumps(dict(split=split, **batch_reports[-1])), flush=True)
        reports[split] = dict(samples=quota, open_seconds=open_seconds, individual_seconds=individual_seconds,
                              coverage=dict(coverage), batch_checks=batch_reports)
        del data
        gc.collect()
        print(json.dumps(dict(split=split, **reports[split])), flush=True)
    report = dict(started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  total_seconds=time.perf_counter()-full_clock, splits=reports,
                  all_10000_individual_samples_and_three_batch_passes_verified=True,
                  independent_feature_mask_fingerprints_match=True, explicit_split_selection_verified=True,
                  torch_imported="torch" in sys.modules, bundle_summary_identity=file_identity(out / "summary.json"),
                  independent_audit_identity=file_identity(out / "independent-replay.json"),
                  limitations="CPU/warm-files only; each open eagerly validates the entire bundle; no cache, model, training or held-out tuning")
    with (destination or out / "loader-verification.json").open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.report)
