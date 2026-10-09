"""Summarize actual D08 annotation, audit, D07 loader cost and external storage."""
import argparse
from collections import Counter
import json
from pathlib import Path

from chess_rl.annotation_benchmark import distribution, read_rows
from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[1]


def run(out, destination):
    settings, summary, audit, loader = [json.loads((out / name).read_text()) for name in
        ("settings.json", "summary.json", "independent-replay.json", "loader-verification.json")]
    assert summary["status"] == "complete" and summary["successful_labels"] == 10000
    assert summary["successful_by_split"] == settings["split_quotas"]
    assert audit["quota_reached"] and audit["successful_labels_replayed"] == 10000
    assert loader["all_10000_individual_samples_and_three_batch_passes_verified"]
    labels = list(read_rows(out / "labels.jsonl"))
    flags = Counter()
    for row in audit["record_checks"]:
        flags.update({k: 1 for k, v in row["raw_score_bound"].items() if v})
    seconds = summary["annotation_loop_seconds"]
    artifacts = {p.name: dict(path=str(p.resolve()), **file_identity(p)) for p in sorted(out.iterdir()) if p.is_file()}
    production_command = json.loads((ROOT / "runs/d08/production.command.json").read_text())
    assert Path(production_command["argv"][-1]).resolve() == out.resolve()
    assert production_command["exit_code"] == 0
    freeze_command = json.loads((ROOT / "runs/d08/freeze.command.json").read_text())
    result = dict(scope="D08 only, automated checks complete; independent milestone review pending",
        bulk_directory=str(out.resolve()), successful_labels=10000, successful_by_split=summary["successful_by_split"],
        failed_attempts=summary["failed_attempts"], interrupted_attempts=summary["interrupted_attempts"],
        retries=summary["retries"], exclusions=summary["excluded"], pending=len(summary["pending_references"]),
        selected_identity=settings["selected_identity"], frozen_at_utc=settings["frozen_at_utc"],
        annotation_started_at_utc=summary["annotation_started_at_utc"], finished_at_utc=summary["finished_at_utc"],
        freeze_seconds=settings["freeze_seconds"], annotation_seconds=seconds,
        freeze_invocation_seconds=freeze_command["duration_seconds"],
        production_invocation_seconds=production_command["duration_seconds"],
        production_setup_and_command_overhead_seconds=production_command["duration_seconds"]-seconds-summary["post_input_audit_seconds"],
        annotation_successful_labels_per_second=10000/seconds, post_input_audit_seconds=summary["post_input_audit_seconds"],
        position_latency_seconds=summary["position_latency_seconds"],
        engine_reported_search_seconds=sum(r["returned_search"]["time"] for r in labels),
        engine_returned_nodes=sum(r["returned_search"]["nodes"] for r in labels),
        naturally_completed_below_node_request=sum(r["returned_search"]["nodes"] < 100000 for r in labels),
        returned_nodes=distribution([r["returned_search"]["nodes"] for r in labels]),
        raw_score_bound_flags=dict(flags), parsed_score_bound_flags={k: sum(bool(r["returned_search"].get(k)) for r in labels) for k in ("lowerbound", "upperbound")},
        coverage=audit["coverage"], distinct_games=settings["distinct_games"],
        maximum_references_per_game=settings["maximum_references_per_game"],
        distinct_rule_positions_by_split=settings["distinct_rule_positions_by_split"],
        cross_split_rule_position_overlaps=audit["cross_split_rule_position_overlaps"],
        independent_audit_seconds=audit["total_seconds"], independent_replay_seconds=audit["replay_seconds"],
        independent_encoding_seconds=audit["encoding_seconds"], independent_legal_mask_seconds=audit["legal_mask_seconds"],
        loader=loader, label_storage_bytes=artifacts["labels.jsonl"]["bytes"],
        full_production_storage_bytes=sum(v["bytes"] for v in artifacts.values()), artifacts=artifacts,
        reused_inputs=settings["artifact_identities"], production_source_identities=settings["production_source_identities"],
        limitations="One sequential CPU run on interactive macOS host; warm-file D07 costs include eager full-bundle validation on each split open; no cache, sustained concurrency, training, strength claim or expansion decision; external /private/tmp artifacts must be retained separately from Git")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: result[k] for k in ("successful_by_split", "annotation_seconds", "label_storage_bytes", "full_production_storage_bytes", "raw_score_bound_flags", "independent_audit_seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.report)
