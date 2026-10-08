"""Summarize completed D06 measurements and conditional linear projections."""
import argparse
import json
from pathlib import Path

from chess_rl.annotation_benchmark import distribution, read_rows
from chess_rl.pgn_replay import file_identity


def build_report(out):
    summary = json.loads((out / "summary.json").read_text())
    replay = json.loads((out / "independent-replay.json").read_text())
    settings = json.loads((out / "settings.json").read_text())
    if summary["status"] != "complete" or summary["pending_references"]:
        raise ValueError("Incomplete benchmark cannot support projections")
    n = summary["successful_labels"]
    if n == 0 or replay["successful_labels_replayed"] != n:
        raise ValueError("No fully audited successful labels")
    attempts = list(read_rows(out / "attempts.jsonl"))
    labels = list(read_rows(out / "labels.jsonl"))
    timing = [a["returned"]["benchmark_timing"] for a in attempts]
    components = {field: distribution([r[field] for r in timing if r[field] is not None]) for field in timing[0] if field != "events"}
    components["executable_rehash_seconds"] = distribution([a["returned"]["executable_rehash_seconds"] for a in attempts if "executable_rehash_seconds" in a["returned"]])
    # Observer starts before executable rehash: subtract it explicitly for startup.
    components["process_handshake_seconds"] = distribution([
        a["returned"]["benchmark_timing"]["startup_to_uciok_seconds"]-a["returned"]["executable_rehash_seconds"]
        for a in attempts if a["returned"]["benchmark_timing"]["startup_to_uciok_seconds"] is not None and "executable_rehash_seconds" in a["returned"]])
    nodes = sum(r["returned_search"]["nodes"] for r in labels)
    engine_time = sum(r["returned_search"]["time"] for r in labels)
    search_wall = components["go_to_bestmove_seconds"]["sum"]
    measured = dict(annotation_loop_seconds=summary["annotation_loop_seconds"],
        annotation_successful_labels_per_second=summary["successful_labels_per_second"],
        invocation_seconds=summary["total_seconds"], invocation_labels_per_second=n/summary["total_seconds"],
        setup_seconds=summary["setup_seconds"], post_input_audit_seconds=summary["post_input_audit_seconds"],
        independent_audit_seconds=replay["total_seconds"],
        independent_audit_setup_membership_join_seconds=replay["setup_membership_join_seconds"],
        audit_inclusive_labels_per_second=n/(summary["total_seconds"]+replay["total_seconds"]),
        position_latency_seconds=summary["position_latency_seconds"], attempt_components_seconds=components,
        worker_replay_validation_and_persistence_residual_seconds=summary["position_latency_seconds"]["sum"]-components["engine_attempt_seconds"]["sum"],
        engine_returned_seconds=engine_time, returned_nodes=nodes,
        successful_searches_per_engine_returned_second=n/engine_time if engine_time else None,
        returned_nodes_per_engine_returned_second=nodes/engine_time if engine_time else None,
        completed_searches_per_observed_search_wall_second=components["go_to_bestmove_seconds"]["count"]/search_wall if search_wall else None,
        returned_search={field: distribution([r["returned_search"][field] for r in labels]) for field in ("depth", "seldepth", "nodes", "nps", "hashfull", "tbhits", "time")},
        returned_nodes_below_request=sum(r["returned_search"]["nodes"] < settings["requested_settings"]["nodes"] for r in labels),
        returned_node_overshoot=distribution([r["returned_search"]["nodes"]-settings["requested_settings"]["nodes"] for r in labels]),
        mate_scores=sum(r["returned_search"]["score"]["relative_mate"] is not None for r in labels),
        score_bounds=sum(bool(r["returned_search"].get("lowerbound") or r["returned_search"].get("upperbound")) for r in labels),
        raw_score_bounds=sum(any(r["raw_score_bound"].values()) for r in replay.get("record_checks", [])),
        replay_seconds=replay["replay_seconds"], encoding_seconds=replay["encoding_seconds"],
        legal_mask_seconds=replay["legal_mask_seconds"], complete_label_audit_seconds=replay["complete_label_audit_seconds"])
    storage = {p.name: file_identity(p) for p in out.iterdir() if p.is_file()}
    scaling_files = ("labels.jsonl", "selected.jsonl", "attempts.jsonl", "latencies.jsonl", "exclusions.jsonl", "uci-transport.log", "independent-replay.json")
    retained_bytes_per_label = sum(storage[name]["bytes"] for name in scaling_files)/n
    projections = []
    for target in (10000, 100000, 300000, 500000, 1000000):
        projections.append(dict(successful_labels=target,
            annotation_worker_hours=target*summary["annotation_loop_seconds"]/n/3600,
            engine_reported_search_hours=target*engine_time/n/3600,
            additional_replay_hours=target*replay["replay_seconds"]/n/3600,
            additional_encoding_hours=target*replay["encoding_seconds"]/n/3600,
            additional_legal_mask_hours=target*replay["legal_mask_seconds"]/n/3600,
            complete_independent_label_audit_hours=target*replay["complete_label_audit_seconds"]/n/3600,
            compact_label_bytes=target*storage["labels.jsonl"]["bytes"]/n,
            full_evidence_bytes=target*retained_bytes_per_label,
            fixed_measured_setup_and_post_input_audit_seconds=summary["setup_seconds"]+summary["post_input_audit_seconds"],
            fixed_measured_independent_audit_setup_seconds=replay["setup_membership_join_seconds"]))
    return dict(scope="D06 only; automated completion, independent milestone review pending",
        selected=summary["selected"], successful_labels=n, attempts=summary["attempts"], retries=summary["retries"],
        failed_attempts=summary["failed_attempts"], interrupted_attempts=summary["interrupted_attempts"],
        excluded=summary["excluded"], pending=len(summary["pending_references"]), coverage=settings["coverage"],
        measured=measured, storage=storage, projections=projections,
        scaling_assumptions=[
            "Same sequential fresh-process D05 worker, teacher settings, host and comparable phase/history distribution; no concurrency speedup assumed",
            "Observed successful yield and retry rate continue; zero observed failures does not establish zero future failures",
            "Linear mean cost and uncompressed bytes per successful label; full evidence includes transport and independent audit records",
            "Fixed corpus scan/hash/join overhead shown separately; expansion audits or larger inputs can change it",
            "Independent label audit contains replay/encoding/mask time; these component costs must not be added twice",
            "Existing canonical games and audited reference tables are reused once, not serialized per label; new-corpus preparation is outside projections",
            "Encoding measured in a serial independent audit with games in memory and warm filesystem; no batch loader/cache/GPU/training costs measured"],
        limitations=["One CPU macOS arm64 run, 1000 distinct training rule positions from 999 games",
            "Uniform hash sampling of unique rule positions changes the phase mix relative to reference-weighted sampling",
            "Engine reported time is quantized milliseconds; observed transport wall includes parser/diagnostic work and engine initialization after go",
            "Startup/reset/teardown timings include host scheduling and transport observation; startup metric excludes executable rehash",
            "Pinned python-chess can omit score bounds placed after WDL; all raw flags are retained in attempt transport and independently reported alongside parsed bounds",
            "No hardware isolation, power/thermal measurement, sustained scale or concurrency validation",
            "Hard-crash recovery/resume are unimplemented; partial bundles cannot be consumed as complete datasets",
            "No final expansion size selected: require subsequent held-out learning curves and training/resource evidence",
            "D07, D08, training, search integration and GPU experiments were not run"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.output_dir)
    with args.report.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("storage", "measured")}, indent=2))


if __name__ == "__main__":
    main()
