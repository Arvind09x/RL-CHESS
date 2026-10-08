"""Selection invariants and timing boundaries; no real engine search in tests."""
import logging
import json
from unittest.mock import patch

import pytest

from chess_rl.annotation_benchmark import TransportClock, distribution, key, select_references


def row(n, position=None, split="train"):
    return dict(game_id=f"g:{n:064x}", sample_ply=n, split=split,
                phase="opening", rule_position=position or f"position-{n} w - -")


def test_selection_order_representative_exclusions_and_distinctness():
    rows = [row(n) for n in range(30)] + [row(40, position=row(2)["rule_position"])]
    excluded = {key(row(2))}
    first, counts = select_references(rows, excluded, 10)
    reverse, _ = select_references(reversed(rows), excluded, 10)
    assert first == reverse and len(first) == 10
    assert len({r["rule_position"] for r in first}) == 10
    assert all(key(r) not in excluded for r in first)
    assert counts == {"opening:w": 30}
    # With full capacity a duplicate identity has the smallest eligible reference.
    all_refs, _ = select_references(rows + [row(41, split="test")], excluded, 30)
    assert row(40, position=row(2)["rule_position"]) in all_refs
    assert all(r["split"] == "train" for r in all_refs)
    all_refs, _ = select_references(rows, set(), 30)
    assert row(2) in all_refs and row(40, position=row(2)["rule_position"]) not in all_refs


@pytest.mark.parametrize("rows, count", [([], 1), ([row(1), row(2, position=row(1)["rule_position"])], 2), ([row(1, split="test")], 1), ([row(1)], 0)])
def test_insufficient_unique_train_capacity_fails(rows, count):
    with pytest.raises(ValueError):
        select_references(rows, set(), count)


def test_distribution_nearest_rank_and_empty():
    assert distribution([])["mean"] is None
    stats = distribution(list(range(1, 101)))
    assert (stats["p50"], stats["p95"], stats["p99"], stats["sum"]) == (50, 95, 99, 5050)


def test_transport_clock_boundaries_and_incomplete_attempt():
    with patch("chess_rl.annotation_benchmark.time.perf_counter", return_value=10):
        clock = TransportClock()
    for offset, direction, line in [(1, ">>", "uciok"), (2, "<<", "setoption name Clear Hash"), (3, ">>", "readyok"), (4, "<<", "go nodes 100000"), (6, ">>", "bestmove e2e4"), (7, "<<", "quit")]:
        with patch("chess_rl.annotation_benchmark.time.perf_counter", return_value=10+offset):
            clock.emit(logging.LogRecord("chess.engine", logging.DEBUG, "", 0, "%s: "+direction+" %s", ("protocol", line), None))
    metrics = clock.metrics(8)
    assert metrics["explicit_reset_to_ready_seconds"] == 1
    assert metrics["go_to_bestmove_seconds"] == 2
    assert metrics["bestmove_to_return_seconds"] == 2
    assert TransportClock().metrics(1)["go_to_bestmove_seconds"] is None


def test_projection_uses_success_denominator_and_keeps_fixed_cost_separate(tmp_path):
    from scripts.report_d06 import build_report
    def save(name, value):
        (tmp_path/name).write_text(json.dumps(value))
    summary = dict(status="complete", pending_references=[], selected=3, successful_labels=2,
        annotation_loop_seconds=20, successful_labels_per_second=.1, total_seconds=25,
        setup_seconds=3, post_input_audit_seconds=2, position_latency_seconds={"sum": 19},
        attempts=3, retries=0, failed_attempts=1, interrupted_attempts=0, excluded=1)
    save("summary.json", summary)
    save("settings.json", dict(coverage={}, requested_settings={"nodes": 100000}))
    save("independent-replay.json", dict(successful_labels_replayed=2, total_seconds=5,
        setup_membership_join_seconds=1, replay_seconds=1, encoding_seconds=2,
        legal_mask_seconds=.5, complete_label_audit_seconds=4))
    timing = dict(engine_attempt_seconds=5, startup_to_uciok_seconds=1,
        go_to_bestmove_seconds=2, events=[])
    attempt = dict(returned=dict(benchmark_timing=timing, executable_rehash_seconds=.25))
    (tmp_path/"attempts.jsonl").write_text((json.dumps(attempt)+"\n")*3)
    search = dict(depth=10, seldepth=12, nodes=100010, nps=50005, hashfull=1,
        tbhits=0, time=2, score=dict(relative_mate=None))
    (tmp_path/"labels.jsonl").write_text((json.dumps(dict(returned_search=search))+"\n")*2)
    for name in ("selected.jsonl", "latencies.jsonl", "exclusions.jsonl", "uci-transport.log"):
        (tmp_path/name).write_text("")
    report = build_report(tmp_path)
    ten_k = report["projections"][0]
    assert ten_k["annotation_worker_hours"] == 10000*10/3600
    assert ten_k["engine_reported_search_hours"] == 10000*2/3600
    assert ten_k["fixed_measured_setup_and_post_input_audit_seconds"] == 5
    assert ten_k["additional_encoding_hours"] == 10000/3600
    assert report["successful_labels"] == 2 and report["attempts"] == 3
    summary["status"] = "interrupted"
    save("summary.json", summary)
    with pytest.raises(ValueError, match="Incomplete"):
        build_report(tmp_path)
