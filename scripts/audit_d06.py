"""D06 independent full membership/label audit; no benchmark/worker imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import chess

from chess_rl.adapter import make_state, encode_position, legal_mask, move_to_id, id_to_move, terminal_result
from chess_rl.canonical_games import canonical_json, canonical_payload, game_id, assign_split
from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[1]
audit_started = datetime.now(timezone.utc).isoformat()
audit_clock = time.perf_counter()
out = Path(sys.argv[1])
settings = json.loads((out / "settings.json").read_text())
summary = json.loads((out / "summary.json").read_text())
manifest = json.loads((ROOT / settings["manifest"]).read_text())
assert summary["status"] == "complete" and not summary["pending_references"]
assert settings["frozen_at_utc"] <= summary["annotation_started_at_utc"]
assert settings["selection_version"] == "d06-rule-sha256-v1" and settings["seed"] == 20261010
assert settings["max_attempts"] == 2 and settings["selected"] == 1000
assert all(settings["frozen_at_utc"] <= a["started_at_utc"] for a in
    (json.loads(raw) for raw in (out / "attempts.jsonl").read_text().splitlines()))
for name, expected in summary["output_identities"].items():
    assert file_identity(out / name) == expected
refs = [json.loads(raw) for raw in (out / "selected.jsonl").read_text().splitlines()]
labels = [json.loads(raw) for raw in (out / "labels.jsonl").read_text().splitlines()]
attempts = [json.loads(raw) for raw in (out / "attempts.jsonl").read_text().splitlines()]
assert len(refs) == 1000 and len({(r["game_id"], r["sample_ply"]) for r in refs}) == len(refs)
refs_by_key = {(r["game_id"], r["sample_ply"]): r for r in refs}
games_path = Path(settings["games_path"])
assert file_identity(games_path) == settings["artifact_identities"]["games.jsonl"]
games = {}
with games_path.open() as handle:
    for raw in handle:
        g = json.loads(raw)
        if g["game_id"] in {r["game_id"] for r in refs}:
            assert g["game_id"] not in games
            games[g["game_id"]] = g
# Independently prove the selected set is exactly the lowest hash ranks.
# No heap/selector imports: test the boundary against EVERY audited train row.
excluded = {tuple(k) for k in settings["excluded_d05_references"]}
assert len(excluded) == 12
for name in ("tiny-real", "tiny-final"):
    saved = [json.loads(x) for x in (ROOT / "runs/d05" / name / "selected.jsonl").read_text().splitlines()]
    assert {(r["game_id"], r["sample_ply"]) for r in saved} == excluded
selected_positions = {r["rule_position"]: r for r in refs}
assert len(selected_positions) == 1000
assert not (set(refs_by_key) & excluded)
def independent_rank(position):
    payload = json.dumps(dict(selection_version="d06-rule-sha256-v1", seed=20261010,
        rule_position=position), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return int(hashlib.sha256(payload.encode()).hexdigest(), 16), position
assert refs == sorted(refs, key=lambda r: independent_rank(r["rule_position"]))
boundary = max(independent_rank(p) for p in selected_positions)
found, smallest, candidate_counts = set(), {}, Counter()
prepared = Path(settings["prepared_path"])
with (prepared / "samples.jsonl").open() as handle:
    for raw in handle:
        r = json.loads(raw)
        k = r["game_id"], r["sample_ply"]
        if r["split"] != "train" or k in excluded:
            continue
        candidate_counts[r["phase"]+":"+r["rule_position"].split()[1]] += 1
        pos = r["rule_position"]
        if pos in selected_positions:
            if pos not in smallest or k < smallest[pos]:
                smallest[pos] = k
            if r == selected_positions[pos]:
                found.add(k)
        else:
            assert independent_rank(pos) > boundary
assert found == set(refs_by_key)
assert all((r["game_id"], r["sample_ply"]) == smallest[pos] for pos, r in selected_positions.items())
assert dict(candidate_counts) == settings["eligible_reference_coverage"]
for name in ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl", "overlap-audit.json"):
    assert file_identity(prepared / name) == settings["artifact_identities"][name]
with (prepared / "protected-heldout.jsonl").open() as handle:
    assert all(json.loads(raw)["rule_position"] not in selected_positions for raw in handle)
with (prepared / "heldout-games.jsonl").open() as handle:
    assert all(json.loads(raw)["game_id"] not in games for raw in handle)
exclusions = [json.loads(x) for x in (out / "exclusions.jsonl").read_text().splitlines()]
label_keys = {(r["game_id"], r["sample_ply"]) for r in labels}
excluded_keys = {(r["reference"]["game_id"], r["reference"]["sample_ply"]) for r in exclusions}
assert not (label_keys & excluded_keys) and label_keys | excluded_keys == set(refs_by_key)
assert len(labels) == summary["successful_labels"] and len(exclusions) == summary["excluded"]
assert len(attempts) == summary["attempts"]
assert sum(a["status"] == "failed" for a in attempts) == summary["failed_attempts"]
assert sum(a["status"] == "interrupted" for a in attempts) == summary["interrupted_attempts"] == 0
assert sum(a["attempt"] > 1 for a in attempts) == summary["retries"]
for k in refs_by_key:
    records = [a for a in attempts if (a["reference"]["game_id"], a["reference"]["sample_ply"]) == k]
    assert 1 <= len(records) <= 2 and [a["attempt"] for a in records] == list(range(1, len(records)+1))
    assert all(a["reference"] == refs_by_key[k] for a in records)
    assert all(a["status"] == "failed" for a in records[:-1])
    assert records[-1]["status"] == ("success" if k in label_keys else "failed")
    if k in excluded_keys:
        assert len(records) == 2
join_seconds = time.perf_counter()-audit_clock
seen, coverage, checked = set(), Counter(), []
replay_times, encode_times, mask_times, audit_times = [], [], [], []
for label in labels:
    label_clock = time.perf_counter()
    key = label["game_id"], label["sample_ply"]
    assert key not in seen
    seen.add(key)
    ref, game = refs_by_key[key], games[key[0]]
    assert {k: label[k] for k in ref if k != "record_version"} == {k: v for k, v in ref.items() if k != "record_version"}
    assert game_id(canonical_payload(game["start_fen"], game["moves_uci"])) == game["game_id"]
    assert assign_split(game["game_id"], manifest["split_seed"]) == game["split"] == label["split"] == "train"
    replay_clock = time.perf_counter()
    state = make_state(game["start_fen"], game["moves_uci"][:key[1]], history_complete=True)
    replay_times.append(time.perf_counter()-replay_clock)
    direct = chess.Board(game["start_fen"])
    for uci in game["moves_uci"][:key[1]]:
        direct.push_uci(uci)
    assert state.board.fen(en_passant="fen") == direct.fen(en_passant="fen") == label["root_fen"]
    assert len(state.board.move_stack) == key[1] and state.history_complete is label["history_complete"] is True
    assert terminal_result(state) is None
    assert " ".join(direct.fen(en_passant="legal").split()[:4]) == ref["rule_position"]
    material = sum(len(direct.pieces(piece, color)) * value for piece, value in
                   ((chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)) for color in chess.COLORS)
    actual_phase = "endgame" if material <= 14 else "opening" if key[1] < 20 else "middlegame"
    assert actual_phase == label["phase"]
    encoding_clock = time.perf_counter()
    encoded = encode_position(state)
    encode_times.append(time.perf_counter()-encoding_clock)
    mask_clock = time.perf_counter()
    mask = legal_mask(state)
    mask_times.append(time.perf_counter()-mask_clock)
    assert encoded.available_snapshots == label["available_snapshots"] == min(8, key[1]+1)
    history = dict(start_fen=direct.root().fen(en_passant="fen"), moves_uci=[m.uci() for m in direct.move_stack], history_complete=True)
    assert hashlib.sha256(canonical_json(history).encode()).hexdigest() == label["replay_sha256"]
    best = chess.Move.from_uci(label["best_move_uci"])
    assert best in direct.legal_moves
    assert move_to_id(state, best) == label["best_move_id"] and id_to_move(state, label["best_move_id"]) == best
    # Independently spell the Black reflection and vocabulary calculation.
    origin, target = best.from_square, best.to_square
    if not direct.turn:
        origin, target = origin ^ 56, target ^ 56
    expected_id = (4096 + 4*(8*chess.square_file(origin)+chess.square_file(target))
                   + (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT).index(best.promotion)) if best.promotion else 64*origin+target
    assert expected_id == label["best_move_id"]
    info = label["returned_search"]
    counts = info["wdl"]["relative"]
    if info["wdl"]["turn"] != ("white" if direct.turn else "black"):
        counts = counts[::-1]
    assert counts == label["teacher_wdl_counts"] and len(counts) == 3
    assert all(type(n) is int and n >= 0 for n in counts) and sum(counts) == 1000
    assert label["teacher_wdl"] == [n / 1000 for n in counts]
    assert info["pv"][0] == best.uci() == label["returned_bestmove_uci"]
    for uci in info["pv"]:
        direct.push_uci(uci)
    assert info["tbhits"] == 0 and info["nodes"] > 0
    for field in ("depth", "seldepth", "nodes", "nps", "hashfull", "tbhits"):
        assert type(info[field]) is int and info[field] >= 0
    assert info["depth"] > 0 and info["time"] >= 0
    assert (info["score"]["relative_cp"] is None) != (info["score"]["relative_mate"] is None)
    assert label["requested_settings"] == manifest["teacher"]
    assert label["teacher_config_id"] == manifest["teacher"]["config_id"]
    assert label["target_provenance"]["executable_sha256"] == manifest["engine"]["executable_sha256"]
    for field in ("dataset_id", "encoding_version", "move_vocab_version", "ending_policy", "wdl_order"):
        assert label[field] == manifest[field]
    matching = [a for a in attempts if (a["reference"]["game_id"], a["reference"]["sample_ply"]) == key and a["status"] == "success"]
    assert len(matching) == 1 and matching[0]["attempt"] == label["successful_attempt"]
    returned = matching[0]["returned"]
    assert returned["completed"] and returned["hash_reset_completed"]
    final_rows = [r for r in returned["returned_rows"] if r.get("multipv") == 1]
    assert final_rows[-1] == info
    for name, source in (("manifest_sha256", "manifest"), ("games_sha256", "games.jsonl"),
                         ("samples_sha256", "samples.jsonl"), ("overlap_audit_sha256", "overlap-audit.json")):
        assert label["target_provenance"][name] == settings["artifact_identities"][source]["sha256"]
    assert label["target_provenance"]["embedded_nnue"] == settings["teacher"]["embedded_nnue"]
    assert returned["principal_response"] == info and returned["bestmove"] == best.uci()
    assert returned["effective_options"]["Threads"] == 1 and returned["effective_options"]["Hash"] == 128
    assert returned["effective_options"]["MultiPV"] == 1 and returned["effective_options"]["UCI_ShowWDL"] is True
    assert returned["effective_options"]["UCI_LimitStrength"] is False
    assert returned["effective_options"]["Ponder"] is False
    for name, asset in zip(("EvalFile", "EvalFileSmall"), manifest["engine"]["embedded_nnue"]):
        assert returned["effective_options"][name] == asset["name"]
    overridden = {"Threads": 1, "Hash": 128, "UCI_ShowWDL": True, "UCI_LimitStrength": False,
        "SyzygyPath": "", "MultiPV": 1, "Ponder": False, "UCI_Chess960": False}
    defaults = {k:v["default"] for k,v in returned["option_capabilities"].items() if v["type"] != "button"}
    assert returned["effective_options"] == {**defaults, **overridden}
    assert returned["engine_id"] == {"name": manifest["engine"]["uci_name"], "author": manifest["engine"]["uci_author"]}
    assert returned["effective_options"]["SyzygyPath"] == "" and returned["effective_options"]["UCI_Chess960"] is False
    if "uci_sent" in returned:
        sent = returned["uci_sent"]
        assert sent.index("setoption name Clear Hash") < sent.index("go nodes 100000")
        initial = state.board.root().fen(en_passant="fen")
        position = "position startpos" if initial == chess.STARTING_FEN else "position fen " + initial
        if state.board.move_stack:
            position += " moves " + " ".join(m.uci() for m in state.board.move_stack)
        assert position in sent and sent.index(position) < sent.index("go nodes 100000")
        raw_final = [line for line in returned["uci_received"] if line.startswith("info depth ") and " pv " in line][-1]
        tokens = raw_final.split()
        for name in ("depth", "seldepth", "nodes", "nps", "hashfull", "tbhits"):
            assert int(tokens[tokens.index(name)+1]) == info[name]
        assert int(tokens[tokens.index("time")+1])/1000 == info["time"]
        assert tokens[tokens.index("pv")+1:] == info["pv"]
        raw_wdl = list(map(int, tokens[tokens.index("wdl")+1:tokens.index("wdl")+4]))
        assert raw_wdl == label["teacher_wdl_counts"]
        score_index = tokens.index("score")
        score_kind = tokens[score_index+1]
        assert score_kind in ("cp", "mate")
        assert int(tokens[score_index+2]) == info["score"]["relative_"+score_kind]
        assert info["score"]["turn"] == ("white" if state.board.turn else "black")
        # Pinned python-chess recognizes bounds immediately after score, but
        # Stockfish can put them after WDL. Keep raw flags independently rather
        # than demanding the parsed dictionary contain an unparsed flag.
        raw_bounds = {bound: bound in tokens for bound in ("lowerbound", "upperbound")}
        for bound in raw_bounds:
            if info.get(bound):
                assert raw_bounds[bound]
        assert returned["uci_received"][-1].split()[1] == best.uci()
        assert sent.count("go nodes 100000") == sent.count("setoption name Clear Hash") == 1
        assert "stop" not in sent
        assert sent[-1] == "quit" and returned["engine_exit_code"] == 0
        assert not returned.get("transport_errors")
    audit_times.append(time.perf_counter()-label_clock)
    coverage[f"{actual_phase}:{'white' if state.board.turn else 'black'}"] += 1
    checked.append(dict(game_id=key[0], sample_ply=key[1], root_fen=label["root_fen"],
        features_sha256=hashlib.sha256(encoded.features.astype("<f4").tobytes()).hexdigest(),
        legal_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        pv_plies=len(info["pv"]), returned_nodes=info["nodes"],
        replay_seconds=replay_times[-1], encoding_seconds=encode_times[-1],
        mask_seconds=mask_times[-1], complete_label_audit_seconds=audit_times[-1],
        raw_score_bound=raw_bounds,
        score_bound={k: info[k] for k in ("lowerbound", "upperbound") if k in info}))
# Raw transport confirms Clear Hash before every go and full-history position commands.
log = (out / "uci-transport.log").read_text()
assert "ERROR" not in log and "WARNING" not in log
go_lines = [line for line in log.splitlines() if ": << go " in line]
hash_lines = [line for line in log.splitlines() if ": << setoption name Clear Hash" in line]
assert len(go_lines) == len(hash_lines) == len(attempts)
for clear, go in zip(hash_lines, go_lines):
    assert log.index(clear) < log.index(go) and "nodes 100000" in go
assert dict(Counter(r["phase"]+":"+r["rule_position"].split()[1] for r in refs)) == settings["coverage"]
assert file_identity(Path(settings["manifest"])) == settings["artifact_identities"]["manifest"]
assert file_identity(Path(settings["split_audit_path"])) == settings["artifact_identities"]["split-audit.json"]
report = dict(started_at_utc=audit_started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
    total_seconds=time.perf_counter()-audit_clock, setup_membership_join_seconds=join_seconds,
    replay_seconds=sum(replay_times), encoding_seconds=sum(encode_times), legal_mask_seconds=sum(mask_times),
    complete_label_audit_seconds=sum(audit_times), selection_boundary_proved=True,
    successful_labels_replayed=len(checked), selected=len(refs), coverage=dict(coverage),
    adapter_and_direct_board_agree=True, independent_move_ids_agree=True,
    raw_hash_reset_and_node_commands_verified=True, all_labels_legal=True,
    record_checks=checked, sample_identities=summary["output_identities"])
(out / "independent-replay.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "record_checks"}, indent=2))
