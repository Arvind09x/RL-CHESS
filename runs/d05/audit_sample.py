"""Independently replay every saved successful label; no D05 worker imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import chess

from chess_rl.adapter import make_state, encode_position, legal_mask, move_to_id, id_to_move, terminal_result
from chess_rl.canonical_games import canonical_json, canonical_payload, game_id, assign_split
from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
settings = json.loads((out / "settings.json").read_text())
summary = json.loads((out / "summary.json").read_text())
manifest = json.loads((ROOT / settings["manifest"]).read_text())
assert summary["status"] == "complete" and not summary["pending_references"]
for name, expected in summary["output_identities"].items():
    assert file_identity(out / name) == expected
refs = [json.loads(raw) for raw in (out / "selected.jsonl").read_text().splitlines()]
labels = [json.loads(raw) for raw in (out / "labels.jsonl").read_text().splitlines()]
attempts = [json.loads(raw) for raw in (out / "attempts.jsonl").read_text().splitlines()]
assert len(refs) <= 12 and len({(r["game_id"], r["sample_ply"]) for r in refs}) == len(refs)
refs_by_key = {(r["game_id"], r["sample_ply"]): r for r in refs}
games_path = Path("/private/tmp/rl-chess-d03-prepared/games.jsonl")
assert file_identity(games_path) == settings["artifact_identities"]["games.jsonl"]
games = {}
with games_path.open() as handle:
    for raw in handle:
        g = json.loads(raw)
        if g["game_id"] in {r["game_id"] for r in refs}:
            assert g["game_id"] not in games
            games[g["game_id"]] = g
seen, coverage, checked = set(), Counter(), []
for label in labels:
    key = label["game_id"], label["sample_ply"]
    assert key not in seen
    seen.add(key)
    ref, game = refs_by_key[key], games[key[0]]
    assert {k: label[k] for k in ref if k != "record_version"} == {k: v for k, v in ref.items() if k != "record_version"}
    assert game_id(canonical_payload(game["start_fen"], game["moves_uci"])) == game["game_id"]
    assert assign_split(game["game_id"], manifest["split_seed"]) == game["split"] == label["split"] == "train"
    state = make_state(game["start_fen"], game["moves_uci"][:key[1]], history_complete=True)
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
    encoded = encode_position(state)
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
    assert label["requested_settings"] == manifest["teacher"]
    assert label["teacher_config_id"] == manifest["teacher"]["config_id"]
    assert label["target_provenance"]["executable_sha256"] == manifest["engine"]["executable_sha256"]
    for field in ("dataset_id", "encoding_version", "move_vocab_version", "ending_policy", "wdl_order"):
        assert label[field] == manifest[field]
    matching = [a for a in attempts if (a["reference"]["game_id"], a["reference"]["sample_ply"]) == key and a["status"] == "success"]
    assert len(matching) == 1 and matching[0]["attempt"] == label["successful_attempt"]
    returned = matching[0]["returned"]
    assert returned["completed"] and returned["hash_reset_completed"]
    assert returned["principal_response"] == info and returned["bestmove"] == best.uci()
    assert returned["effective_options"]["Threads"] == 1 and returned["effective_options"]["Hash"] == 128
    assert returned["effective_options"]["MultiPV"] == 1 and returned["effective_options"]["UCI_ShowWDL"] is True
    assert returned["effective_options"]["SyzygyPath"] == "" and returned["effective_options"]["UCI_Chess960"] is False
    if "uci_sent" in returned:
        sent = returned["uci_sent"]
        assert sent.index("setoption name Clear Hash") < sent.index("go nodes 100000")
        initial = state.board.root().fen(en_passant="fen")
        position = "position startpos" if initial == chess.STARTING_FEN else "position fen " + initial
        if state.board.move_stack:
            position += " moves " + " ".join(m.uci() for m in state.board.move_stack)
        assert position in sent and sent.index(position) < sent.index("go nodes 100000")
        assert sent[-1] == "quit" and returned["engine_exit_code"] == 0
        assert not returned.get("transport_errors")
    coverage[f"{actual_phase}:{'white' if state.board.turn else 'black'}"] += 1
    checked.append(dict(game_id=key[0], sample_ply=key[1], root_fen=label["root_fen"],
        features_sha256=hashlib.sha256(encoded.features.astype("<f4").tobytes()).hexdigest(),
        legal_mask_sha256=hashlib.sha256(legal_mask(state).tobytes()).hexdigest(),
        pv_plies=len(info["pv"]), returned_nodes=info["nodes"],
        score_bound={k: info[k] for k in ("lowerbound", "upperbound") if k in info}))
# Raw transport confirms Clear Hash before every go and full-history position commands.
log = (out / "uci-transport.log").read_text()
assert "ERROR" not in log and "WARNING" not in log
go_lines = [line for line in log.splitlines() if ": << go " in line]
hash_lines = [line for line in log.splitlines() if ": << setoption name Clear Hash" in line]
assert len(go_lines) == len(hash_lines) == len(attempts)
for clear, go in zip(hash_lines, go_lines):
    assert log.index(clear) < log.index(go) and "nodes 100000" in go
report = dict(successful_labels_replayed=len(checked), selected=len(refs), coverage=dict(coverage),
    adapter_and_direct_board_agree=True, independent_move_ids_agree=True,
    raw_hash_reset_and_node_commands_verified=True, all_labels_legal=True,
    record_checks=checked, sample_identities=summary["output_identities"])
(out / "independent-replay.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "record_checks"}, indent=2))
