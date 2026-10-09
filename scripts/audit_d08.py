"""D08 independent full membership/label audit; no worker or selector imports."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import chess
import numpy as np

from chess_rl.adapter import make_state, encode_position, legal_mask, move_to_id, id_to_move, terminal_result
from chess_rl.canonical_games import canonical_json, canonical_payload, game_id, assign_split
from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[1]
audit_started = datetime.now(timezone.utc).isoformat()
audit_clock = time.perf_counter()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
out = args.output_dir
settings = json.loads((out / "settings.json").read_text())
summary = json.loads((out / "summary.json").read_text())
manifest = json.loads((ROOT / settings["manifest"]).read_text())
assert settings["selection_version"] == "d08-reference-sha256-v1" and settings["seed"] == 20261011
quotas = {"train": 9000, "validation": 500, "test": 500}
assert settings["max_attempts"] == 2 and settings["selected"] == 10000
assert settings["split_quotas"] == quotas and settings["no_refill"] is True
assert settings["frozen_at_utc"] <= summary["annotation_started_at_utc"]
for name, expected in {**summary["output_identities"], **summary["evidence_identities"]}.items():
    assert file_identity(out / name) == expected
refs = [json.loads(raw) for raw in (out / "selected.jsonl").read_text().splitlines()]
labels = [json.loads(raw) for raw in (out / "labels.jsonl").read_text().splitlines()]
attempts = [json.loads(raw) for raw in (out / "attempts.jsonl").read_text().splitlines()]
refs_by_key = {(r["game_id"], r["sample_ply"]): r for r in refs}
assert len(refs) == len(refs_by_key) == 10000
assert dict(Counter(r["split"] for r in refs)) == quotas
assert max(Counter(r["game_id"] for r in refs).values()) <= 8
assert all(settings["frozen_at_utc"] <= a["started_at_utc"] <= a["finished_at_utc"] for a in attempts)
games_path = Path(settings["games_path"])
assert file_identity(games_path) == settings["artifact_identities"]["games.jsonl"]
wanted = {r["game_id"] for r in refs}
games = {}
with games_path.open() as handle:
    for raw in handle:
        g = json.loads(raw)
        if g["game_id"] in wanted:
            assert g["game_id"] not in games
            games[g["game_id"]] = g
assert set(games) == wanted
# Independent boundary proof over every audited reference, without selector imports.
excluded = {tuple(k) for k in settings["excluded_prior_references"]}
prior = set()
for path, expected in settings["prior_membership_identities"].items():
    assert file_identity(Path(path)) == expected
    prior.update((r["game_id"], r["sample_ply"]) for r in (json.loads(x) for x in Path(path).read_text().splitlines()))
assert excluded == prior and len(excluded) == 1012 and not set(refs_by_key) & excluded
def independent_rank(ref):
    payload = json.dumps(dict(selection_version="d08-reference-sha256-v1", seed=20261011,
        split=ref["split"], game_id=ref["game_id"], sample_ply=ref["sample_ply"]),
        sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest(), ref["game_id"], ref["sample_ply"]
assert refs == [r for split in quotas for r in sorted((r for r in refs if r["split"] == split), key=independent_rank)]
boundaries = {s: max(independent_rank(r) for r in refs if r["split"] == s) for s in quotas}
found, eligible = set(), Counter()
prepared = Path(settings["prepared_path"])
with (prepared / "samples.jsonl").open() as handle:
    for raw in handle:
        r = json.loads(raw)
        k = r["game_id"], r["sample_ply"]
        if k in excluded:
            continue
        eligible[r["split"]] += 1
        if k in refs_by_key:
            assert r == refs_by_key[k] and k not in found
            found.add(k)
        else:
            assert independent_rank(r) > boundaries[r["split"]]
assert found == set(refs_by_key) and dict(eligible) == settings["eligible_by_split"]
for name in ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl", "overlap-audit.json"):
    assert file_identity(prepared / name) == settings["artifact_identities"][name]
positions = {s: {r["rule_position"] for r in refs if r["split"] == s} for s in quotas}
overlap = {a+"-"+b: len(positions[a] & positions[b]) for a,b in
           (("train","validation"),("train","test"),("validation","test"))}
assert not any(overlap.values())
assert {s: len(p) for s,p in positions.items()} == settings["distinct_rule_positions_by_split"]
held_games, protected_positions, protected_refs = {}, {}, set()
with (prepared / "heldout-games.jsonl").open() as handle:
    for raw in handle:
        row = json.loads(raw)
        if row["game_id"] in wanted:
            held_games[row["game_id"]] = row["split"]
with (prepared / "protected-heldout.jsonl").open() as handle:
    for raw in handle:
        row = json.loads(raw)
        assert row["rule_position"] not in positions["train"]
        protected_positions[row["rule_position"]] = row["split"]
        k = row["game_id"], row["sample_ply"]
        if k in refs_by_key:
            assert row == refs_by_key[k]
            protected_refs.add(k)
for k,r in refs_by_key.items():
    expected = r["split"] if r["split"] != "train" else None
    assert held_games.get(r["game_id"]) == protected_positions.get(r["rule_position"]) == expected
    if expected:
        assert k in protected_refs
exclusions = [json.loads(x) for x in (out / "exclusions.jsonl").read_text().splitlines()]
label_keys = {(r["game_id"], r["sample_ply"]) for r in labels}
excluded_keys = {(r["reference"]["game_id"], r["reference"]["sample_ply"]) for r in exclusions}
pending_keys = {(r["game_id"], r["sample_ply"]) for r in summary["pending_references"]}
assert len(label_keys) == len(labels) and len(excluded_keys) == len(exclusions)
assert not (label_keys & excluded_keys or label_keys & pending_keys or excluded_keys & pending_keys)
assert label_keys | excluded_keys | pending_keys == set(refs_by_key)
assert summary["pending_references"] == [r for r in refs if (r["game_id"],r["sample_ply"]) in pending_keys]
assert len(labels) == summary["successful_labels"] and len(exclusions) == summary["excluded"]
assert len(attempts) == summary["attempts"]
assert sum(a["status"] == "failed" for a in attempts) == summary["failed_attempts"]
assert sum(a["status"] == "interrupted" for a in attempts) == summary["interrupted_attempts"]
assert sum(a["attempt"] > 1 for a in attempts) == summary["retries"]
by_key = {k: [] for k in refs_by_key}
attempt_order, previous_key, previous_finish = [], None, settings["frozen_at_utc"]
for a in attempts:
    k = a["reference"]["game_id"], a["reference"]["sample_ply"]
    assert k in by_key and a["reference"] == refs_by_key[k]
    assert a["requested_settings"] == manifest["teacher"]
    if k != previous_key:
        assert not by_key[k]
        attempt_order.append(k)
    assert previous_finish <= a["started_at_utc"]
    previous_key, previous_finish = k, a["finished_at_utc"]
    by_key[k].append(a)
assert attempt_order == list(refs_by_key)[:len(attempt_order)]
assert [(r["game_id"], r["sample_ply"]) for r in labels] == [k for k in refs_by_key if k in label_keys]
assert [(r["reference"]["game_id"], r["reference"]["sample_ply"]) for r in exclusions] == [k for k in refs_by_key if k in excluded_keys]
for k,records in by_key.items():
    assert len(records) <= 2 and [a["attempt"] for a in records] == list(range(1, len(records)+1))
    assert all(a["status"] == "failed" for a in records[:-1])
    if k in label_keys:
        assert records and records[-1]["status"] == "success"
    elif k in excluded_keys:
        assert len(records) == 2 and records[-1]["status"] == "failed"
    elif records:
        assert records[-1]["status"] in ("interrupted", "failed", "success")
successful_counts = {s: sum(r["split"] == s for r in labels) for s in quotas}
assert successful_counts == summary["successful_by_split"]
assert {s: quotas[s]-successful_counts[s] for s in quotas} == summary["shortfall_by_split"]
assert (summary["status"] == "complete") == (successful_counts == quotas and not pending_keys and not excluded_keys)
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
    assert assign_split(game["game_id"], manifest["split_seed"]) == game["split"] == label["split"] == ref["split"]
    replay_clock = time.perf_counter()
    state = make_state(game["start_fen"], game["moves_uci"][:key[1]], history_complete=True)
    replay_times.append(time.perf_counter()-replay_clock)
    direct = chess.Board(game["start_fen"])
    for uci in game["moves_uci"][:key[1]]:
        direct.push_uci(uci)
    assert state.board.fen(en_passant="fen") == direct.fen(en_passant="fen") == label["root_fen"]
    assert state.board.promoted == direct.promoted and state.board.move_stack == direct.move_stack
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
    independent_mask = np.zeros(4352, dtype=np.bool_)
    for legal in direct.legal_moves:
        origin, target = legal.from_square, legal.to_square
        if not direct.turn:
            origin, target = origin ^ 56, target ^ 56
        identifier = (4096 + 4*(8*(origin % 8)+target % 8)
                      + (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT).index(legal.promotion)
                      if legal.promotion else 64*origin+target)
        independent_mask[identifier] = True
        assert move_to_id(state, legal) == identifier and id_to_move(state, identifier) == legal
    assert np.array_equal(mask, independent_mask)
    info = label["returned_search"]
    counts = info["wdl"]["relative"]
    if info["wdl"]["turn"] != ("white" if direct.turn else "black"):
        counts = counts[::-1]
    assert counts == label["teacher_wdl_counts"] and len(counts) == 3
    assert all(type(n) is int and n >= 0 for n in counts) and sum(counts) == 1000
    assert label["teacher_wdl"] == [n / 1000 for n in counts]
    assert info["pv"][0] == best.uci() == label["returned_bestmove_uci"]
    for uci in info["pv"]:
        pv_move = chess.Move.from_uci(uci)
        assert pv_move and pv_move in direct.legal_moves
        direct.push(pv_move)
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
    matching = [a for a in by_key[key] if a["status"] == "success"]
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
    coverage[f"{label['split']}:{actual_phase}:{'white' if state.board.turn else 'black'}"] += 1
    checked.append(dict(game_id=key[0], sample_ply=key[1], split=label["split"], root_fen=label["root_fen"],
        features_sha256=hashlib.sha256(encoded.features.astype("<f4").tobytes()).hexdigest(),
        legal_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        pv_plies=len(info["pv"]), returned_nodes=info["nodes"],
        replay_seconds=replay_times[-1], encoding_seconds=encode_times[-1],
        mask_seconds=mask_times[-1], complete_label_audit_seconds=audit_times[-1],
        raw_score_bound=raw_bounds,
        score_bound={k: info[k] for k in ("lowerbound", "upperbound") if k in info}))
    if len(checked) % 1000 == 0:
        print(json.dumps(dict(successful_labels_independently_audited=len(checked))), flush=True)
# Raw transport confirms Clear Hash before every go and full-history position commands.
log = (out / "uci-transport.log").read_text()
if summary["failed_attempts"] == 0:
    assert "ERROR" not in log and "WARNING" not in log
go_lines = [line for line in log.splitlines() if ": << go " in line]
hash_lines = [line for line in log.splitlines() if ": << setoption name Clear Hash" in line]
assert len(go_lines) <= len(hash_lines) <= len(attempts)
if summary["failed_attempts"] == summary["interrupted_attempts"] == 0:
    assert len(go_lines) == len(hash_lines) == len(attempts)
for clear, go in zip(hash_lines, go_lines):
    assert "nodes 100000" in go
assert dict(Counter(r["split"]+":"+r["phase"]+":"+r["rule_position"].split()[1] for r in refs)) == settings["coverage"]
assert file_identity(Path(settings["manifest"])) == settings["artifact_identities"]["manifest"]
assert file_identity(Path(settings["split_audit_path"])) == settings["artifact_identities"]["split-audit.json"]
report = dict(started_at_utc=audit_started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
    total_seconds=time.perf_counter()-audit_clock, setup_membership_join_seconds=join_seconds,
    replay_seconds=sum(replay_times), encoding_seconds=sum(encode_times), legal_mask_seconds=sum(mask_times),
    complete_label_audit_seconds=sum(audit_times), selection_boundary_proved=True, cross_split_rule_position_overlaps=overlap,
    successful_by_split=successful_counts, quota_reached=successful_counts == quotas,
    successful_labels_replayed=len(checked), selected=len(refs), coverage=dict(coverage),
    adapter_and_direct_board_agree=True, independent_move_ids_agree=True,
    raw_hash_reset_and_node_commands_verified=True, all_labels_legal=True,
    prior_1012_references_excluded=True, exact_heldout_registry_membership_verified=True,
    record_checks=checked, sample_identities=summary["output_identities"], bundle_summary_identity=file_identity(out / "summary.json"))
with (args.report or out / "independent-replay.json").open("x") as handle:
    handle.write(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "record_checks"}, indent=2))
