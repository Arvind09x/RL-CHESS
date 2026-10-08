"""D03 consolidation of byte-verified D02 intake, before any position sampling.

D02 owns legal mainline replay. This stage validates its completed artifact
identities and record contract, normalizes identity, and retains every source.
No source result is a target. SQLite bounds memory while sorting bulk output.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import chess

from chess_rl.adapter import make_state
from chess_rl.data_manifest import IDENTITY_VERSION, STORAGE_VERSION
from chess_rl.pgn_replay import REPLAY_VERSION, RESULTS, file_identity, json_line

SPLIT_VERSION = "game-sha256-threshold-v1"
SPLITS = ("train", "validation", "test")
SPLIT_DESCRIPTION = (
    'game-sha256-threshold-v1: SHA256 of UTF-8 JSON {game_id,seed,split_version}; '
    'sorted keys, compact separators, ensure_ascii=true, no newline; full digest '
    'as unsigned big-endian integer x; train if 100*x < 90*2**256, validation '
    'if 100*x < 95*2**256, otherwise test; seed=20261008; assignment targets 90/5/5'
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def load_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def canonical_payload(start_fen, moves_uci):
    """Identity bytes for an already legally replayed actual-coordinate mainline.

    Validate the root through the adapter and UCI syntax here. Legal mainline
    attestation comes from the verified D02 summary/table, not this helper.
    """
    if not isinstance(start_fen, str) or len(start_fen.split()) != 6:
        raise ValueError("Identity requires a six-field starting FEN")
    root = make_state(start_fen, history_complete=True).board
    if not isinstance(moves_uci, list):
        raise ValueError("moves_uci must be an array")
    moves = []
    for uci in moves_uci:
        if not isinstance(uci, str):
            raise ValueError("Each UCI must be a string")
        move = chess.Move.from_uci(uci)
        if not move or move.drop is not None or move.uci() != uci:
            raise ValueError("Expected normalized standard actual UCI")
        moves.append(move.uci())
    return canonical_json(dict(identity_version=IDENTITY_VERSION,
                               start_fen=root.fen(en_passant="legal"),
                               moves_uci=moves))


def game_id(payload):
    return "g:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_digest(identity, seed):
    if not isinstance(identity, str) or not re.fullmatch(r"g:[0-9a-f]{64}", identity):
        raise ValueError("Malformed canonical game ID")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Split seed must be uint32")
    return hashlib.sha256(canonical_json(dict(game_id=identity, seed=seed,
                                             split_version=SPLIT_VERSION)).encode("utf-8")).hexdigest()


def assign_split(identity, seed):
    value = int(split_digest(identity, seed), 16)
    if 100 * value < 90 * 2**256:
        return "train"
    if 100 * value < 95 * 2**256:
        return "validation"
    return "test"


def source_metadata(record, manifest, members):
    required = {"record_version", "dataset_id", "encoding_version", "move_vocab_version",
                "ending_policy", "wdl_order", "start_fen", "moves_uci", "history_complete",
                "history_origin", "declared_setup", "source", "pgn_headers", "pgn_result"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("D02 record fields differ from supported contract")
    for key, expected in (("record_version", REPLAY_VERSION), ("history_complete", True),
                          ("history_origin", "declared-pgn-start")):
        if type(record[key]) is not type(expected) or record[key] != expected:
            raise ValueError(f"Unsupported {key}")
    for key in ("dataset_id", "encoding_version", "move_vocab_version", "ending_policy", "wdl_order"):
        expected = getattr(manifest, key)
        if record[key] != (list(expected) if key == "wdl_order" else expected):
            raise ValueError(f"Unsupported {key}")
    source = record["source"]
    if not isinstance(source, dict) or set(source) != {
        "archive_sha256", "member", "member_index", "game_ordinal", "byte_start",
        "byte_end", "line_start", "line_end", "source_sha256"
    }:
        raise ValueError("Unsupported source provenance")
    if source["archive_sha256"] != manifest.archive.downloaded_sha256:
        raise ValueError("Source archive identity mismatch")
    if (not isinstance(source["source_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", source["source_sha256"])):
        raise ValueError("Malformed source SHA-256")
    for key in ("member_index", "game_ordinal", "byte_start", "byte_end", "line_start", "line_end"):
        if type(source[key]) is not int or source[key] < (0 if key == "byte_start" else 1):
            raise ValueError(f"Malformed source {key}")
    member = members.get(source["member_index"])
    if (member is None or not member["pgn"] or source["member"] != member["name"]
            or not source["byte_start"] < source["byte_end"] <= member["bytes"]
            or source["line_start"] > source["line_end"]
            or source["game_ordinal"] > member["counts"]["games"]):
        raise ValueError("Source member/span/ordinal mismatch")
    headers, result = record["pgn_headers"], record["pgn_result"]
    if (type(record["declared_setup"]) is not bool or not isinstance(headers, dict)
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items())
            or not isinstance(result, dict)
            or set(result) != {"header", "movetext", "unfinished", "discrepancy"}):
        raise ValueError("Malformed source headers/result")
    if (result["header"] != headers.get("Result")
            or result["header"] is not None and result["header"] not in RESULTS
            or result["movetext"] not in RESULTS
            or type(result["unfinished"]) is not bool
            or result["unfinished"] != (result["movetext"] == "*")
            or type(result["discrepancy"]) is not bool
            or result["discrepancy"] != ("Result" in headers and headers["Result"] != result["movetext"])
            or record["declared_setup"] != (headers.get("SetUp") == "1")):
        raise ValueError("Inconsistent source result/setup metadata")
    return {key: record[key] for key in
            ("source", "start_fen", "declared_setup", "history_origin", "pgn_headers", "pgn_result")}


def verify_intake(replays, summary_path, expected_summary_sha256, manifest):
    summary_identity = file_identity(summary_path)
    if summary_identity["sha256"] != expected_summary_sha256:
        raise ValueError("Completed D02 summary SHA-256 mismatch")
    summary = load_json(Path(summary_path).read_text(encoding="utf-8"))
    if (summary.get("complete") is not True or summary.get("record_version") != REPLAY_VERSION
            or summary.get("dataset_id") != manifest.dataset_id
            or summary.get("archive") != {"sha256": manifest.archive.downloaded_sha256,
                                          "bytes": manifest.archive.downloaded_bytes}):
        raise ValueError("D02 summary completion/version/archive mismatch")
    identity = file_identity(replays)
    if identity != summary["outputs"]["replays.jsonl"]:
        raise ValueError("D02 replay table identity mismatch")
    return summary, {"summary.json": summary_identity, "replays.jsonl": identity}


def prepare_games(replays, summary_path, expected_summary_sha256, manifest, output_dir, *, progress=None):
    """Verify intake before processing; write complete, deterministically sorted tables.

    A new output directory is required. Any identity/payload conflict aborts,
    with no completed split audit. Originals are opened read-only. Duplicate
    source occurrences retain multiplicity; no provenance is silently dropped.
    """
    if manifest.split_seed != 20261008:
        raise ValueError("This split decision requires seed 20261008")
    if manifest.preparation.split_algorithm not in (None, SPLIT_DESCRIPTION):
        raise ValueError("Manifest split algorithm differs from D03")
    summary, inputs = verify_intake(replays, summary_path, expected_summary_sha256, manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    members = {m["member_index"]: m for m in summary["members"]}
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="rl-chess-d03-") as scratch:
        with sqlite3.connect(str(Path(scratch) / "consolidation.sqlite")) as db:
            db.execute("CREATE TABLE games (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE sources (game_id TEXT NOT NULL, metadata TEXT NOT NULL)")
            observed = hashlib.sha256()
            observed_bytes = 0
            with Path(replays).open("rb") as handle:
                for raw in handle:
                    observed.update(raw)
                    observed_bytes += len(raw)
                    record = load_json(raw)
                    metadata = source_metadata(record, manifest, members)
                    payload = canonical_payload(record["start_fen"], record["moves_uci"])
                    identity = game_id(payload)
                    prior = db.execute("SELECT payload FROM games WHERE id=?", (identity,)).fetchone()
                    if prior is not None and prior[0] != payload:
                        raise ValueError(f"Hash/payload conflict for {identity}")
                    if prior is None:
                        db.execute("INSERT INTO games VALUES (?, ?)", (identity, payload))
                    db.execute("INSERT INTO sources VALUES (?, ?)", (identity, canonical_json(metadata)))
                    counts["accepted"] += 1
                    counts["plies"] += len(record["moves_uci"])
                    counts["unfinished"] += record["pgn_result"]["unfinished"]
                    counts["result_discrepancies"] += record["pgn_result"]["discrepancy"]
                    counts["declared_setups"] += record["declared_setup"]
                    if counts["accepted"] % 10000 == 0:
                        db.commit()
                        if progress:
                            progress(counts["accepted"])
            if {"sha256": observed.hexdigest(), "bytes": observed_bytes} != inputs["replays.jsonl"]:
                raise ValueError("D02 replay bytes changed during processing")
            if any(counts[k] != summary["counts"][k] for k in
                   ("accepted", "plies", "unfinished", "result_discrepancies", "declared_setups")):
                raise ValueError("D02 replay counts differ from completed summary")
            db.execute("CREATE INDEX source_game ON sources(game_id, metadata)")
            db.commit()
            by_split = {s: Counter(games=0, sources=0, plies=0) for s in SPLITS}
            distinct = duplicate_groups = conflicting_results = canonical_plies = 0
            assignments = hashlib.sha256()
            with (out / "games.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for identity, payload in db.execute("SELECT id, payload FROM games ORDER BY id"):
                    replay = load_json(payload)
                    sources = [load_json(row[0]) for row in db.execute(
                        "SELECT metadata FROM sources WHERE game_id=? ORDER BY metadata", (identity,))]
                    results = sorted({v for source in sources for v in
                                      (source["pgn_result"]["header"], source["pgn_result"]["movetext"])
                                      if v is not None and v != "*"})
                    split = assign_split(identity, manifest.split_seed)
                    game = dict(record_version=STORAGE_VERSION, dataset_id=manifest.dataset_id,
                                encoding_version=manifest.encoding_version,
                                move_vocab_version=manifest.move_vocab_version,
                                ending_policy=manifest.ending_policy, wdl_order=list(manifest.wdl_order),
                                **replay, game_id=identity, split=split, history_complete=True,
                                history_origin="declared-pgn-start", sources=sources,
                                result_metadata=dict(observed_completed_results=results,
                                                     conflicting_results=len(results) > 1))
                    handle.write(json_line(game))
                    assignments.update(json_line(dict(game_id=identity, split=split)).encode("utf-8"))
                    distinct += 1
                    duplicate_groups += len(sources) > 1
                    conflicting_results += len(results) > 1
                    canonical_plies += len(replay["moves_uci"])
                    by_split[split].update(games=1, sources=len(sources), plies=len(replay["moves_uci"]))
    audit = dict(audit_version="canonical-split-audit-v1", complete=True,
                 dataset_id=manifest.dataset_id, record_version=STORAGE_VERSION,
                 identity_version=IDENTITY_VERSION, split_version=SPLIT_VERSION,
                 split_algorithm=SPLIT_DESCRIPTION, split_seed=manifest.split_seed,
                 assignment_target_percent=dict(zip(SPLITS, manifest.split_percent)),
                 inputs=inputs, archive=summary["archive"],
                 counts=dict(source_records=counts["accepted"], source_plies=counts["plies"],
                             canonical_games=distinct, canonical_plies=canonical_plies,
                             duplicate_sources=counts["accepted"] - distinct,
                             duplicate_game_groups=duplicate_groups,
                             games_with_conflicting_results=conflicting_results,
                             d02_rejected_sources=summary["counts"]["rejected"]),
                 actual_split_counts={s: dict(by_split[s]) for s in SPLITS},
                 assignments_sha256=assignments.hexdigest(),
                 assignments_serialization='game_id-sorted compact JSONL objects {game_id,split}; sorted keys, ensure_ascii=true',
                 outputs={"games.jsonl": file_identity(out / "games.jsonl")},
                 measurement_scope="canonical games and assignments before position sampling; no position-overlap or coverage audit")
    # Detect mutation of the source summary too; never attest changing intake.
    if file_identity(summary_path) != inputs["summary.json"]:
        raise ValueError("D02 summary changed during processing")
    (out / "split-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
