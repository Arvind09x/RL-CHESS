"""Offline source/replay tests; no archive download, engine or Torch required."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import zipfile

import chess
import chess.pgn
import numpy as np
import pytest

from chess_rl.adapter import encode_position, make_state, terminal_result, apply_move
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import (
    SourceError, SourceFrame, iter_frames, parse_mainline, prepare_archive,
    replay_record, replay_state, source_record, file_identity,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/d02_sources.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_offline_sources_and_round_trips(case):
    if case["reason"]:
        with pytest.raises(SourceError) as error:
            parse_mainline(case["pgn"])
        assert error.value.reason == case["reason"]
        return
    raw = case["pgn"].encode()
    frame = SourceFrame(raw, 150, 150 + len(raw), 10, 10 + raw.count(b"\n") - 1)
    source = source_record(frame, archive_sha256="a"*64, member="fixture.pgn", member_index=2, game_ordinal=7)
    manifest = DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-v1.json")
    record = replay_record(frame, source, manifest)
    assert record["source"] == source
    assert record["source"]["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (source["byte_start"], source["byte_end"], source["game_ordinal"], source["member_index"]) == (150, 150+len(raw), 7, 2)
    assert record["moves_uci"] == case["moves_uci"]
    assert not ({"game_id", "split", "sample_ply", "teacher", "wdl"} & record.keys())
    assert record["start_fen"] == (case["start_fen"] or chess.STARTING_FEN)
    assert record["history_complete"] is True
    state = replay_state(json.loads(json.dumps(record)))
    expected = make_state(case["start_fen"], case["moves_uci"], history_complete=True)
    assert state.initial_fen == expected.initial_fen
    assert state.board.move_stack == expected.board.move_stack
    assert state.board.fen(en_passant="fen") == expected.board.fen(en_passant="fen")
    assert state.history_complete
    np.testing.assert_array_equal(encode_position(state).features, encode_position(expected).features)
    # A fresh public adapter reconstruction reproduces every retained prefix.
    for ply in range(len(case["moves_uci"])+1):
        prefix = make_state(record["start_fen"], record["moves_uci"][:ply], history_complete=True)
        saved = state.board.copy(stack=True)
        while len(saved.move_stack) > ply:
            saved.pop()
        assert prefix.board.fen(en_passant="fen") == saved.fen(en_passant="fen")
    if case["name"] == "result_discrepancy":
        assert record["pgn_result"] == dict(header="1-0", movetext="0-1", unfinished=False, discrepancy=True)
    if case["name"] == "zero_moves_unfinished":
        assert record["pgn_result"] == dict(header=None, movetext="*", unfinished=True, discrepancy=False)


def test_source_replay_beyond_episode_draw():
    case = next(c for c in CASES if c["name"] == "long_repetition_beyond_episode_draw")
    state, _, _ = parse_mainline(case["pgn"])
    assert len(state.board.move_stack) == 49
    prefix = make_state(None, case["moves_uci"][:8])
    assert terminal_result(prefix) is not None
    with pytest.raises(ValueError):
        apply_move(prefix, chess.Move.from_uci("g1f3"))
    # Source legally continues after that draw and preserves all actual moves.
    assert state.board.move_stack[-1].uci() == "e2e4"


def test_python_chess_partial_and_skipped_tokens_never_accepted():
    partial = chess.pgn.read_game(io.StringIO("1. e4 e5 2. Bh6 *"))
    assert partial.errors and len(list(partial.mainline_moves())) == 2
    with pytest.raises(SourceError, match="illegal san"):
        parse_mainline("1. e4 e5 2. Bh6 *")
    skipped = chess.pgn.read_game(io.StringIO("1. e4 garbage e5 *"))
    assert not skipped.errors
    with pytest.raises(SourceError) as error:
        parse_mainline("1. e4 garbage e5 *")
    assert error.value.reason == "syntax"


def fixture_archive(tmp_path, bodies):
    path = tmp_path / "fixtures.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("one.pgn", bodies[0])
        archive.writestr("nested/two.pgn", bodies[1])
        archive.writestr("notice.txt", "source notice\n")
    original = DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-v1.json")
    identity = file_identity(path)
    manifest = replace(original, archive=replace(original.archive, downloaded_sha256=identity["sha256"],
                       downloaded_bytes=identity["bytes"], retrieved_at_utc="2026-10-08T00:00:00Z"))
    return path, manifest


def test_deterministic_all_members_and_accepted_rejected_accounting(tmp_path):
    body = '[Event "one"]\n[Result "*"]\n\n1. e4 e5 *\n\n[Event "bad"]\n\n1. e4 BAD e5 *\n\n[Event "three"]\n\n1. d4 d5 1/2-1/2\n'
    second = '[Event "unicode α"]\n\n1. Nf3 Nf6 *\n\n[Event "bad utf8"]\n\n1. e4 \xff *\n'
    archive, manifest = fixture_archive(tmp_path, [body, second.encode().replace('ÿ'.encode(), b'\xff')])
    reports = [prepare_archive(archive, manifest, tmp_path / name) for name in ("a", "b")]
    assert reports[0] == reports[1]
    for filename in ("replays.jsonl", "rejections.jsonl", "summary.json"):
        assert (tmp_path / "a" / filename).read_bytes() == (tmp_path / "b" / filename).read_bytes()
    report = reports[0]
    assert report["member_count"] == 3 and report["pgn_member_count"] == 2
    assert report["counts"]["games"] == 5
    assert report["counts"]["accepted"] == 3 and report["counts"]["rejected"] == 2
    assert report["rejection_reasons"] == {"encoding": 1, "syntax": 1}
    rows = [json.loads(line) for line in (tmp_path / "a/replays.jsonl").read_text().splitlines()]
    assert [r["source"]["game_ordinal"] for r in rows] == [1, 3, 1]
    with zipfile.ZipFile(archive) as zipped:
        for row in rows:
            source = row["source"]
            blob = zipped.read(source["member"])
            raw = blob[source["byte_start"]:source["byte_end"]]
            assert hashlib.sha256(raw).hexdigest() == source["source_sha256"]
            assert blob[:source["byte_start"]].count(b"\n")+1 == source["line_start"]
            assert replay_state(row).history_complete
        for member in report["members"]:
            assert hashlib.sha256(zipped.read(member["name"])).hexdigest() == member["sha256"]


def test_unrecoverable_comment_tail_never_invents_ordinals(tmp_path):
    body = '1. e4 {lost\n\n[Event "apparently another source"]\n\n1. d4 d5 *\n'
    archive, manifest = fixture_archive(tmp_path, [body, '*\n'])
    report = prepare_archive(archive, manifest, tmp_path / "result")
    assert report["counts"]["games"] == 2
    assert report["counts"]["accepted"] == 1 and report["counts"]["rejected"] == 1
    rejection = json.loads((tmp_path / "result/rejections.jsonl").read_text())
    assert rejection["remaining_boundaries_unknown"]
    assert rejection["source"]["byte_end"] == len(body.encode())


@pytest.mark.parametrize("body", ['1. e4 e5 *\n\n1. d4 d5 *\n', '1. e4 e5 *\n[Event "next"]\n\n1. d4 d5 *\n'])
def test_framing_headerless_and_no_blank_separator(body):
    frames = list(iter_frames(io.BytesIO(body.encode())))
    assert len(frames) == 2
    assert frames[1].byte_start == frames[0].byte_end + (1 if '\n\n' in body[:14] else 0)


def test_archive_identity_and_existing_output_fail_closed(tmp_path):
    archive, manifest = fixture_archive(tmp_path, ['*\n', '*\n'])
    bad = replace(manifest, archive=replace(manifest.archive, downloaded_sha256="0"*64))
    with pytest.raises(ValueError, match="Archive bytes"):
        prepare_archive(archive, bad, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    prepare_archive(archive, manifest, tmp_path / "result")
    with pytest.raises(FileExistsError):
        prepare_archive(archive, manifest, tmp_path / "result")


def test_adapter_rejects_unreplayable_saved_moves():
    with pytest.raises(ValueError, match="Illegal or null"):
        make_state(chess.STARTING_FEN, ['e2e4','e7e5','e4e6'], history_complete=True)
