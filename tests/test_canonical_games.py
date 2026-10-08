"""D03 identity, provenance, fail-closed intake and stable assignment checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import chess
import pytest

from chess_rl import canonical_games as d03
from chess_rl.adapter import make_state
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import SourceFrame, file_identity, json_line, replay_record

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest():
    return DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-v1.json")


def source(manifest, ordinal=1, body='1. e4 e5 2. Nf3 Nc6 1-0', headers='[Result "1-0"]'):
    text = f'[Event "test"]\n' + (headers + '\n' if headers else '') + f'\n{body}\n'
    raw = text.encode()
    provenance = dict(archive_sha256=manifest.archive.downloaded_sha256,
                      member="fixture.pgn", member_index=1, game_ordinal=ordinal,
                      byte_start=ordinal * 1000, byte_end=ordinal * 1000 + len(raw),
                      line_start=ordinal * 10, line_end=ordinal * 10 + 4,
                      source_sha256=hashlib.sha256(raw).hexdigest())
    return replay_record(SourceFrame(raw, 0, len(raw), 1, 5), provenance, manifest)


def intake(tmp_path, manifest, records, name="in"):
    directory = tmp_path / name
    directory.mkdir()
    replay = directory / "replays.jsonl"
    replay.write_text("".join(json_line(r) for r in records))
    counts = dict(accepted=len(records), rejected=0, games=max([r['source']['game_ordinal'] for r in records], default=0),
                  plies=sum(len(r['moves_uci']) for r in records),
                  unfinished=sum(r['pgn_result']['unfinished'] for r in records),
                  result_discrepancies=sum(r['pgn_result']['discrepancy'] for r in records),
                  declared_setups=sum(r['declared_setup'] for r in records))
    summary = directory / "summary.json"
    summary.write_text(json.dumps(dict(complete=True, record_version="pgn-mainline-presplit-v1",
        dataset_id=manifest.dataset_id,
        archive=dict(sha256=manifest.archive.downloaded_sha256, bytes=manifest.archive.downloaded_bytes),
        counts=counts, outputs={"replays.jsonl": file_identity(replay)},
        members=[dict(member_index=1, name="fixture.pgn", bytes=1000000, pgn=True, counts=counts)]), sort_keys=True))
    return replay, summary, file_identity(summary)["sha256"]


def prepare(tmp_path, manifest, records, name="run"):
    inputs = intake(tmp_path, manifest, records, name + "-in")
    out = tmp_path / name
    audit = d03.prepare_games(*inputs, manifest, out)
    games = [json.loads(line) for line in (out / "games.jsonl").read_text().splitlines()]
    return games, audit, out


def test_exact_identity_payload_and_known_hash():
    payload = d03.canonical_payload(chess.STARTING_FEN, ["e2e4", "e7e5"])
    expected = '{"identity_version":"start-moves-sha256-v1","moves_uci":["e2e4","e7e5"],"start_fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
    assert payload == expected
    assert d03.game_id(payload) == 'g:' + hashlib.sha256(expected.encode()).hexdigest()
    assert not payload.endswith('\n')


@pytest.mark.parametrize("field,value", [(4, "1"), (5, "2"), (1, "b"), (2, "-")])
def test_root_identity_distinctions(field, value):
    fields = chess.STARTING_FEN.split()
    fields[field] = value
    assert d03.game_id(d03.canonical_payload(' '.join(fields), [])) != d03.game_id(d03.canonical_payload(chess.STARTING_FEN, []))


def test_mainline_identity_retains_order_and_full_length():
    moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
    first = d03.game_id(d03.canonical_payload(chess.STARTING_FEN, moves))
    assert first != d03.game_id(d03.canonical_payload(chess.STARTING_FEN, moves * 3))
    assert first != d03.game_id(d03.canonical_payload(chess.STARTING_FEN, ['b1c3', 'b8c6', 'c3b1', 'c6b8']))


def test_en_passant_identity_legal_irrelevant_and_pinned():
    irrelevant = 'rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2'
    pinned = 'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 2'
    legal = 'k7/8/8/3pP3/8/8/8/4K3 w - d6 0 2'
    for fen in (irrelevant, pinned):
        assert d03.canonical_payload(fen, []) == d03.canonical_payload(fen.replace(fen.split()[3], '-'), [])
    assert d03.canonical_payload(legal, []) != d03.canonical_payload(legal.replace('d6', '-'), [])


@pytest.mark.parametrize("start,moves", [(chess.STARTING_FEN, ['0000']), (chess.STARTING_FEN, ['P@e4']),
    (chess.STARTING_FEN, ['e2e9']), (chess.STARTING_FEN, [1]), (chess.STARTING_FEN, 'e2e4'),
    (' '.join(chess.STARTING_FEN.split()[:4]), []), ('bad', [])])
def test_identity_rejects_malformed_inputs(start, moves):
    with pytest.raises(ValueError):
        d03.canonical_payload(start, moves)


def test_consolidate_preserves_every_source_and_result(tmp_path, manifest):
    a = source(manifest, 1, headers='[Result "1-0"]\n[Site "https://example.com/a"]\n[White "A"]')
    b = source(manifest, 3, body='1. e4 e5 2. Nf3 Nc6 0-1',
               headers='[Result "1/2-1/2"]\n[LichessURL "https://example.com/b"]\n[White "β"]')
    c = source(manifest, 4, body='1. e4 e5 2. Nf3 Nc6 *', headers='[Site "?"]')
    original = deepcopy([a, b, c, a])
    games, audit, _ = prepare(tmp_path, manifest, [a, b, c, a])
    assert [a, b, c, a] == original
    assert len(games) == 1
    game = games[0]
    assert game['record_version'] == 'replay-jsonl-v1'
    assert game['history_complete'] is True
    assert game['moves_uci'] == a['moves_uci']
    expected = [{k: r[k] for k in ('source', 'start_fen', 'declared_setup', 'history_origin', 'pgn_headers', 'pgn_result')} for r in original]
    assert game['sources'] == sorted(expected, key=d03.canonical_json)
    assert game['result_metadata'] == dict(observed_completed_results=['0-1', '1-0', '1/2-1/2'], conflicting_results=True)
    assert audit['counts']['duplicate_sources'] == 3
    assert audit['counts']['duplicate_game_groups'] == 1
    assert audit['counts']['games_with_conflicting_results'] == 1
    assert sum(v['sources'] for v in audit['actual_split_counts'].values()) == 4
    assert not any('teacher' in k or 'sample' in k for k in game)
    assert make_state(game['start_fen'], game['moves_uci'], history_complete=True).board.fen() == make_state(a['start_fen'], a['moves_uci'], history_complete=True).board.fen()


def test_setup_original_root_retained_after_ep_normalization(tmp_path, manifest):
    fen = 'rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2'
    a = source(manifest, body='2. Nf3 *', headers=f'[SetUp "1"]\n[FEN "{fen}"]')
    b = source(manifest, 2, body='2. Nf3 *', headers=f'[SetUp "1"]\n[FEN "{fen.replace("e6", "-")}"]')
    games, _, _ = prepare(tmp_path, manifest, [a, b])
    assert len(games) == 1
    assert games[0]['start_fen'] == fen.replace('e6', '-')
    assert {s['start_fen'] for s in games[0]['sources']} == {fen, fen.replace('e6', '-')}
    assert all(s['declared_setup'] for s in games[0]['sources'])


def test_repeated_and_reordered_preparation(tmp_path, manifest):
    records = [source(manifest, 1), source(manifest, 2, body='1. d4 d5 *', headers=''), source(manifest, 3)]
    _, first, a = prepare(tmp_path, manifest, records, 'a')
    _, second, b = prepare(tmp_path, manifest, records, 'b')
    _, reordered, c = prepare(tmp_path, manifest, list(reversed(records)), 'c')
    assert (a/'games.jsonl').read_bytes() == (b/'games.jsonl').read_bytes() == (c/'games.jsonl').read_bytes()
    assert (a/'split-audit.json').read_bytes() == (b/'split-audit.json').read_bytes()
    assert first['assignments_sha256'] == second['assignments_sha256'] == reordered['assignments_sha256']
    assert first['inputs'] != reordered['inputs']  # input-byte provenance changes with order


def test_expansion_stability(tmp_path, manifest):
    old = [source(manifest, n, body='*', headers=f'[SetUp "1"]\n[FEN "{chess.STARTING_FEN[:-1]}{n}"]') for n in range(1, 101)]
    previous, _, _ = prepare(tmp_path, manifest, old, 'old')
    assert {g['split'] for g in previous} == set(d03.SPLITS)
    expanded, _, _ = prepare(tmp_path, manifest, [source(manifest, 101, body='1. c4 e5 *', headers=''), *reversed(old), old[0]], 'expanded')
    assignments = {g['game_id']: g['split'] for g in expanded}
    assert all(assignments[g['game_id']] == g['split'] for g in previous)


def test_split_reference_and_targets():
    counts = {s: 0 for s in d03.SPLITS}
    for n in range(2000):
        identity = 'g:' + hashlib.sha256(str(n).encode()).hexdigest()
        raw = f'{{"game_id":"{identity}","seed":20261008,"split_version":"game-sha256-threshold-v1"}}'.encode()
        x = int.from_bytes(hashlib.sha256(raw).digest(), 'big')
        expected = 'train' if 100*x < 90*2**256 else 'validation' if 100*x < 95*2**256 else 'test'
        assert d03.assign_split(identity, 20261008) == expected
        counts[expected] += 1
    assert all(counts.values())
    assert counts != dict(train=1800, validation=100, test=100)


@pytest.mark.parametrize("numerator,expected", [(0, 'train'), (89, 'train'), (90, 'validation'), (94, 'validation'), (95, 'test'), (99, 'test')])
def test_split_threshold_boundaries(monkeypatch, numerator, expected):
    # Ceil the rational boundary: the first integer on/above its threshold.
    x = (numerator*2**256 + 99)//100
    monkeypatch.setattr(d03, 'split_digest', lambda *a: f'{x:064x}')
    assert d03.assign_split('g:' + 'a'*64, 20261008) == expected


def test_hash_payload_conflict_aborts(tmp_path, manifest, monkeypatch):
    inputs = intake(tmp_path, manifest, [source(manifest), source(manifest, 2, body='1. d4 d5 *', headers='')])
    monkeypatch.setattr(d03, 'game_id', lambda payload: 'g:' + '0'*64)
    with pytest.raises(ValueError, match='Hash/payload conflict'):
        d03.prepare_games(*inputs, manifest, tmp_path/'out')
    assert not (tmp_path/'out'/'split-audit.json').exists()


@pytest.mark.parametrize("mutation", ['summary_hash', 'replay_bytes', 'incomplete', 'archive', 'version', 'counts', 'record_version', 'history', 'source_archive', 'source_span', 'result', 'unknown', 'duplicate_key'])
def test_invalid_intake_fails_closed(tmp_path, manifest, mutation):
    record = source(manifest)
    if mutation == 'record_version': record['record_version'] = 'future'
    if mutation == 'history': record['history_complete'] = False
    if mutation == 'source_archive': record['source']['archive_sha256'] = '0'*64
    if mutation == 'source_span': record['source']['byte_end'] = 1
    if mutation == 'result': record['pgn_result']['unfinished'] = True
    if mutation == 'unknown': record['teacher_wdl'] = [1, 0, 0]
    replays, summary, digest = intake(tmp_path, manifest, [record])
    if mutation in ('incomplete', 'archive', 'version', 'counts'):
        data = json.loads(summary.read_text())
        if mutation == 'incomplete': data['complete'] = False
        if mutation == 'archive': data['archive']['sha256'] = '0'*64
        if mutation == 'version': data['record_version'] = 'future'
        if mutation == 'counts': data['counts']['accepted'] += 1
        summary.write_text(json.dumps(data))
        digest = file_identity(summary)['sha256']
    if mutation == 'summary_hash': digest = '0'*64
    if mutation == 'replay_bytes': replays.write_text(replays.read_text() + '\n')
    if mutation == 'duplicate_key':
        replays.write_text(replays.read_text().replace('{', '{"dataset_id":"other",', 1))
        data = json.loads(summary.read_text())
        data['outputs']['replays.jsonl'] = file_identity(replays)
        summary.write_text(json.dumps(data))
        digest = file_identity(summary)['sha256']
    with pytest.raises(ValueError):
        d03.prepare_games(replays, summary, digest, manifest, tmp_path/'out')
    assert not (tmp_path/'out'/'split-audit.json').exists()


def test_existing_directory_and_empty_intake(tmp_path, manifest):
    inputs = intake(tmp_path, manifest, [])
    out = tmp_path/'out'
    audit = d03.prepare_games(*inputs, manifest, out)
    assert audit['counts']['canonical_games'] == 0
    assert (out/'games.jsonl').read_bytes() == b''
    with pytest.raises(FileExistsError):
        d03.prepare_games(*inputs, manifest, out)


def test_import_boundary():
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', 'import chess_rl.canonical_games; import sys; assert not any(k == "torch" or k.startswith("chess_rl.model") or k.startswith("chess_rl.search") for k in sys.modules)'], check=True)
