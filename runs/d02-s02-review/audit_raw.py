"""Selected-archive audit using Event boundaries, independent of D02 framing."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import chess.pgn
from chess_rl.adapter import make_state

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT/'data/manifests/elite-2025-01-v1.json').read_text())
PATHS = [Path('/private/tmp')/name for name in (
    'rl-chess-d02-prepared', 'rl-chess-d02-final-prepared', 'rl-chess-d02-accepted-prepared')]

def identity(path):
    with path.open('rb') as handle:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(handle, 'sha256').hexdigest())

def raw_games(handle):
    # This archive has one Event header for each source, independently censused.
    # Buffer one game, retain exact lines, strip only inter-source blank lines.
    lines = []
    offset = line_number = 0
    start = start_line = 0
    for raw in handle:
        line_number += 1
        if raw.startswith(b'[Event '):
            if lines:
                while lines and not lines[-1].strip():
                    lines.pop()
                yield start, start_line, b''.join(lines), len(lines)
            start, start_line, lines = offset, line_number, []
        lines.append(raw)
        offset += len(raw)
    if lines:
        while lines and not lines[-1].strip():
            lines.pop()
        yield start, start_line, b''.join(lines), len(lines)

archive = Path('/private/tmp/lichess_elite_2025-01.zip')
assert identity(archive) == dict(bytes=MANIFEST['archive']['downloaded_bytes'], sha256=MANIFEST['archive']['downloaded_sha256'])
comparisons = {}
for name in ('replays.jsonl', 'rejections.jsonl', 'summary.json'):
    values = [identity(path/name) for path in PATHS]
    assert all(value == values[0] for value in values)
    comparisons[name] = values[0]

count = plies = sampled = 0
member_hash = hashlib.sha256()
sampled_plies = 0
with zipfile.ZipFile(archive) as zipped, (PATHS[-1]/'replays.jsonl').open() as records:
    info, = zipped.infolist()
    assert info.filename == 'lichess_elite_2025-01.pgn'
    with zipped.open(info) as handle:
        def hashed_lines():
            for raw in handle:
                member_hash.update(raw)
                yield raw
        for start, start_line, raw, nlines in raw_games(hashed_lines()):
            count += 1
            row = json.loads(next(records))
            source = row['source']
            assert source['member_index'] == 1 and source['game_ordinal'] == count
            assert source['member'] == info.filename
            assert source['archive_sha256'] == MANIFEST['archive']['downloaded_sha256']
            assert (source['byte_start'], source['byte_end']) == (start, start+len(raw))
            assert (source['line_start'], source['line_end']) == (start_line, start_line+nlines-1)
            assert source['source_sha256'] == hashlib.sha256(raw).hexdigest()
            plies += len(row['moves_uci'])
            if count in (1, 289776) or count % 1000 == 0:
                game = chess.pgn.read_game(io.StringIO(raw.decode()))
                assert game is not None and not game.errors
                moves = [move.uci() for move in game.mainline_moves()]
                assert moves == row['moves_uci']
                # python-chess fills absent Seven Tag Roster fields. D02 correctly
                # preserves only declared metadata rather than fabricating Site.
                assert all(game.headers[key] == value for key, value in row['pgn_headers'].items())
                assert set(game.headers)-set(row['pgn_headers']) <= {'Event', 'Site', 'Date', 'Round', 'White', 'Black', 'Result'}
                state = make_state(row['start_fen'], moves, history_complete=True)
                assert state.board.fen(en_passant='fen') == game.end().board().fen(en_passant='fen')
                assert state.initial_fen == game.board().fen(en_passant='fen')
                assert state.history_complete
                sampled += 1
                sampled_plies += len(moves)
    assert next(records, None) is None

summary = json.loads((PATHS[-1]/'summary.json').read_text())
assert count == summary['counts']['accepted'] == 289776
assert plies == summary['counts']['plies'] == 25429063
assert member_hash.hexdigest() == summary['members'][0]['sha256']
print(json.dumps(dict(
    independently_framed_sources=count, total_stored_plies=plies,
    all_source_spans_and_hashes_match=True, archive_identity_matches_manifest=True,
    uncompressed_member_sha256=member_hash.hexdigest(),
    fresh_parser_and_adapter_sample_games=sampled, freshly_replayed_sample_plies=sampled_plies,
    sampling='first, last, and every 1000th ordinal',
    byte_identical_three_runs=comparisons,
    limitation='Fresh full-archive legal replay was not rerun; full source/provenance scan and 291 sampled game replays were rerun.'
), indent=2))
