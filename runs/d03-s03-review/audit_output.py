"""Full D03 byte/provenance audit with independently spelled identity/split rules.

No production canonicalization/split functions are imported. Replay a sampled
set through the accepted adapter; full legal mainlines are D02's attestation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import chess

from chess_rl.adapter import make_state

ROOT = Path(__file__).resolve().parents[2]
INTAKE = Path('/private/tmp/rl-chess-d02-accepted-prepared')
OUT = Path(sys.argv[1])


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


assert sha(INTAKE/'summary.json') == '494407e29f64046585151b91698778bb6ce3a0cfe849da0a252c200f85c2444b'
assert sha(INTAKE/'replays.jsonl') == '69c10231f4d569a6b3cb4b26bc22a9535346d68f0a4753bd83c0c92981ea950b'
assert sha(INTAKE/'rejections.jsonl') == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
assert (INTAKE/'summary.json').read_bytes() == (ROOT/'runs/d02/replay-summary.json').read_bytes()
audit = json.loads((OUT/'split-audit.json').read_text())
assert audit['complete'] is True
source_counts = Counter()
split_counts = {s: Counter(games=0, sources=0, plies=0) for s in ('train', 'validation', 'test')}
assignment_hash = hashlib.sha256()
sampled_replays = 0
with tempfile.TemporaryDirectory(prefix='d03-audit-') as scratch:
    with sqlite3.connect(str(Path(scratch)/'expected.sqlite')) as db:
        db.execute('CREATE TABLE expected (id TEXT PRIMARY KEY, payload TEXT)')
        db.execute('CREATE TABLE provenance (id TEXT, metadata TEXT)')
        with (INTAKE/'replays.jsonl').open() as f:
            for n, line in enumerate(f, 1):
                record = json.loads(line)
                payload = compact(dict(identity_version='start-moves-sha256-v1',
                    start_fen=chess.Board(record['start_fen']).fen(en_passant='legal'), moves_uci=record['moves_uci']))
                identity = 'g:' + hashlib.sha256(payload.encode()).hexdigest()
                prior = db.execute('SELECT payload FROM expected WHERE id=?', (identity,)).fetchone()
                assert prior is None or prior[0] == payload
                db.execute('INSERT OR IGNORE INTO expected VALUES (?, ?)', (identity, payload))
                metadata = {k: record[k] for k in ('source', 'start_fen', 'declared_setup', 'history_origin', 'pgn_headers', 'pgn_result')}
                db.execute('INSERT INTO provenance VALUES (?, ?)', (identity, compact(metadata)))
                source_counts['source_records'] += 1
                source_counts['source_plies'] += len(record['moves_uci'])
                if n % 10000 == 0:
                    db.commit()
        db.execute('CREATE INDEX provenance_id ON provenance(id, metadata)')
        db.commit()
        previous = ''
        observed = Counter()
        with (OUT/'games.jsonl').open() as f:
            for n, line in enumerate(f, 1):
                game = json.loads(line)
                assert line == compact(game) + '\n'
                identity = game['game_id']
                assert identity > previous
                previous = identity
                payload = compact({k: game[k] for k in ('identity_version', 'start_fen', 'moves_uci')})
                assert identity == 'g:' + hashlib.sha256(payload.encode()).hexdigest()
                assert db.execute('SELECT payload FROM expected WHERE id=?', (identity,)).fetchone() == (payload,)
                sources = [json.loads(row[0]) for row in db.execute('SELECT metadata FROM provenance WHERE id=? ORDER BY metadata', (identity,))]
                assert game['sources'] == sources
                assert game['record_version'] == 'replay-jsonl-v1'
                assert game['history_complete'] is True and game['history_origin'] == 'declared-pgn-start'
                manifest = json.loads((ROOT/'data/manifests/elite-2025-01-v1.json').read_text()) if n == 1 else manifest
                for k in ('dataset_id', 'encoding_version', 'move_vocab_version', 'ending_policy', 'wdl_order'):
                    assert game[k] == manifest[k]
                split_payload = compact(dict(game_id=identity, seed=20261008, split_version='game-sha256-threshold-v1'))
                x = int.from_bytes(hashlib.sha256(split_payload.encode()).digest(), 'big')
                split = 'train' if 100*x < 90*2**256 else 'validation' if 100*x < 95*2**256 else 'test'
                assert game['split'] == split
                assignment_hash.update((compact(dict(game_id=identity, split=split))+'\n').encode())
                split_counts[split].update(games=1, sources=len(sources), plies=len(game['moves_uci']))
                results = sorted({v for s in sources for v in (s['pgn_result']['header'], s['pgn_result']['movetext']) if v is not None and v != '*'})
                assert game['result_metadata'] == dict(observed_completed_results=results, conflicting_results=len(results)>1)
                observed.update(canonical_games=1, canonical_plies=len(game['moves_uci']),
                    duplicate_sources=len(sources)-1, duplicate_game_groups=int(len(sources)>1),
                    games_with_conflicting_results=int(len(results)>1))
                if n == 1 or n % 1000 == 0:
                    normalized = make_state(game['start_fen'], game['moves_uci'], history_complete=True)
                    for s in sources:
                        original = make_state(s['start_fen'], game['moves_uci'], history_complete=True)
                        assert original.board.fen(en_passant='legal') == normalized.board.fen(en_passant='legal')
                        assert [m.uci() for m in original.board.move_stack] == game['moves_uci']
                    sampled_replays += 1
        assert observed['canonical_games'] == db.execute('SELECT COUNT(*) FROM expected').fetchone()[0]
        source_counts.update(observed)
assert all(audit['counts'][k] == v for k, v in source_counts.items())
assert audit['actual_split_counts'] == {s: dict(c) for s, c in split_counts.items()}
assert audit['assignments_sha256'] == assignment_hash.hexdigest()
assert audit['outputs']['games.jsonl'] == dict(bytes=(OUT/'games.jsonl').stat().st_size, sha256=sha(OUT/'games.jsonl'))
manifest = json.loads((ROOT/'data/manifests/elite-2025-01-v1.json').read_text())
if manifest['preparation']['split_audit_sha256'] is not None:
    assert manifest['preparation']['split_audit_sha256'] == sha(OUT/'split-audit.json')
report = dict(all_source_provenance_occurrences_matched=True, identity_and_split_rules_matched=True,
    original_d02_bytes_preserved=True, sampled_adapter_replays=sampled_replays,
    counts=dict(source_counts), actual_split_counts=audit['actual_split_counts'],
    games_sha256=sha(OUT/'games.jsonl'), split_audit_sha256=sha(OUT/'split-audit.json'),
    assignments_sha256=assignment_hash.hexdigest(), full_legal_replay_rerun=False)
(ROOT/'runs/d03-s03-review/output-verification.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps(report, indent=2, sort_keys=True))
