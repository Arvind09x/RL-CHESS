"""Independent R02 probes: hand-built records, explicit chess expectations, real receipts."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier

import chess
import numpy as np
import pytest

from chess_rl.adapter import encode_position, make_state
from chess_rl.replay import ReplayBuffer, finalize

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('REVIEW_OUTPUT', Path(__file__).resolve().parent))
BLACK_WIN = ['f2f3', 'e7e5', 'g2g4', 'd8h4']
WHITE_WIN = ['e2e4', 'f7f6', 'd2d4', 'g7g5', 'd1h5']
CYCLE = ['g1f3', 'g8f6', 'f3g1', 'f6g8']
LONG = ['g1f3', 'g8f6', 'f3h4', 'f6h5', 'h4f3', 'h5f6', 'f3g1', 'f6g8']

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def move_id(board, move):
    origin = move.from_square ^ (0 if board.turn else 56)
    dest = move.to_square ^ (0 if board.turn else 56)
    if move.promotion:
        return 4096 + 4 * (8 * (origin % 8) + dest % 8) + [5, 4, 3, 2].index(move.promotion)
    return 64 * origin + dest

def source(moves, *, opening=(), fen=chess.STARTING_FEN, cap=None, checkpoint='review-controlled'):
    """Construct records without R02 finalization, R01 auditing or search helpers."""
    template = json.loads((ROOT / 'runs/r01-integration/real/games.json').read_text())[0]
    cfg = copy.deepcopy(template['configuration'])
    cfg['checkpoint']['checkpoint_id'] = checkpoint
    cfg['checkpoint']['provenance'] = {'review': 'hand-built legal trajectory; no neural inference'}
    cfg['settings'] = dict(simulation_budget=3, ply_cap=len(opening)+len(moves) if cap is None else cap,
        c_puct=1.5, unvisited_value=0.0,
        phases=[dict(start_ply=0, temperature=0.0, noise_fraction=0.0, dirichlet_alpha=0.3)])
    cfg['opening'] = dict(opening_id='review-opening', initial_fen=fen, moves_uci=list(opening),
                         history_complete=True, source='independent hand-authored review')
    cfg['opening']['replay_sha256'] = sha(dict(initial_fen=fen, moves_uci=list(opening), history_complete=True))
    rng = np.random.PCG64(cfg['seed']).state
    cfg['rng']['initial_state'] = copy.deepcopy(rng)
    board = chess.Board(fen)
    for uci in opening:
        board.push_uci(uci)
    positions = []
    for uci in moves:
        legal = list(board.legal_moves)
        move = chess.Move.from_uci(uci)
        assert move in legal
        mask = [False] * 4352
        for candidate in legal:
            mask[move_id(board, candidate)] = True
        visits = [0] * 4352
        selected = move_id(board, move)
        visits[selected] = 3
        alternative = next((m for m in legal if m != move), None)
        if alternative:
            visits[selected] = 2
            visits[move_id(board, alternative)] = 1
        positions.append(dict(ply=len(board.move_stack), root_fen=board.fen(en_passant='fen'),
            root_promoted=int(board.promoted), side_to_move='white' if board.turn else 'black',
            legal_mask=mask, root_visits=visits, visit_policy=[v/3 for v in visits],
            completed_simulations=3, move_id=selected, move_uci=uci,
            exploration=dict(temperature=0.0, noise_fraction=0.0, dirichlet_alpha=0.3),
            rng_before=copy.deepcopy(rng), rng_after=copy.deepcopy(rng)))
        board.push(move)
    outcome = board.outcome(claim_draw=False)
    reason = outcome.termination.name if outcome else 'THREEFOLD_REPETITION' if board.is_repetition(3) else 'FIFTY_MOVES' if board.is_fifty_moves() else None
    winner = None if outcome is None or outcome.winner is None else 'white' if outcome.winner else 'black'
    status = 'completed' if reason else 'truncated' if len(board.move_stack) >= cfg['settings']['ply_cap'] else 'interrupted'
    game = dict(schema='r01-game-v1', configuration_sha256=sha(cfg), positions=positions,
        interruptions=[], rng_state=copy.deepcopy(rng), initial_fen=fen,
        moves_uci=list(opening)+list(moves), history_complete=True,
        final_fen=board.fen(en_passant='fen'), final_promoted=int(board.promoted),
        winner=winner, outcome=('1-0' if winner == 'white' else '0-1' if winner == 'black' else '1/2-1/2') if reason else None,
        status=status, ending_reason=reason or ('ply_limit' if status == 'truncated' else 'execution_interrupted'))
    return dict(configuration=cfg, game=game)

@pytest.mark.parametrize('moves,winner', [(BLACK_WIN, 'black'), (WHITE_WIN, 'white'), (CYCLE*2, None)])
def test_known_wdl_and_exact_targets(moves, winner):
    snapshot = source(moves)
    before = canonical(snapshot)
    result = finalize(snapshot)
    assert result['game_id'] == sha([chess.STARTING_FEN, moves])
    assert result['source_id'] == sha(snapshot)
    assert len(result['samples']) == len(moves)
    for ply, sample in enumerate(result['samples']):
        color = 'white' if ply % 2 == 0 else 'black'
        expected = [0,1,0] if winner is None else [1,0,0] if color == winner else [0,0,1]
        assert sample['wdl'] == expected
        assert sample['position'] == snapshot['game']['positions'][ply]
        assert sample['position']['visit_policy'][sample['position']['move_id']] == 2/3
        assert sample['sample_id'] == sha(['r02-sample-v1', result['game_id'], sha(snapshot['configuration']), ply])
    assert canonical(snapshot) == before

def test_history_beyond_eight_and_black_piece_snapshot_perspective():
    snapshot = source(BLACK_WIN, opening=LONG)
    samples = finalize(snapshot)['samples']
    assert {s['position']['side_to_move'] for s in samples} == {'white', 'black'}
    boards = [chess.Board()]
    for uci in LONG + BLACK_WIN:
        board = boards[-1].copy(stack=True)
        board.push_uci(uci)
        boards.append(board)
    for sample in samples:
        ply = sample['position']['ply']
        assert sample['moves_uci'] == (LONG+BLACK_WIN)[:ply]
        assert ply >= 8 and sample['available_snapshots'] == 8
        state = make_state(sample['initial_fen'], sample['moves_uci'], history_complete=True)
        assert [m.uci() for m in state.board.move_stack] == sample['moves_uci']
        actual = encode_position(state).features
        perspective = boards[ply].turn
        for t in range(8):
            historical = boards[ply-t]
            expected = np.zeros((64, 13), dtype=np.float32)
            for square, piece in historical.piece_map().items():
                network_square = square ^ (0 if perspective else 56)
                expected[network_square, piece.piece_type-1+(0 if piece.color == perspective else 6)] = 1
            expected[:,12] = int(historical.is_repetition(2))
            np.testing.assert_array_equal(actual[:,13*t:13*(t+1)], expected)
        assert not np.array_equal(actual, encode_position(make_state(state.board.fen())).features)

@pytest.mark.parametrize('cap,status', [(1,'truncated'), (4,'interrupted')])
def test_unfinished_then_completion(cap, status, tmp_path):
    unfinished = source(BLACK_WIN[:1], cap=cap)
    assert unfinished['game']['status'] == status
    buffer = ReplayBuffer.create(tmp_path/'buffer.sqlite')
    assert buffer.ingest(unfinished)['disposition'] == 'excluded-'+status
    assert buffer.export()['samples'] == []
    assert buffer.ingest(source(BLACK_WIN))['added_samples'] == 4
    receipt = buffer.export()['sources'][0]
    assert receipt['snapshot']['game']['outcome'] is None
    assert receipt['snapshot']['game']['winner'] is None

@pytest.mark.parametrize('zero_first', [True,False])
def test_duplicate_coverage_order_zero_roots_and_stable_restart(zero_first, tmp_path):
    zero = source([], opening=BLACK_WIN)
    full = source(BLACK_WIN)
    a,b = (zero, full) if zero_first else (full, zero)
    assert finalize(zero)['samples'] == []
    buffer = ReplayBuffer.create(tmp_path/'buffer.sqlite')
    buffer.ingest(a)
    assert buffer.ingest(b)['disposition'] == 'duplicate'
    assert buffer.export()['summary']['training_positions'] == (0 if zero_first else 4)
    before = buffer.export()
    reopened = ReplayBuffer(buffer.path)
    assert reopened.ingest(a)['repeated'] and reopened.ingest(b)['repeated']
    assert reopened.export() == before

@pytest.mark.parametrize('same_source', [True,False])
def test_two_simultaneous_writers_first_commit_only(same_source, tmp_path):
    buffer = ReplayBuffer.create(tmp_path/'buffer.sqlite')
    a = source(BLACK_WIN)
    b = copy.deepcopy(a) if same_source else source(BLACK_WIN, checkpoint='other-review-checkpoint')
    barrier = Barrier(2)
    def admit(snapshot):
        barrier.wait(timeout=10)
        return buffer.ingest(snapshot)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(admit, (a,b)))
    assert sum(r['added_samples'] for r in results) == 4
    assert sum(r['repeated'] for r in results) == int(same_source)
    exported = ReplayBuffer(buffer.path).export()
    assert len(exported['sources']) == (1 if same_source else 2)
    assert len(exported['samples']) == 4
    representative = next(r['source_id'] for r in results if r['added_samples'])
    assert all(s['source_id'] == representative for s in exported['samples'])

def test_export_retains_snapshot_during_writer_commit(tmp_path, monkeypatch):
    buffer = ReplayBuffer.create(tmp_path/'buffer.sqlite')
    with sqlite3.connect(buffer.path) as connection:
        connection.execute('PRAGMA journal_mode=WAL')
    original = ReplayBuffer._validate
    fired = False
    def interleave(connection):
        nonlocal fired
        original(connection)
        if not fired:
            fired = True
            buffer.ingest(source(BLACK_WIN))
    monkeypatch.setattr(ReplayBuffer, '_validate', staticmethod(interleave))
    old = buffer.export()
    assert fired and old['summary']['training_positions'] == 0 and old['sources'] == []
    assert buffer.export()['summary']['training_positions'] == 4

@pytest.mark.parametrize('table', ['samples','sources','games','metadata'])
def test_post_open_corruption_rejected_without_writes(table, tmp_path):
    buffer = ReplayBuffer.create(tmp_path/'buffer.sqlite')
    buffer.ingest(source(BLACK_WIN))
    with sqlite3.connect(buffer.path) as connection:
        if table == 'samples':
            connection.execute('DELETE FROM samples WHERE rowid=(SELECT MIN(rowid) FROM samples)')
        elif table == 'sources':
            connection.execute("UPDATE sources SET source_id='changed'")
        elif table == 'games':
            connection.execute('DELETE FROM games')
        else:
            connection.execute("UPDATE metadata SET payload='{}'")
    before = buffer.path.read_bytes()
    with pytest.raises(ValueError): buffer.export()
    with pytest.raises(ValueError): buffer.ingest(source(WHITE_WIN))
    assert buffer.path.read_bytes() == before

def test_missing_existing_and_invalid_create_preserve_artifacts(tmp_path):
    path = tmp_path/'missing.sqlite'
    with pytest.raises(ValueError): ReplayBuffer(path)
    assert not path.exists()
    path.write_bytes(b'accepted sentinel bytes')
    with pytest.raises(FileExistsError): ReplayBuffer.create(path)
    assert path.read_bytes() == b'accepted sentinel bytes'
    # The buffer promises refusal, not a normalized exception class for invalid SQLite bytes.
    with pytest.raises((ValueError, sqlite3.DatabaseError)): ReplayBuffer(path)
    assert path.read_bytes() == b'accepted sentinel bytes'

def test_recorded_four_root_export_and_cli_fresh_process_restart():
    summary = json.loads((ROOT/'runs/r02/summary.json').read_text())
    # Read historical DB in URI read-only mode, avoiding journal or accepted-file writes.
    readonly = sqlite3.connect((ROOT/'runs/r02/replay.sqlite').resolve().as_uri()+'?mode=ro', uri=True)
    with readonly:
        ReplayBuffer._validate(readonly)
        historical_samples = [json.loads(p) for p, in readonly.execute('SELECT payload FROM samples ORDER BY sample_id')]
    readonly.close()
    assert len(historical_samples) == 4
    assert [json.loads(line) for line in (ROOT/'runs/r02/samples.jsonl').read_text().splitlines()] == historical_samples
    assert sha(historical_samples) == summary['samples_sha256']
    database = OUT/'fresh-replay.sqlite'
    for name, archive, create in [('admission','runs/r01/real/games.json',True),
                                  ('restart','runs/r01/real/games.json',False),
                                  ('accepted-main','runs/r01-integration/real/games.json',False)]:
        command = [sys.executable, '-B', 'scripts/run_r02.py', '--records', archive,
                   '--buffer', str(database), '--report', str(OUT/(name+'.json')),
                   '--samples', str(OUT/(name+'.jsonl'))]
        if create: command += ['--create']
        result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
        (OUT/(name+'.stdout.txt')).write_text(result.stdout)
    exported = ReplayBuffer(database).export()
    assert exported['samples'] == historical_samples
    assert sha(exported) == summary['audited_export_sha256'] == '4bdd6f656e159cfe8fb46db65264e876fdee5abe6abb1fc6f85bec50cf6eedf4'
    assert exported['summary'] == dict(source_counts={'admitted':7,'duplicate':1,'excluded-truncated':11,'excluded-interrupted':1}, training_positions=4)
    for name in ('admission','restart','accepted-main'):
        report = json.loads((OUT/(name+'.json')).read_text())
        recorded = json.loads((ROOT/'runs/r02'/('accepted-main-audit.json' if name == 'accepted-main' else name+'.json')).read_text())
        assert report == recorded
        assert [json.loads(line) for line in (OUT/(name+'.jsonl')).read_text().splitlines()] == historical_samples
    assert {(s['position']['side_to_move'], tuple(s['wdl'])) for s in historical_samples} == {
        ('white',(1,0,0)), ('black',(1,0,0)), ('white',(0,1,0)), ('black',(0,1,0))}
