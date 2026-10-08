"""Independent saved-record replay. Imports adapter only, no game/search code."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import chess

from chess_rl.adapter import apply_move, make_state, move_to_id, terminal_result


def main():
    path = Path(sys.argv[1])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    verified = []
    for item in records:
        r = item['record']
        assert r['schema'] == 's05-game-v1' and r['ending_policy'] == 'actual-occurrence-v1'
        assert r['move_vocabulary'] == 'pair4352-v1'
        prefix, moves = r['opening_prefix'], r['moves_uci']
        assert moves[:len(prefix)] == prefix
        state = make_state(r['initial_fen'], prefix, history_complete=r['history_complete'])
        attempts = {a['ply']: a for a in r['attempts']}
        assert len(attempts) == len(r['attempts'])
        completed = 0
        for ply in range(len(prefix), len(moves) + 1):
            a = attempts.get(ply)
            if a is not None:
                assert terminal_result(state) is None and ply < r['ply_cap']
                assert a['requested_simulations'] == r['simulation_budget']
                assert 0 <= a['completed_simulations'] <= a['requested_simulations']
                assert sum(n for _, n, _ in a['visited_edges']) == a['completed_simulations']
                # Independently enumerate actual moves and map shared IDs.
                legal = {move_to_id(state, m): m for m in state.board.legal_moves}
                for i, n, w in a['visited_edges']:
                    assert i in legal and n > 0 and abs(w) <= n
                completed += a['completed_simulations']
                if a['status'] == 'completed':
                    assert a['completed_simulations'] == r['simulation_budget']
                    assert legal[a['move_id']].uci() == a['move_uci']
                else:
                    assert a['status'] in {'failed', 'interrupted'}
                    assert a['move_id'] is a['move_uci'] is None
            if ply == len(moves):
                break
            assert a is not None and a['status'] == 'completed'
            assert a['move_uci'] == moves[ply]
            assert terminal_result(state) is None and ply < r['ply_cap']
            move = chess.Move.from_uci(moves[ply])
            assert move in state.board.legal_moves
            apply_move(state, move_to_id(state, move))
        assert state.board.fen(en_passant='fen') == r['final_fen']
        assert state.board.promoted == r['final_promoted']
        assert [m.uci() for m in state.board.move_stack] == moves
        assert state.initial_fen == r['initial_fen'] and state.history_complete == r['history_complete']
        terminal = terminal_result(state)
        if r['status'] == 'completed':
            assert terminal is not None and terminal.ending_reason == r['ending_reason']
            winner = None if terminal.winner is None else 'white' if terminal.winner else 'black'
            assert r['winner'] == winner
            assert r['outcome'] == ('1/2-1/2' if winner is None else '1-0' if winner == 'white' else '0-1')
        else:
            assert r['outcome'] is r['winner'] is None
            if r['status'] == 'truncated':
                assert terminal is None and len(moves) >= r['ply_cap']
                assert r['ending_reason'] == 'ply_limit'
            else:
                assert r['status'] in {'failed', 'interrupted'} and r['error_type']
        verified.append(dict(game_id=item['game_id'], status=r['status'],
                             retained_plies=len(moves), completed_simulations=completed,
                             ending_reason=r['ending_reason']))
    report = dict(records_verified=len(verified), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  status_counts=dict(Counter(v['status'] for v in verified)), games=verified)
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
