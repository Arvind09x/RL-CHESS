"""Independent review probes; expected backup values use player colors."""
from copy import copy
import hashlib
import json

import chess
import numpy as np
import pytest

from chess_rl.adapter import Evaluation, legal_mask, make_state, move_to_id
from chess_rl.canonical_games import assign_split
from chess_rl.search.puct import Node, backup, expand, traverse

LINE = 'e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6'.split()


@pytest.mark.parametrize('depth', [1, 2, 3, 7, 12])
@pytest.mark.parametrize('value', [-0.8, 0.35])
@pytest.mark.parametrize('black_root', [False, True])
def test_color_oracle_accumulation_and_pending_aliases(depth, value, black_root):
    prefix = ['h2h3'] if black_root else []
    # A mirrored coordinate line gives the same legal progression for Black.
    moves = [chess.Move(chess.square_mirror(chess.Move.from_uci(u).from_square),
                        chess.square_mirror(chess.Move.from_uci(u).to_square)).uci()
             for u in LINE] if black_root else LINE
    def evaluator(states):
        state = states[0]
        index = len(state.board.move_stack) - len(prefix)
        priors = np.zeros(4352, dtype=np.float32)
        priors[move_to_id(state, chess.Move.from_uci(moves[index]))] = 1
        wdl = np.array([(1 + value) / 2, 0, (1 - value) / 2], dtype=np.float32)
        return [Evaluation(priors, wdl, value)]
    root = Node(make_state(moves_uci=prefix))
    expand(root, evaluator)
    for _ in range(depth - 1):
        traverse(root, evaluator=evaluator)
    pending = [traverse(root, evaluator=evaluator), traverse(root, evaluator=evaluator)]
    snapshots = {id(step.edge): (step.edge, step.edge.visit_count, step.edge.value_sum)
                 for path in pending for step in path.path}
    expected = {k: [n, w] for k, (_, n, w) in snapshots.items()}
    assert root.parent_visits == 0
    original_fen = root.state.board.fen()
    original_stack = list(root.state.board.move_stack)
    for resolved in reversed(pending):
        alias = copy(resolved)
        for step in resolved.path:
            signed = value if step.parent.state.board.turn == resolved.leaf.state.board.turn else -value
            expected[id(step.edge)][0] += 1
            expected[id(step.edge)][1] += signed
        assert backup(resolved) == 1 and alias.completed
        with pytest.raises(ValueError):
            backup(alias)
        for key, (edge, _, _) in snapshots.items():
            assert edge.visit_count == expected[key][0]
            assert edge.value_sum == pytest.approx(expected[key][1])
    assert root.parent_visits == 2
    assert root.state.board.fen() == original_fen
    assert root.state.board.move_stack == original_stack


def test_split_assignment_independent_integer_oracle():
    counts = {'train': 0, 'validation': 0, 'test': 0}
    for i in range(2000):
        game_id = 'g:' + hashlib.sha256(str(i).encode()).hexdigest()
        payload = json.dumps({'game_id': game_id, 'seed': 20261008,
                              'split_version': 'game-sha256-threshold-v1'},
                             sort_keys=True, separators=(',', ':'))
        x = int.from_bytes(hashlib.sha256(payload.encode()).digest(), 'big')
        expected = next(label for label, numerator in [('train', 90), ('validation', 95), ('test', 100)]
                        if x * 100 < numerator * (1 << 256))
        assert assign_split(game_id, 20261008) == expected
        counts[expected] += 1
    assert all(counts.values())


@pytest.mark.parametrize('index', [1, 7, 29, 61, 113, 227])
def test_canonical_game_to_search_preserves_replay(index):
    with open('/private/tmp/rl-chess-d03-prepared/games.jsonl') as handle:
        for n, line in enumerate(handle):
            if n == index:
                game = json.loads(line)
                break
    state = make_state(game['start_fen'], game['moves_uci'][:20], history_complete=True)
    before = (state.board.fen(), list(state.board.move_stack), legal_mask(state).tobytes())
    root = Node(state)
    expand(root)
    if root.edges:
        resolved = traverse(root)
        assert backup(resolved) == 1
        assert root.parent_visits == 1
    assert (state.board.fen(), state.board.move_stack, legal_mask(state).tobytes()) == before
