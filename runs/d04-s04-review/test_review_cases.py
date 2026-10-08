"""Reviewer-owned rule and search accounting oracles."""
import json
import random

import chess
import numpy as np
import pytest

from chess_rl.adapter import Evaluation, legal_mask, make_state
from chess_rl.position_sampling import candidates_for_game, select
from chess_rl.search.puct import Node
from chess_rl.search.runner import SearchFailure, TrainingExploration, search


@pytest.mark.parametrize('seed', range(12))
def test_candidate_membership_against_independent_board_rules(seed):
    board = chess.Board()
    board.turn = bool(seed % 2)
    board.halfmove_clock = 99 if seed % 3 == 0 else 0
    fen = board.fen()
    rng = random.Random(seed)
    moves = []
    for _ in range(96):
        if board.outcome(claim_draw=False) is not None:
            break
        move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        board.push(move)
        moves.append(move.uci())
    actual, counts = candidates_for_game(dict(start_fen=fen, moves_uci=moves))
    oracle = chess.Board(fen)
    expected, seen = [], set()
    ended = False
    for ply in range(len(moves)+1):
        terminal = (oracle.outcome(claim_draw=False) is not None
                    or oracle.is_repetition(3) or oracle.is_fifty_moves())
        if terminal:
            ended = True
        if not ended:
            key = ' '.join(oracle.fen(en_passant='legal').split()[:4])
            if key not in seen:
                seen.add(key)
                material = sum(weight*len(oracle.pieces(piece, color))
                               for piece, weight in [(2,3),(3,3),(4,5),(5,9)]
                               for color in [True,False])
                phase = 'endgame' if material <= 14 else 'opening' if ply < 20 else 'middlegame'
                expected.append((ply, phase, key))
        if ply < len(moves):
            oracle.push_uci(moves[ply])
    assert actual == expected
    assert counts['roots'] == len(moves)+1
    chosen = select('independent-game', actual)
    assert len(chosen) <= 8 and len({c[2] for c in chosen}) == len(chosen)
    assert all(c in expected for c in chosen)


@pytest.mark.parametrize('temperature', [0, 0.2, 2])
def test_search_reuse_policy_and_noise_snapshots(temperature):
    state = make_state(moves_uci=['e2e4','e7e5','g1f3','b8c6'])
    original = (state.board.fen(), list(state.board.move_stack))
    root = Node(state)
    first = search(root, 3)
    frozen = first.root_visits.copy()
    priors = {i:e.prior for i,e in root.edges.items()}
    result = search(root, 7, exploration=TrainingExploration(temperature, 0.25, 0.3),
                    rng=np.random.default_rng(125))
    assert result.completed_simulations == 7
    assert sum(int(n) for n in result.root_visits) == 10
    assert np.array_equal(first.root_visits, frozen)
    assert np.array_equal(result.training_policy, result.root_visits.astype(float)/10)
    assert np.all(result.root_visits[~legal_mask(state)] == 0)
    assert {i:e.prior for i,e in root.edges.items()} == priors
    assert root.result.priors[result.move_id] >= 0
    assert result.move in state.board.legal_moves
    reused = search(root, 0)
    assert reused.completed_simulations == 0
    assert reused.move_id == int(np.flatnonzero(reused.root_visits == reused.root_visits.max())[0])
    assert np.array_equal(reused.training_policy, result.training_policy)
    assert (state.board.fen(), state.board.move_stack) == original


def test_failed_search_reports_only_new_completions_and_restores_priors():
    root = Node(make_state())
    baseline = search(root, 3)
    priors = {i:e.prior for i,e in root.edges.items()}
    calls = 0
    def evaluator(states):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('review-injected evaluator failure')
        mask = legal_mask(states[0])
        p = mask.astype(np.float32)/np.count_nonzero(mask)
        return [Evaluation(p.astype(np.float32), np.array([.75,0,.25],dtype=np.float32), .5)]
    with pytest.raises(SearchFailure) as caught:
        search(root, 8, evaluator=evaluator,
               exploration=TrainingExploration(1,.25,.3), rng=np.random.default_rng(27))
    partial = caught.value.partial
    assert partial.completed_simulations == 2
    assert int(partial.root_visits.sum()) == 5
    assert np.all(partial.root_visits >= baseline.root_visits)
    assert partial.move is None and partial.move_id is None
    assert np.array_equal(partial.training_policy, partial.root_visits.astype(float)/5)
    assert {i:e.prior for i,e in root.edges.items()} == priors
    assert search(root, 1).completed_simulations == 1


def test_terminal_with_legal_moves_skips_noise_and_action_rng():
    state = make_state(moves_uci=['g1f3','g8f6','f3g1','f6g8']*2)
    assert state.board.legal_moves.count() > 0
    rng = np.random.default_rng(921)
    before = json.dumps(rng.bit_generator.state,sort_keys=True)
    def forbidden(_):
        raise AssertionError('terminal root invoked evaluator')
    result = search(Node(state),32,evaluator=forbidden,
                    exploration=TrainingExploration(1,.25,.3),rng=rng)
    assert result.completed_simulations == 0 and result.training_policy is None
    assert result.move_id is None and result.move is None and result.root_visits.sum() == 0
    assert before == json.dumps(rng.bit_generator.state,sort_keys=True)
