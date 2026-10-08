"""Reviewer-authored checks beyond the implementation fixtures."""
from dataclasses import replace
import io
import random
import zipfile

import chess
import chess.pgn
import numpy as np
import pytest

from chess_rl.adapter import Evaluation, make_state, move_to_id, terminal_result
from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import SourceError, file_identity, iter_frames, parse_mainline, prepare_archive
from chess_rl.search.evaluator import evaluate
from chess_rl.search.puct import Node, expand, puct_score, select_edge, traverse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("seed", range(16))
def test_random_exported_source_matches_independent_board(seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(10 + seed * 3):
        if board.is_game_over(claim_draw=False):
            break
        board.push(rng.choice(list(board.legal_moves)))
    game = chess.pgn.Game.from_board(board)
    for index, node in enumerate(game.mainline()):
        if index % 7 == 0:
            node.comment = "review comment {with opening brace and unicode α"
        if index % 11 == 0:
            alternatives = [move for move in node.parent.board().legal_moves if move != node.move]
            if alternatives:
                node.parent.add_variation(alternatives[0])
    raw = str(game).encode()
    # Exporter may omit an inner closing brace, making the comment valid PGN.
    frames = list(iter_frames(io.BytesIO(raw)))
    assert len(frames) == 1
    state, _, _ = parse_mainline(raw.decode())
    assert state.board.fen(en_passant="fen") == board.fen(en_passant="fen")
    assert state.board.move_stack == board.move_stack

@pytest.mark.parametrize("text", [
    '1. e4 (1. d4 (1. c4 c5) d5) e5 *',
    '1. e4 ; ignored { bracket [\ne5 *',
    '\ufeff[Event "literal { header"]\n\n1. e4\n% ignored { } [ ]\ne5 *',
])
def test_comment_and_nested_variation_framing(text):
    frame, = iter_frames(io.BytesIO(text.encode()))
    state, _, _ = parse_mainline(frame.raw.decode())
    assert [move.uci() for move in state.board.move_stack] == ['e2e4', 'e7e5']

@pytest.mark.parametrize("tail", [' garbage', ' 2. Nf3', ' (1. d4)', ' 0-1'])
def test_result_never_salvages_trailing_tokens(tail):
    with pytest.raises(SourceError):
        parse_mainline('1. e4 e5 *' + tail)

def test_crc_failure_has_no_complete_summary(tmp_path):
    archive = tmp_path/'bad.zip'
    payload = b'1. e4 e5 *\n'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as handle:
        handle.writestr('one.pgn', payload)
    raw = archive.read_bytes()
    assert raw.count(payload) == 1
    archive.write_bytes(raw.replace(payload, b'1. d4 e5 *\n'))
    manifest = DatasetManifest.load(ROOT/'data/manifests/elite-2025-01-v1.json')
    identity = file_identity(archive)
    manifest = replace(manifest, archive=replace(manifest.archive,
        downloaded_sha256=identity['sha256'], downloaded_bytes=identity['bytes']))
    output = tmp_path/'partial'
    with pytest.raises(zipfile.BadZipFile):
        prepare_archive(archive, manifest, output)
    assert not (output/'summary.json').exists()

def test_search_nonuniform_controlled_two_ply_perspectives_and_scores():
    calls = []
    def controlled(states):
        state, = states
        calls.append(len(state.board.move_stack))
        base, = evaluate(states)
        priors = np.zeros_like(base.priors)
        move = ['e2e4', 'e7e5', 'g1f3'][len(state.board.move_stack)]
        priors[move_to_id(state, chess.Move.from_uci(move))] = 1
        value = 0.8 if state.board.turn else -0.4
        return [Evaluation(priors, [(1+value)/2, 0, (1-value)/2], value)]
    root = Node(make_state())
    expand(root, controlled)
    first = traverse(root, evaluator=controlled)
    assert first.result.value == -0.4 and len(first.path) == 1
    second = traverse(root, evaluator=controlled)
    assert second.result.value == 0.8 and len(second.path) == 2
    assert calls == [0, 1, 2]
    assert root.parent_visits == first.leaf.parent_visits == 0
    # Controlled already-completed stats establish S03's required selection view.
    root_edge = second.path[0].edge
    child_edge = second.path[1].edge
    root_edge.visit_count, root_edge.value_sum = 1, 0.8
    child_edge.visit_count, child_edge.value_sum = 1, -0.8
    assert root.parent_visits == first.leaf.parent_visits == 1
    assert puct_score(root_edge, 1) == pytest.approx(1.55)
    assert puct_score(child_edge, 1) == pytest.approx(-0.05)

def test_failure_does_not_publish_partial_edges_or_visits():
    root = Node(make_state())
    expand(root)
    chosen = select_edge(root)
    def wrong(states):
        priors = np.zeros(4352, dtype=np.float32)
        priors[0] = 1
        return [Evaluation(priors, [0, 1, 0], 0)]
    with pytest.raises(ValueError, match='illegal'):
        traverse(root, evaluator=wrong)
    assert chosen.child is not None and not chosen.child.expanded
    assert chosen.child.edges == {} and root.parent_visits == 0
    retried = traverse(root)
    assert retried.leaf is chosen.child and len(retried.path) == 1
    assert root.parent_visits == retried.leaf.parent_visits == 0
