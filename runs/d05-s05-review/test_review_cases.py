import chess
import chess.engine
import pytest
from chess_rl.adapter import make_state, move_to_id, terminal_result
from chess_rl.stockfish_worker import validate_response
from chess_rl.search.evaluator import evaluate
from chess_rl.search.game import run_game

CASES = [("4k3/P7/8/8/8/8/8/4K3 w - - 0 37", "a7a8"+p) for p in "qrbn"]
CASES += [("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 37", m) for m in ("e1g1", "e1c1")]
CASES += [("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 37", "e5d6")]

@pytest.mark.parametrize("fen,uci", CASES)
@pytest.mark.parametrize("mirror", [False, True])
def test_teacher_and_search_special_moves(fen, uci, mirror):
    direct = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    if mirror:
        direct = direct.mirror()
        move = chess.Move(move.from_square ^ 56, move.to_square ^ 56, promotion=move.promotion)
    state = make_state(direct.fen(en_passant="fen"), history_complete=False)
    assert move in direct.legal_moves
    origin, target = move.from_square, move.to_square
    if not direct.turn:
        origin, target = origin ^ 56, target ^ 56
    expected = (4096 + 4*(8*chess.square_file(origin)+chess.square_file(target)) +
                (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT).index(move.promotion)) if move.promotion else 64*origin+target
    info = dict(multipv=1, wdl=chess.engine.PovWdl(chess.engine.Wdl(600, 300, 100), not direct.turn),
                score=chess.engine.PovScore(chess.engine.Cp(52), not direct.turn), pv=[move],
                depth=8, seldepth=9, nodes=100010, nps=100000, hashfull=0, tbhits=0, time=1.0)
    label = validate_response(state, info, move)
    assert label["best_move_id"] == expected and label["teacher_wdl_counts"] == [100,300,600]
    before = state.board.fen(en_passant="fen")
    def selected(states):
        outputs = evaluate(states)
        for root, output in zip(states, outputs):
            if terminal_result(root) is None and move in root.board.legal_moves:
                output.priors[:] = 0
                output.priors[move_to_id(root, move)] = 1
        return outputs
    game = run_game(state, 1, ply_cap=1, evaluator=selected, evaluator_id="independent-special-move-probe")
    direct.push(move)
    assert game.moves_uci == (move.uci(),)
    assert game.attempts[0].move_id == expected
    assert game.final_fen == direct.fen(en_passant="fen") and game.final_promoted == direct.promoted
    assert game.history_complete is False and state.board.fen(en_passant="fen") == before
    assert not state.board.move_stack
    if game.status != "completed":
        assert game.status == "truncated" and game.outcome is None
