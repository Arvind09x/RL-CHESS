"""Controlled E01 checks; no real P/R checkpoints or strength experiments."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import chess
import numpy as np
import pytest

from chess_rl.adapter import (Evaluation, apply_move, make_state, move_to_id, terminal_result)
from chess_rl.evaluation import (Checkpoint, Decision, GreedyPlayer, Opening, OpeningSets,
    RandomLegalPlayer, SearchPlayer, SearchSettings, StockfishPin, StockfishPlayer, run_match, summarize)
from chess_rl.evaluation.replay import audit_match
from chess_rl.search.evaluator import evaluate
from chess_rl.search.puct import SearchConfig
import chess_rl.evaluation.matches as matches


def book(state=None):
    return OpeningSets("controlled-book-v1", (Opening.from_state("v", state or make_state(moves_uci=["e2e4"]),
        source="controlled validation fixture"),), (Opening.from_state("t", make_state(moves_uci=["d2d4"]),
        source="controlled reserved test fixture"),))


def search_player(role="P", evaluator=evaluate, **kwargs):
    return SearchPlayer(f"fixture-{role}", role, "controlled-evaluator-v1", evaluator,
                        kwargs.pop("settings", SearchSettings(2)), **kwargs)


def pr(openings=None, **kwargs):
    return run_match(kwargs.pop("left", search_player()), kwargs.pop("right", search_player("R")),
        openings or book(), split=kwargs.pop("split", "validation"), match_id="fixture-pr",
        comparison="P-versus-R", seed=kwargs.pop("seed", 17), ply_cap=kwargs.pop("ply_cap", 3),
        protocol_only=kwargs.pop("protocol_only", True), **kwargs)


class Controlled:
    def __init__(self, action, player_id="controlled"):
        self.action, self.player_id = action, player_id

    def metadata(self):
        return dict(player_id=self.player_id, kind="controlled", version="test-v1", settings=dict(action="injected"))

    def choose(self, state, *, seed):
        move = self.action(state) if callable(self.action) else self.action
        return Decision(move_to_id(state, chess.Move.from_uci(move)), dict(controlled=True))


def fixed(right, state=None, **kwargs):
    return run_match(kwargs.pop("left", search_player()), right, book(state), split="validation", match_id="fixture-fixed",
        comparison="fixed-opponent", seed=18, ply_cap=kwargs.pop("ply_cap", 2), protocol_only=True, **kwargs)


def one_hot(action):
    def evaluator(states):
        output = []
        for state in states:
            terminal = terminal_result(state)
            if terminal is not None:
                output.append(terminal)
            else:
                move = action(state) if callable(action) else action
                if chess.Move.from_uci(move) not in state.board.legal_moves:
                    output.extend(evaluate([state]))
                    continue
                priors = np.zeros(4352, dtype=np.float32)
                priors[move_to_id(state, chess.Move.from_uci(move))] = 1
                output.append(Evaluation(priors, np.array([0, 1, 0], dtype=np.float32), 0))
        return output
    return evaluator


def test_pairing_routing_replay_and_reproducibility():
    calls = {"P": [], "R": []}
    def ev(role):
        def fn(states):
            calls[role].append([(s.board.turn, tuple(m.uci() for m in s.board.move_stack)) for s in states])
            return evaluate(states)
        return fn
    result = pr(left=search_player(evaluator=ev("P")), right=search_player("R", ev("R")))
    record = result.to_dict()
    first, second = record["games"]
    assert first["pair_id"] == second["pair_id"]
    assert first["opening"] == second["opening"]
    assert first["players"]["white"]["role"] == second["players"]["black"]["role"] == "P"
    assert first["attempts"][0]["player_id"] == "fixture-R"  # Black to move after e4.
    assert second["attempts"][0]["player_id"] == "fixture-P"
    assert calls["P"] and calls["R"]
    assert audit_match(record)["passed"]
    assert result.to_dict() == pr(left=search_player(evaluator=ev("P")), right=search_player("R", ev("R"))).to_dict()
    assert record["configuration"]["left_player"]["checkpoint"] is None
    record["games"][0]["moves_uci"].append("bad")
    assert audit_match(result.to_dict())["passed"]  # Snapshot ownership.


@pytest.mark.parametrize("settings", [SearchSettings(3), SearchSettings(2, SearchConfig(2)),
                                      SearchSettings(2, SearchConfig(unvisited_value=0.1))])
def test_pr_rejects_unequal_settings(settings):
    with pytest.raises(ValueError, match="identical"):
        pr(right=search_player("R", settings=settings))


@pytest.mark.parametrize("left,right", [(search_player("R"), search_player()),
                                        (search_player(), RandomLegalPlayer())])
def test_pr_explicit_roles_and_search_type(left, right):
    with pytest.raises(ValueError):
        pr(left=left, right=right)


def test_checkpoint_metadata_and_unavailable_guard(tmp_path):
    with pytest.raises(ValueError, match="checkpoint identity"):
        pr(protocol_only=False)
    path = tmp_path / "controlled-bytes.fixture"
    path.write_bytes(b"not a trained checkpoint")
    checkpoint = Checkpoint.from_file("explicit-fixture-bytes", path, kind="fixture")
    player = search_player(checkpoint=checkpoint)
    result = pr(left=player).to_dict()
    assert result["configuration"]["left_player"]["checkpoint"] == dict(
        checkpoint_id="explicit-fixture-bytes", path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kind="fixture")
    with pytest.raises(ValueError, match="checkpoint identity"):
        pr(left=player, protocol_only=False)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        pr(left=player)
    with pytest.raises(ValueError, match="kind"):
        search_player(checkpoint=Checkpoint("wrong-role", str(path), "0" * 64, "rl-R"))


def test_opening_separation_and_transpositions():
    o = book().validation[0]
    with pytest.raises(ValueError, match="unique"):
        OpeningSets("bad", (o,), (o,))
    with pytest.raises(ValueError, match="distinct"):
        OpeningSets("bad", (o,), (Opening.from_state("other", o.state(), source="duplicate"),))
    a = make_state(moves_uci=["g1f3", "g8f6", "b1c3", "b8c6"])
    b = make_state(moves_uci=["b1c3", "b8c6", "g1f3", "g8f6"])
    with pytest.raises(ValueError, match="distinct"):
        OpeningSets("transposition", (Opening.from_state("a", a, source="fixture"),),
                    (Opening.from_state("b", b, source="fixture"),))
    test = pr(split="test", ply_cap=1).to_dict()
    assert all(g["opening"]["opening_id"] == "t" for g in test["games"])
    assert audit_match(test)["passed"]


def test_opening_rejects_post_terminal_history():
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + ["g1f3"]
    with pytest.raises(ValueError, match="after"):
        book(make_state(moves_uci=moves))


@pytest.mark.parametrize("failure", [RuntimeError("fault"), KeyboardInterrupt("stop"), SystemExit("exit")])
def test_failure_pair_preservation_and_no_draw(failure):
    def broken(states):
        raise failure
    result = pr(left=search_player(evaluator=broken), ply_cap=2).to_dict()
    expected = "failed" if isinstance(failure, RuntimeError) else "interrupted"
    assert [g["status"] for g in result["games"]] == ["truncated", expected]
    assert all(g["outcome"] is None for g in result["games"])
    assert result["summary"]["completed_game_denominator"] == 0
    assert result["summary"]["completed_game_score"] is None
    assert result["summary"]["incomplete_pairs"] == 1
    assert audit_match(result)["passed"]


def test_partial_search_completions_on_failure():
    calls = 0
    def broken(states):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("partial")
        return evaluate(states)
    record = pr(left=search_player(evaluator=broken), right=search_player("R", broken), ply_cap=2).to_dict()
    failed = record["games"][0]["attempts"][0]
    assert failed["details"]["completed_simulations"] == 1
    assert audit_match(record)["completed_search_simulations"] >= 1


def test_completed_single_leg_and_pair_denominators():
    state = make_state("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    good = one_hot("g6g7")
    def bad(states):
        raise RuntimeError("fixture failure")
    record = pr(book(state), left=search_player(evaluator=good), right=search_player("R", bad), ply_cap=1).to_dict()
    assert [g["status"] for g in record["games"]] == ["completed", "failed"]
    assert record["summary"]["completed_game_wdl"] == dict(win=1, draw=0, loss=0)
    assert record["summary"]["completed_game_denominator"] == 1
    assert record["summary"]["complete_pairs"] == 0
    assert audit_match(record)["passed"]


@pytest.mark.parametrize("fen,reason", [
    ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "CHECKMATE"),
    ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", "STALEMATE"),
    ("8/8/8/8/8/2k5/8/R3K3 w - - 100 1", "FIFTY_MOVES")])
def test_terminal_before_cap(fen, reason):
    def forbidden(states):
        raise AssertionError("terminal evaluator call")
    record = pr(book(make_state(fen)), left=search_player(evaluator=forbidden),
                right=search_player("R", forbidden), ply_cap=0).to_dict()
    assert all(g["status"] == "completed" and g["ending_reason"] == reason and not g["attempts"] for g in record["games"])
    assert audit_match(record)["passed"]


def test_actual_repetition_full_history_and_intended_claim():
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    terminal = pr(book(make_state(moves_uci=cycle * 2)), ply_cap=8).to_dict()
    assert all(g["ending_reason"] == "THREEFOLD_REPETITION" for g in terminal["games"])
    assert terminal["games"][0]["history_complete"] is True
    assert audit_match(terminal)["passed"]
    state = make_state(moves_uci=cycle + cycle[:3])
    result = pr(book(state), left=search_player(evaluator=one_hot("f6g8")),
                right=search_player("R", one_hot("f6g8")), ply_cap=8).to_dict()
    assert all(len(g["attempts"]) == 1 and g["status"] == "completed" for g in result["games"])
    assert audit_match(result)["passed"]
    intended = pr(book(state), ply_cap=7).to_dict()
    assert all(g["status"] == "truncated" for g in intended["games"])


SPECIAL = [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
    ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6"),
    ("7k/8/8/8/3Pp3/8/8/K7 b - d3 0 1", "e4d3"),
] + [("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8" + p) for p in "qrbn"] + [
    ("7k/8/8/8/8/8/p7/7K b - - 0 1", "a2a1" + p) for p in "qrbn"]


@pytest.mark.parametrize("fen,uci", SPECIAL)
def test_special_moves_both_player_routes(fen, uci):
    state = make_state(fen)
    record = fixed(Controlled(uci), state, left=search_player(evaluator=one_hot(uci)), ply_cap=1).to_dict()
    assert all(g["moves_uci"] == [uci] and g["history_complete"] is False for g in record["games"])
    assert {g["attempts"][0]["player_id"] for g in record["games"]} == {"fixture-P", "controlled"}
    assert not state.board.move_stack
    assert audit_match(record)["generated_plies"] == 2


@pytest.mark.parametrize("error", [RuntimeError("apply"), KeyboardInterrupt("apply")])
def test_candidate_application_atomic(monkeypatch, error):
    original = matches.apply_move
    def bad(state, move_id):
        original(state, move_id)
        raise error
    openings = book()  # Build before fault injection.
    monkeypatch.setattr(matches, "apply_move", bad)
    record = pr(openings).to_dict()
    assert all(g["moves_uci"] == ["e2e4"] and g["attempts"][0]["applied"] is False for g in record["games"])
    monkeypatch.setattr(matches, "apply_move", original)
    assert audit_match(record)["passed"]


def test_bad_player_mutation_and_illegal_return():
    def mutate(state):
        state.board.push(next(iter(state.board.legal_moves)))
        return next(iter(state.board.legal_moves)).uci()
    record = fixed(Controlled(mutate)).to_dict()
    assert record["games"][0]["status"] == "failed"
    assert "mutated" in record["games"][0]["error"]["message"]
    assert audit_match(record)["passed"]
    class Illegal(Controlled):
        def choose(self, state, *, seed):
            return Decision(0, {})
    record = fixed(Illegal("unused")).to_dict()
    assert record["games"][0]["status"] == "failed"
    assert audit_match(record)["passed"]


def test_random_seed_and_sorted_legal_support():
    player = RandomLegalPlayer()
    state = make_state()
    decisions = [player.choose(state, seed=n).move_id for n in range(30)]
    assert decisions == [player.choose(state, seed=n).move_id for n in range(30)]
    assert len(set(decisions)) > 1
    assert fixed(player).to_dict() == fixed(player).to_dict()
    assert audit_match(fixed(player).to_dict())["passed"]


@pytest.mark.parametrize("fen,uci", [
    ("7k/8/8/8/8/8/q7/R6K w - - 0 1", "a1a2"),
    ("r6k/Q7/8/8/8/8/8/7K b - - 0 1", "a8a7"),
    ("7k/8/8/3pP3/8/8/8/K7 w - d6 0 1", "e5d6"),
    ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8q"),
    ("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", "g6g7")])
def test_greedy_perspective_special_material_and_mate(fen, uci):
    state = make_state(fen)
    decision = GreedyPlayer().choose(state, seed=0)
    assert decision.move_id == move_to_id(state, chess.Move.from_uci(uci))
    assert GreedyPlayer().metadata()["settings"]["piece_values"] == dict(pawn=1, knight=3, bishop=3, rook=5, queen=9, king=0)


def test_greedy_tie_smallest_shared_id():
    state = make_state()
    assert GreedyPlayer().choose(state, seed=4).move_id == min(move_to_id(state, m) for m in state.board.legal_moves)
    assert audit_match(fixed(GreedyPlayer()).to_dict())["passed"]


@pytest.mark.parametrize("field", ["pair", "route", "seed", "fen", "outcome", "budget", "opening", "summary", "missing-search-trace"])
def test_replay_rejects_corruption(field):
    record = pr().to_dict()
    game = record["games"][0]
    if field == "pair": game["pair_id"] = "bad"
    if field == "route": game["attempts"][0]["player_id"] = "bad"
    if field == "seed": game["attempts"][0]["seed"] += 1
    if field == "fen": game["final_fen"] = chess.STARTING_FEN
    if field == "outcome": game["outcome"] = "1/2-1/2"
    if field == "budget": game["attempts"][0]["details"]["completed_simulations"] += 1
    if field == "opening": game["opening"]["source"] = "bad"
    if field == "missing-search-trace": game["attempts"][0]["details"] = {}
    if field == "summary": record["summary"]["completed_game_denominator"] = 2
    with pytest.raises(ValueError): audit_match(record)


def test_uncertainty_resamples_complete_pairs_only():
    record = pr(book(make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")), ply_cap=0).to_dict()
    games = record["games"]
    more = deepcopy(games)
    for g in more: g["pair_id"] += "other"
    summary = summarize(games + more, "fixture-P", seed=4)
    assert summary["complete_pairs"] == 2
    assert summary["paired_uncertainty"]["interval"] == [0.5, 0.5]
    assert summarize(games + more, "fixture-P", seed=4) == summary


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(seed=True), dict(ply_cap=-1), dict(ply_cap=True)])
def test_protocol_argument_validation(kwargs):
    with pytest.raises(ValueError): pr(**kwargs)


def test_stockfish_pin_and_small_budget_validation(tmp_path):
    pin = StockfishPin(str(tmp_path / "absent"), "0" * 64, "Stockfish fixture", "fixture", "fixture")
    with pytest.raises(ValueError): StockfishPlayer(pin, nodes=100000)
    with pytest.raises(ValueError): StockfishPlayer(pin, nodes=True)
    with pytest.raises(ValueError): StockfishPin("x", "bad", "Stockfish", "17.1", "commit")
    record = fixed(StockfishPlayer(pin)).to_dict()
    assert record["games"][0]["status"] == "failed"
    assert audit_match(record)["passed"]


def test_stockfish_transport_records_history_settings_nodes(monkeypatch, tmp_path):
    import chess.engine
    path = tmp_path / "engine.fixture"
    path.write_bytes(b"controlled-engine-bytes")
    pin = StockfishPin(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "Stockfish fixture", "fixture", "fixture")
    calls = []
    class Option:
        default = 0
        type = "spin"
        def parse(self, value): return value
    class Fake:
        id = dict(name="Stockfish fixture", author="fixture")
        options = {k: Option() for k in ("Threads", "Hash", "Skill Level", "UCI_LimitStrength", "UCI_ShowWDL",
            "SyzygyPath", "MultiPV", "Ponder", "UCI_Chess960", "Clear Hash")}
        options["Clear Hash"] = SimpleNamespace(type="button")
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def configure(self, options): calls.append(options)
        def ping(self): calls.append("ping")
        def play(self, board, limit, **kwargs):
            assert [m.uci() for m in board.move_stack] == ["e2e4"]
            assert limit.nodes == 32 and kwargs["ponder"] is False
            return SimpleNamespace(move=chess.Move.from_uci("e7e5"), info=dict(nodes=35, depth=2, tbhits=0), resigned=False)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **kw: Fake())
    player = StockfishPlayer(pin, nodes=32)
    decision = player.choose(make_state(moves_uci=["e2e4"]), seed=5)
    assert decision.details["returned_info"]["nodes"] == 35  # Record overshoot.
    assert calls[0]["Hash"] == 16 and calls[0]["UCI_ShowWDL"] is False
    assert calls[1:] == [{"Clear Hash": None}, "ping"]
    assert decision.details["effective_options"]["MultiPV"] == 1
    assert player.metadata()["settings"]["configuration_namespace"].startswith("evaluation-opponent")


@pytest.mark.parametrize("error", [RuntimeError("second root"), KeyboardInterrupt("second root")])
def test_fault_after_actual_move_preserves_history(error):
    calls = 0
    def fault(states):
        nonlocal calls
        calls += 1
        if calls == 4:  # First 2-simulation root completes with 3 calls.
            raise error
        return evaluate(states)
    result = pr(left=search_player(evaluator=fault), right=search_player("R", fault), ply_cap=4).to_dict()
    assert len(result["games"][0]["moves_uci"]) == 2
    assert result["games"][0]["attempts"][0]["applied"] is True
    assert result["games"][0]["attempts"][1]["applied"] is False
    assert audit_match(result)["passed"]


def test_protocol_import_boundary_subprocess():
    import os
    import subprocess
    import sys
    code = """
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch'} or fullname.startswith(('chess_rl.model', 'chess_rl.stockfish_worker', 'chess_rl.training')):
            raise ImportError('blocked ' + fullname)
sys.meta_path.insert(0, Block())
from chess_rl.adapter import make_state
from chess_rl.evaluation import Opening, OpeningSets, RandomLegalPlayer, SearchPlayer, SearchSettings, run_match
from chess_rl.search.evaluator import evaluate
v=Opening.from_state('v',make_state(moves_uci=['e2e4']),source='fixture')
t=Opening.from_state('t',make_state(moves_uci=['d2d4']),source='fixture')
p=SearchPlayer('dummy','fixture','s01-uniform-dummy-v1',evaluate,SearchSettings(1))
r=run_match(p,RandomLegalPlayer(),OpeningSets('fixture',(v,),(t,)),split='validation',match_id='boundary',comparison='fixed-opponent',seed=1,ply_cap=2,protocol_only=True)
assert len(r.games)==2
"""
    result = subprocess.run([sys.executable, '-B', '-c', code], env=dict(os.environ, PYTHONPATH='src:.'), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr



def test_native_search_scalars_and_explicit_opening_provenance():
    settings = SearchSettings(2, SearchConfig(np.float32(1.5), np.float32(0)))
    assert audit_match(pr(left=search_player(settings=settings), right=search_player("R", settings=settings)).to_dict())["passed"]
    with pytest.raises(ValueError, match="explicit boolean"):
        Opening("ambiguous", chess.STARTING_FEN, (), None, "fixture")
    record = pr(book(make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")), ply_cap=0).to_dict()
    record["games"][0]["outcome"] = None
    with pytest.raises(ValueError, match="consistent rule outcome"):
        summarize(record["games"], "fixture-P")
