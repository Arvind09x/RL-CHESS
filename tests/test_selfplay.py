"""R01 move boundaries, complete history, atomic recovery and owned RNG."""
import copy
from dataclasses import asdict
import json
from types import SimpleNamespace

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl import selfplay as impl
from chess_rl.adapter import Evaluation, encode_position, make_state, move_to_id
from chess_rl.contracts import AdapterConfig
from chess_rl.evaluation.matches import Opening
from chess_rl.search.evaluator import evaluate
from chess_rl.selfplay import Phase, Session, Settings, audit, summarize
from chess_rl.training.checkpoints import ModelConfig
from .helpers import CASTLING, PROMOTION, BLACK_PROMOTION, CYCLE, LONG_CYCLE
from .test_evaluator import terminal_cases


@pytest.fixture(autouse=True)
def cpu():
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(old, warn_only=warn)


@pytest.fixture
def fixture_checkpoint(tmp_path, monkeypatch):
    """Explicit dummy evaluator for rules/fault tests; real artifact runs are separate."""
    path = tmp_path / "fixture.pt"
    path.write_bytes(b"R01 controlled orchestration fixture, not neural weights")
    class FixtureEvaluator:
        def __init__(self, path):
            self.checkpoint = SimpleNamespace(checkpoint_id="controlled-fixture", config=ModelConfig(),
                contracts=asdict(AdapterConfig()), provenance={"fixture": "dummy-search"})
        def __call__(self, states):
            assert len(states) == 1
            return evaluate(states)
    monkeypatch.setattr(impl, "RealEvaluator", FixtureEvaluator)
    return path


def opening(state=None):
    return Opening.from_state("controlled", state or make_state(), source="hand-authored R01 fixture")


def create(tmp_path, checkpoint, state=None, *, cap=5, budget=3, name="game", phases=None):
    return Session.create(tmp_path / name, checkpoint, opening(state), seed=971,
        opening_selection_seed=73, settings=Settings(budget, cap, phases=phases or (
            Phase(0, 1.0, .25), Phase(3, 0.0, 0.0))))


def core(snapshot):
    result = copy.deepcopy(snapshot["game"])
    result.pop("interruptions")
    return result


@pytest.mark.parametrize("index", [0, 1, 2, 3, 5, 6, 7, 8, 10])
def test_terminal_precedence_at_initial_cap(tmp_path, fixture_checkpoint, index):
    state, reason, _ = terminal_cases()[index]
    # Declared setup games attest complete history; unlike isolated-FEN probes.
    state.history_complete = True
    session = create(tmp_path, fixture_checkpoint, state, cap=len(state.board.move_stack))
    session.evaluator = lambda states: pytest.fail("terminal inference")
    result = session.run(stop_after_moves=0)
    assert result["game"]["status"] == "completed"
    assert result["game"]["ending_reason"] == reason
    assert not result["game"]["positions"]
    assert Session.load(session.directory).snapshot() == result


def controlled(uci):
    def call(states):
        result = evaluate(states)[0]
        state = states[0]
        move = chess.Move.from_uci(uci)
        if isinstance(result, Evaluation) and move in state.board.legal_moves:
            priors = np.zeros(4352, dtype=np.float32)
            priors[move_to_id(state, move)] = 1
            return [Evaluation(priors, result.wdl, result.value)]
        return [result]
    return call


@pytest.mark.parametrize("fen,uci,reason", [
    ("7k/8/5KQ1/8/8/8/8/8 w - - 99 1", "g6g7", "CHECKMATE"),
    ("8/8/8/8/8/5kq1/8/7K b - - 99 1", "g3g2", "CHECKMATE"),
    ("4k3/8/8/8/8/8/8/R3K3 w - - 99 1", "a1a2", "FIFTY_MOVES"),
    ("7k/5K2/8/6Q1/8/8/8/8 w - - 0 1", "g5g6", "STALEMATE"),
])
def test_last_move_terminal_beats_cap(tmp_path, fixture_checkpoint, fen, uci, reason):
    session = create(tmp_path, fixture_checkpoint, make_state(fen, history_complete=True),
        budget=1, cap=1, phases=(Phase(0, 0., 0.),))
    session.evaluator = controlled(uci)
    result = session.run()["game"]
    assert result["moves_uci"] == [uci]
    assert result["status"] == "completed" and result["ending_reason"] == reason
    audit(session.configuration, result)


@pytest.mark.parametrize("fen,uci", [
    (CASTLING.format("w"), "e1g1"), (CASTLING.format("w"), "e1c1"),
    (CASTLING.format("b"), "e8g8"), (CASTLING.format("b"), "e8c8"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3"),
] + [(fen, prefix + piece) for fen, prefix in [(PROMOTION, "e7e8"),
                                             (BLACK_PROMOTION, "e2e1")] for piece in "qrbn"])
def test_special_move_roundtrip_durable(tmp_path, fixture_checkpoint, fen, uci):
    state = make_state(fen, history_complete=True)
    before = state.board.fen(), list(state.board.move_stack)
    session = create(tmp_path, fixture_checkpoint, state, budget=1, cap=2,
                     phases=(Phase(0, 0., 0.),))
    session.evaluator = controlled(uci)
    session.run(stop_after_moves=1)
    reload = Session.load(session.directory)
    replay = audit(reload.configuration, reload.record)
    expected = state.board.copy(stack=True)
    expected.push_uci(uci)
    assert replay.board.fen(en_passant="fen") == expected.fen(en_passant="fen")
    assert replay.board.promoted == expected.promoted
    assert replay.board.move_stack == expected.move_stack
    assert (state.board.fen(), state.board.move_stack) == before
    assert reload.record["positions"][0]["move_uci"] == uci


def test_reached_repetition_at_final_cap_and_intended_only(tmp_path, fixture_checkpoint):
    state = make_state(moves_uci=CYCLE + CYCLE[:3])
    capped = create(tmp_path, fixture_checkpoint, state, cap=7, name="intended")
    assert capped.run()["game"]["status"] == "truncated"
    session = create(tmp_path, fixture_checkpoint, state, cap=8, budget=1,
                     phases=(Phase(0, 0., 0.),))
    session.evaluator = controlled(CYCLE[-1])
    result = session.run()["game"]
    assert result["ending_reason"] == "THREEFOLD_REPETITION" and result["status"] == "completed"


def test_long_complete_history_survives_reload_and_features(tmp_path, fixture_checkpoint):
    state = make_state(moves_uci=LONG_CYCLE + ["e2e4", "e7e5", "g1f3"])
    before = encode_position(state).features.copy()
    session = create(tmp_path, fixture_checkpoint, state, cap=len(state.board.move_stack) + 2)
    session.run(stop_after_moves=0)
    restored = Session.load(session.directory)
    replay = audit(restored.configuration, restored.record)
    assert replay.board.move_stack == state.board.move_stack
    np.testing.assert_array_equal(encode_position(replay).features, before)
    assert restored.configuration["opening"]["moves_uci"] == [m.uci() for m in state.board.move_stack]


@pytest.mark.parametrize("stop", [0, 1, 2, 4])
def test_deterministic_continuation_noise_temperature(tmp_path, fixture_checkpoint, stop):
    uninterrupted = create(tmp_path, fixture_checkpoint, name="full").run()
    session = create(tmp_path, fixture_checkpoint, name="split")
    interrupted = session.run(stop_after_moves=stop)
    assert interrupted["game"]["status"] == "interrupted"
    assert interrupted["game"]["outcome"] is None
    restored = Session.load(session.directory)
    assert restored.snapshot() == interrupted
    resumed = restored.run()
    assert core(resumed) == core(uninterrupted)
    assert resumed["configuration"] == uninterrupted["configuration"]
    assert resumed["game"]["status"] == "truncated"


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("call", [1, 3, 6])
def test_mid_search_fault_restores_boundary_rng(tmp_path, fixture_checkpoint, exception, call):
    reference = create(tmp_path, fixture_checkpoint, name="full").run()
    session = create(tmp_path, fixture_checkpoint, name="fault")
    initial = core(session.snapshot())
    original = session.evaluator
    calls = 0
    def fault(states):
        nonlocal calls
        calls += 1
        if calls == call:
            raise exception("forced fault")
        return original(states)
    session.evaluator = fault
    stopped = session.run()
    assert stopped["game"]["status"] == "interrupted" and stopped["game"]["outcome"] is None
    assert stopped["game"]["interruptions"][-1]["error_type"] == exception.__name__
    if call <= 3:
        assert core(stopped) == initial
    assert core(Session.load(session.directory).run()) == core(reference)


def test_post_push_interrupt_does_not_commit_candidate(tmp_path, fixture_checkpoint, monkeypatch):
    session = create(tmp_path, fixture_checkpoint)
    before = core(session.snapshot())
    original = impl.apply_move
    def fail(state, move_id):
        original(state, move_id)
        if state.initial_fen == chess.STARTING_FEN:
            raise KeyboardInterrupt("after push")
    monkeypatch.setattr(impl, "apply_move", fail)
    assert core(session.run()) == before
    monkeypatch.setattr(impl, "apply_move", original)
    assert Session.load(session.directory).run()["game"]["status"] == "truncated"


@pytest.mark.parametrize("published", [False, True])
def test_uncertain_publication_reloads_whole_old_or_new_boundary(tmp_path, fixture_checkpoint, monkeypatch, published):
    reference = create(tmp_path, fixture_checkpoint, name="full").run()
    session = create(tmp_path, fixture_checkpoint, name="crash")
    original = impl._write
    def fail(path, value, **kwargs):
        if published:
            original(path, value, **kwargs)
        raise OSError("simulated process loss at publication")
    monkeypatch.setattr(impl, "_write", fail)
    with pytest.raises(OSError):
        session.run()
    monkeypatch.setattr(impl, "_write", original)
    restored = Session.load(session.directory)
    assert len(restored.record["positions"]) == int(published)
    assert core(restored.run()) == core(reference)


def test_stale_session_and_immutable_freeze(tmp_path, fixture_checkpoint):
    session = create(tmp_path, fixture_checkpoint)
    other = Session.load(session.directory)
    freeze = (session.directory / "configuration.json").read_bytes()
    session.run(stop_after_moves=1)
    with pytest.raises(ValueError, match="stale"):
        other.run()
    assert (session.directory / "configuration.json").read_bytes() == freeze
    with pytest.raises(FileExistsError):
        create(tmp_path, fixture_checkpoint)


def test_checkpoint_runtime_corruption_and_history_drift_rejected(tmp_path, fixture_checkpoint):
    session = create(tmp_path, fixture_checkpoint)
    raw = fixture_checkpoint.read_bytes()
    fixture_checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        Session.load(session.directory)
    fixture_checkpoint.write_bytes(raw)
    torch.set_num_threads(2)
    with pytest.raises(ValueError, match="runtime"):
        Session.load(session.directory)
    torch.set_num_threads(1)
    bad = session.snapshot()["game"]
    bad["moves_uci"] = ["e2e4"]
    with pytest.raises(ValueError):
        audit(session.configuration, bad)
    file = session.directory / "boundary.json"
    value = json.loads(file.read_text())
    value["content"]["outcome"] = "1/2-1/2"
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        Session.load(session.directory)


def test_complete_history_required(tmp_path, fixture_checkpoint):
    with pytest.raises(ValueError, match="complete"):
        create(tmp_path, fixture_checkpoint, make_state(chess.STARTING_FEN))


def test_summary_duplicate_completed_only(tmp_path, fixture_checkpoint):
    terminal = make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", history_complete=True)
    complete = create(tmp_path, fixture_checkpoint, terminal, name="complete").run()
    interrupted = create(tmp_path, fixture_checkpoint, name="interrupted").run(stop_after_moves=0)
    truncated = create(tmp_path, fixture_checkpoint, name="truncated", cap=0).run()
    report = summarize([complete, complete, interrupted, truncated])
    assert report["status_counts"] == dict(completed=2, truncated=1, interrupted=1)
    assert report["duplicate_completed_games"] == 1
    assert report["duplicate_completed_rate"] == .5
    assert report["status_rates"]["interrupted"] == .25
    bad = copy.deepcopy(interrupted)
    bad["game"]["outcome"] = "1/2-1/2"
    with pytest.raises(ValueError):
        summarize([bad])


@pytest.mark.parametrize("kwargs", [dict(simulation_budget=0), dict(ply_cap=-1),
    dict(simulation_budget=True), dict(phases=()), dict(phases=(Phase(1, 1., 0.),)),
    dict(phases=(Phase(0, 1., 0.), Phase(0, 0., 0.)))])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)
