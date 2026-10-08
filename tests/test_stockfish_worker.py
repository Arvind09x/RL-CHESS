"""Controlled D05 fixtures: no Stockfish is needed by the regression suite."""
import asyncio
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path

import chess
import chess.engine
import pytest

from chess_rl.adapter import make_state
from chess_rl.canonical_games import canonical_payload, game_id, assign_split
from chess_rl.data_manifest import DatasetManifest
from chess_rl.position_sampling import phase, rule_position, sample_record
from chess_rl.stockfish_worker import (AnnotationWorker, AttemptFailure, configure_options,
    engine_attempt, replay_identity, replay_root, validate_response, verify_artifacts,
    verify_teacher_bytes)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest():
    return DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-d04-v1.json")


def fixture_game(manifest, ply=0, moves=None):
    moves = moves or ["e2e4", "e7e5", "g1f3", "b8c6"]
    # Find an explicitly assigned train game, without changing split rules.
    for clock in range(1, 40):
        fen = chess.STARTING_FEN.rsplit(" ", 1)[0] + f" {clock}"
        identity = game_id(canonical_payload(fen, moves))
        if assign_split(identity, manifest.split_seed) == "train":
            break
    game = dict(game_id=identity, split="train", start_fen=fen, moves_uci=moves,
        dataset_id=manifest.dataset_id, encoding_version=manifest.encoding_version,
        move_vocab_version=manifest.move_vocab_version, ending_policy=manifest.ending_policy,
        history_complete=True, history_origin="declared-pgn-start", record_version="replay-jsonl-v1",
        wdl_order=list(manifest.wdl_order))
    state = make_state(fen, moves[:ply], history_complete=True)
    ref = sample_record(identity, "train", (ply, phase(state.board, ply), rule_position(state.board)))
    return game, ref, state


def response(state, turn=chess.WHITE):
    move = next(iter(state.board.legal_moves))
    return dict(multipv=1, wdl=chess.engine.PovWdl(chess.engine.Wdl(700, 200, 100), turn),
        pv=[move], score=chess.engine.PovScore(chess.engine.Cp(125), turn),
        depth=12, seldepth=17, nodes=100123, nps=90000, hashfull=1, tbhits=0, time=1.1), move


@pytest.mark.parametrize("ply", [0, 1])
@pytest.mark.parametrize("turn", [chess.WHITE, chess.BLACK])
def test_perspective_and_roundtrip(manifest, ply, turn):
    _, _, state = fixture_game(manifest, ply)
    info, move = response(state, turn)
    before = replay_identity(state)
    label = validate_response(state, info, move)
    assert label["teacher_wdl_counts"] == ([700, 200, 100] if state.board.turn == turn else [100, 200, 700])
    assert sum(label["teacher_wdl"]) == pytest.approx(1)
    assert replay_identity(state) == before


@pytest.mark.parametrize("field,value", [
    ("wdl", None), ("wdl", [700, 200, 100]),
    ("wdl", chess.engine.PovWdl(chess.engine.Wdl(-1, 500, 501), True)),
    ("wdl", chess.engine.PovWdl(chess.engine.Wdl(700, 200, 99), True)),
    ("pv", []), ("pv", [chess.Move.from_uci("e2e5")]),
    ("pv", [chess.Move.null()]), ("score", None), ("multipv", 2),
    ("multipv", True), ("score", chess.engine.PovScore(chess.engine.Cp(float("nan")), True)),
    ("wdl", chess.engine.PovWdl(chess.engine.Wdl(700, 200, 100), None)),
    ("nodes", True), ("nodes", 0), ("time", float("nan")), ("tbhits", 1)])
def test_invalid_responses(manifest, field, value):
    _, _, state = fixture_game(manifest)
    info, move = response(state)
    info[field] = value
    with pytest.raises(ValueError):
        validate_response(state, info, move)


@pytest.mark.parametrize("best", [None, chess.Move.null(), chess.Move.from_uci("e2e5"), chess.Move.from_uci("e2e4")])
def test_invalid_bestmove(manifest, best):
    _, _, state = fixture_game(manifest)
    info, _ = response(state)
    with pytest.raises(ValueError):
        validate_response(state, info, best)


def test_illegal_pv_suffix(manifest):
    _, _, state = fixture_game(manifest)
    info, move = response(state)
    info["pv"].append(move)
    with pytest.raises(ValueError, match="PV"):
        validate_response(state, info, move)


@pytest.mark.parametrize("failure", [TimeoutError("deadline"), AttemptFailure("missing", {"wdl": None}), ValueError("malformed")])
def test_retry_preserves_failures_and_deduplicates(manifest, failure):
    game, ref, state = fixture_game(manifest)
    attempts = []

    async def attempt(root, trace):
        trace.update(hash_reset_completed=True, completed=True)
        if not attempts:
            raise failure
        return response(root)

    worker = AnnotationWorker(manifest, {}, attempt, attempts.append)
    label = asyncio.run(worker.annotate(ref, game))
    assert label["successful_attempt"] == 2
    assert [r["status"] for r in attempts] == ["failed", "success"]
    assert attempts[0]["error"]["type"] == type(failure).__name__
    with pytest.raises(ValueError, match="Duplicate"):
        asyncio.run(worker.annotate(ref, game))
    assert len(attempts) == 2


def test_exhaustion(manifest):
    game, ref, _ = fixture_game(manifest)
    attempts = []

    async def attempt(root, trace):
        raise TimeoutError("not completed")

    worker = AnnotationWorker(manifest, {}, attempt, attempts.append, max_attempts=3)
    assert asyncio.run(worker.annotate(ref, game)) is None
    assert len(attempts) == 3 and all(r["status"] == "failed" for r in attempts)


@pytest.mark.parametrize("error", [KeyboardInterrupt, asyncio.CancelledError])
def test_interruption_preserved_and_propagated(manifest, error):
    game, ref, _ = fixture_game(manifest)
    attempts = []

    async def attempt(root, trace):
        trace["partial"] = "never a target"
        raise error()

    worker = AnnotationWorker(manifest, {}, attempt, attempts.append)
    with pytest.raises(error):
        asyncio.run(worker.annotate(ref, game))
    assert len(attempts) == 1 and attempts[0]["status"] == "interrupted"
    assert attempts[0]["returned"]["partial"] == "never a target"


@pytest.mark.parametrize("field", ["completed", "hash_reset_completed"])
def test_partial_or_no_reset_never_success(manifest, field):
    game, ref, _ = fixture_game(manifest)
    attempts = []

    async def attempt(root, trace):
        trace.update(completed=True, hash_reset_completed=True)
        trace[field] = False
        return response(root)

    assert asyncio.run(AnnotationWorker(manifest, {}, attempt, attempts.append).annotate(ref, game)) is None
    assert all(r["status"] == "failed" for r in attempts)


def test_root_rejections_and_history_identity(manifest):
    game, ref, state = fixture_game(manifest, 4)
    assert len(replay_root(ref, game, manifest).board.move_stack) == 4
    assert replay_identity(state) != replay_identity(make_state(state.board.fen(), history_complete=True))
    for bad in (dict(ref, sample_ply=True), dict(ref, sample_ply=1000), dict(ref, split="test"),
                dict(ref, phase="endgame"), dict(ref, rule_position="mismatch")):
        with pytest.raises(ValueError):
            replay_root(bad, game, manifest)
    game, ref, _ = fixture_game(manifest, 4, ["f2f3", "e7e5", "g2g4", "d8h4"])
    with pytest.raises(ValueError, match="Terminal"):
        replay_root(ref, game, manifest)


def test_mutation_rejected(manifest):
    game, ref, _ = fixture_game(manifest)
    attempts = []

    async def attempt(root, trace):
        trace.update(completed=True, hash_reset_completed=True)
        root.board.push(next(iter(root.board.legal_moves)))
        return response(root)

    assert asyncio.run(AnnotationWorker(manifest, {}, attempt, attempts.append).annotate(ref, game)) is None


class FakeAnalysis:
    def __init__(self, state, mode):
        self.info, self.move = response(state)
        self.mode = mode

    async def __aiter__(self):
        if self.mode == "malformed_uci":
            # Exercise the real pinned parser: valid later output cannot erase
            # a malformed WDL diagnostic in this attempt.
            chess.engine._parse_uci_info("depth 9 wdl garbage 0 0", chess.Board(), chess.engine.INFO_ALL)
        if self.mode == "truncated_pv":
            logging.getLogger("chess.engine").debug("%s: >> %s", "fixture", "info depth 9 multipv 1 pv g8f6 malformed")
        yield self.info
        if self.mode == "timeout":
            await asyncio.sleep(1)
        elif self.mode == "missing_final_wdl":
            yield {k: v for k, v in self.info.items() if k != "wdl"}
        elif self.mode == "interruption":
            raise asyncio.CancelledError()

    async def wait(self):
        return chess.engine.BestMove(self.move, None)


@pytest.mark.parametrize("mode", ["success", "timeout", "interruption", "missing_final_wdl", "malformed_uci", "truncated_pv", "shutdown_interruption"])
def test_transport_resets_and_completion(manifest, monkeypatch, mode):
    _, _, state = fixture_game(manifest, 1)
    events = []
    class Transport:
        def close(self):
            events.append("close")
    class Protocol:
        transport = Transport()
        id = dict(name=manifest.engine.uci_name, author=manifest.engine.uci_author)
        options = {name: chess.engine.Option(name, "spin", 1, 1, 1024, None) for name in ("Threads", "Hash", "MultiPV")}
        options.update({name: chess.engine.Option(name, "check", False, None, None, None) for name in ("UCI_ShowWDL", "Ponder", "UCI_Chess960", "UCI_LimitStrength")})
        options["Clear Hash"] = chess.engine.Option("Clear Hash", "button", None, None, None, None)
        options["SyzygyPath"] = chess.engine.Option("SyzygyPath", "string", "", None, None, None)
        for name, asset in zip(("EvalFile", "EvalFileSmall"), manifest.engine.embedded_nnue):
            options[name] = chess.engine.Option(name, "string", asset.name, None, None, None)
        async def configure(self, options):
            assert "MultiPV" not in options and "Ponder" not in options and "UCI_Chess960" not in options
            events.append(options)
        async def ping(self):
            events.append("ping")
        async def analysis(self, board, limit, **kwargs):
            assert len(board.move_stack) == 1 and limit.nodes == 100000
            assert kwargs["multipv"] == 1
            events.append("search")
            return FakeAnalysis(state, mode)
        async def quit(self):
            events.append("quit")
            if mode == "shutdown_interruption":
                raise asyncio.CancelledError()
    async def open_engine(*args, **kwargs):
        protocol = Protocol()
        protocol.returncode = asyncio.get_running_loop().create_future()
        protocol.returncode.set_result(0)
        return protocol.transport, protocol
    monkeypatch.setattr(chess.engine, "popen_uci", open_engine)
    trace = {}
    if mode in ("timeout", "interruption", "malformed_uci", "truncated_pv", "shutdown_interruption"):
        error = {"timeout": TimeoutError, "interruption": asyncio.CancelledError, "malformed_uci": AttemptFailure,
                 "truncated_pv": AttemptFailure, "shutdown_interruption": asyncio.CancelledError}[mode]
        with pytest.raises(error):
            asyncio.run(engine_attempt(state, manifest, Path("fake"), .02, trace))
        assert trace["returned_rows"]
        if mode in ("malformed_uci", "truncated_pv"):
            assert trace["transport_errors"]
        elif mode != "shutdown_interruption":
            assert not trace.get("completed")
    else:
        info, move = asyncio.run(engine_attempt(state, manifest, Path("fake"), .02, trace))
        if mode == "missing_final_wdl":
            with pytest.raises(ValueError, match="WDL"):
                validate_response(state, info, move)
        else:
            validate_response(state, info, move)
    assert events.index({"Clear Hash": None}) < events.index("ping") < events.index("search") < events.index("close")


def test_historical_manifest_blocked():
    manifest = DatasetManifest.load(ROOT / "data/manifests/elite-2025-01-v1.json")
    with pytest.raises(ValueError):
        AnnotationWorker(manifest, {}, None, None)


@pytest.mark.parametrize("bound", [0, 4, True, 1.5])
def test_retry_bound(manifest, bound):
    with pytest.raises(ValueError):
        AnnotationWorker(manifest, {}, None, None, max_attempts=bound)


def sealed_artifacts(tmp_path, manifest, monkeypatch):
    from chess_rl.pgn_replay import file_identity
    inputs = {"games.jsonl": {"sha256": "f"*64, "bytes": 10},
              "split-audit.json": {"sha256": manifest.preparation.split_audit_sha256, "bytes": 10}}
    for name in ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl"):
        (tmp_path / name).write_text("sealed fixture " + name)
    audit = dict(audit_version="position-overlap-audit-v1", complete=True,
        dataset_id=manifest.dataset_id, sampling_version="phase-spread-sha256-v1",
        sampling_seed=manifest.sampling_seed, sampling_algorithm=manifest.preparation.sampling_algorithm,
        phase_boundaries=manifest.preparation.phase_boundaries, overlap_identity=manifest.overlap_policy,
        frozen_registry_version="protected-heldout-v1", cross_split_overlaps_after=0,
        labels_produced=0, assignments_sha256="frozen assignments", inputs=inputs,
        outputs={name: file_identity(tmp_path / name) for name in
                 ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl")})
    (tmp_path / "overlap-audit.json").write_text(json.dumps(audit))
    manifest = replace(manifest, preparation=replace(manifest.preparation,
        overlap_audit_sha256=file_identity(tmp_path / "overlap-audit.json")["sha256"],
        protected_heldout_registry_sha256=file_identity(tmp_path / "protected-heldout.jsonl")["sha256"]))
    (tmp_path / "manifest.json").write_text(manifest.to_json())
    monkeypatch.setattr("chess_rl.stockfish_worker.verify_intake", lambda *args: (
        {"assignments_sha256": "frozen assignments"}, dict(inputs)))
    return manifest


@pytest.mark.parametrize("tamper", [None, "samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl", "overlap-audit.json", "manifest.json"])
def test_artifact_gate(manifest, tmp_path, monkeypatch, tamper):
    manifest = sealed_artifacts(tmp_path, manifest, monkeypatch)
    if tamper:
        with (tmp_path / tamper).open("a") as handle:
            handle.write("tamper")
    def verify():
        return verify_artifacts(manifest, tmp_path / "manifest.json", tmp_path / "games.jsonl", tmp_path / "split-audit.json", tmp_path)
    if tamper:
        with pytest.raises(ValueError):
            verify()
    else:
        identities = verify()
        assert "protected-heldout.jsonl" in identities and "heldout-games.jsonl" in identities


@pytest.mark.parametrize("tamper", [None, "binary", "asset"])
def test_teacher_identity_gate(manifest, tmp_path, tamper):
    blobs = [b"first embedded asset", b"second embedded asset"]
    binary = b"header" + b"".join(blobs)
    assets = tuple(replace(asset, sha256=hashlib.sha256(blob).hexdigest(),
        name=f"nn-{hashlib.sha256(blob).hexdigest()[:12]}.nnue",
        source_url=f"https://tests.stockfishchess.org/api/nn/nn-{hashlib.sha256(blob).hexdigest()[:12]}.nnue")
        for asset, blob in zip(manifest.engine.embedded_nnue, blobs))
    manifest = replace(manifest, engine=replace(manifest.engine,
        executable_sha256=hashlib.sha256(binary).hexdigest(), embedded_nnue=assets))
    executable = tmp_path / "engine"
    executable.write_bytes(binary + (b"changed" if tamper == "binary" else b""))
    offsets = [6, 6+len(blobs[0])]
    evidence = dict(executable_sha256=manifest.engine.executable_sha256,
        nnue=[dict(name=a.name, sha256=a.sha256, bytes=len(blob), embedded_offset=offset)
              for a, blob, offset in zip(assets, blobs, offsets)])
    if tamper == "asset":
        evidence["nnue"][0]["embedded_offset"] += 1
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence))
    if tamper:
        with pytest.raises(ValueError):
            verify_teacher_bytes(manifest, executable, path)
    else:
        assert len(verify_teacher_bytes(manifest, executable, path)["embedded_nnue"]) == 2


def test_same_overlap_key_is_not_a_teacher_cache(manifest):
    game, first, _ = fixture_game(manifest, 0, ["g1f3", "g8f6", "f3g1", "f6g8"])
    _, second, _ = fixture_game(manifest, 4, ["g1f3", "g8f6", "f3g1", "f6g8"])
    assert first["rule_position"] == second["rule_position"]
    calls, attempts = [], []
    async def attempt(root, trace):
        calls.append(replay_identity(root))
        trace.update(completed=True, hash_reset_completed=True)
        return response(root)
    worker = AnnotationWorker(manifest, {}, attempt, attempts.append)
    a = asyncio.run(worker.annotate(first, game))
    b = asyncio.run(worker.annotate(second, game))
    assert len(calls) == 2 and calls[0] != calls[1]
    assert a["root_fen"] != b["root_fen"] and a["replay_sha256"] != b["replay_sha256"]
