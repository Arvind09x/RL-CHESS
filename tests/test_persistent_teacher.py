"""Process reuse, reset order, invalid-response recovery and loader admission."""
import asyncio
import json
from pathlib import Path

import chess
import pytest

from chess_rl.data_manifest import DatasetManifest
from chess_rl.stockfish_worker import PersistentEngine
from chess_rl.supervised import open_dataset
from .test_stockfish_worker import FakeAnalysis, fixture_game
from .test_supervised import fixture_bundle, read_rows, write_rows, seal_bundle


def test_reuse_reset_and_restart_after_invalid_response(tmp_path, monkeypatch):
    from dataclasses import replace
    from chess_rl.pgn_replay import file_identity
    executable = tmp_path / "engine"
    executable.write_bytes(b"controlled engine")
    manifest = DatasetManifest.load(Path(__file__).resolve().parents[1] / "data/manifests/elite-2025-01-d04-v1.json")
    manifest = replace(manifest, engine=replace(manifest.engine, executable_sha256=file_identity(executable)["sha256"]))
    _, _, state = fixture_game(manifest, 1)
    events, protocols = [], []
    class Transport:
        def close(self): events.append("close")
    class Protocol:
        id = {}
        options = {}
        transport = Transport()
        mode = "success"
        def send_line(self, line): events.append(line)
        async def configure(self, options): events.append(options)
        async def ping(self): events.append("ready")
        async def analysis(self, board, limit, **kwargs):
            assert kwargs["game"] is not None
            events.append("search")
            return FakeAnalysis(state, self.mode)
        async def quit(self): events.append("quit")
    async def spawn(*args, **kwargs):
        protocol = Protocol()
        protocols.append(protocol)
        return protocol.transport, protocol
    monkeypatch.setattr(chess.engine, "popen_uci", spawn)
    monkeypatch.setattr("chess_rl.stockfish_worker.configure_options", lambda *a: {})
    async def run():
        async with PersistentEngine(manifest, executable) as pool:
            for _ in range(2):
                trace = {}
                await pool.attempt(state, trace)
                assert trace["completed"] and trace["process_healthy_after_search"]
            assert len(protocols) == 1
            protocols[0].mode = "missing_final_wdl"
            with pytest.raises(ValueError, match="WDL"):
                await pool.attempt(state, {})
            await pool.attempt(state, {})
            assert len(protocols) == 2
    asyncio.run(run())
    for i, event in enumerate(events):
        if event == "search":
            assert events[i-3:i] == ["ucinewgame", {"Clear Hash": None}, "ready"]
    assert events.count("close") == 2


@pytest.mark.parametrize("bad", [None, "missing_reset", "unhealthy"])
def test_persistent_label_requires_matching_process_policy_and_raw_evidence(tmp_path, bad):
    args = fixture_bundle(tmp_path)
    bundle = args["bundle"]
    settings = json.loads((bundle / "settings.json").read_text())
    settings["process_policy"] = "persistent-clear-hash-v1"
    (bundle / "settings.json").write_text(json.dumps(settings))
    labels, attempts = read_rows(bundle / "labels.jsonl"), read_rows(bundle / "attempts.jsonl")
    manifest = DatasetManifest.load(args["manifest_path"])
    for label, attempt in zip(labels, attempts):
        label["target_provenance"]["process_policy"] = settings["process_policy"]
        ref = attempt["reference"]
        games = read_rows(args["games_path"])
        game = next(g for g in games if g["game_id"] == ref["game_id"])
        position = "position fen " + game["start_fen"]
        if game["start_fen"] == chess.STARTING_FEN:
            position = "position startpos"
        if ref["sample_ply"]:
            position += " moves " + " ".join(game["moves_uci"][:ref["sample_ply"]])
        returned = attempt["returned"]
        returned.update(process_policy=settings["process_policy"], process_healthy_after_search=True,
            uci_sent=["ucinewgame", "setoption name Clear Hash", "isready", position,
                      f"go nodes {manifest.teacher.nodes}", "isready"])
        if bad == "missing_reset": returned["uci_sent"].pop(0)
        if bad == "unhealthy": returned["process_healthy_after_search"] = False
    write_rows(bundle / "labels.jsonl", labels)
    write_rows(bundle / "attempts.jsonl", attempts)
    seal_bundle(bundle)
    if bad:
        with pytest.raises(ValueError, match="Persistent"):
            open_dataset(**args, split="train")
    else:
        assert len(open_dataset(**args, split="train")) == 3
