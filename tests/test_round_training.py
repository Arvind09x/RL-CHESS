"""Round-1 training: compact self-play arrays, both recipes, checkpoints, resume and selection (CPU)."""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.adapter import encode_position, make_state
from chess_rl.dataset_cache import CachedSupervisedDataset
from chess_rl.replay import ReplayBuffer
from chess_rl.selfplay import Phase
from chess_rl.supervised import open_dataset
from chess_rl.training import checkpoints as C
from chess_rl.training import round_training as R
from chess_rl.training.checkpoints import (CheckpointError, DataOrder, ModelConfig, Progress, load_inference,
                                           save_checkpoint)
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedSources, MixedState
from scripts import train_mixed as T
from scripts.train_supervised import validate
from .test_checkpoints import assert_rng_equal, assert_tree_equal, rng_snapshot
from .test_dataset_cache import build
from .test_replay import BLACK_WIN, WHITE_WIN, played
from .test_selfplay import cpu, create, fixture_checkpoint
from .test_supervised import EVIDENCE, ROOT, fixture_bundle

MODEL = ModelConfig(d_policy=23, d_value=19, dropout=0.23)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def specifications():
    # The validation game is a whole self-play game below (blocked); the test game shares the
    # position after e2e4 f7f6 with the white-win self-play game (one protected position).
    return [('train', chess.STARTING_FEN, ['e2e4', 'e7e5', 'g1f3', 'b8c6'], [1, 2, 3], None),
            ('validation', chess.STARTING_FEN, list(BLACK_WIN), [1, 2], None),
            ('test', chess.STARTING_FEN, ['e2e4', 'f7f6', 'b1c3', 'e7e5'], [2], None)]


def p_checkpoint(path, train):
    torch.manual_seed(7)
    model = MODEL.build()
    optimizer = R.build_round_optimizer(model)
    order = DataOrder("fixture-P", DataOrder.reference_hash(train), tuple(range(len(train))), 0, 1, False, 0)
    save_checkpoint(path, model=model, optimizer=optimizer, checkpoint_id="fixture-P", provenance={"fixture": "P"},
                    weights=R.SETTINGS.weights, progress=Progress(0, 0, 0), data_order=order)
    return path


@pytest.fixture
def world(tmp_path, fixture_checkpoint):
    """Supervised bundle with both caches, a real small P, and a replay buffer of games 'played by P'."""
    paths = fixture_bundle(tmp_path, specifications())
    summary_path = Path(paths["bundle"]) / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["loader_acceptance"] = {"train": 3, "validation": 2, "test": 1}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    build(paths, "train", "validation")
    train = open_dataset(**paths, split="train").freeze()
    p = p_checkpoint(tmp_path / "p.pt", train)
    games = [json.loads(line) for line in Path(paths["games_path"]).read_text().splitlines()]
    validation_game = next(g for g in games if g["split"] == "validation")
    buffer = ReplayBuffer.create(tmp_path / "replay.sqlite")
    blocked = played(tmp_path, p, BLACK_WIN, name="blocked",
                     state=make_state(validation_game["start_fen"], history_complete=True))
    buffer.ingest(blocked.snapshot())
    buffer.ingest(played(tmp_path, p, WHITE_WIN, name="white").snapshot())
    session = create(tmp_path, p, make_state('4k3/8/8/8/8/8/8/R3K3 w - - 99 1', history_complete=True),
                     name='soft-draw', cap=1, budget=3, phases=(Phase(0, 0., 0.),))
    buffer.ingest(session.run())
    buffer.ingest(create(tmp_path, p, name='truncated', cap=1, budget=1).run())
    config = tmp_path / "configs" / "data.json"
    config.parent.mkdir()
    config.write_text(json.dumps(dict(version="cuda-portable-data-v1", paths={
        name: os.path.relpath(value, config.parent) for name, value in paths.items()})))
    return dict(tmp=tmp_path, paths=paths, train=train, p=p, p_sha=R._sha256(p), buffer=tmp_path / "replay.sqlite",
                config=config)


def round_data(world, name="r1-data", holdout=R.held_out):
    cached = CachedSupervisedDataset.open(split="train", **world["paths"])
    return R.build_round_data(world["tmp"] / name, buffer=ReplayBuffer(world["buffer"]),
                              prepared=world["paths"]["prepared"], identity=cached.source_identity,
                              checkpoint_sha256=world["p_sha"], source={"fixture": True}, holdout=holdout)


def library_sources(world):
    prepared = world["paths"]["prepared"]
    return MixedSources(open_dataset(**world["paths"], split="train"), ReplayBuffer(world["buffer"]),
                        heldout_games_path=prepared / "heldout-games.jsonl",
                        protected_positions_path=prepared / "protected-heldout.jsonl")


def training_model(world):
    model = load_inference(world["p"]).model
    model.train()
    return model


# Arrays and filtering.

def test_holdout_rule_is_fixed():
    assert R.held_out(hashlib.sha256(b"1").hexdigest())
    assert not R.held_out("0" * 64) and not R.held_out("f" * 64)
    assert int(hashlib.sha256(("r1-holdout-v1" + "0" * 64).encode()).hexdigest()[:16], 16) % 10 == 1


def test_filtering_matches_mixed_sources_and_counts_are_separate(world):
    data = round_data(world, holdout=None)
    library = library_sources(world)
    assert data.train.sample_ids == library.replay_membership
    counts = data.manifest["counts"]
    assert counts["blocked_games"] == 1 and counts["blocked_game_positions"] == len(BLACK_WIN)
    assert counts["protected_positions"] == 3 and counts["protected_positions_outside_blocked_games"] == 1
    assert len(library.excluded_heldout) == counts["blocked_game_positions"] + counts["protected_positions_outside_blocked_games"]
    assert counts["training_positions"] == len(data.train) == len(WHITE_WIN) - 1 + 1
    assert len(data.holdout) == 0 and data.manifest["holdout"]["rule"] is None
    _, ours = library.batches([0], range(len(data.train)))
    mine = data.train.batch(range(len(data.train)))
    for name in ("sample_ids", "features", "legal_mask", "policy_distributions", "wdl"):
        a, b = getattr(ours, name), getattr(mine, name)
        assert (a == b) if name == "sample_ids" else (a.dtype == b.dtype and np.array_equal(a, b))


def test_arrays_round_trip_exactly_to_the_buffer_samples(world):
    data = round_data(world, holdout=None)
    samples = {s["sample_id"]: s for s in ReplayBuffer(world["buffer"]).iter_samples()}
    batch = data.train.batch(range(len(data.train)))
    for row, sample_id in enumerate(data.train.sample_ids):
        sample = samples[sample_id]
        state = make_state(sample["initial_fen"], sample["moves_uci"], history_complete=True)
        assert np.array_equal(batch.features[row], encode_position(state).features)
        assert np.array_equal(batch.legal_mask[row], np.array(sample["position"]["legal_mask"], dtype=bool))
        assert np.array_equal(batch.policy_distributions[row], np.array(sample["position"]["visit_policy"], dtype=np.float32))
        assert np.array_equal(batch.wdl[row], np.array(sample["wdl"], dtype=np.float32))
        assert data.train.game_ids[row] == sample["game_id"] and data.train.plies[row] == sample["position"]["ply"]


def test_holdout_takes_whole_games_after_filtering(world):
    white = next(s["game_id"] for s in ReplayBuffer(world["buffer"]).iter_samples() if s["moves_uci"][:1] == ["e2e4"])
    data = round_data(world, holdout=lambda g: g == white)
    counts = data.manifest["counts"]
    assert set(data.holdout.game_ids) == {white} and white not in data.train.game_ids
    assert counts["holdout_games"] == 1 and counts["holdout_positions"] == len(WHITE_WIN) - 1
    assert counts["training_positions"] == len(data.train) == 1
    production = round_data(world, name="r1-production")
    assert production.manifest["holdout"]["rule"] == R.HOLDOUT_RULE
    assert all(R.held_out(g) for g in production.holdout.game_ids)
    assert not any(R.held_out(g) for g in production.train.game_ids)


def test_round_data_is_verified_on_open_and_never_rewritten(world):
    data = round_data(world, holdout=None)
    with pytest.raises(ValueError, match="never rewritten"):
        round_data(world, holdout=None)
    copy = world["tmp"] / "tampered"
    shutil.copytree(data.directory, copy)
    path = copy / "train" / "policy_values.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="altered"):
        R.RoundData.open(copy)
    registries = dict(data.manifest["registries"])
    registries["heldout-games.jsonl"] = dict(registries["heldout-games.jsonl"], sha256="0" * 64)
    with pytest.raises(ValueError, match="registries"):
        R.RoundData.open(data.directory, registries=registries)


def test_registry_identity_must_match_the_dataset(world):
    cached = CachedSupervisedDataset.open(split="train", **world["paths"])
    identity = cached.source_identity
    identity["authoritative"]["protected-heldout.jsonl"] = dict(identity["authoritative"]["protected-heldout.jsonl"], bytes=1)
    with pytest.raises(ValueError, match="registry identity"):
        R.build_round_data(world["tmp"] / "x", buffer=ReplayBuffer(world["buffer"]), prepared=world["paths"]["prepared"],
                           identity=identity, checkpoint_sha256=world["p_sha"], source={})
    assert not (world["tmp"] / "x").exists()


# Updates.

def test_mix_step_is_bit_identical_to_the_library_mixed_step(world):
    data = round_data(world, holdout=None)
    library = library_sources(world)
    cached = CachedSupervisedDataset.open(split="train", **world["paths"])
    lib_state = MixedState.initial(library, R.SETTINGS, seed=R.SAMPLING_SEED)
    state = MixedState.initial(R.sampler_sources(cached, data), R.SETTINGS, seed=R.SAMPLING_SEED)
    lib_model, model = training_model(world), training_model(world)
    lib_opt, opt = R.build_round_optimizer(lib_model), R.build_round_optimizer(model)
    lib_metrics, metrics = [], []
    torch.manual_seed(11)
    for _ in range(3):
        m, lib_state = mixed_step(lib_model, lib_opt, library, lib_state, max_gradient_norm=10.0)
        lib_metrics.append(m)
    torch.manual_seed(11)
    for _ in range(3):
        record, state = R.round_step(model, opt, recipe="mix", supervised=cached, rows=data.train, state=state,
                                     max_gradient_norm=10.0)
        metrics.append(record)
    assert_tree_equal(lib_model.state_dict(), model.state_dict())
    assert_tree_equal(lib_opt.state_dict(), opt.state_dict())
    for a, b in zip(lib_metrics, metrics):
        assert b["losses"]["total_loss"] == a.total_loss and b["gradient_norm"] == a.gradient_norm
        assert b["losses"]["stockfish"]["policy_loss"] == a.supervised.policy_loss
        assert b["losses"]["selfplay"]["wdl_loss"] == a.selfplay.wdl_loss and b["clipped"] == a.clipped
    assert (state.supervised, state.selfplay, state.sampling_rng, state.updates) == \
           (lib_state.supervised, lib_state.selfplay, lib_state.sampling_rng, lib_state.updates)


def test_pure_and_mix_draw_identical_selfplay_sequences_across_wraps(world):
    data = round_data(world, holdout=None)
    cached = CachedSupervisedDataset.open(split="train", **world["paths"])
    initial = MixedState.initial(R.sampler_sources(cached, data), R.SETTINGS, seed=R.SAMPLING_SEED)
    runs = {}
    for recipe in R.RECIPES:
        model = training_model(world)
        opt = R.build_round_optimizer(model)
        state, records = initial, []
        torch.manual_seed(3)
        for _ in range(3):
            record, state = R.round_step(model, opt, recipe=recipe, supervised=cached, rows=data.train, state=state,
                                         max_gradient_norm=10.0)
            records.append(record)
        runs[recipe] = (state, records)
    (mix_state, mix), (pure_state, pure) = runs["mix"], runs["pure"]
    assert mix_state == pure_state and pure_state.selfplay.epoch > 3  # 24 draws per update from 5 rows
    assert [r["selfplay_indices"] for r in mix] == [r["selfplay_indices"] for r in pure]
    assert [r["stockfish_indices"] for r in mix] == [r["discarded_stockfish_indices"] for r in pure]
    assert all(r["trained_positions"] == dict(selfplay=24, stockfish=8) and r["discarded_sampler_indices"] == 0 for r in mix)
    assert all(r["trained_positions"] == dict(selfplay=24, stockfish=0) and r["discarded_sampler_indices"] == 8 for r in pure)
    assert pure[0]["losses"]["total_loss"] == pytest.approx(
        pure[0]["losses"]["selfplay"]["policy_loss"] + pure[0]["losses"]["selfplay"]["wdl_loss"])


def test_failed_input_does_not_advance_the_sampler(world):
    data = round_data(world, holdout=None)
    cached = CachedSupervisedDataset.open(split="train", **world["paths"])
    state = MixedState.initial(R.sampler_sources(cached, data), R.SETTINGS, seed=R.SAMPLING_SEED)
    model = training_model(world)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        R.round_step(model, opt, recipe="pure", supervised=cached, rows=data.train, state=state, max_gradient_norm=10.0)
    assert state.updates == 0 and all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


# Checkpoints and restore.

def saved_round(world, tmp_path, recipe="pure", updates=2):
    data = round_data(world, holdout=None)
    cached = CachedSupervisedDataset.open(split="train", **world["paths"])
    state = MixedState.initial(R.sampler_sources(cached, data), R.SETTINGS, seed=R.SAMPLING_SEED)
    torch.manual_seed(5)
    model = training_model(world)
    opt = R.build_round_optimizer(model)
    for _ in range(updates):
        _, state = R.round_step(model, opt, recipe=recipe, supervised=cached, rows=data.train, state=state,
                                max_gradient_norm=10.0)
    provenance = dict(recipe=recipe, fixture=True)
    path = tmp_path / "round.pt"
    save_checkpoint(path, model=model, optimizer=opt, checkpoint_id="fixture-round", provenance=provenance,
                    weights=R.SETTINGS.weights, progress=Progress(updates, 0, 32 * updates), mixed_state=state)
    expected = dict(expected_config=MODEL, expected_provenance=provenance, expected_settings=R.SETTINGS,
                    expected_identity=R.sampler_identity(state))
    return path, expected, model, opt, state


def test_restore_round_restores_everything_and_installs_rng_last(world, tmp_path):
    path, expected, model, opt, state = saved_round(world, tmp_path)
    payload = torch.load(path, weights_only=True)
    torch.manual_seed(999)
    restored = R.restore_round(path, **expected)
    assert restored.updates == 2 and restored.state == state and restored.model.training
    assert_tree_equal(restored.model.state_dict(), model.state_dict())
    assert_tree_equal(restored.optimizer.state_dict(), opt.state_dict())
    for parameter, values in restored.optimizer.state.items():
        assert values["step"].device.type == "cpu" and values["step"].item() == 2
        assert all(t.device == parameter.device for n, t in values.items() if n != "step")
    assert torch.equal(torch.get_rng_state(), payload["training"]["rng"]["torch"])
    assert C._rng_state()["python"] == payload["training"]["rng"]["python"]


@pytest.mark.parametrize("change", ["provenance", "settings", "identity", "config", "runtime"])
def test_restore_round_refuses_mismatches_and_leaves_rng_unchanged(world, tmp_path, monkeypatch, change):
    path, expected, *_ = saved_round(world, tmp_path)
    if change == "provenance":
        expected["expected_provenance"] = dict(expected["expected_provenance"], recipe="mix")
    elif change == "settings":
        expected["expected_settings"] = MixedSettings(supervised_fraction=0.5, batch_size=32, supervised_batch_size=16)
    elif change == "identity":
        expected["expected_identity"] = dict(expected["expected_identity"], replay_id="0" * 64)
    elif change == "config":
        expected["expected_config"] = replace(MODEL, dropout=0.1)
    else:
        real = R.runtime_config
        monkeypatch.setattr(R, "runtime_config", lambda: dict(real(), threads=real()["threads"] + 1))
    random_state = rng_snapshot()
    with pytest.raises(CheckpointError):
        R.restore_round(path, **expected)
    assert_rng_equal(random_state, rng_snapshot())


def test_round_checkpoints_load_for_inference(world, tmp_path):
    path, _, model, *_ = saved_round(world, tmp_path, recipe="mix")
    loaded = load_inference(path)
    model.eval()
    features = torch.tensor(np.stack([open_dataset(**world["paths"], split="validation").freeze()[i].features
                                      for i in range(2)]))
    with torch.inference_mode():
        for a, b in zip(model(features), loaded.model(features)):
            assert torch.equal(a, b)
    assert loaded.provenance["recipe"] == "mix" and not loaded.model.training


# Selection, statistics and run files.

def test_candidate_selection():
    e = lambda u, d: dict(update=u, checkpoint=f"update-{u:06d}.pt", delta_total=d)
    done = R.select_candidate([e(250, 0.01), e(300, 0.06)], completed=True)
    assert done["candidate"] == "update-000300.pt" and done["final_flag"] and not done["stopped"]
    quiet = R.select_candidate([e(250, 0.01), e(300, 0.04)], completed=True)
    assert quiet["candidate"] == "update-000300.pt" and not quiet["final_flag"]
    stop = R.select_candidate([e(250, 0.1), e(500, 0.25), e(750, 0.2501)], completed=False)
    assert stop["stopped"] and stop["stopped_at"] == 750 and stop["candidate"] == "update-000500.pt"
    first = R.select_candidate([e(250, 0.3)], completed=False)
    assert first["candidate"] == "P" and first["candidate_update"] == 0
    with pytest.raises(ValueError):
        R.select_candidate([e(250, 0.1)], completed=False)


def test_clustered_bootstrap_resamples_games_not_positions():
    values = np.array([1.0] * 100 + [0.0])
    groups = ["a"] * 100 + ["b"]
    result = R.clustered_bootstrap(values, groups)
    assert result["mean"] == pytest.approx(100 / 101) and result["games"] == 2 and result["positions"] == 101
    assert result["low"] == 0.0 and result["high"] == 1.0
    assert R.clustered_bootstrap(values, groups) == result
    flat = R.clustered_bootstrap(np.full(10, 0.5), list(range(10)))
    assert flat["low"] == flat["high"] == flat["mean"] == 0.5


def test_metrics_segments_supersede_redone_updates_and_ignore_a_partial_line(tmp_path):
    rec = lambda u, x=0: json.dumps(dict(event="update", update=u, x=x))
    head = lambda n, u: json.dumps(dict(event="segment", segment=n, from_update=u))
    (tmp_path / "metrics-001-from-000000.jsonl").write_text("\n".join([head(1, 0), rec(1), rec(2), rec(3)]) + "\n")
    (tmp_path / "metrics-002-from-000002.jsonl").write_text("\n".join([head(2, 2), rec(3, 1), rec(4)]) + "\n" + '{"event": "upd')
    records, superseded, partial = R.read_metrics(tmp_path)
    assert [r["update"] for r in records] == [1, 2, 3, 4] and records[2]["x"] == 1
    assert superseded == 1 and partial == 1


def test_publication_never_overwrites(tmp_path):
    R.publish_json(tmp_path / "a.json", {"x": 1})
    with pytest.raises(FileExistsError):
        R.publish_json(tmp_path / "a.json", {"x": 2})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}


# The command line, end to end in separate processes.

def cli(*args, check=True):
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train_mixed.py"), *map(str, args)],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def train_args(world, data, output, *extra):
    return ["train", "--recipe", "pure", "--data-config", world["config"], "--round-data", data.directory,
            "--p-checkpoint", world["p"], "--p-sha256", world["p_sha"], "--output", output, "--device", "cpu",
            "--updates", 6, "--checkpoint-every", 2, "--skip-disk-check", *extra]


def test_uninterrupted_resumed_and_crashed_runs_are_bit_identical(world):
    draw = next(s["game_id"] for s in ReplayBuffer(world["buffer"]).iter_samples() if s["initial_fen"].startswith("4k3"))
    data = round_data(world, holdout=lambda g: g == draw)  # one whole game held out for evaluation
    assert len(data.holdout) == 1 and len(data.train) == len(WHITE_WIN) - 1
    tmp = world["tmp"]
    a, b, c = tmp / "a", tmp / "b", tmp / "c"
    cli(*train_args(world, data, a))
    cli(*train_args(world, data, b, "--stop-after", 2))
    assert not (b / "summary.json").exists() and (b / "update-000002.json").exists()
    cli(*train_args(world, data, b))
    result = cli("compare", a, b, "--update", 6)
    assert json.loads(result.stdout)["identical"]
    # Crash after a checkpoint but before its sidecar, with metrics lines past the last pair.
    cli(*train_args(world, data, c, "--stop-after", 4))
    shutil.copy(a / "update-000006.pt", c / "update-000006.pt")
    segment = sorted(c.glob("metrics-*.jsonl"))[-1]
    lines = [l for l in sorted(a.glob("metrics-*.jsonl"))[0].read_text().splitlines() if '"update": 5,' in l
             or '"update": 6,' in l]
    with segment.open("a") as stream:
        stream.write("\n".join(lines) + "\n" + '{"event": "upd')
    cli(*train_args(world, data, c))
    assert len(list((c / "orphans").glob("update-000006.pt.orphan-*"))) == 1
    header = json.loads(sorted(c.glob("metrics-*.jsonl"))[-1].read_text().splitlines()[0])
    assert header["from_update"] == 4 and header["superseded_lines"] == 2 and header["partial_lines"] == 1
    assert json.loads(cli("compare", a, c, "--update", 6).stdout)["identical"]
    # The reference is validate() on P, measured in this run.
    reference = json.loads((a / "reference.json").read_text())
    validation = CachedSupervisedDataset.open(split="validation", **world["paths"])
    assert reference["validation"] == json.loads(json.dumps(validate(load_inference(world["p"]).model, validation, 32)))
    summary = json.loads((a / "summary.json").read_text())
    assert summary["completed"] and summary["candidate"]["checkpoint"] == "update-000006.pt"
    assert summary["trained_positions"] == dict(selfplay=6 * 24, stockfish=0)
    assert summary["discarded_sampler_indices"] == 6 * 8 and summary["recipe"] == "pure"
    assert summary["selfplay"]["seen_twice"] == 6 * 24 - len(data.train)
    loaded = load_inference(a / "update-000006.pt")
    assert loaded.provenance["recipe"] == "pure"
    assert loaded.provenance["recipe_details"]["trained_positions_per_update"] == dict(selfplay=24, stockfish=0)
    assert cli("inference", a / "update-000006.pt", "--data-config", world["config"], "--device", "cpu").returncode == 0
    # A finished run is left alone; a changed setting is refused.
    before = {p.name: p.stat().st_mtime_ns for p in a.iterdir()}
    assert "already-complete" in cli(*train_args(world, data, a)).stdout
    assert {p.name: p.stat().st_mtime_ns for p in a.iterdir()} == before
    d = tmp / "d"
    cli(*train_args(world, data, d, "--stop-after", 2))
    refused = cli(*[x if x != 2 else 3 for x in train_args(world, data, d)], check=False)
    assert refused.returncode == 2 and "other settings" in refused.stderr


def test_mix_run_reproduces_p_or_stops_before_training(world):
    data = round_data(world, holdout=None)
    out = world["tmp"] / "mix"
    args = train_args(world, data, out)
    args[args.index("pure")] = "mix"
    refused = cli(*args, "--expect-p-total", 99.0, check=False)
    assert refused.returncode == 2 and "does not reproduce" in refused.stderr
    assert not list(out.glob("update-*.pt"))
    reference = json.loads((out / "reference.json").read_text()) if (out / "reference.json").exists() else None
    assert reference is None
    out2 = world["tmp"] / "mix2"
    args = train_args(world, data, out2)
    args[args.index("pure")] = "mix"
    cli(*args)
    summary = json.loads((out2 / "summary.json").read_text())
    assert summary["trained_positions"] == dict(selfplay=6 * 24, stockfish=6 * 8) and summary["discarded_sampler_indices"] == 0
    sidecar = json.loads((out2 / "update-000002.json").read_text())
    assert sidecar["delta_total"] == sidecar["evaluation"]["validation"]["total_loss"] - \
        json.loads((out2 / "reference.json").read_text())["validation"]["total_loss"]
    labels = sidecar["evaluation"]["paired_labels"]
    assert labels == R.paired_labels(sidecar["evaluation"]["paired"]["validation_total"])
    assert summary["evaluations"][0]["paired_labels"] == labels
    assert summary["final_paired_labels"] == json.loads((out2 / "update-000006.json").read_text())["evaluation"]["paired_labels"]


def test_data_command_checks_the_pointer_and_verification(world, monkeypatch, capsys):
    buffer = world["tmp"] / "buffer"
    buffer.mkdir()
    shutil.copy(world["buffer"], buffer / "replay.sqlite")
    summary = ReplayBuffer(buffer / "replay.sqlite").summary()
    verified = dict(buffer=str(buffer), games=3, checkpoint_sha256=world["p_sha"], halves=[dict(half="h", games=4)],
                    source_counts=summary["source_counts"], training_positions=summary["training_positions"],
                    sqlite_bytes=(buffer / "replay.sqlite").stat().st_size, audit_seconds=1.0)
    pointer = dict(schema=T.POINTER_SCHEMA, branch="extended", buffer=str(buffer), replay=str(buffer / "replay.sqlite"),
                   halves=["h"], games=3, training_positions_before_holdout=summary["training_positions"],
                   checks=dict(extended=verified), core_choice=None, extension_choice=None)
    path = world["tmp"] / "pointer.json"
    path.write_text(json.dumps(pointer))
    monkeypatch.setattr(T, "verify_buffer", lambda b, h: dict(verified, audit_seconds=2.0))
    args = ["data", "--pointer", path, "--data-config", world["config"], "--p-checkpoint", world["p"],
            "--p-sha256", world["p_sha"], "--output", world["tmp"] / "built", "--expected-branch", "extended",
            "--expected-games", 3, "--expected-positions", summary["training_positions"], "--skip-disk-check"]
    assert T.main(list(map(str, args))) == 0
    built = R.RoundData.open(world["tmp"] / "built")
    assert built.manifest["source"]["verified"]["audit_seconds"] == 2.0
    assert built.manifest["source"]["pointer"]["sha256"] == R._sha256(path)
    assert T.main(list(map(str, args))) == 0 and '"event": "reused"' in capsys.readouterr().out
    wrong = list(map(str, args))
    wrong[wrong.index("--expected-games") + 1] = "4"
    wrong[wrong.index("--output") + 1] = str(world["tmp"] / "other")
    assert T.main(wrong) == 2
    monkeypatch.setattr(T, "verify_buffer", lambda b, h: dict(verified, training_positions=1))
    wrong = list(map(str, args))
    wrong[wrong.index("--output") + 1] = str(world["tmp"] / "other")
    assert T.main(wrong) == 2 and not (world["tmp"] / "other").exists()


def test_exact_comparison_reports_any_difference():
    a = dict(w=torch.zeros(3), x=[1.0, 2], y={"z": (1, 2)})
    assert T.tree_differences(a, dict(w=torch.zeros(3), x=[1.0, 2], y={"z": (1, 2)}), "t", []) == []
    flipped = torch.zeros(3)
    flipped[1] = torch.finfo(torch.float32).tiny
    assert T.tree_differences(a, dict(a, w=flipped), "t", []) == ["t.w"]
    assert T.tree_differences(a, dict(a, w=torch.zeros(3, dtype=torch.float64)), "t", []) == ["t.w"]
    assert T.tree_differences(a, dict(a, x=[1.0 + 1e-16 * 2, 2]), "t", []) == ["t.x[0]"]
    assert T.tree_differences(a, dict(a, y={"z": [1, 2]}), "t", []) == ["t.y.z length/type"]


def test_paired_labels_answer_two_different_questions():
    interval = lambda low, high: dict(R.clustered_bootstrap([0.0], ["g"]), low=low, high=high)
    supported = R.paired_labels(interval(0.02, 0.07))
    assert supported["increase_supported"] and not supported["increase_above_flag_supported"]
    above = R.paired_labels(interval(0.06, 0.1))
    assert above["increase_supported"] and above["increase_above_flag_supported"]
    neither = R.paired_labels(interval(-0.01, 0.3))
    assert not neither["increase_supported"] and not neither["increase_above_flag_supported"]
    edge = R.paired_labels(interval(0.05, 0.2))
    assert edge["increase_supported"] and not edge["increase_above_flag_supported"]
    assert not R.paired_labels(interval(0.0, 0.1))["increase_supported"]
    assert "regardless" in supported["rule"] and "validation games" in supported["interval"]


def test_baseline_measures_p_only_prints_its_report_and_must_reproduce(world):
    data = round_data(world, holdout=None)
    out = world["tmp"] / "p-baseline"
    args = ["baseline", "--data-config", world["config"], "--round-data", data.directory, "--p-checkpoint", world["p"],
            "--p-sha256", world["p_sha"], "--output", out, "--device", "cpu"]
    first = cli(*args)
    assert "P BASELINE" in first.stdout and "across validation games" in first.stdout
    assert '"event": "baseline-recorded"' in first.stdout and not list(world["tmp"].glob("**/update-*.pt"))
    record = json.loads((out / "baseline.json").read_text())
    validation = CachedSupervisedDataset.open(split="validation", **world["paths"])
    assert record["validation"] == json.loads(json.dumps(validate(load_inference(world["p"]).model, validation, 32)))
    spread = record["across_validation_games"]
    assert spread["mean_total_interval"]["method"] == "game-clustered percentile bootstrap"
    assert spread["note"] == R.UNCERTAINTY_NOTE
    again = cli(*args)
    assert '"event": "baseline-reproduced"' in again.stdout
    assert (out / "baseline.json").read_text() == json.dumps(record, indent=2, sort_keys=True) + "\n"
    refused = cli(*args[:-4], "--output", world["tmp"] / "p-baseline-2", "--device", "cpu",
                  "--expect-p-total", 99.0, check=False)
    assert refused.returncode == 2 and "NO (training refuses to start)" in refused.stdout


def test_exploratory_candidate_keeps_the_original_rule_result_alongside():
    e = lambda u, d: dict(update=u, checkpoint=f"update-{u:06d}.pt", delta_total=d)
    evaluations = [e(250, 0.4957), e(500, 0.3), e(750, 0.2)]
    exploratory = R.exploratory_candidate(evaluations, completed=True)
    assert exploratory["candidate"] == "update-000750.pt" and exploratory["exploratory"]
    assert exploratory["threshold_breaches"] == [250, 500] and exploratory["final_flag"]
    original = R.select_candidate(evaluations, completed=True)
    assert original["stopped"] and original["stopped_at"] == 250 and original["candidate"] == "P"
    with pytest.raises(ValueError):
        R.exploratory_candidate(evaluations, completed=False)
    assert R.stopped_training(dict(hard_stop=True)) and not R.stopped_training(dict(hard_stop=True, training_stopped=False))


def patched_cli(stop, *args, check=True):
    """The command line with the stop threshold lowered so that every checkpoint breaches it."""
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    code = ("import sys; from chess_rl.training import round_training as R; R.STOP_DELTA = float(sys.argv[1]); "
            "from scripts import train_mixed as T; sys.exit(T.main(sys.argv[2:]))")
    result = subprocess.run([sys.executable, "-c", code, str(stop), *map(str, args)],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_exploratory_run_records_breaches_without_stopping_and_matches_the_stopped_run(world):
    data = round_data(world, holdout=None)
    stopped, full = world["tmp"] / "stopped", world["tmp"] / "full"
    note = "A6 test amendment"
    patched_cli(-10.0, *train_args(world, data, stopped))
    s = json.loads((stopped / "summary.json").read_text())
    assert not s["completed"] and s["selection"]["stopped_at"] == 2 and s["candidate"]["checkpoint"] == "P"
    assert s["hard_stop_enforced"] and not s["exploratory"] and s["threshold_breaches"] == [2]
    patched_cli(-10.0, *train_args(world, data, full, "--no-hard-stop", "--amendment-note", note, "--stop-after", 2))
    assert json.loads(cli("compare", stopped, full, "--update", 2, "--state-only").stdout)["identical"]
    plain = cli("compare", stopped, full, "--update", 2, check=False)
    assert plain.returncode == 1 and json.loads(plain.stdout)["differences"]["everything else"]
    patched_cli(-10.0, *train_args(world, data, full, "--no-hard-stop", "--amendment-note", note))
    f = json.loads((full / "summary.json").read_text())
    assert f["completed"] and f["exploratory"] and not f["hard_stop_enforced"] and f["updates"] == 6
    assert f["candidate"]["checkpoint"] == "update-000006.pt" and f["selection"]["threshold_breaches"] == [2, 4, 6]
    assert f["threshold_breaches"] == [2, 4, 6]
    assert f["original_rule_selection"]["candidate"] == "P" and f["original_rule_selection"]["stopped_at"] == 2
    for u in (2, 4, 6):
        sidecar = json.loads((full / f"update-{u:06d}.json").read_text())
        assert sidecar["hard_stop"] and not sidecar["training_stopped"]
    provenance = load_inference(full / "update-000006.pt").provenance
    assert provenance["exploratory"]["amendment"] == note and provenance["thresholds"]["hard_stop_enforced"] is False
    assert "exploratory" not in load_inference(stopped / "update-000002.pt").provenance
    # The stop setting is part of the run: resuming without it, or switching it on later, is refused.
    other = world["tmp"] / "other"
    patched_cli(-10.0, *train_args(world, data, other, "--no-hard-stop", "--amendment-note", note, "--stop-after", 2))
    refused = patched_cli(-10.0, *train_args(world, data, other), check=False)
    assert refused.returncode == 2 and "other settings" in refused.stderr
    lone = cli(*train_args(world, data, world["tmp"] / "lone", "--no-hard-stop"), check=False)
    assert lone.returncode == 2 and "need --amendment-note" in lone.stderr


def passes_args(world, data, output, passes, *extra):
    args = train_args(world, data, output)
    at = args.index("--updates")
    del args[at:at + 2]
    return [*args, "--passes", passes, *extra]


def test_multi_pass_run_records_its_amendment_and_matches_one_pass_up_to_a_common_checkpoint(world):
    data = round_data(world, holdout=None)
    note = "A7 test amendment: second pass, diagnostic only"
    two = world["tmp"] / "two"
    cli(*passes_args(world, data, two, 2, "--amendment-note", note))
    s = json.loads((two / "summary.json").read_text())
    assert s["total_updates"] == math.ceil(2 * len(data.train) / 24) and s["completed"] and not s["exploratory"]
    provenance = load_inference(two / f"update-{s['total_updates']:06d}.pt").provenance
    assert provenance["passes_over_selfplay"] == 2 and provenance["amendment"] == note
    assert "hard_stop_enforced" not in provenance["thresholds"] and "exploratory" not in provenance
    # A run with a longer planned length trains identically up to a checkpoint both runs share.
    one, longer = world["tmp"] / "one", world["tmp"] / "longer"
    cli(*train_args(world, data, one))
    cli(*passes_args(world, data, longer, 30, "--amendment-note", note))
    total = json.loads((longer / "summary.json").read_text())["total_updates"]
    assert total == math.ceil(30 * len(data.train) / 24) > 6
    assert "amendment" not in load_inference(one / "update-000006.pt").provenance
    assert json.loads(cli("compare", one, longer, "--update", 6, "--state-only").stdout)["identical"]
    assert cli("compare", one, longer, "--update", 6, check=False).returncode == 1
    bad = cli(*train_args(world, data, world["tmp"] / "bad"), "--passes", 2, "--amendment-note", note, check=False)
    assert bad.returncode == 2 and "cannot be combined" in bad.stderr
    alone = cli(*train_args(world, data, world["tmp"] / "alone"), "--amendment-note", note, check=False)
    assert alone.returncode == 2 and "give one of them" in alone.stderr
    missing = cli(*passes_args(world, data, world["tmp"] / "missing", 2), check=False)
    assert missing.returncode == 2 and "need --amendment-note" in missing.stderr
