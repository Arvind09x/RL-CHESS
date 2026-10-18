"""Round-2 training: union of round data sets, the round-2 data command, start vs reference, fallback (CPU)."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.adapter import make_state
from chess_rl.dataset_cache import CachedSupervisedDataset
from chess_rl.replay import ReplayBuffer
from chess_rl.selfplay import Phase
from chess_rl.training import round2_training as U
from chess_rl.training import round_training as R
from chess_rl.training.checkpoints import load_inference
from chess_rl.training.mixed_state import MixedState
from scripts import train_round2 as T2
from .test_replay import BLACK_WIN, WHITE_WIN, played
from .test_round_training import ROOT, cli, one_thread, round_data, world  # noqa: F401 (fixtures)
from .test_selfplay import cpu, create, fixture_checkpoint  # noqa: F401 (fixtures)

# Round-2 games: none of round 1's (game ids differ), decisive both ways plus a soft draw.
FOOLS_MATE = ["g2g4", "e7e5", "f2f3", "d8h4"]
SCHOLARS_MATE = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]


def cli2(*args, check=True, patch="pass"):
    """scripts/train_round2.py in a new process; patch is Python run first (e.g. lowering the stop)."""
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    code = (f"import sys; from chess_rl.training import round_training as R; from chess_rl.training import "
            f"round2_training as U; {patch}; from scripts import train_round2 as T; sys.exit(T.main(sys.argv[1:]))")
    result = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True, env=env,
                            cwd=ROOT)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def rounds(world, monkeypatch):
    """Round 1 (arrays from P's games, real hold-out rule), R1-mix (a finished round-1 mix run), and a
    round-2 replay buffer of games 'played by R1-mix' with its round2-buffer-v1 record and final selection."""
    tmp = world["tmp"]
    r1 = round_data(world, name="r1-data")
    r1mix = tmp / "r1-mix"
    cli("train", "--recipe", "mix", "--data-config", world["config"], "--round-data", r1.directory,
        "--p-checkpoint", world["p"], "--p-sha256", world["p_sha"], "--output", r1mix, "--device", "cpu",
        "--updates", 2, "--checkpoint-every", 2, "--skip-disk-check")
    start = r1mix / "update-000002.pt"
    start_sha = R._sha256(start)
    buffer_dir = tmp / "round2-buffer"
    buffer_dir.mkdir()
    buffer = ReplayBuffer.create(buffer_dir / "replay.sqlite")
    buffer.ingest(played(tmp, start, FOOLS_MATE, name="r2-black").snapshot())
    buffer.ingest(played(tmp, start, SCHOLARS_MATE, name="r2-white").snapshot())
    buffer.ingest(create(tmp, start, make_state('4k3/8/8/8/8/8/8/3QK3 w - - 99 1', history_complete=True),
                         name='r2-soft-draw', cap=1, budget=3, phases=(Phase(0, 0., 0.),)).run())
    summary = buffer.summary()
    verified = dict(buffer=str(buffer_dir), games=3, checkpoint_sha256=start_sha,
                    halves=[dict(half="h2", games=3)], source_counts=summary["source_counts"],
                    training_positions=summary["training_positions"],
                    sqlite_bytes=(buffer_dir / "replay.sqlite").stat().st_size, audit_seconds=1.0)
    record = dict(schema=T2.RECORD_SCHEMA, start_checkpoint=dict(path=str(start), sha256=start_sha),
                  buffer=str(buffer_dir), replay=str(buffer_dir / "replay.sqlite"), halves=["h2"], games=3,
                  training_positions_before_holdout=summary["training_positions"], checks=verified)
    record_path = tmp / "round2-buffer.json"
    record_path.write_text(json.dumps(record))
    selection = tmp / "selection.json"
    selection.write_text(json.dumps(dict(final=True, status="final", selected=dict(path=str(start), sha256=start_sha))))
    monkeypatch.setattr(T2, "verify_buffer", lambda b, h: dict(verified, audit_seconds=2.0))
    return dict(world, r1=r1, r1mix=r1mix, start=start, start_sha=start_sha, buffer_dir=buffer_dir, record=record_path,
                selection=selection, verified=verified)


def data_args(r, output, **override):
    values = dict(record=r["record"], selection=r["selection"], start=r["start"], start_sha=r["start_sha"])
    values.update(override)
    return list(map(str, ["data", "--record", values["record"], "--selection", values["selection"],
                          "--data-config", r["config"], "--start-checkpoint", values["start"],
                          "--start-sha256", values["start_sha"], "--output", output, "--expected-games", 3,
                          "--skip-disk-check"]))


def train_args(r, r2, output, *extra, recipe="mix", start=None, start_sha=None):
    return ["train", "--recipe", recipe, "--data-config", r["config"],
            "--round-data", r["r1"].directory, r["p_sha"], "--round-data", r2, r["start_sha"],
            "--start-checkpoint", start or r["start"], "--start-sha256", start_sha or r["start_sha"],
            "--p-checkpoint", r["p"], "--p-sha256", r["p_sha"], "--output", output, "--device", "cpu",
            "--updates", 4, "--checkpoint-every", 2, "--skip-disk-check", *extra]


def build_r2(r, name="r2-data"):
    assert T2.main(data_args(r, r["tmp"] / name)) == 0
    return r["tmp"] / name


# The union of round data sets.

class Rows:
    """Row-set stand-in with only the identities the union checks read."""

    def __init__(self, samples, games):
        self.sample_ids, self.game_ids, self.plies = tuple(samples), tuple(games), tuple(range(len(samples)))

    def __len__(self):
        return len(self.sample_ids)


def test_union_keeps_round_order_and_batches_exactly_like_each_part(rounds):
    r2 = R.RoundData.open(build_r2(rounds))
    r1 = rounds["r1"]
    union = U.RoundUnion.open([(r1.directory, rounds["p_sha"]), (r2.directory, rounds["start_sha"])])
    assert union.train.sample_ids == r1.train.sample_ids + r2.train.sample_ids
    assert union.train.game_ids == r1.train.game_ids + r2.train.game_ids
    assert len(union.train) == len(r1.train) + len(r2.train) and len(r2.train) > 0
    assert [p["checkpoint_sha256"] for p in union.manifest["parts"]] == [rounds["p_sha"], rounds["start_sha"]]
    # Any order, mixing rounds, equals each owning part's batch row by row.
    n1 = len(r1.train)
    order = [len(union.train) - 1, 0, n1, n1 - 1] + list(range(1, len(union.train) - 1))
    batch = union.train.batch(order)
    for row, i in enumerate(order):
        part, local = (r1.train, i) if i < n1 else (r2.train, i - n1)
        one = part.batch([local])
        assert batch.sample_ids[row] == one.sample_ids[0]
        for name in ("features", "legal_mask", "policy_distributions", "wdl"):
            assert np.array_equal(getattr(batch, name)[row], getattr(one, name)[0])
    with pytest.raises(ValueError):
        union.train.batch([len(union.train)])
    # One pass of the sampler covers every union position exactly once.
    cached = CachedSupervisedDataset.open(split="train", **rounds["paths"])
    state = MixedState.initial(R.sampler_sources(cached, union), R.SETTINGS, seed=U.SAMPLING_SEED)
    drawn = []
    for _ in range(math.ceil(len(union.train) / R.SETTINGS.selfplay_batch_size)):
        _, ri, state = state.next_selection()
        drawn += ri
    assert sorted(drawn[:len(union.train)]) == list(range(len(union.train)))
    assert U.recipe_details("mix")["name"] == "R2-mix" and U.recipe_details("pure")["name"] == "R2-pure"


def test_union_refuses_repeats_mixed_sources_and_the_wrong_player(rounds, tmp_path):
    r1 = rounds["r1"]
    with pytest.raises(ValueError, match="listed twice"):
        U.RoundUnion.open([(r1.directory, rounds["p_sha"]), (r1.directory, rounds["p_sha"])])
    with pytest.raises(ValueError, match="another checkpoint"):
        U.RoundUnion.open([(r1.directory, rounds["start_sha"])])
    # The same buffer built with another hold-out rule: same samples, different set.
    other = round_data(rounds, name="r1-other", holdout=None)
    with pytest.raises(ValueError, match="hold-out rules"):
        U.RoundUnion.open([(r1.directory, rounds["p_sha"]), (other.directory, rounds["p_sha"])])
    for key, value in (("supervised_source_id", "0" * 64), ("registries", {"x": 1})):
        copy = tmp_path / f"r1-{key}"
        shutil.copytree(r1.directory, copy)
        manifest = json.loads((copy / "manifest.json").read_text())
        manifest[key] = value
        (copy / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="different"):
            U.RoundUnion.open([(r1.directory, rounds["p_sha"]), (copy, rounds["p_sha"])])
    # Repeated samples, a game in two rounds, a game in both sets.
    with pytest.raises(ValueError, match="sample ids repeat"):
        U.union_rows([Rows("ab", "gg"), Rows("bc", "hh")], [Rows("z", "q")])
    with pytest.raises(ValueError, match="training game appears in two rounds"):
        U.union_rows([Rows("ab", "gg"), Rows("cd", "gh")], [Rows("z", "q")])
    with pytest.raises(ValueError, match="hold-out game appears in two rounds"):
        U.union_rows([Rows("a", "g")], [Rows("b", "h"), Rows("c", "h")])
    with pytest.raises(ValueError, match="training set and the hold-out set"):
        U.union_rows([Rows("a", "g")], [Rows("b", "g")])
    train, holdout = U.union_rows([Rows("a", "g"), Rows("b", "h")], [Rows("c", "i"), Rows("de", "jj")])
    assert holdout.part_slices() == [slice(0, 1), slice(1, 3)]
    values = dict(total=np.arange(3.0))
    assert [list(v["total"]) for v in U.holdout_by_round(values, holdout)] == [[0.0], [1.0, 2.0]]


# The round-2 data command.

def test_data_command_checks_the_record_and_selection_and_reopens_without_the_buffer(rounds, capsys):
    tmp = rounds["tmp"]
    # A record whose start sha256 is not the selection's, and a selection naming another network.
    wrong = json.loads(rounds["selection"].read_text())
    wrong["selected"]["sha256"] = rounds["p_sha"]
    other = tmp / "other-selection.json"
    other.write_text(json.dumps(wrong))
    assert T2.main(data_args(rounds, tmp / "no1", selection=other)) == 2
    record = json.loads(rounds["record"].read_text())
    record["start_checkpoint"]["sha256"] = rounds["p_sha"]
    bad_record = tmp / "bad-record.json"
    bad_record.write_text(json.dumps(record))
    assert T2.main(data_args(rounds, tmp / "no2", record=bad_record)) == 2
    unfinished = json.loads(rounds["selection"].read_text())
    unfinished.update(final=False, status="provisional")
    provisional = tmp / "provisional.json"
    provisional.write_text(json.dumps(unfinished))
    assert T2.main(data_args(rounds, tmp / "no3", selection=provisional)) == 2
    assert not any((tmp / n).exists() for n in ("no1", "no2", "no3"))
    # Built once; every sample played by R1-mix; source recorded.
    r2 = build_r2(rounds)
    data = R.RoundData.open(r2)
    assert data.manifest["checkpoint_sha256"] == rounds["start_sha"] and data.manifest["holdout"]["rule"] == R.HOLDOUT_RULE
    source = data.manifest["source"]
    assert source["record"]["sha256"] == R._sha256(rounds["record"])
    assert source["selection"]["sha256"] == R._sha256(rounds["selection"])
    assert source["verified"]["audit_seconds"] == 2.0
    # With the buffer deleted, a rerun reopens the arrays and never opens the buffer.
    shutil.rmtree(rounds["buffer_dir"])
    capsys.readouterr()
    assert T2.main(data_args(rounds, r2)) == 0 and '"event": "reused"' in capsys.readouterr().out
    # Reopening checks the record, the selection and R1-mix.
    edited = json.loads(rounds["record"].read_text())
    edited["note"] = "edited"
    changed = tmp / "changed-record.json"
    changed.write_text(json.dumps(edited))
    assert T2.main(data_args(rounds, r2, record=changed)) == 2
    assert T2.main(data_args(rounds, r2, selection=other)) == 2
    assert T2.main(data_args(rounds, r2, start=rounds["p"], start_sha=rounds["p_sha"])) == 2


def test_data_command_refuses_a_sample_played_by_another_checkpoint(rounds, monkeypatch):
    tmp = rounds["tmp"]
    mixed = tmp / "mixed-buffer"
    mixed.mkdir()
    shutil.copy(rounds["buffer_dir"] / "replay.sqlite", mixed / "replay.sqlite")
    buffer = ReplayBuffer(mixed / "replay.sqlite")
    buffer.ingest(played(tmp, rounds["p"], WHITE_WIN, name="by-p").snapshot())  # decisive, so it has training samples
    summary = buffer.summary()
    assert summary["training_positions"] > json.loads(rounds["record"].read_text())["training_positions_before_holdout"]
    verified = dict(rounds["verified"], buffer=str(mixed), games=4, source_counts=summary["source_counts"],
                    training_positions=summary["training_positions"])
    record = dict(json.loads(rounds["record"].read_text()), buffer=str(mixed), replay=str(mixed / "replay.sqlite"),
                  games=4, training_positions_before_holdout=summary["training_positions"], checks=verified)
    path = tmp / "mixed-record.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(T2, "verify_buffer", lambda b, h: dict(verified, audit_seconds=2.0))
    args = data_args(rounds, tmp / "mixed-data", record=path)
    args[args.index("--expected-games") + 1] = "4"
    with pytest.raises(ValueError, match="played by another checkpoint"):
        T2.main(args)
    assert not (tmp / "mixed-data").exists() and not list(tmp.glob(".mixed-data.partial-*"))


# Training.

def test_runs_resume_exactly_load_for_inference_and_refuse_changes(rounds):
    r2 = build_r2(rounds)
    tmp = rounds["tmp"]
    a, b = tmp / "a", tmp / "b"
    cli2(*train_args(rounds, r2, a))
    cli2(*train_args(rounds, r2, b, "--stop-after", 2))
    assert not (b / "summary.json").exists()
    cli2(*train_args(rounds, r2, b))
    assert json.loads(cli("compare", a, b, "--update", 4).stdout)["identical"]
    assert cli("inference", b / "update-000004.pt", "--data-config", rounds["config"], "--device", "cpu").returncode == 0
    loaded = load_inference(b / "update-000004.pt")
    assert loaded.checkpoint_id == "r2-mix-update-000004"
    provenance = loaded.provenance
    assert provenance["schema"] == U.RUN_VERSION and provenance["recipe_details"]["name"] == "R2-mix"
    assert provenance["initial_checkpoint"]["sha256"] == rounds["start_sha"]
    assert provenance["forgetting_reference"]["sha256"] == rounds["p_sha"]
    assert provenance["sampler"]["seed"] == U.SAMPLING_SEED and provenance["run_seed"] == U.RUN_SEED
    assert [p["checkpoint_sha256"] for p in provenance["round_data"]["union"]["parts"]] == [rounds["p_sha"], rounds["start_sha"]]
    assert json.loads((a / "update-000002.json").read_text())["version"] == R.SIDECAR_VERSION
    # Training starts from R1-mix, not P: the first update's weights move away from R1-mix only.
    reference = json.loads((a / "reference.json").read_text())
    start_sidecar = json.loads((rounds["start"].with_suffix(".json")).read_text())
    assert reference["start"]["sidecar_validation"] == start_sidecar["evaluation"]["validation"]
    assert abs(reference["start"]["difference_from_sidecar"]["total_loss"]) <= 0.001
    assert reference["start"]["delta_versus_p"]["total_loss"] == pytest.approx(start_sidecar["delta_total"], abs=1e-9)
    sidecar = json.loads((a / "update-000002.json").read_text())
    assert sidecar["delta_total"] == sidecar["evaluation"]["validation"]["total_loss"] - reference["p"]["validation"]["total_loss"]
    assert sidecar["evaluation"]["versus_start"]["delta"]["total_loss"] == pytest.approx(
        sidecar["evaluation"]["validation"]["total_loss"] - reference["start"]["validation"]["total_loss"])
    by_round = sidecar["evaluation"]["holdout"]["by_round"]
    assert [x["round"] for x in by_round] == [1, 2]
    summary = json.loads((a / "summary.json").read_text())
    assert summary["selectable"] and summary["role"] == "selectable" and summary["completed"]
    assert summary["candidate"]["path"] == str(a / "update-000004.pt") and summary["candidate"]["source"] == "this run"
    assert summary["selfplay"]["by_round"] == [len(rounds["r1"].train), len(R.RoundData.open(r2).train)]
    # A changed seed or data set refuses to resume.
    c = tmp / "c"
    cli2(*train_args(rounds, r2, c, "--stop-after", 2))
    seed = cli2(*train_args(rounds, r2, c), patch="U.SAMPLING_SEED = 1", check=False)
    assert seed.returncode == 2 and "other settings" in seed.stderr
    r2b = tmp / "r2-copy"
    shutil.copytree(r2, r2b)
    moved = cli2(*train_args(rounds, r2b, c), check=False)
    assert moved.returncode == 2 and "other settings" in moved.stderr


def test_start_checks_and_the_fallback_to_the_start_checkpoint(rounds):
    r2 = build_r2(rounds)
    tmp = rounds["tmp"]
    # Start equal to the reference, round data in the wrong order, a tampered start sidecar.
    same = cli2(*train_args(rounds, r2, tmp / "same", start=rounds["p"], start_sha=rounds["p_sha"]), check=False)
    assert same.returncode == 2 and "must differ" in same.stderr
    swapped = train_args(rounds, r2, tmp / "swapped")
    at = swapped.index("--round-data")
    swapped[at:at + 6] = ["--round-data", r2, rounds["start_sha"], "--round-data", rounds["r1"].directory, rounds["p_sha"]]
    wrong_order = cli2(*swapped, check=False)
    assert wrong_order.returncode == 2 and "round 1 (played by P) first" in wrong_order.stderr
    tampered = tmp / "tampered"
    tampered.mkdir()
    shutil.copy(rounds["start"], tampered / "update-000002.pt")
    sidecar = json.loads(rounds["start"].with_suffix(".json").read_text())
    sidecar["evaluation"]["validation"]["total_loss"] += 0.01
    (tampered / "update-000002.json").write_text(json.dumps(sidecar))
    refused = cli2(*train_args(rounds, r2, tmp / "t", start=tampered / "update-000002.pt"), check=False)
    assert refused.returncode == 2 and "differs from its sidecar" in refused.stderr
    assert not list((tmp / "t").glob("update-*.pt"))
    # R2-mix: the first checkpoint breaches the stop with nothing eligible, so the candidate is R1-mix.
    stopped = tmp / "stopped"
    cli2(*train_args(rounds, r2, stopped), patch="R.STOP_DELTA = -10.0")
    s = json.loads((stopped / "summary.json").read_text())
    assert not s["completed"] and s["updates"] == 2 and s["selection"]["fallback_to_start"]
    assert s["selection"]["candidate"] == "start" and s["original_rule_selection"]["stopped_at"] == 2
    assert s["candidate"] == dict(path=str(rounds["start"]), sha256=rounds["start_sha"], update=0,
                                  checkpoint_id=load_inference(rounds["start"]).checkpoint_id,
                                  source="start checkpoint (fallback)")
    assert "R1-mix" in s["selection"]["reason"] and "so P" not in s["selection"]["reason"]
    # R2-pure: no hard stop; every breach recorded, the original rule's candidate beside the final one.
    pure = tmp / "pure"
    note = "round-2 test decision: R2-pure report-only, no hard stop"
    lone = cli2(*train_args(rounds, r2, pure, "--no-hard-stop", recipe="pure"), check=False)
    assert lone.returncode == 2 and "amendment-note" in lone.stderr
    cli2(*train_args(rounds, r2, pure, "--no-hard-stop", "--amendment-note", note, recipe="pure"),
         patch="R.STOP_DELTA = -10.0")
    p = json.loads((pure / "summary.json").read_text())
    assert p["completed"] and p["updates"] == 4 and p["exploratory"] and not p["selectable"] and p["role"] == "report-only"
    assert p["threshold_breaches"] == [2, 4] and p["candidate"]["path"] == str(pure / "update-000004.pt")
    assert p["original_rule_selection"]["candidate"] == "start" and p["original_rule_selection"]["stopped_at"] == 2
    assert p["trained_positions"]["stockfish"] == 0 and p["discarded_sampler_indices"] == 4 * 8
    loaded = load_inference(pure / "update-000004.pt")
    assert loaded.checkpoint_id == "r2-pure-update-000004" and loaded.provenance["exploratory"]["amendment"] == note
    for u in (2, 4):
        record = json.loads((pure / f"update-{u:06d}.json").read_text())
        assert record["hard_stop"] and not record["training_stopped"]


def test_start_candidate_mapping_keeps_a_real_candidate():
    e = lambda u, d: dict(update=u, checkpoint=f"update-{u:06d}.pt", delta_total=d)
    original = R.select_candidate([e(250, 0.1), e(500, 0.3)], completed=False)
    mapped = U.start_candidate(original, start_path="s.pt", start_sha256="a" * 64, start_checkpoint_id="r1")
    assert mapped["candidate"] == "update-000250.pt" and not mapped["fallback_to_start"]
    fallback = U.start_candidate(R.select_candidate([e(250, 0.3)], completed=False), start_path="s.pt",
                                 start_sha256="a" * 64, start_checkpoint_id="r1")
    assert fallback["candidate"] == "start" and fallback["start"]["sha256"] == "a" * 64
    assert U.candidate_record(fallback, output="x", sha256=None)["path"] == "s.pt"
