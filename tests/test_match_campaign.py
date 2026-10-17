"""Match driver: frozen openings and candidates, plans, real worker processes, resume, reports, selection."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

import chess
import pytest

torch = pytest.importorskip("torch")

from chess_rl.adapter import make_state
from chess_rl.evaluation import match_campaign as M
from chess_rl.evaluation.matches import Opening, digest, summarize
from chess_rl.evaluation.players import SearchSettings
from chess_rl.position_sampling import rule_position
from chess_rl.training.checkpoints import DataOrder, ModelConfig, Progress, save_checkpoint
from chess_rl.training.trainer import LossWeights

ROOT = Path(__file__).resolve().parents[1]
# Fixture openings are 6 to 12 plies deep; the cap leaves each game a few moves.
CAP = 16


# Fixture data: a tiny held-out registry with the real file formats.

def _game(index, plies=24):
    board = chess.Board()
    for ply in range(plies):
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        if not moves:
            break
        pick = int(hashlib.sha256(f"{index}:{ply}".encode()).hexdigest(), 16) % len(moves)
        board.push(moves[pick])
    return [m.uci() for m in board.move_stack]


def write_registry(directory, *, games=30, extra_rows=()):
    directory = Path(directory)
    prepared = directory / "prepared"
    prepared.mkdir(parents=True)
    records, held, protected, owner = [], [], [], {}
    for i in range(games):
        split = ("validation", "test", "train")[i % 3]
        moves = _game(i)
        game_id = "g:" + hashlib.sha256(json.dumps(moves).encode()).hexdigest()
        records.append(dict(game_id=game_id, split=split, start_fen=chess.STARTING_FEN, history_complete=True,
                            moves_uci=moves))
        if split == "train":
            continue
        held.append(dict(game_id=game_id, split=split))
        for ply in (3, 6, 9, 12, 16):
            key = rule_position(make_state(moves_uci=moves[:ply]).board)
            if owner.setdefault(key, split) != split:
                continue
            protected.append(dict(game_id=game_id, split=split, sample_ply=ply, phase="opening", rule_position=key,
                                  record_version="position-reference-v1"))
    protected.extend(extra_rows)
    lines = lambda rows: "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (directory / "games.jsonl").write_text(lines(records))
    (prepared / "heldout-games.jsonl").write_text(lines(held))
    (prepared / "protected-heldout.jsonl").write_text(lines(protected))
    sha = hashlib.sha256((prepared / "protected-heldout.jsonl").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(dict(preparation=dict(protected_heldout_registry_sha256=sha))))
    return dict(registry_dir=prepared, games_path=directory / "games.jsonl", manifest_path=directory / "manifest.json",
                records=records, protected=protected)


def freeze(data, path, count=3):
    return M.freeze_openings(path, registry_dir=data["registry_dir"], games_path=data["games_path"],
                             manifest_path=data["manifest_path"], count=count, log=lambda _: None)


# Fixture checkpoints with the real round-1 provenance fields.

def save_model(path, seed, provenance):
    torch.manual_seed(seed)
    model = ModelConfig(d_policy=8, d_value=8, dropout=0.0).build()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, model=model, optimizer=torch.optim.Adam(model.parameters()),
                    checkpoint_id=f"fixture-{path.stem}-{seed}", provenance=provenance, weights=LossWeights(),
                    progress=Progress(0, 0, 0), data_order=DataOrder("fixture", "0" * 64, (0,), 0, 1, False, 0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_root(base):
    """A repository-shaped root: evaluation sources copied, fixture checkpoints and run summaries."""
    root = Path(base)
    (root / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "scripts/run_match.py", root / "scripts/run_match.py")
    p_sha = save_model(root / "artifacts/p/p.pt", 1, {"purpose": "fixture P"})
    spec = dict(P=dict(path="artifacts/p/p.pt", sha256=p_sha), candidates=[], excluded=[
        dict(name="R1-mix-2pass", path="runs/train/r1-mix-2pass", reason="diagnostic second pass (A7)")],
        tie_order=["R1-mix"], decided="fixture")
    for name, recipe, seed, extra, eligibility in (
            ("R1-mix", "mix", 2, {}, "selectable"),
            ("R1-pure-full", "pure", 3, dict(exploratory=dict(amendment="A6"),
                                             thresholds=dict(hard_stop_enforced=False)), "report-only")):
        path = f"runs/train/{name.lower()}/update-000002.pt"
        sha = save_model(root / path, seed, dict(recipe=recipe, initial_checkpoint=dict(sha256=p_sha), **extra))
        summary = dict(candidate=dict(checkpoint="update-000002.pt", sha256=sha, update=2),
                       selection=dict(candidate="update-000002.pt"), **({"exploratory": True} if extra else {}))
        (root / path).with_name("summary.json").write_text(json.dumps(summary))
        spec["candidates"].append(dict(name=name, path=path, sha256=sha, summary=str(Path(path).with_name("summary.json")),
                                       recipe=recipe, eligibility=eligibility, note="fixture"))
    spec["P"]["name"] = "P"
    return root, spec


@pytest.fixture(scope="module")
def runtime():
    return M.worker_runtime("cpu")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    base = tmp_path_factory.mktemp("world")
    root, spec = make_root(base / "root")
    data = write_registry(base / "data")
    return dict(root=root, spec=spec, data=data, base=base)


def setup_matches(world, name, runtime=None):
    matches = world["base"] / name / "runs/matches"
    freeze(world["data"], matches / "openings.json")
    M.freeze_candidates(matches / "candidates.json", root=world["root"], spec=world["spec"], log=lambda _: None)
    return matches


def plan(world, matches, runtime, *, kind="validation", candidate="R1-mix", count=2, workers=2, ply_cap=CAP,
         simulations=2, **extra):
    return M.create_plan(matches, root=world["root"], kind=kind, candidate=candidate, openings_path=matches / "openings.json",
                         candidates_path=matches / "candidates.json", device="cpu", workers=workers,
                         simulations=simulations, ply_cap=ply_cap, protocol="fixture", count=count, runtime=runtime,
                         log=lambda _: None, **extra)


def records(matches, match_id):
    with sqlite3.connect(matches / match_id / "results.sqlite3") as db:
        rows = db.execute("SELECT game_id, payload FROM games ORDER BY game_id").fetchall()
    return {g: json.loads(p) for g, p in rows}


# Openings.

def test_opening_freeze_uses_only_registry_roots_with_full_history(world, tmp_path):
    data = world["data"]
    frozen = freeze(data, tmp_path / "a/openings.json")
    sets = frozen["sets"]
    by_key = {}
    for row in data["protected"]:
        by_key.setdefault(row["rule_position"], set()).add(row["split"])
    games = {g["game_id"]: g for g in data["records"]}
    for split in ("validation", "test"):
        for o, row in zip(getattr(sets, split), frozen["record"]["registry_rows"][split], strict=True):
            assert by_key[o.rule_key()] == {split} and row["split"] == split
            assert M.PLY_BAND[0] <= row["sample_ply"] <= M.PLY_BAND[1]
            assert o.history_complete and o.initial_fen == chess.STARTING_FEN
            assert list(o.moves_uci) == games[row["game_id"]]["moves_uci"][:row["sample_ply"]]
            assert rule_position(o.state().board) == row["rule_position"]
    assert not {o.rule_key() for o in sets.validation} & {o.rule_key() for o in sets.test}
    rows = frozen["record"]["registry_rows"]["validation"] + frozen["record"]["registry_rows"]["test"]
    assert len({r["game_id"] for r in rows}) == len(rows)  # at most one root per game
    again = freeze(data, tmp_path / "b/openings.json")
    assert again["record"]["sets"] == frozen["record"]["sets"]
    assert [o.opening_id for o in frozen["smoke"]] == list(M.SMOKE_OPENING_IDS)
    with pytest.raises(ValueError, match="never rewritten"):
        freeze(data, tmp_path / "a/openings.json")


def test_a_non_registry_position_of_a_held_out_game_is_never_an_opening(world, tmp_path):
    data = world["data"]
    registered = {r["rule_position"] for r in data["protected"]}
    game = next(g for g in data["records"] if g["split"] == "validation")
    root = make_state(moves_uci=game["moves_uci"][:7])
    assert rule_position(root.board) not in registered  # ply 7 of a held-out game, not in the registry
    frozen = freeze(data, tmp_path / "o.json")
    assert all(o.rule_key() in registered for o in frozen["sets"].validation + frozen["sets"].test)
    # Rows outside the ply band are never used even though they are protected.
    assert all(r["sample_ply"] in (6, 9, 12) for split in ("validation", "test")
               for r in frozen["record"]["registry_rows"][split])


def test_freeze_refuses_a_smoke_root_in_the_registry_or_a_changed_registry(world, tmp_path):
    smoke = M.opening(M.smoke_openings()[0])
    row = dict(game_id="g:x", split="test", sample_ply=11, phase="opening", rule_position=smoke.rule_key(),
               record_version="position-reference-v1")
    data = write_registry(tmp_path / "d", extra_rows=[row])
    with pytest.raises(ValueError, match="smoke opening"):
        freeze(data, tmp_path / "o.json")
    data = write_registry(tmp_path / "e")
    with data["registry_dir"].joinpath("protected-heldout.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="manifest"):
        freeze(data, tmp_path / "p.json")


def test_tampered_opening_record_is_rejected(world, tmp_path):
    path = tmp_path / "o.json"
    freeze(world["data"], path)
    value = json.loads(path.read_text())
    value["sets"]["validation"][0]["moves_uci"] = value["sets"]["validation"][0]["moves_uci"][:-1]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        M.load_openings(path)


# Candidates.

def test_candidate_freeze_checks_eligibility_bytes_and_lineage(world, tmp_path):
    root, spec = world["root"], world["spec"]
    frozen = M.freeze_candidates(tmp_path / "c.json", root=root, spec=spec, log=lambda _: None)
    assert {c["name"]: c["eligibility"] for c in frozen["record"]["candidates"]} == {
        "R1-mix": "selectable", "R1-pure-full": "report-only"}
    assert frozen["record"]["rules"]["provisional_min_pairs"] == 48 and frozen["record"]["rules"]["final_pairs"] == 64
    bad = deepcopy(spec)
    bad["candidates"][1]["eligibility"] = "selectable"
    with pytest.raises(ValueError, match="exploratory"):
        M.freeze_candidates(tmp_path / "1.json", root=root, spec=bad, log=lambda _: None)
    bad = deepcopy(spec)
    bad["candidates"][0]["eligibility"] = "report-only"
    with pytest.raises(ValueError, match="report-only"):
        M.freeze_candidates(tmp_path / "2.json", root=root, spec=bad, log=lambda _: None)
    bad = deepcopy(spec)
    bad["candidates"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes"):
        M.freeze_candidates(tmp_path / "3.json", root=root, spec=bad, log=lambda _: None)
    bad = deepcopy(spec)
    bad["P"] = dict(spec["P"], path=spec["candidates"][0]["path"], sha256=spec["candidates"][0]["sha256"])
    with pytest.raises(ValueError, match="not trained from P"):
        M.freeze_candidates(tmp_path / "4.json", root=root, spec=bad, log=lambda _: None)
    bad = deepcopy(spec)
    bad["tie_order"] = ["R1-pure-full"]
    with pytest.raises(ValueError, match="tie order"):
        M.freeze_candidates(tmp_path / "5.json", root=root, spec=bad, log=lambda _: None)
    with pytest.raises(ValueError, match="never rewritten"):
        M.freeze_candidates(tmp_path / "c.json", root=root, spec=spec, log=lambda _: None)


def test_round1_candidate_record_matches_the_decision():
    spec = M.ROUND1_CANDIDATES
    assert spec["P"]["sha256"].startswith("cf7442ee")
    assert {c["name"]: c["eligibility"] for c in spec["candidates"]} == {
        "R1-mix": "selectable", "R1-pure-full": "report-only"}
    assert spec["tie_order"] == ("R1-mix",)
    assert {e["name"] for e in spec["excluded"]} == {"R1-mix-2pass", "R1-pure-full-2pass"}
    assert (M.SIMULATIONS, M.PLY_CAP, M.WORKERS, M.PROVISIONAL_MIN_PAIRS, M.FINAL_PAIRS) == (128, 512, 5, 48, 64)


# Plans.

def test_plan_freezes_pairing_settings_and_refuses_changes(world, runtime):
    matches = setup_matches(world, "plans")
    mix = plan(world, matches, runtime)
    pure = plan(world, matches, runtime, candidate="R1-pure-full")
    assert mix["opening_ids"] == pure["opening_ids"] and mix["settings"] == pure["settings"]
    assert mix["settings"] == SearchSettings(2).to_dict() and mix["settings"]["exploration"] is None
    assert [p["metadata"]["player_id"] for p in mix["players"]] == ["P", "R1-mix"]
    assert mix["score_player_id"] == "R1-mix"
    schedule = mix["schedule"]
    for first, second in zip(schedule[::2], schedule[1::2], strict=True):
        assert first["opening"] == second["opening"] and first["pair_id"] == second["pair_id"]
        assert (first["leg"], second["leg"]) == (0, 1)
    assert len({j["game_seed"] for j in schedule}) == len(schedule)
    with pytest.raises(ValueError, match="never rewritten"):
        plan(world, matches, runtime)
    with pytest.raises(ValueError, match="128 simulations"):
        M.create_plan(matches, root=world["root"], kind="validation", candidate="R1-mix",
                      openings_path=matches / "openings.json", candidates_path=matches / "candidates.json",
                      device="cpu", workers=5, simulations=2, ply_cap=512, runtime=runtime, log=lambda _: None)
    with pytest.raises(ValueError, match="not a frozen"):
        plan(world, matches, runtime, candidate="R1-mix-2pass")
    directory = matches / mix["match_id"]
    original = (directory / "plan.json").read_text()
    for field, value in (("ply_cap", CAP + 1), ("workers", 3), ("seed", 1), ("planned_pairs", 1),
                         ("settings", SearchSettings(3).to_dict()), ("opening_ids", mix["opening_ids"][:1])):
        changed = json.loads(original)
        changed[field] = value
        (directory / "plan.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            M.load_plan(directory, world["root"])
    (directory / "plan.json").write_text(original)
    M.load_plan(directory, world["root"])
    # A different plan cannot reuse this store.
    M.initialize_store(directory / "results.sqlite3", M.load_plan(directory, world["root"]))
    with pytest.raises(ValueError, match="different frozen plan"):
        M.initialize_store(directory / "results.sqlite3", M.load_plan(matches / pure["match_id"], world["root"]))


def test_plan_refuses_changed_source_and_checkpoint_bytes(world, runtime, tmp_path):
    root = tmp_path / "root"
    shutil.copytree(world["root"], root)
    w = dict(world, root=root, base=tmp_path)
    matches = setup_matches(w, "m")
    p = plan(w, matches, runtime)
    source = root / "src/chess_rl/evaluation/matches.py"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="evaluation source differs"):
        M.load_plan(matches / p["match_id"], root)
    source.write_text(source.read_text()[:-1])
    original = (matches / "openings.json").read_bytes()
    (matches / "openings.json").write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="openings.json differs"):
        M.run(matches / p["match_id"], root, log=lambda _: None)
    (matches / "openings.json").write_bytes(original)
    checkpoint = root / w["spec"]["candidates"][0]["path"]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        M.run(matches / p["match_id"], root, log=lambda _: None)


# Running, resuming and validating stored games.

def run_cli(root, matches, *args):
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen([sys.executable, str(ROOT / "scripts/run_match.py"), "--root", str(root),
                             "--matches", str(matches), *args], env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@pytest.fixture(scope="module")
def played(world, runtime):
    """Both fixture validation matches played to the end, uninterrupted."""
    matches = setup_matches(world, "played")
    plans = {n: plan(world, matches, runtime, candidate=n) for n in ("R1-mix", "R1-pure-full")}
    for p in plans.values():
        M.run(matches / p["match_id"], world["root"], log=lambda _: None)
    return dict(matches=matches, plans=plans)


def test_killed_run_resumes_without_replaying_and_matches_an_uninterrupted_run(world, runtime, played):
    matches = setup_matches(world, "killed")
    p = plan(world, matches, runtime, workers=1)
    process = run_cli(world["root"], matches, "run", "--match-id", p["match_id"])
    database = matches / p["match_id"] / "results.sqlite3"
    deadline = time.time() + 240
    first = {}
    while time.time() < deadline and not first:
        time.sleep(0.05)
        if database.exists():
            try:
                first = records(matches, p["match_id"])
            except sqlite3.OperationalError:
                pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()
    assert 0 < len(first) < p["planned_games"]
    M.run(matches / p["match_id"], world["root"], log=lambda _: None)
    resumed = records(matches, p["match_id"])
    for game_id, value in first.items():
        assert resumed[game_id] == value  # stored games are never replayed or replaced
    reference = records(played["matches"], p["match_id"])
    assert sorted(resumed) == sorted(reference) == sorted(j["game_id"] for j in p["schedule"])
    for game_id in resumed:
        assert resumed[game_id]["record"] == reference[game_id]["record"]
    # The selectable candidate's match is finished, so its report is available without the report-only match.
    assert M.match_report(matches / p["match_id"], world["root"])["eligibility"] == "selectable"


def test_stored_game_validation_rejects_tampering(world, played):
    p = played["plans"]["R1-mix"]
    values = records(played["matches"], p["match_id"])
    jobs = {j["game_id"]: j for j in p["schedule"]}
    game_id = sorted(values)[0]
    record, job = values[game_id]["record"], jobs[game_id]
    M.validate_game(record, job, p)
    assert record["status"] == "truncated" and len(record["attempts"]) == CAP - len(job["opening"]["moves_uci"])

    def rejected(change, plan=p, job=job):
        bad = deepcopy(record)
        change(bad)
        with pytest.raises(ValueError):
            M.validate_game(bad, job, plan)

    def other_move(bad):
        attempt = bad["attempts"][0]
        alternatives = [e for e in attempt["details"]["visited_edges"] if e[0] != attempt["move_id"]]
        attempt["move_id"] = alternatives[0][0] if alternatives else attempt["move_id"] + 1

    rejected(other_move)
    rejected(lambda bad: bad["moves_uci"].pop())

    def less_visited(bad):
        # Same total visits, but another edge now has the maximum: the applied move is not the max-visit move.
        attempt = bad["attempts"][1]
        edges, chosen = attempt["details"]["visited_edges"], attempt["move_id"]
        count = next(e[1] for e in edges if e[0] == chosen)
        for edge in edges:
            if edge[0] == chosen:
                edge[1] = 0
        edges.append([chosen + 1, count, 0.0])

    rejected(less_visited)
    rejected(lambda bad: bad.update(winner="white", outcome="1-0"))  # a scored truncation
    rejected(lambda bad: None, plan=dict(p, settings=SearchSettings(3).to_dict()))  # wrong budget
    rejected(lambda bad: None, job=jobs[game_id.replace("leg0", "leg1") if "leg0" in game_id
                                       else game_id.replace("leg1", "leg0")])  # the other leg's seed and leg
    # Colour routing: the same game with White and Black swapped.
    swapped = deepcopy(record)
    swapped["players"]["white"], swapped["players"]["black"] = swapped["players"]["black"], swapped["players"]["white"]
    with pytest.raises(ValueError, match="routing"):
        M.validate_game(swapped, job, p)


def test_status_has_no_scores_and_reports_refuse_unfinished_schedules(world, runtime, played):
    matches = setup_matches(world, "unfinished")
    p = plan(world, matches, runtime)
    s = M.status(matches / p["match_id"], world["root"])
    assert s["recorded_games"] == 0 and s["pending_games"] == 4 and not s["finished"]
    assert "score" not in json.dumps(s)
    with pytest.raises(ValueError, match="pending"):
        M.match_report(matches / p["match_id"], world["root"])
    with pytest.raises(ValueError):
        M.select(matches, world["root"], candidates_path=matches / "candidates.json", log=lambda _: None)


def test_report_numbers_equal_summarize(world, played):
    p = played["plans"]["R1-mix"]
    report = M.match_report(played["matches"] / p["match_id"], world["root"])
    values = records(played["matches"], p["match_id"])
    games = [values[j["game_id"]]["record"] for j in p["schedule"]]
    assert report["summary"] == summarize(games, "R1-mix", seed=p["statistics"]["seed"],
                                          bootstrap_samples=p["statistics"]["samples"])
    assert report["score_player_id"] == "R1-mix" and report["timing"]["games"] == 4
    pure = played["plans"]["R1-pure-full"]
    side = M.match_report(played["matches"] / pure["match_id"], world["root"])
    assert side["eligibility"] == "report-only" and side["score_player_id"] == "R1-pure-full"
    assert side["comparison_with_selectable_on_common_pairs"]["candidates"] == ["R1-mix", "R1-pure-full"]


def test_smoke_report_shows_only_statuses_lengths_and_timing(world, runtime):
    matches = setup_matches(world, "smoke")
    p = plan(world, matches, runtime, kind="smoke", count=None)
    assert p["opening_ids"] == list(M.SMOKE_OPENING_IDS) and p["planned_games"] == 4
    M.run(matches / p["match_id"], world["root"], log=lambda _: None)
    report = M.match_report(matches / p["match_id"], world["root"])
    text = json.dumps(report)
    for word in ("winner", '"outcome"', "wdl", "score", "1-0", "0-1", "1/2-1/2", "checkmate", "game_id", "leg",
                 "ending_reason", "white", "black"):
        assert word not in text
    assert set(report["status_counts"]) == {"completed", "truncated", "failed", "interrupted"}
    assert report["plies_played_sorted"] == sorted(report["plies_played_sorted"])
    assert report["timing"]["games"] == 4 and report["projection_hours_at_5_workers"]


# Score orientation and selection.

def fake_game(match, opening_id, leg, winner_id, left="P", right="R", status="completed"):
    white, black = (left, right) if leg == 0 else (right, left)
    winner = None if winner_id is None else "white" if winner_id == white else "black"
    done = status == "completed"
    return dict(pair_id=f"{match}:{opening_id}", leg=leg, status=status,
                winner=winner if done else None,
                outcome=({None: "1/2-1/2", "white": "1-0", "black": "0-1"}[winner] if done else None),
                players=dict(white=dict(player_id=white), black=dict(player_id=black)))


def fake_match(name, results, *, protocol="fixture", ids=None):
    """results: per opening, (leg-0 winner, leg-1 winner) as 'R', 'P', None (draw) or 'T' (truncated)."""
    ids = ids or [f"o{i}" for i in range(len(results))]
    match_id = f"validation-P-vs-{name}"
    games = []
    for o, legs in zip(ids, results, strict=True):
        for leg, w in enumerate(legs):
            games.append(fake_game(match_id, o, leg, {"R": name, "P": "P", None: None, "T": None, "F": None}[w], right=name,
                                   status="truncated" if w == "T" else "failed" if w == "F" else "completed"))
    plan = dict(kind="validation", candidates_file="cf", opening_ids=ids, openings_set_sha256="s", settings="x",
                ply_cap=4, protocol=protocol, seed=1, score_player_id=name, runtime="rt", sources="src",
                device="cpu", workers=2, match_id=match_id, sha256="p" + name,
                planned_pairs=len(ids), planned_games=2 * len(ids), statistics=dict(seed=3, samples=500),
                players=[dict(sha256="pp"), dict(sha256=f"sha-{name}")])
    return plan, games


def fake_candidates(eligibility, tie_order):
    by_name = {n: dict(name=n, eligibility=e, sha256=f"sha-{n}", path=f"{n}.pt", checkpoint_id=n)
               for n, e in eligibility.items()}
    rules = dict(selection=M.SELECTION_RULE)
    return dict(file="cf", by_name=by_name,
                record=dict(candidates=list(by_name.values()), tie_order=list(tie_order), rules=rules))


def test_score_orientation_is_R_against_P():
    plan, games = fake_match("R1-mix", [("R", "R"), ("R", None), ("P", "R")])
    r = summarize(games, "R1-mix", seed=1)
    p = summarize(games, "P", seed=1)
    # Pair scores for R: 1, 0.75, 0.5.
    assert r["complete_pair_score"] == pytest.approx(0.75) and p["complete_pair_score"] == pytest.approx(0.25)
    assert r["completed_game_wdl"] == dict(win=4, draw=1, loss=1)


def test_selection_picks_the_stronger_selectable_candidate_and_applies_the_tie_rule():
    strong = [("R", "R"), ("R", None), ("R", "P"), ("R", "R")]
    weak = [("P", "P"), ("R", "P"), (None, None), ("P", "R")]
    cands = fake_candidates(dict(A="selectable", B="selectable"), ["A"])
    out = M.decide(cands, dict(A=fake_match("A", weak), B=fake_match("B", strong)))
    assert out["final"] and out["selected"]["name"] == "B"
    assert out["selected"]["validation_score"] > 0.5 and out["demo"]["player"] == "B"
    out = M.decide(cands, dict(A=fake_match("A", strong), B=fake_match("B", strong)))
    assert out["selected"]["name"] == "A" and out["selected"]["tied_with"] == ["B"]
    out = M.decide(cands, dict(A=fake_match("A", weak), B=fake_match("B", weak)))
    assert out["selected"]["name"] == "A" and out["demo"]["player"] == "P"


def test_report_only_candidates_are_never_selected():
    cands = fake_candidates({"R1-mix": "selectable", "R1-pure-full": "report-only"}, ["R1-mix"])
    out = M.decide(cands, {"R1-mix": fake_match("R1-mix", [("P", "P")] * 4),
                           "R1-pure-full": fake_match("R1-pure-full", [("R", "R")] * 4)})
    assert out["selected"]["name"] == "R1-mix"
    assert out["report_only"]["R1-pure-full"]["own_match"]["complete_pair_score"] == 1.0
    assert out["demo"]["player"] == "P"


def test_selection_uses_only_pairs_complete_in_every_selectable_match_and_the_thresholds():
    cands = fake_candidates(dict(A="selectable", B="selectable"), ["A"])
    # Eight pairs; fixture thresholds: provisional at ceil(0.75 x 8) = 6, final needs all 8.
    a = [("R", "R")] * 7 + [("T", "R")]
    b = [("R", "R")] * 6 + [("T", "P"), ("P", "P")]
    out = M.decide(cands, dict(A=fake_match("A", a), B=fake_match("B", b)))
    assert out["common_count"] == 6 and not out["final"] and out["selected"] is None
    assert out["status"].startswith("provisional")
    assert out["scores_on_common_set"]["A"]["pairs"] == out["scores_on_common_set"]["B"]["pairs"] == 6
    b = [("R", "R")] * 5 + [("T", "P"), ("T", "P"), ("P", "P")]
    out = M.decide(cands, dict(A=fake_match("A", a), B=fake_match("B", b)))
    assert out["status"].startswith("insufficient") and out["scores_on_common_set"] is None
    assert out["own_match"] is None
    f = [("R", "R")] * 7 + [("F", "R")]
    out = M.decide(cands, dict(A=fake_match("A", [("R", "R")] * 8), B=fake_match("B", f)))
    assert out["status"].startswith("blocked") and out["selected"] is None
    unfinished = fake_match("B", [("R", "R")] * 8)
    with pytest.raises(ValueError, match="unfinished"):
        M.decide(cands, dict(A=fake_match("A", a), B=(unfinished[0], unfinished[1][:-1])))
    other = fake_match("B", [("R", "R")] * 8, ids=[f"x{i}" for i in range(8)])
    with pytest.raises(ValueError, match="frozen protocol"):
        M.decide(cands, dict(A=fake_match("A", a), B=other))
    with pytest.raises(ValueError, match="every selectable"):
        M.decide(cands, dict(A=fake_match("A", a)))


def test_a_report_only_match_never_blocks_shrinks_or_changes_the_selection():
    cands = fake_candidates({"R1-mix": "selectable", "R1-pure-full": "report-only"}, ["R1-mix"])
    mix = fake_match("R1-mix", [("R", "P")] * 8)
    # Failures and truncations in the report-only match.
    pure = fake_match("R1-pure-full", [("F", "R"), ("T", "T")] + [("R", "R")] * 6)
    out = M.decide(cands, {"R1-mix": mix, "R1-pure-full": pure})
    assert out["final"] and out["selected"]["name"] == "R1-mix" and out["common_count"] == 8
    assert out["decided_by"] == ["R1-mix"] and "R1-pure-full" not in out["failures"]
    side = out["report_only"]["R1-pure-full"]
    assert side["failures"] == 1 and side["comparison"]["common_count"] == 6
    assert set(side["comparison"]["scores"]) == {"R1-mix", "R1-pure-full"}
    # Unfinished or absent: still the same selection.
    for matches in ({"R1-mix": mix, "R1-pure-full": (pure[0], pure[1][:-3])}, {"R1-mix": mix}):
        out = M.decide(cands, matches)
        assert out["final"] and out["selected"]["name"] == "R1-mix"
        assert out["report_only"]["R1-pure-full"]["status"].startswith("pending")


def test_production_thresholds_are_48_and_64():
    cands = fake_candidates(dict(A="selectable", B="report-only"), ["A"])
    b = [("T", "T")] * 64  # a report-only match with no complete pair changes nothing
    for complete, status in ((64, "final"), (50, "provisional"), (48, "provisional"), (47, "insufficient")):
        a = [("R", "P")] * complete + [("T", "R")] * (64 - complete)
        out = M.decide(cands, dict(A=fake_match("A", a, protocol="production"),
                                   B=fake_match("B", b, protocol="production")))
        assert out["common_count"] == complete and out["status"].startswith(status)
        assert out["final"] == (status == "final")


def test_select_and_test_plan_rules_end_to_end(world, runtime, played):
    matches = played["matches"]
    selection = M.select(matches, world["root"], candidates_path=matches / "candidates.json", log=lambda _: None)
    # Every fixture game is truncated at the cap: no complete pair, so nothing is selected, no comparison or
    # per-match score is shown, and no final record is written (a later decision is not blocked).
    assert selection["common_count"] == 0 and not selection["final"]
    assert selection["scores_on_common_set"] is None and selection["own_match"] is None
    assert not (matches / "selection.json").exists() and (matches / "selection-status.json").exists()
    assert M.select(matches, world["root"], candidates_path=matches / "candidates.json",
                    log=lambda _: None)["status"] == selection["status"]
    with pytest.raises(ValueError, match="needs the recorded selection"):
        plan(world, matches, runtime, kind="test", candidate=None, selection_path=matches / "selection.json")
    # A final selection record (as the rule would write it) enables exactly one test plan for that R.
    other = world["base"] / "final/runs/matches"
    shutil.copytree(matches, other, ignore=shutil.ignore_patterns("selection.json", "validation-*"))
    cands = M.load_candidates(other / "candidates.json")
    spec = cands["by_name"]["R1-mix"]
    M.publish_once(other / "selection.json", M.sealed(dict(
        schema=M.SELECTION_SCHEMA, protocol="fixture", candidates_file=cands["file"], final=True, status="final",
        selected=dict(name="R1-mix", path=spec["path"], sha256=spec["sha256"], checkpoint_id=spec["checkpoint_id"]))))
    with pytest.raises(ValueError, match="only the selected"):
        plan(world, other, runtime, kind="test", candidate="R1-pure-full", selection_path=other / "selection.json")
    test = plan(world, other, runtime, kind="test", candidate=None, selection_path=other / "selection.json")
    assert test["split"] == "test" and test["score_player_id"] == "R1-mix"
    sets = M.load_openings(other / "openings.json")["sets"]
    assert test["opening_ids"] == [o.opening_id for o in sets.test][:2]
    with pytest.raises(ValueError, match="played once"):
        plan(world, other, runtime, kind="test", candidate=None, selection_path=other / "selection.json")
    rung = plan(world, other, runtime, kind="rung", candidate=None, opponent="material-greedy",
                selection_path=other / "selection.json", count=1)
    assert rung["score_player_id"] == "R1-mix" and rung["players"][1]["name"] == "material-greedy"
    M.run(other / rung["match_id"], world["root"], log=lambda _: None)
    report = M.match_report(other / rung["match_id"], world["root"])
    assert report["summary"]["left_player_id"] == "R1-mix"


def test_stockfish_rung_requires_the_exact_pinned_bytes(world, runtime, tmp_path):
    engine = tmp_path / "engine"
    engine.write_bytes(b"not the pinned engine")
    matches = setup_matches(world, "stockfish")
    cands = M.load_candidates(matches / "candidates.json")
    spec = cands["by_name"]["R1-mix"]
    M.publish_once(matches / "selection.json", M.sealed(dict(
        schema=M.SELECTION_SCHEMA, protocol="production", candidates_file=cands["file"], final=True, status="final",
        selected=dict(name="R1-mix", path=spec["path"], sha256=spec["sha256"], checkpoint_id=spec["checkpoint_id"]))))
    kwargs = dict(root=world["root"], kind="rung", openings_path=matches / "openings.json",
                  candidates_path=matches / "candidates.json", device="cpu", workers=1,
                  selection_path=matches / "selection.json", runtime=runtime, log=lambda _: None)
    with pytest.raises(ValueError, match="checksum"):
        M.create_plan(matches, opponent="stockfish-1k", stockfish=engine, **kwargs)
    with pytest.raises(ValueError, match="needs --stockfish"):
        M.create_plan(matches, opponent="stockfish-10k", **kwargs)
    assert M.STOCKFISH_PIN["sha256"] == "3bd05ad6e2fcfb7384aca5d4ebd50a493c07b32fc3838b65d17405c97ab45bc4"


def test_claim_rule_and_attrition():
    def summary(interval, complete):
        return dict(complete_pairs=complete, paired_uncertainty=None if interval is None else dict(interval=interval))
    assert M.test_claim(summary([0.52, 0.7], 64), 64, "production", 0)["result"] == "R improves over P"
    assert M.test_claim(summary([0.3, 0.49], 64), 64, "production", 0)["result"] == "regression"
    assert M.test_claim(summary([0.45, 0.6], 64), 64, "production", 0)["result"].startswith("neither")
    assert M.test_claim(summary([0.5, 0.7], 64), 64, "production", 0)["result"].startswith("neither")
    provisional = M.test_claim(summary([0.52, 0.7], 50), 64, "production", 0)
    assert provisional["result"] == "R improves over P" and provisional["provisional"]
    assert M.test_claim(summary([0.52, 0.7], 47), 64, "production", 0)["result"] == "none"
    assert M.test_claim(summary([0.52, 0.7], 64), 64, "production", 1)["result"] == "none"
