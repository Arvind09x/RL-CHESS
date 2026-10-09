"""Freeze then execute bounded actual T03-checkpoint games and crash/resume audit."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

import numpy as np
import torch

from chess_rl.adapter import encode_position, make_state
from chess_rl.evaluation.matches import Opening, digest
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.selfplay import Phase, Session, Settings, _checkpoint, audit, runtime, summarize
from tests.helpers import CASTLING, PROMOTION, BLACK_PROMOTION, CYCLE, LONG_CYCLE


ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sources():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]}


def specs():
    result = []
    def add(name, state, cap, budget=4, phases=None, seed=20261010):
        result.append(dict(name=name, opening=asdict(Opening.from_state(name, state,
            source="hand-authored declared-setup/standard-game R01 correctness fixtures; no book sampling")),
            settings=asdict(Settings(budget, cap, phases=phases or (Phase(0, 1., .25), Phase(4, 0., .25)))),
            seed=seed, opening_selection_seed=20261010))
    add("standard-a", make_state(), 8)
    add("standard-b", make_state(), 8, seed=20261011)
    add("opening-e4", make_state(moves_uci=["e2e4", "e7e5", "g1f3", "b8c6"]), 8)
    long = make_state(moves_uci=LONG_CYCLE + ["e2e4", "e7e5", "g1f3"])
    add("long-history", long, len(long.board.move_stack) + 4)
    for color, fen in (("white", "7k/8/5KQ1/8/8/8/8/8 w - - 99 1"),
                       ("black", "8/8/8/8/8/5kq1/8/7K b - - 99 1")):
        add(color + "-mate-cap", make_state(fen, history_complete=True), 1, 64, (Phase(0, 0., 0.),))
    for color in ("w", "b"):
        add("fifty-cap-" + color, make_state(f"4k3/8/8/8/8/8/8/R3K3 {color} - - 99 1",
            history_complete=True), 1, 4, (Phase(0, 0., 0.),))
        add("castling-" + color, make_state(CASTLING.format(color), history_complete=True), 1, 3)
        add("promotion-" + color, make_state(PROMOTION if color == "w" else BLACK_PROMOTION,
                                             history_complete=True), 1, 3)
    add("ep-white", make_state(moves_uci=["e2e4", "a7a6", "e4e5", "d7d5"]), 5, 3)
    add("ep-black", make_state(moves_uci=["a2a3", "e7e5", "d2d4", "e5e4", "f2f4"]), 6, 3)
    terminal = make_state("7k/6Q1/5K2/8/8/8/8/8 b - - 150 1", history_complete=True)
    add("terminal-mate", terminal, 0)
    add("terminal-mate-duplicate", terminal, 0)
    add("terminal-stalemate", make_state("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", history_complete=True), 0)
    add("reached-threefold", make_state(moves_uci=CYCLE * 2), 8)
    add("intended-threefold-only", make_state(moves_uci=CYCLE + CYCLE[:3]), 7)
    return result


def create(directory, checkpoint, spec):
    return Session.create(directory, checkpoint, Opening(**spec["opening"]),
        settings=Settings.from_dict(spec["settings"]), seed=spec["seed"],
        opening_selection_seed=spec["opening_selection_seed"])


def core(value):
    result = copy.deepcopy(value["game"])
    result.pop("interruptions")
    return result


def child(mode, directory):
    # Deliberately scramble all global RNGs; only the recorded owned PCG64
    # stream drives exploration. Eval-mode inference consumes none of these.
    random.seed(99991)
    np.random.seed(99992)
    torch.manual_seed(99993)
    session = Session.load(directory)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    if mode == "kill":
        original = session.evaluator
        calls = 0
        def kill(states):
            nonlocal calls
            calls += 1
            if calls == 3:
                # Root noise has consumed private RNG; one leaf backup has
                # completed. Hard loss must still recover the prior boundary.
                os.kill(os.getpid(), signal.SIGKILL)
            return original(states)
        session.evaluator = kill
    result = session.run()
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1]) and now[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    save(directory / "fresh-process-result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "fixtures", "resume", "kill"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode in ("resume", "kill"):
        child(args.mode, args.output)
        return
    if args.mode == "freeze":
        args.output.mkdir(parents=True, exist_ok=False)
        evaluator = RealEvaluator(args.checkpoint)
        long = make_state(moves_uci=LONG_CYCLE + ["e2e4", "e7e5", "g1f3"])
        resume_spec = dict(name="resume-long-history", opening=asdict(Opening.from_state("resume-opening", long,
            source="hand-authored complete standard-game prefix for CPU continuation")),
            settings=asdict(Settings(4, len(long.board.move_stack) + 6,
                phases=(Phase(0, 1., .25), Phase(len(long.board.move_stack) + 3, 0., .25)))),
            seed=20261012, opening_selection_seed=20261010)
        plan = dict(schema="r01-frozen-real-suite-v1", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
            checkpoint=_checkpoint(args.checkpoint, evaluator), runtime=runtime(), source_sha256=sources(),
            games=specs(), interruption=dict(spec=resume_spec, cooperative_after_new_moves=2,
                forced_kill_evaluator_call=3, resume="fresh process; exact committed record/RNG comparison"),
            limitations="Fixed bounded plumbing fixtures; caps frozen; no tuning, scale or playing-strength claim",
            comparison="zero tolerance for boards/history/visits/policies/actions/configuration/RNG")
        save(args.output / "frozen-plan.json", plan)
        print(json.dumps(dict(games=len(plan["games"]), plan_sha256=digest(plan)), indent=2))
        return
    plan = json.loads((args.output / "frozen-plan.json").read_text())
    assert plan["runtime"] == runtime() and plan["source_sha256"] == sources()
    evaluator = RealEvaluator(args.checkpoint)
    assert json.loads(json.dumps(_checkpoint(args.checkpoint, evaluator))) == plan["checkpoint"]
    save(args.output / "execution-start.json", dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_plan_sha256=digest(plan), source_sha256=sources(), runtime=runtime()))
    games = []
    shapes = []
    for spec in plan["games"]:
        session = create(args.output / spec["name"], args.checkpoint, spec)
        hook = session.evaluator.checkpoint.model.register_forward_pre_hook(
            lambda module, inputs: shapes.append(list(inputs[0].shape)))
        result = session.run()
        hook.remove()
        assert Session.load(session.directory).snapshot() == result
        audit(result["configuration"], result["game"])
        games.append(dict(fixture=spec["name"], **result))
    spec = plan["interruption"]["spec"]
    reference = create(args.output / "continuation-reference", args.checkpoint, spec).run()
    session = create(args.output / "continuation-split", args.checkpoint, spec)
    initial = session.snapshot()
    before_features = encode_position(audit(initial["configuration"], initial["game"])).features
    stopped = session.run(stop_after_moves=plan["interruption"]["cooperative_after_new_moves"])
    assert stopped["game"]["status"] == "interrupted" and stopped["game"]["outcome"] is None
    save(args.output / "interrupted-snapshot.json", stopped)
    commands = []
    for mode in ("kill", "resume"):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), mode, "--output", str(session.directory)]
        result = subprocess.run(command, capture_output=True)
        commands.append(dict(mode=mode, argv=command, returncode=result.returncode,
            stdout=result.stdout.decode(), stderr=result.stderr.decode()))
        if mode == "kill":
            assert result.returncode == -signal.SIGKILL, commands
            assert Session.load(session.directory).snapshot() == stopped
        else:
            assert result.returncode == 0, commands
    resumed = json.loads((session.directory / "fresh-process-result.json").read_text())
    assert resumed["configuration"] == reference["configuration"]
    assert core(resumed) == core(reference)
    assert resumed["game"]["moves_uci"][:len(spec["opening"]["moves_uci"])] == spec["opening"]["moves_uci"]
    np.testing.assert_array_equal(before_features, encode_position(Opening(**spec["opening"]).state()).features)
    assert all(shape == [1, 64, 128] for shape in shapes) and shapes
    games.append(dict(fixture="forced-interruption-before-resume", **stopped))
    save(args.output / "games.json", games)
    generated = [g for g in games if g["game"]["positions"]]
    report = dict(frozen_plan_sha256=digest(plan), scope=plan["limitations"],
        summary=summarize(games), generated_game_summary=summarize(generated),
        generated_plies=sum(len(g["game"]["positions"]) for g in games),
        committed_simulations=sum(sum(p["root_visits"]) for g in games for p in g["game"]["positions"]),
        inference=dict(forwards=len(shapes), shapes=sorted({tuple(shape) for shape in shapes}),
                       all_batch_of_one=True),
        continuation=dict(commands=commands, full_initial_history_preserved=True, opening_prefix_preserved=True,
            frozen_checkpoint_search_schedule_preserved=True, owned_rng_exact=True, global_rng_untouched=True,
            hard_kill_recovered_prior_atomic_boundary=True, exact_positions_visits_actions_policies=True,
            reference_core_sha256=digest(core(reference)), resumed_core_sha256=digest(core(resumed)),
            resumed_status=resumed["game"]["status"],
            counted_interruption="one distinct stopped game in snapshot summary; resumed reference comparison is separate"),
        checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == plan["checkpoint"]["sha256"])
    assert report["checkpoint_unchanged"]
    save(args.output / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
