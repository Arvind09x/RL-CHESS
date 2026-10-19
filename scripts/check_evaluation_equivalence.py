"""Supporting evidence for the round-2 cross-revision audit (docs/round2-match-audit.md;
chess_rl.evaluation.match_campaign.CROSS_REVISION_EXCEPTION). It supports the code audit; it is not proof of
B200 equivalence: it runs on CPU, on a finite set of positions.

R1-mix's validation match was played at an older code revision than R2-mix's. This script extracts that
revision from git and refuses unless EVERY evaluation source pinned by the sealed round-1 plan hashes
identically in the extracted tree, then runs P and R1-mix under BOTH trees, in separate isolated interpreters,
on the 64 frozen validation roots and the positions reached by alternating full-budget search decisions from
them (P as White, R1-mix as Black). Network outputs (legal policy, WDL, value) and search decisions (move and
every visited edge's visits and value sum) must be byte-identical at every position. The sealed record binds
the old and new hashes of every pinned source, the checkpoints, the reference plan, the openings, the settings
and the numerical environment; `run_match.py --round 2 select` requires it, with full coverage, before it
admits the exception. A partial run needs --smoke and is never accepted by the selection.

    .venv/bin/python scripts/check_evaluation_equivalence.py --old-revision 1bacbfe --plies 3

CPU only; expect tens of minutes for the full run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chess_rl.evaluation import match_campaign as M  # noqa: E402
from chess_rl.evaluation.matches import digest  # noqa: E402

WORKER = r'''
import hashlib, json, sys
src = sys.argv[1]
sys.path.insert(0, src)
import chess_rl
assert chess_rl.__file__.startswith(src), chess_rl.__file__
from chess_rl.runtime import configure
spec = json.load(open(sys.argv[2]))
configure(spec["device"], threads=spec["threads"])
import numpy as np
from chess_rl.adapter import apply_move, make_state, terminal_result
from chess_rl.evaluation.matches import turn_seed
from chess_rl.evaluation.players import Checkpoint, SearchPlayer, SearchSettings
from chess_rl.search.real_evaluator import RealEvaluator

def h(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()

players = {}
for name, entry in spec["players"].items():
    evaluator = RealEvaluator(entry["path"], device=spec["device"])
    assert evaluator.checkpoint.checkpoint_id == entry["checkpoint_id"], (name, evaluator.checkpoint.checkpoint_id)
    role = "P" if entry["kind"] == "pretrained-P" else "R"
    players[name] = SearchPlayer(name, role, "real-evaluator:" + entry["sha256"], evaluator,
                                 SearchSettings(spec["simulations"]),
                                 Checkpoint(entry["checkpoint_id"], entry["path"], entry["sha256"], entry["kind"]))
white, black = spec["white"], spec["black"]
out = []
for root in spec["roots"]:
    state = make_state(root["initial_fen"], root["moves_uci"], history_complete=True)
    for ply in range(spec["plies"] + 1):
        record = dict(root=root["opening_id"], ply=ply, fen=state.board.fen(en_passant="fen"), networks={}, decision=None)
        for name, player in players.items():
            result = player.evaluator([state])[0]
            record["networks"][name] = dict(priors=h(result.priors), wdl=h(result.wdl), value=float(result.value))
        if ply == spec["plies"] or terminal_result(state) is not None:
            out.append(record)
            break
        mover = players[white if state.board.turn else black]
        decision = mover.choose(state, seed=turn_seed(spec["seed"], len(state.board.move_stack), mover.player_id))
        edges = [[int(i), int(n), float(v)] for i, n, v in decision.details["visited_edges"]]
        record["decision"] = dict(player=mover.player_id, move_id=int(decision.move_id), move_uci=decision.details["move_uci"],
                                  edges=edges, simulations=int(decision.details["completed_simulations"]))
        out.append(record)
        apply_move(state, decision.move_id)
import torch
json.dump(dict(records=out, torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
          open(sys.argv[3], "w"), sort_keys=True)
'''


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_revision(root, revision, destination):
    """The older revision's src/ and scripts/run_match.py, via git archive (git is used only here)."""
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", revision + "^{commit}"],
                            check=True, capture_output=True, text=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit, "src", "scripts/run_match.py"],
                             check=True, capture_output=True).stdout
    destination.mkdir(parents=True)
    subprocess.run(["tar", "-x", "-C", str(destination)], input=archive, check=True)
    return commit


def run_worker(python, src, spec, output):
    script = output.parent / f"worker-{output.stem}.py"
    script.write_text(WORKER)
    spec_path = output.parent / f"spec-{output.stem}.json"
    spec_path.write_text(json.dumps(spec))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PYTHONHASHSEED"] = "0"
    subprocess.run([python, "-I", str(script), str(src), str(spec_path), str(output)], check=True, env=env,
                   cwd=str(output.parent))
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan", default="runs/matches/validation-P-vs-R1-mix",
                        help="the sealed round-1 validation plan whose pinned sources define the old revision")
    parser.add_argument("--old-revision", required=True, help="git revision of the round-1 validation code")
    parser.add_argument("--output", default="runs/matches/" + M.EQUIVALENCE_FILE)
    parser.add_argument("--plies", type=int, default=4, help="search decisions per root (P White, R1-mix Black)")
    parser.add_argument("--roots", type=int, help="first N validation roots (default: all 64; fewer needs --smoke)")
    parser.add_argument("--smoke", action="store_true", help="allow a partial run; the record is marked and never "
                        "accepted by the selection")
    parser.add_argument("--simulations", type=int, default=M.SIMULATIONS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026101908)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--keep", action="store_true", help="keep the temporary trees and worker outputs")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    if output.exists():
        raise SystemExit(f"{output} already exists; evidence is never rewritten")
    full = (args.roots is None or args.roots >= M.EQUIVALENCE_REQUIRED["roots"]) and \
        args.plies >= M.EQUIVALENCE_REQUIRED["min_plies"] and args.simulations == M.EQUIVALENCE_REQUIRED["simulations"]
    if not full and not args.smoke:
        raise SystemExit("a partial run (fewer than 64 roots, fewer than 2 plies or not 128 simulations) needs --smoke")
    started = time.time()
    plan = M.read_sealed(root / args.plan / "plan.json", M.PLAN_SCHEMA)
    if plan["kind"] != "validation":
        raise SystemExit("the reference plan must be a validation plan")
    openings = M.load_openings(root / "runs/matches/openings.json")
    if openings["file"] != plan["openings_file"]:
        raise SystemExit("openings.json differs from the plan's frozen openings")
    files = M.CROSS_REVISION_EXCEPTION["files"]
    for name, old in files.items():
        if plan["sources"].get(name) != old:
            raise SystemExit(f"{name}: the plan pins {plan['sources'].get(name)}, not the recorded round-1 hash")
    work = Path(tempfile.mkdtemp(prefix="equivalence-", dir=root / "artifacts")) if (root / "artifacts").exists() \
        else Path(tempfile.mkdtemp(prefix="equivalence-"))
    try:
        old_tree = work / "old"
        commit = extract_revision(root, args.old_revision, old_tree)
        # Every evaluation source pinned by the plan must hash identically in the extracted tree.
        for name, sha in plan["sources"].items():
            if not (old_tree / name).exists() or sha_file(old_tree / name) != sha:
                raise SystemExit(f"{name}: {args.old_revision} does not match the plan's pinned hash")
        current = M.source_hashes(root, sorted(plan["sources"]))
        differing = {n: dict(old=plan["sources"][n], new=current[n]) for n in sorted(current) if current[n] != plan["sources"][n]}
        unexpected = sorted(set(differing) - set(files))
        if unexpected:
            raise SystemExit(f"evaluation sources outside the exception differ from the plan: {unexpected}")
        players = {}
        for entry in plan["players"]:
            players[entry["name"]] = dict(path=str(root / entry["path"]), sha256=entry["sha256"],
                                          checkpoint_id=entry["checkpoint_id"], kind=entry["kind"])
            if sha_file(root / entry["path"]) != entry["sha256"]:
                raise SystemExit(f"{entry['path']}: checkpoint bytes differ from the plan")
        if set(players) != {"P", "R1-mix"}:
            raise SystemExit(f"the reference plan must be P vs R1-mix, not {sorted(players)}")
        roots = [o.to_dict() for o in openings["sets"].validation][: args.roots]
        spec = dict(device=args.device, threads=args.threads, simulations=args.simulations, plies=args.plies,
                    seed=args.seed, players=players, white="P", black="R1-mix", roots=roots)
        results = {}
        for label, src in (("old", old_tree / "src"), ("new", root / "src")):
            t = time.time()
            results[label] = run_worker(args.python, src, spec, work / f"{label}.json")
            results[label]["seconds"] = round(time.time() - t, 1)
            print(json.dumps(dict(event="evaluated", revision=label, positions=len(results[label]["records"]),
                                  seconds=results[label]["seconds"])), flush=True)
        old_records, new_records = results["old"]["records"], results["new"]["records"]
        mismatches = []
        if len(old_records) != len(new_records):
            mismatches.append(dict(reason="different number of positions", old=len(old_records), new=len(new_records)))
        for a, b in zip(old_records, new_records):
            if a != b:
                mismatches.append(dict(root=a["root"], ply=a["ply"], old=a, new=b))
        decisions = sum(r["decision"] is not None for r in new_records)
        record = dict(schema=M.EQUIVALENCE_SCHEMA, reference_plan=dict(match_id=plan["match_id"], sha256=plan["sha256"]),
                      openings_file=openings["file"], openings_set_sha256=plan["openings_set_sha256"],
                      old_revision=commit, old_revision_argument=args.old_revision,
                      new_revision=dict(working_tree=str(root), note="hashes below identify the evaluation sources"),
                      old_sources={n: plan["sources"][n] for n in files}, new_sources={n: current[n] for n in files},
                      old_pinned_sources=dict(plan["sources"]), new_pinned_sources=dict(current),
                      smoke=not full,
                      differing_files=differing, unchanged_files=sorted(n for n in current if n not in differing),
                      checkpoints={n: dict(path=plan_entry["path"], sha256=plan_entry["sha256"],
                                           checkpoint_id=plan_entry["checkpoint_id"]) for n, plan_entry in
                                   ((e["name"], e) for e in plan["players"])},
                      settings=dict(device=args.device, threads=args.threads, simulations=args.simulations,
                                    plies=args.plies, roots=len(roots), seed=args.seed, white="P", black="R1-mix"),
                      note=("supporting evidence for the compatibility audit on CPU; not proof of B200 equivalence"),
                      comparisons=len(new_records), network_evaluations=2 * len(new_records), search_decisions=decisions,
                      mismatches=len(mismatches), mismatch_details=mismatches[:20], passed=not mismatches,
                      runtime={k: results["new"][k] for k in ("torch", "numpy", "python")},
                      runtime_old={k: results["old"][k] for k in ("torch", "numpy", "python")},
                      seconds=dict(old=results["old"]["seconds"], new=results["new"]["seconds"],
                                   total=round(time.time() - started, 1)),
                      records_sha256=dict(old=digest(old_records), new=digest(new_records)),
                      created_utc=M._now())
        M.publish_once(output, M.sealed(record))
        print(json.dumps(dict(event="evidence", path=str(output), passed=record["passed"], comparisons=record["comparisons"],
                              search_decisions=decisions, mismatches=record["mismatches"]), indent=2))
        if mismatches:
            raise SystemExit("the two revisions differ; the cross-revision exception must not be used")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
