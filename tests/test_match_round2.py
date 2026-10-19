"""Round-2 match records: candidate freeze from the round-1 selection, validation plans for the round-2 runs,
selection with reused round-1 matches under the audited cross-revision exception, and the test-plan guard."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

torch = pytest.importorskip("torch")

from chess_rl.evaluation import match_campaign as M
from chess_rl.evaluation.matches import digest

from tests.test_match_campaign import (CAP, fake_candidates, fake_match, plan, played, run_cli, runtime,  # noqa: F401
                                       save_model, setup_matches, world)

EXCEPTED = sorted(M.CROSS_REVISION_EXCEPTION["files"])


def final_selection(matches, name="R1-mix"):
    """A final round-1 selection record as the rule writes it (fixture games never complete a pair)."""
    cands = M.load_candidates(matches / "candidates.json")
    spec = cands["by_name"][name]
    M.publish_once(matches / "selection.json", M.sealed(dict(
        schema=M.SELECTION_SCHEMA, protocol="fixture", candidates_file=cands["file"], final=True, status="final",
        selected=dict(name=name, path=spec["path"], sha256=spec["sha256"], checkpoint_id=spec["checkpoint_id"]))))


def round2_summary(path, sha, *, role, exploratory, start_sha, by_round=None):
    return dict(name="R2-" + role, role=role, completed=True, exploratory=exploratory,
                candidate=dict(path=str(path), sha256=sha, update=4, source="this run"),
                selection=dict(candidate=Path(path).name), start_checkpoint=dict(sha256=start_sha),
                final_delta_total=0.1, round_sampling=None if by_round is None else dict(by_round=by_round))


def round2_spec(world, matches, *, mix_extra=None, pure_role="report-only"):
    """Fixture round-2 runs trained from R1-mix, written next to the round-1 fixtures; returns the spec."""
    root = world["root"]
    r1 = M.load_candidates(matches / "candidates.json")["by_name"]["R1-mix"]
    runs = {}
    for name, recipe, seed, extra in (
            ("R2-mix", "mix", 12, dict(passes_over_selfplay=None, thresholds=dict(hard_stop_enforced=True), **(mix_extra or {}))),
            ("R2-pure", "pure", 13, dict(exploratory=dict(amendment="decision 3"), thresholds=dict(hard_stop_enforced=False)))):
        path = f"runs/train/{name.lower()}-fixture/update-000004.pt"
        sha = save_model(root / path, seed, dict(recipe=recipe, initial_checkpoint=dict(sha256=r1["sha256"]), **extra))
        exploratory = recipe == "pure"
        role = "selectable" if name == "R2-mix" else pure_role
        summary = round2_summary(path, sha, role=role, exploratory=exploratory, start_sha=r1["sha256"],
                                 by_round=[dict(cycles_completed=0, effective_passes=0.2), dict(cycles_completed=1, effective_passes=1.0005)])
        (root / path).with_name("summary.json").write_text(json.dumps(summary))
        runs[name] = dict(name=name, path=path, sha256=sha, summary=str(Path(path).with_name("summary.json")),
                          recipe=recipe, eligibility=role, note="fixture")
    rel = lambda p: str(Path(p).relative_to(root)) if str(p).startswith(str(root)) else str(p)
    return dict(round=2, previous_round=dict(candidates=rel(matches / "candidates.json"), selection=rel(matches / "selection.json")),
                P=dict(path=world["spec"]["P"]["path"], sha256=world["spec"]["P"]["sha256"]),
                start=dict(name="R1-mix", path=r1["path"], sha256=r1["sha256"]),
                candidates=(dict(name="R1-mix", carried_from_round=1, eligibility="selectable",
                                 validation_match="validation-P-vs-R1-mix", note="fixture"),
                            runs["R2-mix"],
                            dict(name="R1-pure-full", carried_from_round=1, eligibility="report-only",
                                 validation_match="validation-P-vs-R1-pure-full", note="fixture"),
                            runs["R2-pure"]),
                excluded=(), tie_order=("R1-mix", "R2-mix"), decided="fixture")


@pytest.fixture(scope="module")
def round2(world, runtime, played):
    """A matches directory with the played round-1 fixture matches, a final round-1 selection, and the round-2
    candidate record frozen from fixture R2 runs; the round-1 records live under the world root so the spec's
    relative paths resolve."""
    matches = world["root"] / "runs/matches"
    shutil.copytree(played["matches"], matches)
    final_selection(matches)
    spec = round2_spec(world, matches)
    record = M.freeze_round2_candidates(matches / "candidates-round2.json", root=world["root"], spec=spec, log=lambda _: None)
    return dict(matches=matches, spec=spec, record=record)


def test_round2_candidate_record_matches_the_decision():
    spec = M.ROUND2_CANDIDATES
    names = [c["name"] for c in spec["candidates"]]
    assert names == ["R1-mix", "R2-mix", "R1-pure-full", "R2-pure"]
    assert [c["eligibility"] for c in spec["candidates"]] == ["selectable", "selectable", "report-only", "report-only"]
    assert spec["tie_order"] == ("R1-mix", "R2-mix") and spec["start"]["name"] == "R1-mix"
    assert spec["start"]["sha256"] == M.ROUND1_CANDIDATES["candidates"][0]["sha256"]
    assert spec["P"] == M.ROUND1_CANDIDATES["P"]
    for c in spec["candidates"]:
        if c.get("carried_from_round") == 1:
            assert c["name"] in {x["name"] for x in M.ROUND1_CANDIDATES["candidates"]}
        else:
            assert len(c["sha256"]) == 64 and c["path"].startswith("runs/train/r2-") and c["summary"].endswith("summary.json")
    assert set(M.CROSS_REVISION_EXCEPTION["files"]) == {"scripts/run_match.py", "src/chess_rl/evaluation/match_campaign.py",
                                                         "src/chess_rl/training/checkpoints.py"}


def test_round2_freeze_carries_round1_entries_and_checks_lineage(world, round2):
    record = round2["record"]["record"]
    by_name = round2["record"]["by_name"]
    assert record["round"] == 2 and record["schema"] == M.CANDIDATES_SCHEMA
    r1 = M.load_candidates(round2["matches"] / "candidates.json")
    assert record["previous_round"]["candidates_file"] == r1["file"] and record["previous_round"]["selected"] == "R1-mix"
    assert by_name["R1-mix"]["carried_from_round"] == 1 and by_name["R1-mix"]["sha256"] == r1["by_name"]["R1-mix"]["sha256"]
    assert by_name["R1-mix"]["validation_plan_sha256"] == M.read_sealed(
        round2["matches"] / "validation-P-vs-R1-mix/plan.json", M.PLAN_SCHEMA)["sha256"]
    assert by_name["R2-mix"]["round"] == 2 and by_name["R2-mix"]["kind"] == "rl-R" and not by_name["R2-mix"]["exploratory"]
    assert by_name["R2-pure"]["exploratory"] and by_name["R2-pure"]["eligibility"] == "report-only"
    assert record["rules"]["cross_revision"] == M.CROSS_REVISION_EXCEPTION
    with pytest.raises(ValueError, match="never rewritten"):
        M.freeze_round2_candidates(round2["matches"] / "candidates-round2.json", root=world["root"], spec=round2["spec"],
                                   log=lambda _: None)


def test_round2_freeze_refuses_wrong_start_role_or_reuse(world, round2, tmp_path):
    root, matches = world["root"], round2["matches"]
    base = round2["spec"]
    # Trained from P instead of R1-mix.
    bad = deepcopy(base)
    path = "runs/train/r2-from-p/update-000004.pt"
    sha = save_model(root / path, 21, dict(recipe="mix", initial_checkpoint=dict(sha256=base["P"]["sha256"]),
                                            thresholds=dict(hard_stop_enforced=True)))
    (root / path).with_name("summary.json").write_text(json.dumps(round2_summary(path, sha, role="selectable", exploratory=False,
                                                                                 start_sha=base["P"]["sha256"])))
    bad["candidates"] = (base["candidates"][0], dict(base["candidates"][1], path=path, sha256=sha,
                                                     summary=str(Path(path).with_name("summary.json"))), *base["candidates"][2:])
    with pytest.raises(ValueError, match="not trained from R1-mix"):
        M.freeze_round2_candidates(tmp_path / "a.json", root=root, spec=bad, log=lambda _: None)
    # The summary's role disagrees with the eligibility.
    bad = deepcopy(base)
    summary_path = root / base["candidates"][1]["summary"]
    original = summary_path.read_text()
    summary_path.write_text(json.dumps(dict(json.loads(original), role="report-only")))
    try:
        with pytest.raises(ValueError, match="another candidate, role or start"):
            M.freeze_round2_candidates(tmp_path / "b.json", root=root, spec=bad, log=lambda _: None)
    finally:
        summary_path.write_text(original)
    # A selectable run that re-sampled round 2 beyond one pass.
    summary_path.write_text(json.dumps(dict(json.loads(original), round_sampling=dict(by_round=[
        dict(cycles_completed=0, effective_passes=0.4), dict(cycles_completed=1, effective_passes=1.7)]))))
    try:
        with pytest.raises(ValueError, match="beyond one pass"):
            M.freeze_round2_candidates(tmp_path / "c.json", root=root, spec=base, log=lambda _: None)
    finally:
        summary_path.write_text(original)
    # Carrying over a round-1 name with another eligibility, or without the final selection.
    bad = deepcopy(base)
    bad["candidates"] = (dict(base["candidates"][0], eligibility="report-only"), *base["candidates"][1:])
    with pytest.raises(ValueError, match="tie order|not a round-1 candidate"):
        M.freeze_round2_candidates(tmp_path / "d.json", root=root, spec=bad, log=lambda _: None)
    other = tmp_path / "runs/matches"
    shutil.copytree(matches, other, ignore=shutil.ignore_patterns("selection.json", "candidates-round2.json"))
    bad = deepcopy(base)
    bad["previous_round"] = dict(candidates=str(other / "candidates.json"), selection=str(other / "selection.json"))
    with pytest.raises(FileNotFoundError):
        M.freeze_round2_candidates(tmp_path / "e.json", root=root, spec=bad, log=lambda _: None)


def test_round2_validation_plan_runs_and_reports_with_its_own_record(world, runtime, round2):
    matches, root = round2["matches"], world["root"]
    p = M.create_plan(matches, root=root, kind="validation", candidate="R2-mix", openings_path=matches / "openings.json",
                      candidates_path=matches / "candidates-round2.json", device="cpu", workers=2, simulations=2, ply_cap=CAP,
                      protocol="fixture", count=2, runtime=runtime, log=lambda _: None)
    assert p["match_id"] == "validation-P-vs-R2-mix" and p["candidates_file"] == round2["record"]["file"]
    assert p["players"][1]["name"] == "R2-mix" and p["players"][0]["name"] == "P" and p["score_player_id"] == "R2-mix"
    with pytest.raises(ValueError, match="not a frozen validation candidate"):
        M.create_plan(matches, root=root, kind="validation", candidate="R2-mix", openings_path=matches / "openings.json",
                      candidates_path=matches / "candidates.json", device="cpu", workers=2, simulations=2, ply_cap=CAP,
                      protocol="fixture", count=2, runtime=runtime, log=lambda _: None)
    # verify_inputs finds the round-2 record by identity; the run and the report work with it.
    M.verify_inputs(p, root, matches)
    M.run(matches / p["match_id"], root, log=lambda _: None)
    assert M.status(matches / p["match_id"], root)["finished"]
    report = M.match_report(matches / p["match_id"], root)
    assert report["score_player_id"] == "R2-mix" and report["eligibility"] == "selectable"
    # The round-1 report still resolves its own (round-1) record.
    assert M.match_report(matches / "validation-P-vs-R1-mix", root)["eligibility"] == "selectable"


def test_round2_cli_round_option(world, round2, tmp_path):
    matches = tmp_path / "runs/matches"
    shutil.copytree(round2["matches"], matches, ignore=shutil.ignore_patterns("candidates-round2.json", "validation-P-vs-R2-*"))
    proc = run_cli(world["root"], matches, "--round", "2", "select")
    out, err = proc.communicate(timeout=120)
    assert proc.returncode != 0 and "candidates-round2.json" in err
    proc = run_cli(world["root"], matches, "--round", "1", "status", "--match-id", "validation-P-vs-R1-mix")
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0 and json.loads(out)["finished"]


def test_test_and_rung_plans_bind_to_the_latest_round(world, runtime, round2):
    matches = round2["matches"]
    with pytest.raises(ValueError, match="round-2 candidate record exists"):
        M.create_plan(matches, root=world["root"], kind="test", openings_path=matches / "openings.json",
                      candidates_path=matches / "candidates.json", device="cpu", workers=2, simulations=2, ply_cap=CAP,
                      protocol="fixture", count=1, runtime=runtime, selection_path=matches / "selection.json", log=lambda _: None)
    with pytest.raises(ValueError, match="needs the recorded selection"):
        M.create_plan(matches, root=world["root"], kind="rung", opponent="material-greedy", openings_path=matches / "openings.json",
                      candidates_path=matches / "candidates-round2.json", device="cpu", workers=2, simulations=2, ply_cap=CAP,
                      protocol="fixture", count=1, runtime=runtime, selection_path=matches / "selection-round2.json",
                      log=lambda _: None)
    assert not list(matches.glob("test-*")) and not (matches / "test-plan-claim.json").exists()


# Selection rule with reused round-1 matches and the cross-revision exception (pure decide(), fake matches).

def round2_fakes(sources_old, sources_new, *, r1_results, r2_results, pure_results=None):
    cands = fake_candidates(dict(**{"R1-mix": "selectable", "R2-mix": "selectable", "R1-pure-full": "report-only",
                                    "R2-pure": "report-only"}), ("R1-mix", "R2-mix"))
    cands["record"].update(round=2, previous_round=dict(candidates_file="cf1", candidates_sha256="x", selected="R1-mix"))
    cands["record"]["P"] = dict(sha256="pp")
    for n in ("R1-mix", "R1-pure-full"):
        cands["by_name"][n].update(carried_from_round=1, validation_match=f"validation-P-vs-{n}", validation_plan_sha256="p" + n)
    matches = {}
    for n, results, sources, cf, created in (("R1-mix", r1_results, sources_old, "cf1", "2026-10-17T20:32:09.055+00:00"),
                                             ("R2-mix", r2_results, sources_new, "cf", "2026-10-19T12:00:00.000+00:00")):
        p, games = fake_match(n, results)
        p.update(sources=sources, candidates_file=cf, created_utc=created, openings_file="of")
        matches[n] = (p, games)
    if pure_results is not None:
        p, games = fake_match("R1-pure-full", pure_results)
        p.update(sources=sources_old, candidates_file="cf1", created_utc="2026-10-17T20:32:12.776+00:00", openings_file="of")
        matches["R1-pure-full"] = (p, games)
    return cands, matches


ADDED = "src/chess_rl/search/batched_evaluator.py"
OLD = dict({n: M.CROSS_REVISION_EXCEPTION["files"][n] for n in EXCEPTED}, **{"src/chess_rl/search/puct.py": "same",
                                                                            "src/chess_rl/model/network.py": "same"})
NEW = dict({n: "new-" + n for n in EXCEPTED}, **{"src/chess_rl/search/puct.py": "same", "src/chess_rl/model/network.py": "same",
                                                 ADDED: M.CROSS_REVISION_EXCEPTION["added_files"][ADDED]})


def evidence(matches_root, old=OLD, new=NEW, **override):
    record = dict(schema=M.EQUIVALENCE_SCHEMA, old_sources={n: old[n] for n in EXCEPTED}, new_sources={n: new[n] for n in EXCEPTED},
                  old_pinned_sources=dict(old), new_pinned_sources={n: new.get(n) for n in old},
                  passed=True, comparisons=320, network_evaluations=640, search_decisions=192, mismatches=0, smoke=False,
                  old_revision="1bacbfe", new_revision="tree",
                  settings=dict(roots=64, plies=3, simulations=128, white="P", black="R1-mix", device="cpu", threads=4, seed=1),
                  checkpoints={"P": dict(sha256="pp"), "R1-mix": dict(sha256="sha-R1-mix")},
                  reference_plan=dict(match_id="validation-P-vs-R1-mix", sha256="pR1-mix"), openings_file="of",
                  openings_set_sha256="s")
    record.update(override)
    path = Path(matches_root) / M.EQUIVALENCE_FILE
    if path.exists():
        path.unlink()
    M.publish_once(path, M.sealed(record))
    return path


def test_round2_selection_reuses_r1_mix_under_the_exception_with_evidence(tmp_path):
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "P"), ("R", "R"), ("P", "P")],
                                  r2_results=[("R", "R"), ("R", None), ("P", "P")])
    with pytest.raises(ValueError, match="equivalence evidence"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path)
    out = M.decide(cands, matches, matches_root=tmp_path)
    assert out["final"] and out["selected"]["name"] == "R2-mix" and out["round"] == 2
    assert out["reused_matches"] == {"R1-mix": "validation-P-vs-R1-mix"}
    audit = out["cross_revision"]
    assert audit["reference_plan"] == "validation-P-vs-R2-mix" and audit["evidence"]["comparisons"] == 320
    assert sorted(audit["plans"]["R1-mix"]["differing_files"]) == sorted(EXCEPTED + [ADDED])
    assert audit["evidence"]["search_decisions"] == 192 and audit["evidence"]["settings"]["roots"] == 64
    assert audit["plans"]["R2-mix"]["differing_files"] == {}
    assert audit["exception"]["files"] == M.CROSS_REVISION_EXCEPTION["files"]
    # Tie: R1-mix first in tie order.
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "R"), ("P", "P")], r2_results=[("R", "P"), ("R", "P")])
    out = M.decide(cands, matches, matches_root=tmp_path)
    assert out["selected"]["name"] == "R1-mix" and out["selected"]["tied_with"] == ["R2-mix"]
    # A report-only round-1 match is reported beside the selection and never blocks.
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "R"), ("P", "P")], r2_results=[("R", "R"), ("R", "P")],
                                  pure_results=[("F", "R"), ("R", "R")])
    out = M.decide(cands, matches, matches_root=tmp_path)
    assert out["final"] and out["selected"]["name"] == "R2-mix" and out["report_only"]["R1-pure-full"]["failures"] == 1


def test_round2_selection_refuses_uncovered_or_unproven_source_differences(tmp_path):
    # A game-play file differs: no exception, rerun required.
    changed = dict(NEW, **{"src/chess_rl/search/puct.py": "changed"})
    cands, matches = round2_fakes(OLD, changed, r1_results=[("R", "R")], r2_results=[("R", "R")])
    evidence(tmp_path)
    with pytest.raises(ValueError, match="puct.py differs .* not covered"):
        M.decide(cands, matches, matches_root=tmp_path)
    # An excepted file at a hash other than the recorded round-1 one is not covered either.
    old_other = dict(OLD, **{"scripts/run_match.py": "some-other-old-hash"})
    cands, matches = round2_fakes(old_other, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")])
    with pytest.raises(ValueError, match="run_match.py differs .* not covered"):
        M.decide(cands, matches, matches_root=tmp_path)
    # Evidence for another revision pair, or failed evidence, is refused.
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")])
    evidence(tmp_path, new=dict(NEW, **{"scripts/run_match.py": "yet-another"}))
    with pytest.raises(ValueError, match="another pair of revisions"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path, passed=False, mismatches=3)
    with pytest.raises(ValueError, match="does not establish equivalence"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path, checkpoints={"P": dict(sha256="pp")})
    with pytest.raises(ValueError, match="other checkpoints"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path, checkpoints={"P": dict(sha256="pp"), "R1-mix": dict(sha256="another-r1-mix")})
    with pytest.raises(ValueError, match="other checkpoints"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path, reference_plan=dict(match_id="validation-P-vs-R1-mix", sha256="not-the-sealed-plan"))
    with pytest.raises(ValueError, match="other checkpoints"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path, openings_file="other-openings")
    with pytest.raises(ValueError, match="other checkpoints"):
        M.decide(cands, matches, matches_root=tmp_path)
    # Coverage: a smoke run, fewer roots, fewer plies, other simulations or missing decisions are refused.
    for bad in (dict(smoke=True), dict(settings=dict(roots=2, plies=3, simulations=128, white="P", black="R1-mix")),
                dict(settings=dict(roots=64, plies=1, simulations=128, white="P", black="R1-mix")),
                dict(settings=dict(roots=64, plies=3, simulations=64, white="P", black="R1-mix")),
                dict(search_decisions=10), dict(network_evaluations=639), dict(comparisons=10, network_evaluations=20)):
        evidence(tmp_path, **bad)
        with pytest.raises(ValueError, match="coverage is incomplete"):
            M.decide(cands, matches, matches_root=tmp_path)
    # The full pinned source maps must match both plans exactly.
    evidence(tmp_path, old_pinned_sources=dict(OLD, **{"src/chess_rl/model/network.py": "other"}))
    with pytest.raises(ValueError, match="another pair of revisions"):
        M.decide(cands, matches, matches_root=tmp_path)
    evidence(tmp_path, new_pinned_sources={n: "x" for n in OLD})
    with pytest.raises(ValueError, match="another pair of revisions"):
        M.decide(cands, matches, matches_root=tmp_path)
    # Added files: only the audited list at exact hashes; a pinned file missing from the new plans is never admitted.
    evidence(tmp_path)
    cands, matches = round2_fakes(OLD, dict(NEW, **{ADDED: "other-hash"}), r1_results=[("R", "R")], r2_results=[("R", "R")])
    with pytest.raises(ValueError, match="audited list of added files"):
        M.decide(cands, matches, matches_root=tmp_path)
    cands, matches = round2_fakes(OLD, dict(NEW, **{"src/chess_rl/evaluation/unlisted.py": "h"}), r1_results=[("R", "R")],
                                  r2_results=[("R", "R")])
    with pytest.raises(ValueError, match="audited list of added files"):
        M.decide(cands, matches, matches_root=tmp_path)
    removed = {n: h for n, h in NEW.items() if n != "src/chess_rl/model/network.py"}
    cands, matches = round2_fakes(OLD, removed, r1_results=[("R", "R")], r2_results=[("R", "R")])
    with pytest.raises(ValueError, match="missing from the round-2 plans"):
        M.decide(cands, matches, matches_root=tmp_path)
    # Identical sources need no evidence at all.
    cands, matches = round2_fakes(NEW, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")])
    (tmp_path / M.EQUIVALENCE_FILE).unlink()
    out = M.decide(cands, matches, matches_root=tmp_path)
    assert out["cross_revision"]["evidence"] is None and out["final"]


def test_round2_selection_checks_the_reused_plan_identity_and_candidates_file(tmp_path):
    evidence(tmp_path)
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")])
    matches["R1-mix"][0]["sha256"] = "tampered"
    with pytest.raises(ValueError, match="R1-mix: validation plan differs"):
        M.decide(cands, matches, matches_root=tmp_path)
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")])
    matches["R2-mix"][0]["candidates_file"] = "cf1"  # a round-2 plan must pin the round-2 record
    with pytest.raises(ValueError, match="R2-mix: validation plan differs"):
        M.decide(cands, matches, matches_root=tmp_path)
    cands, matches = round2_fakes(OLD, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")])
    matches["R1-mix"][0]["players"][0]["sha256"] = "another-P"
    with pytest.raises(ValueError, match="R1-mix: validation plan differs"):
        M.decide(cands, matches, matches_root=tmp_path)
    with pytest.raises(ValueError, match="needs the matches root"):
        M.decide(*round2_fakes(OLD, NEW, r1_results=[("R", "R")], r2_results=[("R", "R")]))


def test_select_round2_writes_its_own_files_and_requires_its_record(world, round2, tmp_path):
    matches = tmp_path / "runs/matches"
    shutil.copytree(round2["matches"], matches)
    with pytest.raises(ValueError, match="not the round-2 candidate record"):
        M.select(matches, world["root"], candidates_path=matches / "candidates.json", round_number=2, log=lambda _: None)
    with pytest.raises(ValueError, match="no validation match planned"):
        # R2-mix's fixture match may or may not have been copied; remove it to hit the rule deterministically.
        shutil.rmtree(matches / "validation-P-vs-R2-mix", ignore_errors=True)
        M.select(matches, world["root"], candidates_path=matches / "candidates-round2.json", round_number=2, log=lambda _: None)


# The validation job's control flow, with the Python runtime, git, flock and awk stubbed (no GPU, no games).

JOB = ROOT_JOB = Path(__file__).resolve().parents[1] / "jobs/b200-selfplay/match-validation-round2.sbatch"
FAKE_PY = r'''#!/bin/bash
# Stub of the pinned Python: logs every invocation and produces the files the job waits for.
log="$FAKE_LOG"; echo "$*" >> "$log"
case "$*" in
  *" candidates") mkdir -p runs/matches; echo '{}' > runs/matches/candidates-round2.json ;;
  *" plan "*) c=$(echo "$*" | sed -E 's/.*--candidate ([^ ]+).*/\1/'); mkdir -p runs/matches/validation-P-vs-$c
             echo '{}' > runs/matches/validation-P-vs-$c/plan.json ;;
  *" run --match-id "*) m=${*##*--match-id }; touch runs/matches/$m/played ;;
  *" select") test -e runs/matches/evaluation-equivalence-round2.json || { echo "selection not final: no evidence" >&2; exit 1; }
              echo '{}' > runs/matches/selection-round2.json ;;
esac
exit 0
'''


def job_world(tmp_path, *, evidence_present):
    repo = tmp_path / "repo"
    (repo / "jobs/b200").mkdir(parents=True)
    (repo / "jobs/b200-selfplay").mkdir()
    (repo / "runtime/cuda-cu128").mkdir(parents=True)
    (repo / ".venv-cuda-cu128/bin").mkdir(parents=True)
    (repo / "runs/matches").mkdir(parents=True)
    (repo / "artifacts/slurm").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("fixture\n")
    shutil.copy2(JOB.parent.parent / "b200/common.sh", repo / "jobs/b200/common.sh")
    shutil.copy2(JOB, repo / "jobs/b200-selfplay/match-validation-round2.sbatch")
    for name in ("openings.json", "candidates.json", "selection.json"):
        (repo / "runs/matches" / name).write_text("{}\n")
    if evidence_present:
        (repo / "runs/matches" / M.EQUIVALENCE_FILE).write_text("{}\n")
    py = repo / ".venv-cuda-cu128/bin/python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("git", "echo fixture-revision\n"), ("flock", "exit 0\n"), ("awk", "exit 0\n")):
        (stubs / name).write_text("#!/bin/bash\n" + body)
        (stubs / name).chmod(0o755)
    return repo, stubs


def run_job(repo, stubs, log):
    import os
    import subprocess
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(repo), SLURM_JOB_ID="1", SLURM_JOB_PARTITION="gpu-small",
               SLURM_JOB_ACCOUNT="student", FAKE_LOG=str(log), PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(["bash", str(repo / "jobs/b200-selfplay/match-validation-round2.sbatch")], env=env, cwd=str(repo),
                          capture_output=True, text=True)


def job_steps(log):
    steps = []
    for line in log.read_text().splitlines():
        if "run_match.py" not in line:
            continue
        words = line.split("run_match.py --round 2 ")[1].split()
        steps.append(" ".join(w for w in words if not w.startswith("--") or w == "--match-id")
                     .replace(" --match-id", "").replace(" validation ", " ").replace("cuda:0 5", "").strip())
    return steps


def test_job_runs_r2_mix_then_selection_then_r2_pure(tmp_path):
    repo, stubs = job_world(tmp_path, evidence_present=True)
    log = tmp_path / "calls.log"
    result = run_job(repo, stubs, log)
    assert result.returncode == 0, result.stdout + result.stderr
    steps = job_steps(log)
    assert steps == ["candidates", "plan R2-mix", "plan R2-pure", "run validation-P-vs-R2-mix", "select",
                     "run validation-P-vs-R2-pure", "report validation-P-vs-R2-mix", "report validation-P-vs-R2-pure"]
    assert (repo / "runs/matches/selection-round2.json").exists()
    # A resubmit skips the frozen record and plans, and an existing selection is never redone.
    log.unlink()
    assert run_job(repo, stubs, log).returncode == 0
    assert job_steps(log) == ["run validation-P-vs-R2-mix", "run validation-P-vs-R2-pure",
                              "report validation-P-vs-R2-mix", "report validation-P-vs-R2-pure"]


def test_job_without_evidence_skips_selection_and_still_plays_r2_pure(tmp_path):
    repo, stubs = job_world(tmp_path, evidence_present=False)
    log = tmp_path / "calls.log"
    result = run_job(repo, stubs, log)
    assert result.returncode == 0, result.stdout + result.stderr
    steps = job_steps(log)
    assert "select" not in steps and "run validation-P-vs-R2-pure" in steps
    assert steps.index("run validation-P-vs-R2-mix") < steps.index("run validation-P-vs-R2-pure")
    assert "no runs/matches/evaluation-equivalence-round2.json yet" in result.stderr
    assert not (repo / "runs/matches/selection-round2.json").exists()
