"""Index executed R02 evidence after preservation, without touching accepted runs."""
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import xml.etree.ElementTree as ET

import chess
import numpy as np
import torch

from chess_rl.evaluation.matches import digest
from chess_rl.replay import ReplayBuffer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/r02"


def read(name):
    return json.loads((OUT / name).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    admission, restart, accepted = [read(n) for n in ("admission.json", "restart.json", "accepted-main-audit.json")]
    exported = ReplayBuffer(OUT / "replay.sqlite").export()
    assert digest(exported) == admission["audited_export_sha256"] == restart["audited_export_sha256"] == accepted["audited_export_sha256"]
    assert admission["summary"]["training_positions"] == 4
    assert all(r["repeated"] and r["added_samples"] == 0 for r in restart["receipts"] + accepted["receipts"])
    assert admission["records_sha256"] == accepted["records_sha256"]
    assert [json.loads(line) for line in (OUT / "samples.jsonl").read_text().splitlines()] == exported["samples"]
    assert read("preservation-before.json")["primary"] == read("preservation-after.json")["primary"]
    tests = {}
    for name in ("focused-final.xml", "regression-model-final.xml", "regression-adapter.xml"):
        suite = ET.parse(OUT / name).getroot().find("testsuite")
        tests[name] = {key: int(suite.attrib[key]) for key in ("tests", "errors", "failures", "skipped")}
        tests[name]["passed"] = tests[name]["tests"] - tests[name]["skipped"]
        assert tests[name]["errors"] == tests[name]["failures"] == 0
    assert tests["focused-final.xml"]["passed"] == 44
    assert tests["regression-model-final.xml"]["passed"] == 1497
    assert tests["regression-adapter.xml"]["passed"] == 1023
    commands = {name: read(name + ".command.json") for name in ("regression-model-final", "regression-adapter")}
    historical = {"src/chess_rl/replay.py": OUT / "development/replay-before-consistent-open.py",
                  "tests/test_replay.py": OUT / "development/test-replay-before-consistent-open.py"}
    for name, command in commands.items():
        assert command["exit_code"] == 0
        assert all(sha(historical.get(path, ROOT / path) if name == "regression-adapter" else ROOT / path) == value
                   for path, value in command["sources"].items())
    summary = dict(local_date="2026-10-10", timezone="Asia/Kolkata", scope="R02 only; pending independent review",
        base="48bce1ecb469d07d9d5a57aff7d9cb4896a7e281", branch="codex/r02-replay", checkout=str(ROOT),
        retained_R01_implementation="2e31236c33a114490c59be837459803c7b4f7299",
        tests=tests, runtime=dict(python=platform.python_version(), numpy=np.__version__,
            torch=str(torch.__version__), chess=chess.__version__, sqlite=sqlite3.sqlite_version),
        admission=admission["summary"], manifest=exported["manifest"],
        audited_export_sha256=digest(exported), samples_sha256=digest(exported["samples"]),
        restart_exact=True, accepted_main_records_exact=True,
        preservation=dict(all_preserved=True, accepted_R01_checkout=True, accepted_main_tree=True,
            primary_user_patch_and_index=True, external_files=len(read("preservation-after.json")["external_files"])),
        limitations=["bounded buffer; full reaudit per operation", "representative depends on first committed completed source",
            "R01 evidence has winner/draw roots; controlled full games verify loser roots for both colors",
            "no training or R03", "no scaling, strength, GPU or hardware/power-loss claim"],
        harness_development=["Preservation checker initially compared primary to historical pre-integration R01 HEAD.",
            "Located accepted R01 main integration 48bce1e and fast-forwarded the isolated branch to that base.",
            "Corrected preservation checks to compare the accepted restored plan bytes and current patch/index; "
            "historical pre-integration diff headers are not the current integrated diff headers.",
            "Final inspection added a consistent read transaction on buffer open and a forced concurrent-commit fixture. "
            "Earlier source/test versions and their passing checks are preserved under development; "
            "final model regression covers the corrected source. Adapter production code is unchanged."],
        evidence={str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "summary.json"},
        new_sources={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src/chess_rl/replay.py",
            ROOT / "tests/test_replay.py", ROOT / "scripts/run_r02.py", ROOT / "docs/r02-replay.md")})
    with (OUT / "summary.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tests": tests, "admission": summary["admission"], "preservation": summary["preservation"]}, indent=2))


if __name__ == "__main__":
    main()
