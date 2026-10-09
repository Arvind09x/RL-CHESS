"""Verify final T04 evidence and unchanged accepted tree before committing."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from run_checks import ROOT, OUT, identity, preservation, write

BASE = "4fa48d4afef91a3bea125fb5f9f85b4d5159fc28"


def main():
    before = json.loads((OUT / "preservation-before.json").read_text())
    assert preservation() == before
    assert before["primary_head"] == BASE
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == BASE
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "codex/t04-real-evaluator"
    for name in ("focused-final", "fixture", "model", "adapter", "replay"):
        command = json.loads((OUT / f"{name}.command.json").read_text())
        assert command["returncode"] == 0
        for source, expected in command["sources"].items():
            assert identity(ROOT / source) == expected, (name, source)
    tests = {}
    for name in ("focused-final", "model", "adapter"):
        root = ET.parse(OUT / f"{name}.xml").getroot()
        totals = {key: sum(int(suite.get(key, 0)) for suite in root.findall("testsuite"))
                  for key in ("tests", "failures", "errors", "skipped")}
        assert totals["failures"] == totals["errors"] == 0
        totals["passed"] = totals["tests"] - totals["skipped"]
        tests[name] = totals
    assert tests["focused-final"]["passed"] == 94
    assert tests["model"]["passed"] == 1378
    assert tests["adapter"]["passed"] == 1023 and tests["adapter"]["skipped"] == 10
    records = json.loads((ROOT / "tests/fixtures/adapter_records.json").read_text())
    assert json.loads((OUT / "replay.stdout.txt").read_text()) == [record["expected"] for record in records]
    unchanged, appends = 0, {}
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True).splitlines()
    for path in paths:
        original = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
        actual = (ROOT / path).read_bytes()
        if path in ("README.md", "coding-plan.md"):
            assert actual.startswith(original)
            appends[path] = dict(original_bytes=len(original), append_bytes=len(actual) - len(original),
                                 t04_checkbox_pending=b"| [ ] | T04 /" in actual if path == "coding-plan.md" else None)
        else:
            assert actual == original, path
            unchanged += 1
    failed = json.loads((OUT / "focused.command.json").read_text())
    assert failed["returncode"] == 1
    fixture = json.loads((OUT / "fixture.json").read_text())
    summary = dict(base_commit=BASE, branch="codex/t04-real-evaluator", isolated_checkout=str(ROOT),
        finished_at_utc=datetime.now(timezone.utc).isoformat(), review_status="Independent review pending; T04 checkbox unchecked",
        scope="T04 only; stop before R01 and performance campaigns", tests=tests, t04_cases=34,
        frozen_adapter_fixtures=len(records), accepted_files_exactly_unchanged=unchanged,
        original_documents_preserved_with_appends=appends, primary_and_checkpoint_unchanged=True,
        checkpoint=fixture["checkpoint"], tolerances=fixture["tolerances"],
        batch_max_absolute_error=fixture["batch_max_absolute_error"],
        independent_conversion_max_absolute_error=fixture["independent_conversion_max_absolute_error"],
        bounded_search_nonterminal_roots=len(fixture["bounded_search"]), simulations_per_root=3,
        bounded_terminal_searches=fixture["terminal_searches_bypassed"], mixed_batch_size=31,
        mixed_forward_shapes=fixture["mixed_forward_shapes"], input_histories_and_rng_unchanged=True,
        initial_failed_run="focused: 93 passed, one overly strict bitwise batch comparison; corrected test, unchanged tolerances; retained evidence",
        runtime=fixture["runtime"], limitations="Pinned CPU numerical agreement only; no concurrency, accelerator or playing-strength claim",
        evidence={str(path.relative_to(OUT)): identity(path) for path in sorted(OUT.iterdir()) if path.is_file()},
        implementation={str(path.relative_to(ROOT)): identity(path) for path in
            [ROOT / "src/chess_rl/search/real_evaluator.py", ROOT / "tests/test_real_evaluator.py", ROOT / "scripts/check_t04.py"]})
    write("preservation-final.json", preservation())
    write("summary.json", summary)
    print(json.dumps(dict(tests=tests, unchanged_accepted_files=unchanged, status=summary["review_status"]), indent=2))


if __name__ == "__main__": main()
