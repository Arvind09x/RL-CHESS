"""Verify final R01 evidence identities and write its review index once."""
import base64
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from verify import OUT, ROOT, git, identity, save


def main():
    commands = {}
    for path in sorted(OUT.glob("*.command.json")):
        command = json.loads(path.read_text())
        assert command["exit_code"] == 0
        for name, expected in command["tested_sources"].items():
            assert identity(ROOT / name) == expected, name
        for label in ("stdout", "stderr"):
            output = json.loads(path.with_name(path.name.replace(".command.json", f".{label}.json")).read_text())
            raw = base64.b64decode(output["data"])
            assert len(raw) == output["bytes"] and hashlib.sha256(raw).hexdigest() == output["sha256"]
        commands[path.stem.removesuffix(".command")] = dict(argv=command["argv"],
            exit_code=command["exit_code"], duration_seconds=command["duration_seconds"])
    suites = {}
    for path in sorted(OUT.glob("*.xml")):
        suite = ET.parse(path).getroot().find("testsuite")
        counts = {k: int(suite.attrib[k]) for k in ("tests", "errors", "failures", "skipped")}
        assert counts["errors"] == counts["failures"] == 0
        counts["passed"] = counts["tests"] - counts["skipped"]
        suites[path.name] = counts
    assert suites["regression-model.xml"]["passed"] == 1453
    assert suites["regression-adapter.xml"]["passed"] == 1023
    assert suites["focused.xml"]["passed"] == 308
    for kind in ("model", "adapter"):
        fresh = json.loads((OUT / f"adapter-replay-{kind}.stdout.json").read_text())
        prior = json.loads((ROOT / f"runs/t03-t04-integration/frozen-adapter-{kind}.stdout.json").read_text())
        assert base64.b64decode(fresh["data"]) == base64.b64decode(prior["data"])
    before = json.loads((OUT / "preservation-before.json").read_text())
    after = json.loads((OUT / "preservation-after.json").read_text())
    assert before == after
    base = json.loads((OUT / "setup.json").read_text())["base_commit"]
    preserved = 0
    for name in git("ls-tree", "-r", "--name-only", base).decode().splitlines():
        accepted = git("show", base + ":" + name)
        current = (ROOT / name).read_bytes()
        if name in ("README.md", "coding-plan.md"):
            assert current.startswith(accepted)
        else:
            assert accepted == current, name
        preserved += 1
    real = json.loads((OUT / "real/summary.json").read_text())
    plan = json.loads((OUT / "real/frozen-plan.json").read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
               for name, expected in plan["source_sha256"].items())
    result = dict(scope="Only R01; independent review pending; commit and stop without integration",
        base_commit=base, branch=git("branch", "--show-current").decode().strip(),
        checkout=str(ROOT), shared_dependency_interfaces_unchanged=True,
        no_R02_R03_R04_or_P_selection=True, commands=commands, suites=suites,
        real_checkpoint=plan["checkpoint"], real_fixture_results=real,
        preservation=dict(primary_exact_head_status_patch_and_tracked_bytes=True,
            base_files_verified=preserved, base_documentation_exact_prefix_preserved=True,
            unchanged_indexed_external_files=len(after["external_files"]),
            checkpoint_unchanged=True),
        limitations="Pinned macOS arm64 sequential deterministic CPU; no strength, scale, GPU or power-loss claim")
    save(OUT / "summary.json", result)
    print(json.dumps(dict(suites=suites, real_summary=real["summary"], preserved_base_files=preserved), indent=2))


if __name__ == "__main__":
    main()
