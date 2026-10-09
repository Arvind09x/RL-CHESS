"""Create a small review index; retain raw command output and external bytes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.check_cuda_support import TOLERANCES
from scripts.stage_cuda_data import identity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"runs/cuda-support"
ARTIFACTS = ROOT.parent/"external-artifacts"


def main():
    commands = {}
    for path in sorted(OUT.glob("*.command.json")):
        record = json.loads(path.read_text())
        assert record["exit_code"] == 0 and record["source_unchanged"]
        for stream, expected in record["output"].items():
            assert identity(OUT/f"{path.name.removesuffix('.command.json')}.{stream}.json") == expected
        commands[path.name] = record
    tests = {}
    for path in sorted(OUT.glob("*.xml")):
        suite = ET.parse(path).getroot().find("testsuite")
        tests[path.name] = {k:int(suite.attrib[k]) for k in ("tests", "failures", "errors", "skipped")}
        assert tests[path.name]["failures"] == tests[path.name]["errors"] == 0
    # Historical stdout containers carry a lossless base64 representation.
    import base64
    historical = json.loads((ROOT/"runs/t03-t04-integration/frozen-adapter-model.stdout.json").read_text())
    baseline = base64.b64decode(historical["data"]).decode()
    assert json.loads((OUT/"frozen-adapter-model.stdout.json").read_text()) == baseline
    assert json.loads((OUT/"frozen-adapter-only.stdout.json").read_text()) == baseline
    artifacts = {}
    for name in ("cuda-support-cpu", "cuda-support-portable-cpu"):
        directory = ARTIFACTS/name
        for file in ("summary.json", "config.json", "uninterrupted.json", "resumed.json"):
            (OUT/f"{name}-{file}").write_bytes((directory/file).read_bytes())
        artifacts[name] = {str(p):identity(p) for p in directory.rglob("*") if p.is_file()}
    staged = ARTIFACTS/"cuda-support-portable"
    for file in ("data-config.json", "staging-evidence.json"):
        (OUT/f"portable-{file}").write_bytes((staged/file).read_bytes())
    artifacts["cuda-support-portable"] = {str(p):identity(p) for p in staged.rglob("*") if p.is_file()}
    source_names = ["src/chess_rl/training/checkpoints.py", "src/chess_rl/search/real_evaluator.py",
        "scripts/check_cuda_support.py", "scripts/stage_cuda_data.py", "tests/test_cuda_support.py",
        "tests/test_cuda_staging.py", "runtime/cuda/requirements.in", "runtime/cuda/requirements.lock",
        "runtime/cuda/setup.sh", "runtime/cuda/build-constraints.txt", "runtime/cuda/local-frozen-data.json",
        "docs/cuda-support.md"]
    summary = dict(branch="codex/cuda-support", base_commit="ea0f8a1",
        prepared_utc=datetime.now(timezone.utc).isoformat(), independent_review="pending",
        scope="Backend preparation only; default CPU preserved; no main integration or R01 edits",
        tests=tests, command_count=len(commands), all_commands_passed=True,
        final_sources={name:identity(ROOT/name) for name in source_names},
        hardware_verified=False, hardware_reason="Local macOS arm64, CPU Torch, CUDA unavailable; no authorized CUDA host supplied",
        outstanding=["Linux CUDA environment installation", "CUDA forward/backward, save/reload, continuation and evaluator commands"],
        tolerances_fixed_before_execution=TOLERANCES, external_artifacts=artifacts,
        actual_t03_cpu_checkpoint=json.loads((OUT/"accepted-cpu-artifact.json").read_text()),
        preservation_report="preservation.json", no_cross_cpu_cuda_exact_agreement_claim=True)
    (OUT/"commands.json").write_text(json.dumps(commands, indent=2, sort_keys=True) + "\n")
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("All indexed commands passed; frozen replay exact; review summary and external artifact hashes saved")


if __name__ == "__main__": main()
