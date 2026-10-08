"""Validate D01 metadata and optionally verify a teacher via idle UCI only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from chess_rl.data_manifest import DatasetManifest

ROOT = Path(__file__).resolve().parents[1]


def verify_teacher(manifest, executable):
    binary = executable.read_bytes()
    digest = hashlib.sha256(binary).hexdigest()
    if digest != manifest.engine.executable_sha256:
        raise ValueError("teacher executable checksum differs from manifest")
    teacher = manifest.teacher
    commands = ["uci", f"setoption name Threads value {teacher.threads}",
                f"setoption name Hash value {teacher.hash_mb}",
                f"setoption name MultiPV value {teacher.multipv}",
                "setoption name UCI_ShowWDL value true", "setoption name UCI_Chess960 value false",
                "setoption name UCI_LimitStrength value false", "setoption name Ponder value false",
                "setoption name SyzygyPath value <empty>", "setoption name Clear Hash",
                "isready", "compiler", "quit"]
    # Avoid ambient cwd assets; the complete embedded NNUE bytes were audited
    # independently at build. Idle handshakes do not evaluate/load-check a root.
    with tempfile.TemporaryDirectory(prefix="d01-uci-") as directory:
        result = subprocess.run([str(executable.resolve())], input="\n".join(commands) + "\n",
                                text=True, capture_output=True, timeout=30, cwd=directory)
    if result.returncode or result.stderr:
        raise ValueError(f"teacher handshake failed: exit {result.returncode}: {result.stderr}")
    lines = result.stdout.splitlines()
    if "uciok" not in lines or "readyok" not in lines:
        raise ValueError("teacher did not finish UCI/readiness handshake")
    for name, value in (("name", manifest.engine.uci_name), ("author", manifest.engine.uci_author)):
        if f"id {name} {value}" not in lines:
            raise ValueError(f"teacher UCI {name} mismatch")
    options = {}
    for line in lines:
        match = re.fullmatch(r"option name (.+?) type (.+)", line)
        if match:
            options[match[1]] = match[2]
    for name, requested in (("Threads", teacher.threads), ("Hash", teacher.hash_mb),
                            ("MultiPV", teacher.multipv)):
        match = re.fullmatch(r"spin default \d+ min (\d+) max (\d+)", options.get(name, ""))
        if not match or not int(match[1]) <= requested <= int(match[2]):
            raise ValueError(f"required UCI spin capability unavailable: {name}")
    if options.get("UCI_ShowWDL") not in ("check default false", "check default true"):
        raise ValueError("required UCI_ShowWDL capability unavailable")
    if options.get("Clear Hash") != "button":
        raise ValueError("required Clear Hash capability unavailable")
    if any("Unknown command" in line or "No such option" in line for line in lines):
        raise ValueError("teacher rejected configuration commands")
    # Build evidence hashes the full blobs and verifies their bytes in the binary.
    # Here the options must still identify those pinned embedded defaults.
    asset_checks = []
    for option, asset in zip(("EvalFile", "EvalFileSmall"), manifest.engine.embedded_nnue):
        if options.get(option) != f"string default {asset.name}":
            raise ValueError(f"teacher NNUE option mismatch: {option}")
        asset_checks.append({"name": asset.name, "sha256": asset.sha256, "embedded": asset.embedded})
    return {"executable": str(executable.resolve()), "executable_sha256": digest,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "commands_sent": commands, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr, "options": options,
            "assets": asset_checks, "search_commands_sent": False,
            "limitation": "idle UCI handshake; NNUE byte embedding audited at build; no search, WDL output or performance verification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/elite-2025-01-v1.json")
    parser.add_argument("--teacher-executable", type=Path)
    parser.add_argument("--require-annotation-ready", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = DatasetManifest.load(args.manifest)
    if DatasetManifest.from_json(manifest.to_json()) != manifest:
        raise ValueError("manifest round trip failed")
    if args.require_annotation_ready:
        manifest.require_annotation_ready()
    report = {"dataset_id": manifest.dataset_id, "schema_version": manifest.schema_version,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "preparation_ready": True, "annotation_ready": not manifest.annotation_blockers(),
              "annotation_blockers": list(manifest.annotation_blockers()),
              "archive_url": manifest.archive.url, "teacher": None}
    if args.teacher_executable:
        report["teacher"] = verify_teacher(manifest, args.teacher_executable)
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
