"""Attest a Stockfish build for use as the D09 teacher on another machine.

Writes a teacher-evidence JSON in the schema the loader and annotation worker
accept. The manifest's binary stays the reference build; this attestation lets
an identical-source build (same release tag, source commit, source archive and
embedded networks) label positions, with label equivalence demonstrated
separately on overlapping references. No searches are run here.

Usage:
  attest_stockfish.py --manifest M --executable E --source-archive sf_17.1.tar.gz
                      --net-dir Stockfish-sf_17.1/src --output evidence.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from chess_rl.data_manifest import DatasetManifest
from chess_rl.pgn_replay import file_identity


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(executable):
    """uci/isready/quit transcript; the engine must identify as the manifest's release."""
    commands = "uci\ncompiler\nisready\nquit\n"
    proc = subprocess.run([str(executable)], input=commands, capture_output=True, text=True, timeout=60)
    return dict(argv=[str(executable)], commands_sent=commands.split(), exit_code=proc.returncode,
                stdout=proc.stdout, stderr=proc.stderr)


def locate_networks(binary, manifest, net_dir):
    """Find each embedded network's byte range by searching the binary for the net file bytes."""
    records = []
    for asset in manifest.engine.embedded_nnue:
        net = net_dir / asset.name
        if not net.exists():
            raise SystemExit(f"missing network file {net}; the Stockfish build downloads it into src/")
        blob = net.read_bytes()
        if hashlib.sha256(blob).hexdigest() != asset.sha256:
            raise SystemExit(f"network file {asset.name} does not match the manifest hash")
        offset = binary.find(blob)
        if offset < 0 or binary.find(blob, offset + 1) >= 0:
            raise SystemExit(f"embedded network {asset.name} not found exactly once in the executable")
        records.append(dict(name=asset.name, sha256=asset.sha256, bytes=len(blob), embedded_offset=offset,
                            source_url=asset.source_url))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True, help="the sf_17.1 tarball the build came from")
    parser.add_argument("--net-dir", type=Path, required=True, help="directory holding the downloaded .nnue files")
    parser.add_argument("--build-command", default="", help="the exact make invocation used")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = DatasetManifest.load(args.manifest)
    engine = manifest.engine
    archive_sha = sha256(args.source_archive)
    if archive_sha != engine.package_sha256:
        raise SystemExit(f"source archive hash {archive_sha} differs from the manifest's {engine.package_sha256}")
    binary = args.executable.read_bytes()
    identity = file_identity(args.executable)
    transcript = preflight(args.executable)
    if transcript["exit_code"] != 0 or f"id name {engine.uci_name}" not in transcript["stdout"]:
        raise SystemExit("engine preflight failed or UCI name differs from the manifest")
    compiler = next((line for line in transcript["stdout"].splitlines() if line.startswith("Compiled by")), "")
    evidence = dict(
        schema="teacher-build-attestation-v1",
        executable=str(args.executable.resolve()), executable_bytes=identity["bytes"],
        executable_sha256=identity["sha256"],
        reference_build=identity["sha256"] == engine.executable_sha256,
        alternate_build=dict(release_tag=engine.release_tag, source_commit=engine.source_commit,
            source_archive_sha256=archive_sha, platform=f"{platform.system()} {platform.release()} {platform.machine()}",
            build=args.build_command),
        source_archive_url=engine.download_url, source_archive_sha256=archive_sha,
        source_archive_bytes=args.source_archive.stat().st_size,
        nnue=locate_networks(binary, manifest, args.net_dir),
        compiler=compiler, preflight=transcript, searches_run=False, syzygy="disabled",
        separately_required_runtime_assets=[], python=sys.version, attested_on=platform.platform())
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(executable_sha256=identity["sha256"], reference_build=evidence["reference_build"],
                          networks=[n["name"] for n in evidence["nnue"]], output=str(args.output))))


if __name__ == "__main__":
    main()
