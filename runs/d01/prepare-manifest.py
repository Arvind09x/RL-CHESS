"""D01 evidence construction from already retrieved bytes; no game data or search."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from chess_rl.data_manifest import (ArchiveProvenance, DatasetManifest, EngineAsset,
                                   EngineProvenance, PreparationEvidence, TeacherConfig)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/d01'
SOURCE = Path('/private/tmp/Stockfish-sf_17.1')
EXE = SOURCE / 'src/stockfish'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')

index = Path('/private/tmp/rl-chess-d01-index.html')
head = Path('/private/tmp/rl-chess-d01-archive-head.txt')
url = 'https://database.nikonoel.fr/lichess_elite_2025-01.zip'
assert re.search(r'<a[^>]+href="' + re.escape(url) + r'"[^>]*>January 2025', index.read_text())
assert 'HTTP/2 200' in head.read_text()
verified = datetime.fromtimestamp(head.stat().st_mtime, timezone.utc).isoformat()
save('archive-provenance.json', {
    'index_url': 'https://database.nikonoel.fr/', 'index_sha256': digest(index),
    'selected_link': {'text': 'January 2025', 'href': url},
    'verification': 'index link plus HEAD only; zero archive body bytes retrieved',
    'verified_at_utc': verified, 'head_response': head.read_text(),
    'downloaded_archive_checksum': None,
    'limitation': 'HEAD content length/ETag are transport observations, not checksum/coverage measurements',
})
release = json.loads(Path('/private/tmp/rl-chess-d01-release.json').read_text())
tag = json.loads(Path('/private/tmp/rl-chess-d01-stockfish-tag.json').read_text())
assert tag['object']['sha'] == '03e27488f3d21d8ff4dbf3065603afa21dbd0ef3'
source_archive = Path('/private/tmp/rl-chess-stockfish-17.1-source.tar.gz')
source_url = 'https://codeload.github.com/official-stockfish/Stockfish/tar.gz/refs/tags/sf_17.1'
commands = ['uci', 'compiler', 'isready', 'quit']
r = subprocess.run([str(EXE)], input='\n'.join(commands)+'\n', text=True,
                   capture_output=True, timeout=30, cwd='/private/tmp')
assert r.returncode == 0 and not r.stderr
assert 'id name Stockfish 17.1\n' in r.stdout and '\nreadyok\n' in r.stdout
observed = datetime.now(timezone.utc).isoformat()
binary = EXE.read_bytes()
assets = []
asset_evidence = []
for name in ('nn-1c0000000000.nnue', 'nn-37f18f62d772.nnue'):
    p = SOURCE / 'src' / name
    blob = p.read_bytes()
    sha = digest(p)
    assert name == f'nn-{sha[:12]}.nnue'
    offset = binary.find(blob)
    assert offset >= 0, f'{name} not embedded in executable'
    assets.append(EngineAsset(name, f'https://tests.stockfishchess.org/api/nn/{name}', sha, True))
    asset_evidence.append({'name': name, 'sha256': sha, 'bytes': len(blob),
                           'source_url': assets[-1].source_url, 'embedded_offset': offset})
save('stockfish-provenance.json', {
    'release_url': release['html_url'], 'tag': tag, 'source_archive_url': source_url,
    'source_archive_sha256': digest(source_archive), 'source_archive_bytes': source_archive.stat().st_size,
    'build_command_record': 'runs/d01/stockfish-build-command.json',
    'binary_asset_attempted': next({k: a.get(k) for k in ('id', 'name', 'size', 'browser_download_url')}
                                   for a in release['assets'] if a['id'] == 242039612),
    'reason_for_source_build': 'official binary release asset host reset connections (curl exit 35)',
    'executable': str(EXE), 'executable_sha256': digest(EXE), 'executable_bytes': len(binary),
    'nnue': asset_evidence, 'separately_required_runtime_assets': [], 'syzygy': 'disabled',
    'preflight': {'argv': [str(EXE)], 'commands_sent': commands, 'exit_code': r.returncode,
                  'stdout': r.stdout, 'stderr': r.stderr, 'verified_at_utc': observed},
    'file': subprocess.check_output(['file', str(EXE)], text=True),
    'dynamic_libraries': subprocess.check_output(['otool', '-L', str(EXE)], text=True),
    'compiler': subprocess.check_output(['clang++', '--version'], text=True),
    'searches_run': False,
})
manifest = DatasetManifest(
    dataset_id='lichess-elite-2025-01-pilot-v1',
    archive=ArchiveProvenance('2025-01', url, 'https://database.nikonoel.fr/', verified,
                             'runs/d01/archive-provenance.json', None, None, None),
    engine=EngineProvenance('17.1', 'sf_17.1', tag['object']['sha'], 'macOS 14.5 arm64',
                           'local official release source build; make -j2 build ARCH=apple-silicon COMP=clang; Apple clang 15.0.0; O3/LTO/NEON_DOTPROD; no PGO',
                           source_url, digest(source_archive), 'Stockfish-sf_17.1/src/stockfish',
                           digest(EXE), 'Stockfish 17.1', 'the Stockfish developers (see AUTHORS file)',
                           observed, 'runs/d01/stockfish-provenance.json',
                           'embedded NNUE; no separate runtime assets; Syzygy disabled', tuple(assets)),
    teacher=TeacherConfig(), preparation=PreparationEvidence(None, None, None, None, None, None),
    split_seed=20261008, sampling_seed=20261009,
)
path = ROOT / 'data/manifests/elite-2025-01-v1.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(manifest.to_json())
print('Prepared', path, 'with', len(manifest.annotation_blockers()), 'annotation blockers')
