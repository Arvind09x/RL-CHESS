"""D04 exact-base preservation and four-field manifest scope audit."""
import hashlib
import json
from pathlib import Path
import subprocess
from chess_rl.pgn_replay import file_identity

ROOT = Path(__file__).resolve().parents[2]
BASE = '37f286618642150b0ffa904abee0cfaac3ad75e8'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


allowed = {'README.md','coding-plan.md'}
base_paths = git('ls-tree','-r','--name-only',BASE).decode().splitlines()
checks = {}
for name in base_paths:
    old = git('show',f'{BASE}:{name}')
    new = (ROOT/name).read_bytes()
    if name not in allowed:
        assert old==new,name
    checks[name] = dict(base_sha256=hashlib.sha256(old).hexdigest(),current_sha256=hashlib.sha256(new).hexdigest(),byte_identical=old==new)
assert git('merge-base',BASE,'HEAD').decode().strip()==BASE
assert git('branch','--show-current').decode().strip()=='codex/d04-position-sampling'
original = json.loads((ROOT/'data/manifests/elite-2025-01-v1.json').read_text())
prepared = json.loads((ROOT/'data/manifests/elite-2025-01-d04-v1.json').read_text())
old_p = original.pop('preparation')
new_p = prepared.pop('preparation')
assert original==prepared
resolved = {'sampling_algorithm','phase_boundaries','overlap_audit_sha256','protected_heldout_registry_sha256'}
assert {k:v for k,v in old_p.items() if k not in resolved}=={k:v for k,v in new_p.items() if k not in resolved}
assert all(old_p[k] is None and new_p[k] is not None for k in resolved)
assert (ROOT/'runs/d04/replay-model.stdout.txt').read_bytes()==(ROOT/'runs/d04/replay-adapter.stdout.txt').read_bytes()
assert (ROOT/'runs/d04/replay-adapter.stdout.txt').read_bytes()==(ROOT/'runs/d03-s03-review/replay-adapter.stdout.txt').read_bytes()
external = {}
for path, identity in {
 '/private/tmp/rl-chess-d02-accepted-prepared/replays.jsonl':dict(bytes=541174538,sha256='69c10231f4d569a6b3cb4b26bc22a9535346d68f0a4753bd83c0c92981ea950b'),
 '/private/tmp/rl-chess-d03-prepared/games.jsonl':dict(bytes=638770177,sha256='f9f69d255b652f45276ee6163857b3a9c8054ac5ff037f36c94e9ecb039f9df5'),
 '/private/tmp/rl-chess-d03-prepared/split-audit.json':file_identity(ROOT/'runs/d03/split-audit.json'),
 '/private/tmp/lichess_elite_2025-01.zip':dict(bytes=82724777,sha256=original['archive']['downloaded_sha256']),
}.items():
    actual = file_identity(path)
    assert actual==identity,path
    external[path]=actual
teacher = file_identity('/private/tmp/Stockfish-sf_17.1/src/stockfish')
assert teacher['sha256']==original['engine']['executable_sha256']
external['/private/tmp/Stockfish-sf_17.1/src/stockfish']=teacher
new_paths = (set(git('diff','--name-only',BASE).decode().splitlines())-set(base_paths)) | set(git('ls-files','--others','--exclude-standard').decode().splitlines())
new_allowed = {'src/chess_rl/position_sampling.py','src/chess_rl/position_expansion.py',
 'scripts/prepare_d04.py','scripts/expand_d04.py','tests/test_position_sampling.py',
 'docs/d04-position-sampling.md','data/manifests/elite-2025-01-d04-v1.json'}
assert all(p in new_allowed or p.startswith('runs/d04/') for p in new_paths),new_paths
assert not {p for p in new_paths if p.endswith('.sqlite') or p.endswith('samples.jsonl')}
report = dict(base_commit=BASE,branch='codex/d04-position-sampling',checkout=str(ROOT),
              accepted_files=checks,external_identities=external,
              byte_identical_base_files=sum(c['byte_identical'] for c in checks.values()),
              original_manifest_unchanged=True,new_manifest_only_four_d04_fields=True,
              accepted_code_tests_dependency_pins_teacher_and_historical_evidence_preserved=True,
              adapter_reports_match_accepted=True)
(ROOT/'runs/d04/preservation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='accepted_files'},indent=2))
