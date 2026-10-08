"""Combined review preservation and authoritative output identity checks."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = '37f286618642150b0ffa904abee0cfaac3ad75e8'


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def identity(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(bytes=path.stat().st_size, sha256=digest)


checks = {}
for name in git('ls-tree','-r','--name-only',BASE).decode().splitlines():
    original = git('show',f'{BASE}:{name}')
    current = (ROOT/name).read_bytes()
    if name not in {'README.md','coding-plan.md'}:
        assert original == current, name
    checks[name] = dict(base_sha256=hashlib.sha256(original).hexdigest(),
                        current_sha256=hashlib.sha256(current).hexdigest(), unchanged=original == current)
for commit, checkout in [
    ('1042f1aa7f37298f2ef3a1b44eb2dceea4ed3f52','/private/tmp/rl-chess-d04'),
    ('4b7eee35f98f655296ced1263e10e5bd82541e0f','/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/s04-root-policy')]:
    assert git('rev-parse','HEAD',cwd=checkout).decode().strip() == commit
    assert not git('status','--porcelain',cwd=checkout).strip()
    for name in git('diff','--name-only',BASE,commit).decode().splitlines():
        original = git('show',f'{commit}:{name}')
        current = (ROOT/name).read_bytes()
        if name in {'README.md','docs/d04-position-sampling.md','docs/s04-root-policy.md'}:
            assert current.startswith(original),name
        elif name != 'coding-plan.md':
            assert current == original,name
before = json.loads((ROOT/'data/manifests/elite-2025-01-v1.json').read_text())
after = json.loads((ROOT/'data/manifests/elite-2025-01-d04-v1.json').read_text())
for field in ('sampling_algorithm','phase_boundaries','overlap_audit_sha256','protected_heldout_registry_sha256'):
    before['preparation'][field] = after['preparation'][field]
assert before == after,'Manifest changes extend beyond four D04 decisions'
bulk = Path('/private/tmp/rl-chess-d04-prepared')
audit_identity = identity(bulk/'overlap-audit.json')
assert audit_identity['sha256'] == after['preparation']['overlap_audit_sha256']
audit = json.loads((bulk/'overlap-audit.json').read_text())
outputs = {}
for name, expected in audit['outputs'].items():
    actual = identity(bulk/name)
    assert actual == expected
    assert actual == identity(Path('/private/tmp/rl-chess-d04-reselected')/name)
    outputs[name] = actual
assert outputs['protected-heldout.jsonl']['sha256'] == after['preparation']['protected_heldout_registry_sha256']
assert (OUT/'replay-adapter.stdout.txt').read_bytes() == (OUT/'replay-model.stdout.txt').read_bytes()
report = dict(base_commit=BASE, base_files=len(checks),
              unchanged_base_files=sum(item['unchanged'] for item in checks.values()),
              original_task_code_and_evidence_preserved=True, original_checkouts_clean=True,
              new_manifest_changes_only_four_D04_fields=True, accepted_original_manifest_unchanged=True,
              repeated_selection_outputs_identical=True, both_adapter_replays_identical=True,
              overlap_audit_identity=audit_identity, outputs=outputs,files=checks)
(OUT/'preservation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
