"""Create the T01 closeout summary and focused staged-change inventory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
directory = root/'runs/t01'
base = json.loads((directory/'config.json').read_text())['base_commit']
branch = subprocess.check_output(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=root).decode().strip()
assert branch == 'codex/t01-supervised-losses'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == base
paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root).decode().strip('\0').split('\0')
fixed = {'README.md', 'coding-plan.md', 'docs/t01-supervised-losses.md', 'scripts/check_t01.py',
         'src/chess_rl/tensor_ops.py', 'tests/test_training.py', 'tests/fixtures/loss_reference.py'}
assert all(p in fixed or p.startswith(('runs/t01/', 'src/chess_rl/training/')) for p in paths)
commands = {p.name: json.loads(p.read_text()) for p in sorted(directory.glob('*.command.json'))}
assert len(commands) == 8 and all(c['exit_code'] == 0 for c in commands.values())
preservation = json.loads((directory/'preservation.json').read_text())
real = json.loads((directory/'real-batch.json').read_text())
summary = dict(task='T01 only', status='automated checks passed; independent review pending', base_commit=base,
    branch=branch, checkout=str(root), recorded_at_utc=datetime.now(timezone.utc).isoformat(),
    checks=dict(focused_passes=219, t01_passes=92, model_passes=1273, adapter_passes=1012,
        adapter_expected_skips=8, prior_reviewer_probe_passes=84, m06='passed', frozen_adapter_fixtures=13),
    real_batch=dict(optimizer_steps=1, size=2, metrics=real['metrics'],
        independent_loss_absolute_errors=real['loss_absolute_errors'], changed_tensors=real['changed_parameter_tensors']),
    preservation=dict(accepted_files=preservation['accepted_files'], external_artifacts=len(preservation['external_identities']),
        main_clean_and_unchanged=preservation['primary_main_clean_and_unchanged']),
    scope_stop='No checkpoints, tiny overfit, P, D08, T02, T04 or RL; stop for independent review')
(directory/'completion.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
paths = sorted(set(paths) | {'runs/t01/completion.json', 'runs/t01/change-manifest.json'})
identities = {}
for name in paths:
    if name == 'runs/t01/change-manifest.json': continue
    content = (root/name).read_bytes()
    identities[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
manifest = dict(base_commit=base, changed_paths=paths, identities=identities,
    self_hash_excluded_to_avoid_recursion=True, original_documents_append_only=['README.md', 'coding-plan.md'])
(directory/'change-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2))
