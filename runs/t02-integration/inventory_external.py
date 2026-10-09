"""Hash every retained workspace artifact, excluding only this run's new directory."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from run_checks import ROOT, OUT, ARTIFACTS, identity, save

def inventory():
    return {str(p): identity(p) for p in sorted((ROOT.parent/'external-artifacts').rglob('*'))
            if p.is_file() and ARTIFACTS not in p.parents}

current = inventory()
if sys.argv[1] == 'before':
    previous = json.loads((ROOT/'runs/d08-t01-integration/summary.json').read_text())
    for key in ('fresh_loader_report', 'fresh_replay_report'):
        expected = previous['d08'][key]
        assert current[expected['path']] == {k:expected[k] for k in ('bytes', 'sha256')}
    save(OUT/'workspace-external-before.json', dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), files=current))
else:
    original = json.loads((OUT/'workspace-external-before.json').read_text())['files']
    assert current == original
    save(OUT/'workspace-external-after.json', dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        files=current, exact_same_files_and_bytes=True))
print(f'{len(current)} historical workspace artifact files verified')
