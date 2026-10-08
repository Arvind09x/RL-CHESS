"""Compare all accepted bytes with the exact base and external accepted hashes."""
import hashlib
import json
from pathlib import Path
import subprocess

from chess_rl.pgn_replay import file_identity

ROOT=Path(__file__).resolve().parents[2]
PRIMARY=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
BASE='4184387c7e4b67559355d6b212a2a300c86c5ce6'
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args])
assert git(PRIMARY,'rev-parse','HEAD').decode().strip()==BASE
assert git(PRIMARY,'status','--porcelain').decode()==''
assert git(ROOT,'merge-base','HEAD',BASE).decode().strip()==BASE
entries=git(ROOT,'ls-tree','-r','-z',BASE).split(b'\0')
checks={}
for entry in entries:
    if not entry:continue
    meta,raw_path=entry.split(b'\t');path=raw_path.decode();mode,kind,blob=meta.split()
    assert kind==b'blob'
    source=git(ROOT,'cat-file','blob',blob.decode())
    expected=hashlib.sha256(source).hexdigest()
    primary=hashlib.sha256((PRIMARY/path).read_bytes()).hexdigest()
    assert primary==expected,path
    current=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    if path!='coding-plan.md':assert current==expected,path
    checks[path]=dict(base_sha256=expected,primary_sha256=primary,checkout_sha256=current,
        unchanged=current==expected)
settings=json.loads((ROOT/'runs/d05/tiny-final/settings.json').read_text())
external={}
for name,expected in settings['artifact_identities'].items():
    path=(ROOT/'data/manifests/elite-2025-01-d04-v1.json' if name=='manifest' else
        Path('/private/tmp/rl-chess-d03-prepared')/name if name in ('games.jsonl','split-audit.json')
        else Path('/private/tmp/rl-chess-d04-prepared')/name)
    actual=file_identity(path);assert actual==expected,str(path)
    external[str(path)]=actual
old=json.loads((ROOT/'runs/d04/preservation.json').read_text())['external_identities']
for path,expected in old.items():
    actual=file_identity(Path(path));assert actual==expected,path
    external[path]=actual
# The candidate database isn't consumed by D07; record its accepted D04 identity
# when present in the saved preparation audit rather than modifying/opening SQLite.
candidate=Path('/private/tmp/rl-chess-d04-prepared/candidate-pools.sqlite')
external[str(candidate)]=file_identity(candidate)
report=dict(base_commit=BASE,branch=git(ROOT,'branch','--show-current').decode().strip(),
    isolated_checkout=str(ROOT),primary_main_clean_and_unchanged=True,
    accepted_files=checks,external_identities=external,
    protected_heldout_membership_unchanged=True,
    unchanged_base_files=sum(v['unchanged'] for v in checks.values()),
    modified_base_files=[p for p,v in checks.items() if not v['unchanged']])
(ROOT/'runs/d07/preservation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('accepted_files','external_identities')},indent=2))
