"""Guarded preservation checks; no production source changes."""
import base64,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRIMARY=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
SOURCE=ROOT.parent/'r04-local-cycle'
REVIEW=ROOT.parent/'r04-independent-review-20261010-9LCZHH'
ART=ROOT.parent/'external-artifacts/r04-integration-20261010'
ENV=os.environ|dict(GIT_OPTIONAL_LOCKS='0',PYTHONDONTWRITEBYTECODE='1')
def git(*args,cwd=ROOT):
    return subprocess.check_output(['git','--no-optional-locks',*args],cwd=cwd,env=ENV)
def identity(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def save(path,v):
    with Path(path).open('x') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
def before(): return json.loads((OUT/'preservation-before.json').read_text())
def historical():
    b=before()
    assert git('rev-parse','HEAD',cwd=SOURCE).decode().strip()==b['source_head']
    assert not git('status','--porcelain=v1','-uall',cwd=SOURCE)
    for n,v in b['source_tracked'].items():assert identity(SOURCE/n)==v,n
    for n,v in b['supplied_review'].items():assert identity(REVIEW/n)==v,n
    for n,v in b['external'].items():assert identity(n)==v,n
    return dict(source_files=len(b['source_tracked']),review_files=len(b['supplied_review']),historical_external_files=len(b['external']),all_exact=True)
def primary_unchanged():
    p=before()['primary']
    assert git('branch','--show-current',cwd=PRIMARY)==b'main\n'
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()==p['head']
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode()==p['status']
    assert (PRIMARY/'.git/index').read_bytes()==base64.b64decode(p['index_base64'])
    assert git('ls-files','--stage','-z',cwd=PRIMARY)==base64.b64decode(p['index_entries_base64'])
    assert git('diff','--binary',cwd=PRIMARY)==base64.b64decode(p['patch_base64'])
    assert git('diff','--cached','--binary',cwd=PRIMARY)==base64.b64decode(p['staged_patch_base64'])==b''
    assert (PRIMARY/'coding-plan.md').read_bytes()==base64.b64decode(p['plan_base64'])
    for n,v in p['tracked'].items():assert identity(PRIMARY/n)==v,n
    assert git('stash','list','--format=%H',cwd=PRIMARY).decode().splitlines()==p['stash_list']
    return dict(primary_tree_exact=True,plan_file_and_patch_exact=True,raw_index_exact=True,staged_entries_exact=True)
def accepted_tree():
    for n,v in json.loads((OUT/'accepted-tree-inventory.json').read_text()).items():
        if n not in ('README.md','coding-plan.md'):assert identity(ROOT/n)==v,n
def normalized(patch):
    return b'\n'.join(l for l in patch.split(b'\n') if not l.startswith(b'index '))
def expected_plan():
    data=(ROOT/'coding-plan.md').read_bytes()
    for task in ('D06','D07','E01'):
        old=f'| [ ] | {task} /'.encode()
        assert data.count(old)==1
        data=data.replace(old,f'| [x] | {task} /'.encode(),1)
    original=base64.b64decode(before()['primary']['plan_base64'])
    assert data[:len(original)].replace(b'| [x] | R04 /',b'| [ ] | R04 /',1)==original
    return data
def restored(target):
    expected=expected_plan()
    for n in git('ls-tree','-r','--name-only',target).decode().splitlines():
        if n=='coding-plan.md':assert (PRIMARY/n).read_bytes()==expected
        else:assert identity(PRIMARY/n)==identity(ROOT/n),n
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY)==b' M coding-plan.md\n'
    assert not git('diff','--cached','--binary',cwd=PRIMARY)
    assert git('write-tree',cwd=PRIMARY)==git('rev-parse','HEAD^{tree}',cwd=PRIMARY)
    assert normalized(git('diff','--binary',cwd=PRIMARY))==normalized(base64.b64decode(before()['primary']['patch_base64']))
    assert b'| [x] | R04 /' in git('show','HEAD:coding-plan.md',cwd=PRIMARY)
    assert b'**R04 and Gate I accepted**' in git('show','HEAD:coding-plan.md',cwd=PRIMARY)

