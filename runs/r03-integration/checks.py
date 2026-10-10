"""Fresh integration receipts and exact preservation gates."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRIMARY=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
SOURCE=ROOT.parent/'r03-mixed-update'
REVIEW=ROOT.parent/'independent-review-r03-20261010'
EXTERNAL=ROOT.parent/'external-artifacts/r03-integration-20261010'
MODEL=str(PRIMARY/'.venv/bin/python')
ADAPTER=str(PRIMARY/'.venv-acceptance/bin/python')
OVERRIDES=dict(PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',GIT_OPTIONAL_LOCKS='0',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
ENV=dict(os.environ,**OVERRIDES)
def git(*args,cwd=ROOT):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(cwd),*args],env=ENV)
def identity(path):
    path=Path(path); h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())
def save(path,value):
    with Path(path).open('x') as f: json.dump(value,f,sort_keys=True,indent=2); f.write('\n')
def code_inventory():
    paths=[p for directory in ('src','tests','scripts') for p in (ROOT/directory).rglob('*.py')]
    paths += [ROOT/'pyproject.toml',ROOT/'uv.lock',ROOT/'runs/r03/frozen-plan.json']
    return {str(p.relative_to(ROOT)):identity(p) for p in paths}
def run(name,argv):
    assert not (OUT/(name+'.command.json')).exists()
    before=code_inventory(); start=datetime.now(timezone.utc).isoformat(); tic=time.monotonic()
    p=subprocess.run(list(map(str,argv)),cwd=ROOT,env=ENV,capture_output=True)
    for kind,data in [('stdout',p.stdout),('stderr',p.stderr)]:
        with (OUT/(name+'.'+kind+'.txt')).open('xb') as f: f.write(data)
    after=code_inventory()
    save(OUT/(name+'.command.json'),dict(argv=list(map(str,argv)),cwd=str(ROOT),environment=OVERRIDES,
        started_at_utc=start,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=time.monotonic()-tic,exit_code=p.returncode,
        source_hashes_before=before,source_hashes_after=after,sources_unchanged=before==after,
        stdout=identity(OUT/(name+'.stdout.txt')),stderr=identity(OUT/(name+'.stderr.txt'))))
    print(name,p.returncode,p.stdout.decode()[-800:],flush=True)
    if p.returncode: print(p.stderr.decode()[-2500:],flush=True)
    assert p.returncode==0 and before==after,name
def external_preserved():
    before=json.loads((OUT/'preservation-before.json').read_text())
    assert git('rev-parse','HEAD',cwd=SOURCE).decode().strip()==before['source_head']
    assert git('status','--porcelain=v1','-uall',cwd=SOURCE).decode()==before['source_status']
    for n,item in before['source_tracked'].items(): assert identity(SOURCE/n)==item,n
    for n,item in before['review'].items(): assert identity(REVIEW/n)==item,n
    for n,item in before['external'].items(): assert identity(n)==item,n
    for n,item in json.loads((ROOT/'runs/r03-acceptance/supplied-evidence-index.json').read_text()).items():
        assert identity(ROOT/'runs/r03-acceptance'/n)==item,n
    return dict(source_tracked=len(before['source_tracked']),review_files=len(before['review']),
        historical_external_files=len(before['external']))
def preserved():
    before=json.loads((OUT/'preservation-before.json').read_text()); p=before['primary']
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()==p['head']
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode()==p['status']
    assert (PRIMARY/'.git/index').read_bytes()==base64.b64decode(p['index_base64'])
    assert git('ls-files','--stage','-z',cwd=PRIMARY)==base64.b64decode(p['index_entries_base64'])
    assert git('diff','--binary',cwd=PRIMARY)==base64.b64decode(p['patch_base64'])
    assert git('diff','--cached','--binary',cwd=PRIMARY)==base64.b64decode(p['staged_patch_base64'])
    assert (PRIMARY/'coding-plan.md').read_bytes()==base64.b64decode(p['plan_base64'])
    for n,item in p['tracked'].items(): assert identity(PRIMARY/n)==item,n
    for n,item in json.loads((OUT/'accepted-tree-inventory.json').read_text()).items():
        assert identity(ROOT/n)==item,n
    return dict(primary_raw_index_exact=True,primary_bytes_and_patch_exact=True,
        accepted_tree_exact=True,**external_preserved())
def main():
    EXTERNAL.mkdir(exist_ok=False)
    save(OUT/'environment.json',dict(model=MODEL,adapter=ADAPTER,external=str(EXTERNAL),
        dependencies_installed=False,overrides=OVERRIDES))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[]
        for name,python,extra in [('model',MODEL,['-p','runs.m06.diagnostic_plugin']),('adapter',ADAPTER,[])]:
            futures.append(pool.submit(run,name,[python,'-B','-m','pytest','--import-mode=importlib',
                '-p','no:cacheprovider','--basetemp='+str(EXTERNAL/(name+'-temp')),
                '--junitxml='+str(OUT/(name+'.xml')),*extra]))
        for f in futures: f.result()
    run('supplied-probes',[MODEL,'-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider',
        '--basetemp='+str(EXTERNAL/'probes-temp'),'--junitxml='+str(OUT/'supplied-probes.xml'),
        ROOT/'runs/r03-acceptance/test_reviewer_probes.py',ROOT/'runs/r03-acceptance/test_failure_boundary.py'])
    for name,python in [('model-replay',MODEL),('adapter-replay',ADAPTER)]:
        run(name,[python,'-B','scripts/check_adapter.py'])
    run('legacy-continuation',[MODEL,'-B','scripts/check_t02.py','--directory',EXTERNAL/'legacy-continuation'])
    for mode in ('full','prefix','resume'):
        run('mixed-'+mode,[MODEL,'-B','scripts/run_r03.py','--mode',mode,'--output',OUT/(mode+'.json'),
            '--checkpoint',EXTERNAL/'mixed-boundary.pt'])
    full,prefix,resume=[json.loads((OUT/(m+'.json')).read_text()) for m in ('full','prefix','resume')]
    historical=json.loads((ROOT/'runs/r03/bounded-full-final.json').read_text())
    comparisons=dict(prefix_exact=prefix['steps']==full['steps'][:1],
        continuation_exact=resume['steps']==full['steps'][1:],final_exact=resume['final']==full['final'],
        historical_steps_exact=full['steps']==historical['steps'],historical_final_exact=full['final']==historical['final'],
        checkpoint_matches_historical=identity(EXTERNAL/'mixed-boundary.pt')==json.loads((ROOT/'runs/r03/summary.json').read_text())['checkpoint']['identity'])
    assert all(comparisons.values()); save(OUT/'comparisons.json',comparisons)
    run('source-and-legacy-audit',[MODEL,'-B',OUT/'audit_sources.py'])
    save(OUT/'preservation-after-verification.json',preserved())
    print('Fresh integration verification and exact preservation passed',flush=True)
if __name__=='__main__': main()
