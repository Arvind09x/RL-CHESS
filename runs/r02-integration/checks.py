"""Fresh bounded integration checks; refuse overwriting evidence."""
import base64
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
SOURCE=ROOT.parent/'r02-replay'
ENV=dict(os.environ, PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
def git(*args,cwd=ROOT):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(cwd),*args],env=ENV)
def identity(path):
    data=Path(path).read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def save(path,value):
    with Path(path).open('x') as stream: json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')
def run(name,argv,extra=None):
    env=dict(ENV,**(extra or {})); start=time.monotonic()
    result=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env,capture_output=True)
    for kind,data in [('stdout',result.stdout),('stderr',result.stderr)]:
        with (OUT/(name+'.'+kind+'.txt')).open('xb') as stream: stream.write(data)
    save(OUT/(name+'.command.json'),dict(argv=list(map(str,argv)),cwd=str(ROOT),
        environment={k:env[k] for k in ENV if k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','GIT_OPTIONAL_LOCKS',
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS') or k=='REVIEW_OUTPUT'},
        extra_environment=extra or {},duration_seconds=time.monotonic()-start,exit_code=result.returncode))
    print(result.stdout.decode()[-4500:],flush=True)
    if result.returncode: print(result.stderr.decode()[-3000:],flush=True); raise SystemExit(result.returncode)
def preserved():
    before=json.loads((OUT/'preservation-before.json').read_text())
    p=before['primary']
    assert git('rev-parse','HEAD',cwd=PRIMARY).decode().strip()==p['head']
    assert git('status','--porcelain=v1','-uall',cwd=PRIMARY).decode()==p['status']
    assert (PRIMARY/'.git/index').read_bytes()==base64.b64decode(p['index_base64'])
    assert git('diff','--binary',cwd=PRIMARY)==base64.b64decode(p['patch_base64'])
    assert git('diff','--cached','--binary',cwd=PRIMARY)==base64.b64decode(p['staged_patch_base64'])
    assert (PRIMARY/'coding-plan.md').read_bytes()==base64.b64decode(p['plan_base64'])
    for n,item in p['tracked'].items(): assert identity(PRIMARY/n)==item,n
    assert git('rev-parse','HEAD',cwd=SOURCE).decode().strip()==before['source_head']
    assert git('status','--porcelain=v1','-uall',cwd=SOURCE).decode()==before['source_status']
    for n,item in before['source_tracked'].items(): assert identity(SOURCE/n)==item,n
    for n,item in before['review'].items():
        assert identity(SOURCE/'runs/r02-review-20261010'/n)==item,n
        assert identity(OUT/'supplied-review'/n)==item,n
    for n,item in before['r01_tracked'].items(): assert identity(ROOT.parent/'r01-self-play'/n)==item,n
    for n,item in before['external'].items(): assert identity(n)==item,n
    for n,item in json.loads((OUT/'accepted-tree-inventory.json').read_text()).items(): assert identity(ROOT/n)==item,n
    return dict(primary_original_bytes_patch_and_index_exact=True,accepted_candidate_tree_exact=True,
        original_r01_tree_exact=True,original_r02_tree_exact=True,review_files_exact=len(before['review']),
        external_artifacts_exact=len(before['external']))
if __name__=='__main__':
    name=sys.argv[1]
    if name=='preservation':
        result=preserved(); save(OUT/'preservation-after-verification.json',result); print(json.dumps(result,indent=2))
    else:
        python=PRIMARY/('.venv-acceptance/bin/python' if name=='adapter' else '.venv/bin/python')
        argv=[python,'-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider',
            '--basetemp='+str(OUT/'tmp'/name),'--junitxml='+str(OUT/(name+'.xml'))]
        extra=None
        if name=='model': argv+=['-p','runs.m06.diagnostic_plugin']
        elif name=='supplied-probes':
            output=OUT/'supplied-probes-artifacts'; output.mkdir()
            argv+=[OUT/'test_supplied_review.py']; extra=dict(REVIEW_OUTPUT=str(output))
        else: assert name=='adapter'
        run(name,argv,extra)
