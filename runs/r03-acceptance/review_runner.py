"""Independent R03 review receipts; writes only to this separate directory."""
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/Users/arvindsrinivass/Documents/ChatGPT/RL Project 2/r03-mixed-update')
OUT = Path(__file__).resolve().parent
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')
MODEL = str(PRIMARY/'.venv/bin/python')
ADAPTER = str(PRIMARY/'.venv-acceptance/bin/python')
ENV = dict(PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1', GIT_OPTIONAL_LOCKS='0',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')

def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root).decode()

def snapshot():
    result = {}
    for name, root in [('reviewed',ROOT),('primary',PRIMARY)]:
        result[name] = dict(head=git(root,'rev-parse','HEAD'), status=git(root,'status','--porcelain=v1'),
            diff=git(root,'diff','--binary'), staged=git(root,'diff','--cached','--binary'),
            tracked={p:identity(root/p) for p in git(root,'ls-files').splitlines() if (root/p).is_file()})
    plan = json.loads((ROOT/'runs/r03/frozen-plan.json').read_text())
    result['frozen_inputs'] = {p:identity(p) for p in plan['inputs']}
    assert result['frozen_inputs'] == plan['inputs']
    return result

def run(name, argv):
    target = OUT/(name+'.command.json')
    assert not target.exists(), target
    start = datetime.now(timezone.utc).isoformat(); tic = time.monotonic()
    p = subprocess.run(argv,cwd=ROOT,env=dict(os.environ,**ENV),capture_output=True)
    (OUT/(name+'.stdout.txt')).write_bytes(p.stdout)
    (OUT/(name+'.stderr.txt')).write_bytes(p.stderr)
    target.write_text(json.dumps(dict(argv=argv,cwd=str(ROOT),environment=ENV,
        started_at_utc=start,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=time.monotonic()-tic,exit_code=p.returncode,
        stdout=identity(OUT/(name+'.stdout.txt')),stderr=identity(OUT/(name+'.stderr.txt'))),indent=2)+'\n')
    print(name,p.returncode,flush=True)
    assert p.returncode == 0, name

if __name__ == '__main__':
    before = snapshot()
    (OUT/'preservation-before.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
    suites = [('model',MODEL,['-p','runs.m06.diagnostic_plugin']),('adapter',ADAPTER,[])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run,name,[python,'-B','-m','pytest','--import-mode=importlib',
            '-p','no:cacheprovider','--basetemp='+str(OUT/(name+'-temp')),
            '--junitxml='+str(OUT/(name+'.xml')),*extra]) for name,python,extra in suites]
        for f in futures: f.result()
    checkpoint = OUT/'boundary.pt'
    for mode in ('full','prefix','resume'):
        run(mode,[MODEL,'-B','scripts/run_r03.py','--mode',mode,'--output',str(OUT/(mode+'.json')),
            '--checkpoint',str(checkpoint)])
    full,prefix,resume = [json.loads((OUT/(m+'.json')).read_text()) for m in ('full','prefix','resume')]
    comparisons = dict(prefix_exact=prefix['steps']==full['steps'][:1],
        continuation_exact=resume['steps']==full['steps'][1:],final_exact=resume['final']==full['final'])
    historical = json.loads((ROOT/'runs/r03/bounded-full-final.json').read_text())
    comparisons['historical_steps_exact'] = full['steps']==historical['steps']
    comparisons['historical_final_exact'] = full['final']==historical['final']
    assert all(comparisons.values()), comparisons
    (OUT/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    after = snapshot()
    (OUT/'preservation-after.json').write_text(json.dumps(after,sort_keys=True,indent=2)+'\n')
    assert before == after
    print('exact continuation and preservation verified',flush=True)
