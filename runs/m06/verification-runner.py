"""Exact M06 offline verification commands; never overwrites prior evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/private/tmp/rl-chess-m06-change')
SOURCE=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
MODEL=str(SOURCE/'.venv/bin/python')
ADAPTER=str(SOURCE/'.venv-acceptance/bin/python')
OUT=ROOT/'runs/m06'
commands=[]

def run(argv,name,overrides=None):
    env=os.environ.copy()
    overrides={'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),**(overrides or {})}
    env.update(overrides)
    started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
    result=subprocess.run(argv,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    # Keep exact stdout separately while readable logs satisfy Git whitespace checks.
    raw_path = OUT/'checker-raw-output.json'
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    raw[name] = result.stdout
    raw_path.write_text(json.dumps(raw, indent=2)+'\n')
    (OUT/name).write_text(result.stdout.rstrip('\n')+'\n')
    commands.append({'argv':argv,'cwd':str(ROOT),'environment_overrides':overrides,
                     'started_at_utc':started,'duration_seconds':round(time.monotonic()-t,3),
                     'exit_code':result.returncode,'output':'runs/m06/'+name})
    (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(name,'exit',result.returncode,result.stdout[-200:],flush=True)
    if result.returncode:raise SystemExit(result.returncode)

run([MODEL,'scripts/check_m06.py','--lczero-checkout','/private/tmp/rl-chess-f03-lczero',
     '--output',str(OUT/'relative-acceptance.json')],'01-m06-check.log')
run([MODEL,'runs/m06/diagnostic.py','scripts/check_m05.py','--output',str(OUT/'m05-rerun.json')],'02-m05-diagnostic-check.log')
for milestone in (4,3,2,1):
    run([MODEL,f'scripts/check_m0{milestone}.py','--maia3-checkout','/private/tmp/rl-chess-f03-maia3',
         '--output',str(OUT/f'm0{milestone}-rerun.json')],f'0{7-milestone}-m0{milestone}-check.log')
run([MODEL,'-m','pytest','-p','no:cacheprovider','-p','runs.m06.diagnostic_plugin',
     '--junitxml='+str(OUT/'pytest-model.xml')],'07-model-tests.log')
run([ADAPTER,'-m','pytest','-p','no:cacheprovider','--junitxml='+str(OUT/'pytest-adapter-only.xml')],'08-adapter-only-tests.log')
run([MODEL,'scripts/check_adapter.py'],'09-adapter-replay-model.json')
run([ADAPTER,'scripts/check_adapter.py'],'10-adapter-replay-only.json')
run([ADAPTER,'-c',"import importlib.util,sys; assert importlib.util.find_spec('torch') is None; import chess_rl.adapter; assert 'torch' not in sys.modules; print('No installed/imported Torch')"],'11-adapter-import.log')
run(['uv','pip','check','--python',MODEL],'12-model-dependencies.log',{'UV_CACHE_DIR':'/private/tmp/rl-chess-m06-uv-cache'})
run(['uv','pip','check','--python',ADAPTER],'13-adapter-dependencies.log',{'UV_CACHE_DIR':'/private/tmp/rl-chess-m06-uv-cache'})
run([MODEL,'scripts/check_upstream.py','--maia3-checkout','/private/tmp/rl-chess-f03-maia3',
     '--lczero-checkout','/private/tmp/rl-chess-f03-lczero'],'14-upstream-check.log')
