import hashlib, json, os, subprocess, time
from pathlib import Path
root=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
out=Path('/private/tmp/rl-chess-m02-stage/runs/m02')
out.mkdir(parents=True,exist_ok=True)
commands=[]
def run(command, name, overrides=None):
 start=time.time()
 env=os.environ.copy()
 env.update(overrides or {})
 result=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
 (out/name).write_text(result.stdout)
 commands.append({'command':command,'cwd':str(root),'environment_overrides':overrides or {},'exit_code':result.returncode,'output':'runs/m02/'+name,'duration_seconds':round(time.time()-start,3)})
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print(name,result.returncode,result.stdout[-400:],flush=True)
 if result.returncode: raise SystemExit(result.returncode)
run(['.venv/bin/python','scripts/check_m02.py','--maia3-checkout','/private/tmp/rl-chess-f03-maia3','--output',str(out/'backbone-acceptance.json')],'01-m02-check.log')
run(['.venv/bin/python','scripts/check_m01.py','--maia3-checkout','/private/tmp/rl-chess-f03-maia3','--output',str(out/'m01-rerun.json')],'02-m01-check.log')
run(['.venv/bin/python','-m','pytest','-p','no:cacheprovider','--junitxml='+str(out/'pytest-model.xml')],'03-model-tests.log')
run(['.venv-acceptance/bin/python','-m','pytest','-p','no:cacheprovider','--junitxml='+str(out/'pytest-adapter-only.xml')],'04-adapter-only-tests.log')
run(['.venv/bin/python','scripts/check_adapter.py'],'05-adapter-replay-model.json')
run(['.venv-acceptance/bin/python','scripts/check_adapter.py'],'06-adapter-replay-only.json')
run(['.venv-acceptance/bin/python','-c',"import importlib.util, sys; assert importlib.util.find_spec('torch') is None; import chess_rl.adapter; assert 'torch' not in sys.modules; print('Adapter-only environment: no installed or imported torch')"],'07-adapter-import.log')
run(['uv','pip','check','--python','.venv/bin/python'],'08-model-dependencies.log',{'UV_CACHE_DIR':'/private/tmp/rl-chess-m02-uv-cache'})
run(['uv','pip','check','--python','.venv-acceptance/bin/python'],'09-adapter-dependencies.log',{'UV_CACHE_DIR':'/private/tmp/rl-chess-m02-uv-cache'})
run(['.venv/bin/python','scripts/check_upstream.py','--maia3-checkout','/private/tmp/rl-chess-f03-maia3','--lczero-checkout','/private/tmp/rl-chess-f03-lczero'],'10-upstream-check.log')
# Preserve original plans and every historical artifact/adapter/pin from the starting commit.
baseline='0d0359a'
files=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline],cwd=root,text=True).splitlines()
protected=[f for f in files if f.startswith(('runs/','src/chess_rl/adapter/','docs/notices/')) or f in ('interface.md','plan1.md','docs/upstream-lock.json','pyproject.toml','uv.lock','src/chess_rl/model/gab.py','tests/gab_checks.py','tests/test_gab.py','tests/fixtures/gab_reference.py','scripts/check_m01.py')]
checks={}
for file in protected:
 old=subprocess.check_output(['git','show',baseline+':'+file],cwd=root)
 new=(root/file).read_bytes()
 assert old==new,file
 checks[file]=hashlib.sha256(new).hexdigest()
prior=json.loads((root/'runs/m01/original-plan-preservation.json').read_text())
original=Path(prior['directory'])
hashes={f:hashlib.sha256((original/f).read_bytes()).hexdigest() for f in prior['sha256']}
assert hashes==prior['sha256']
report={'baseline_commit':subprocess.check_output(['git','rev-parse',baseline],cwd=root,text=True).strip(),'protected_repository_sha256':checks,'original_planning_directory':str(original),'original_plan_sha256':hashes,'original_hashes_match_historical_m01_evidence':True,'repository_interface_and_plan1_unchanged':True,'historical_runs_adapter_m01_pins_notices_and_dependencies_unchanged':True}
(out/'preservation-check.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print('Preservation checks passed:',len(checks),'protected repository files and 3 original planning files',flush=True)
