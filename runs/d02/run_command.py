"""Run one actual command and retain its argv, environment, exit and raw output."""
from datetime import datetime,timezone
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
name=sys.argv[1]
argv=sys.argv[2:]
env={'PYTHONPATH':str(root/'src')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1'}
start=datetime.now(timezone.utc).isoformat()
t=time.monotonic()
with (root/'runs/d02'/f'{name}.log').open('w') as out:
 result=subprocess.run(argv,cwd=root,env={**os.environ,**env},stdout=out,stderr=subprocess.STDOUT)
record={'argv':argv,'cwd':str(root),'environment_overrides':env,'started_at_utc':start,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.monotonic()-t,'exit_code':result.returncode,'output':f'runs/d02/{name}.log'}
(root/'runs/d02'/f'{name}-command.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
sys.exit(result.returncode)
