"""Lossless integration receipts bind isolated code and pinned environment."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PRIMARY=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
PYTHON=PRIMARY/'.venv/bin/python'
OVERRIDES=dict(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',GIT_OPTIONAL_LOCKS='0')
def identity(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def run(name,argv):
    code={str(p.relative_to(ROOT)):identity(p) for d in ('src','scripts','tests') for p in (ROOT/d).rglob('*.py')}
    code.update({str(p.relative_to(ROOT)):identity(p) for p in OUT.glob('*.py')})
    stamp=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
    with (OUT/(name+'.stdout.txt')).open('xb') as stdout,(OUT/(name+'.stderr.txt')).open('xb') as stderr:
        result=subprocess.run([str(v) for v in argv],cwd=ROOT,env=os.environ|OVERRIDES,stdout=stdout,stderr=stderr)
    unchanged=all(identity(ROOT/n)==v for n,v in code.items())
    receipt=dict(argv=[str(v) for v in argv],cwd=str(ROOT),started_at_utc=stamp,
        ended_at_utc=datetime.now(timezone.utc).isoformat(),duration_seconds=time.monotonic()-tick,
        exit_code=result.returncode,environment_overrides=OVERRIDES,tested_sources=code,sources_unchanged=unchanged,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        outputs={k:identity(OUT/(name+'.'+k+'.txt')) for k in ('stdout','stderr')})
    with (OUT/(name+'.command.json')).open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print(name+': exit='+str(result.returncode),flush=True)
    if result.returncode:print((OUT/(name+'.stderr.txt')).read_text(),flush=True)
    assert result.returncode==0 and unchanged
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='cycle':
        for phase in ('audit','regenerate','boundary','full','prefix','resume','evaluate','reload'):
            run(phase,[PYTHON,'-B',OUT/'verify_chain.py',phase])
    elif mode in ('model','adapter'):
        py=PYTHON if mode=='model' else PRIMARY/'.venv-acceptance/bin/python'
        argv=[py,'-B','-m','pytest','--import-mode=importlib','-p','no:cacheprovider','--junitxml='+str(OUT/(mode+'.xml'))]
        if mode=='model':argv.extend(['-p','runs.m06.diagnostic_plugin'])
        run('regression-'+mode,argv)

