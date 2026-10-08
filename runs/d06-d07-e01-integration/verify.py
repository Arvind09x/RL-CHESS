"""Verify the reviewed combined snapshot without altering historical/bulk evidence."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, subprocess, threading, time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
PRIMARY=Path('/Users/arvindsrinivass/Desktop/RL-Chess')
OUT=Path(os.environ.get('RL_CHESS_INTEGRATION_OUTPUT',str(Path('/private/tmp/rl-chess-combined-verification')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))))
BULK=OUT/'external-audits'
BASE='4184387c7e4b67559355d6b212a2a300c86c5ce6'
SOURCES={'D06':'ac424ecfafc3770023fba196b6158e65d943f199','D07':'b6e3d0eec171aa52e8fbc3ddedda9ec128daf330','E01':'ccea776af7ee74814305e09a5df24578b18511e1'}
ENV=dict(os.environ,PYTHONPATH=f'{ROOT}/src:{ROOT}',PYTHONDONTWRITEBYTECODE='1')
COMMANDS=[];LOCK=threading.Lock()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def identity(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return dict(bytes=path.stat().st_size,sha256=h.hexdigest())
def run(name,argv):
    argv=list(map(str,argv));start=datetime.now(timezone.utc).isoformat();begin=time.monotonic()
    result=subprocess.run(argv,cwd=ROOT,env=ENV,capture_output=True)
    stdout=OUT/f'{name}.stdout.txt';stderr=OUT/f'{name}.stderr.txt'
    stdout.write_bytes(result.stdout);stderr.write_bytes(result.stderr)
    record=dict(name=name,argv=argv,cwd=str(ROOT),started_at_utc=start,duration_seconds=time.monotonic()-begin,
        exit_code=result.returncode,environment={k:ENV[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE')},
        stdout=identity(stdout),stderr=identity(stderr))
    with LOCK:
        COMMANDS.append(record);dump(OUT/'commands.json',COMMANDS)
    print(f'{name}: exit {result.returncode}, {record["duration_seconds"]:.2f}s',flush=True)
    if result.returncode:
        print(result.stdout.decode(errors='replace')[-6000:]+result.stderr.decode(errors='replace')[-2000:],flush=True)
        raise RuntimeError(name)
    return result.stdout

def preserve():
    checks={};appended={};basefiles=git('ls-tree','-r','--name-only',BASE).decode().splitlines()
    for name in basefiles:
        expected=git('show',f'{BASE}:{name}');actual=(ROOT/name).read_bytes()
        if name in ('README.md','coding-plan.md'):
            suffixes=[]
            for commit in SOURCES.values():
                source=git('show',f'{commit}:{name}')
                assert source.startswith(expected),name
                suffixes.append(source[len(expected):])
            prefix=expected+b''.join(suffixes)
            assert actual.startswith(prefix),name
            appended[name]=dict(base_preserved=True,all_source_entries_verbatim=True)
        else:
            assert actual==expected,name
        checks[name]=identity(ROOT/name)
    added={}
    for label,commit in SOURCES.items():
        for name in git('diff','--name-only',BASE,commit).decode().splitlines():
            if name in ('README.md','coding-plan.md'):continue
            expected=git('show',f'{commit}:{name}')
            assert (ROOT/name).read_bytes()==expected,name
            added[name]=dict(source=label,sha256=hashlib.sha256(expected).hexdigest())
        subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True)
    external={}
    for task in ('d06','d07'):
        for path,expected in json.loads((ROOT/f'runs/{task}/preservation.json').read_text())['external_identities'].items():
            if path in external:assert external[path]==expected,path
            external[path]=expected
    index=json.loads((ROOT/'runs/d06/artifact-index.json').read_text())
    for name,expected in index['artifacts'].items():external[str(Path(index['bulk_directory'])/name)]=expected
    for path,expected in external.items():assert identity(Path(path))==expected,path
    manifest=json.loads((ROOT/'data/manifests/elite-2025-01-d04-v1.json').read_text())
    teacher=manifest['teacher']
    assert teacher['nodes']==100000 and teacher['hash_mb']==128 and teacher['threads']==1
    assert teacher['show_wdl'] is True and teacher['multipv']==1 and teacher['tablebases']=='disabled'
    tracked=git('ls-files').decode().splitlines()
    assert not any(p.startswith(('data/raw/','data/processed/','data/labels/','data/annotations/','checkpoints/')) for p in tracked)
    report=dict(base_commit=BASE,base_files=len(basefiles),byte_identical_base_files=len(basefiles)-len(appended),
        appended_documents=appended,accepted_base_files=checks,exact_source_files=added,
        original_external_identities=external,protected_heldout_artifacts_preserved=True,teacher_settings=teacher,
        bulk_artifacts_outside_git=True,source_commits_are_ancestors=True)
    dump(OUT/'preservation.json',report)
    print(f'Preservation: {len(basefiles)-len(appended)} base files unchanged, {len(added)} source files exact, {len(external)} external identities match',flush=True)
    return report

def environment_checks(label,venv):
    py=PRIMARY/venv/'bin/python'
    run('environment-'+label,[py,'-B','-c',"import importlib.metadata as m,platform,json,sys; print(json.dumps(dict(python=sys.version,platform=platform.platform(),executable=sys.executable,versions={n:m.version(n) for n in ['numpy','python-chess','chess','pytest']}),indent=2))"])
    plugin=['-p','runs.m06.diagnostic_plugin'] if label=='model' else []
    xml=OUT/f'regressions-{label}.xml'
    run('regressions-'+label,[py,'-B','-m','pytest',*plugin,'--import-mode=importlib','-p','no:cacheprovider',f'--junitxml={xml}'])
    suite=ET.parse(xml).getroot().find('testsuite').attrib
    assert int(suite['failures'])==int(suite['errors'])==0
    passed=int(suite['tests'])-int(suite['skipped'])
    assert (passed,int(suite['skipped']))==((1181,0) if label=='model' else (1012,7)),suite
    probes=[f'runs/{t}/test_review_cases.py' for t in ('d02-s02-review','d03-s03-review','d04-s04-review','d05-s05-review')]
    px=OUT/f'accepted-review-probes-{label}.xml'
    run('accepted-review-probes-'+label,[py,'-B','-m','pytest',*probes,'--import-mode=importlib','-p','no:cacheprovider',f'--junitxml={px}'])
    p=ET.parse(px).getroot().find('testsuite').attrib
    assert (int(p['tests']),int(p['skipped']),int(p['failures']),int(p['errors']))==(84,0,0,0)
    frozen=json.loads(run('frozen-adapter-'+label,[py,'-B','scripts/check_adapter.py']))
    assert len(frozen)==13
    assert frozen==json.loads((ROOT/'runs/d05-s05-review/adapter-frozen-replay.log').read_text())
    run('d07-roots-'+label,[py,'-B','runs/d07/verify_sample.py',OUT/f'd07-roots-{label}.json'])
    roots=json.loads((OUT/f'd07-roots-{label}.json').read_text());assert len(roots['records'])==12
    s05=ROOT/'runs/s05/20261008T132541185409Z/games-adapter/games.jsonl'
    run('s05-replay-'+label,[py,'-B','runs/s05/replay_games.py',s05,OUT/f's05-replay-{label}.json'])
    e01=ROOT/'runs/e01/20261008T150821584550Z'
    audits={}
    for producer in ('adapter','model'):
        name=f'e01-{producer}-in-{label}'
        run(name,[py,'-B','scripts/replay_e01.py',e01/f'matches-{producer}/matches.jsonl','--output',OUT/f'{name}.json'])
        audit=json.loads((OUT/f'{name}.json').read_text())
        assert audit['passed'] and audit['games']==48 and audit['matches']==24
        audits[producer]=dict(matches=24,games=48,passed=True)
    return dict(regression_passes=passed,expected_skips=int(suite['skipped']),reviewer_probes=84,
        frozen_fixtures=13,d07_roots=12,e01=audits)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs={label:pool.submit(environment_checks,label,venv) for label,venv in [('model','.venv'),('adapter','.venv-acceptance')]}
        counts={label:f.result() for label,f in fs.items()}
    # Read-only symlinks to original bulk inputs; the new audit alone writes here.
    target=BULK/'d06';target.mkdir(parents=True,exist_ok=False)
    for path in Path('/private/tmp/rl-chess-d06-benchmark').iterdir():
        if path.name!='independent-replay.json':(target/path.name).symlink_to(path)
    py=PRIMARY/'.venv-acceptance/bin/python'
    stdout=run('d06-full-label-audit',[py,'-B','scripts/audit_d06.py',target])
    audit=json.loads(stdout)
    assert audit['successful_labels_replayed']==1000 and audit['selection_boundary_proved'] and audit['all_labels_legal']
    dump(OUT/'d06-replay-summary.json',audit)
    run('manifest-teacher-gate',[py,'-B','scripts/check_d01.py','--teacher-executable','/private/tmp/Stockfish-sf_17.1/src/stockfish'])
    run('prepared-manifest-teacher-gate',[py,'-B','scripts/check_d01.py','--manifest','data/manifests/elite-2025-01-d04-v1.json','--require-annotation-ready','--teacher-executable','/private/tmp/Stockfish-sf_17.1/src/stockfish'])
    run('upstream-pins-offline',[py,'-B','scripts/check_upstream.py','--maia3-checkout','/private/tmp/rl-chess-f03-maia3','--lczero-checkout','/private/tmp/rl-chess-f03-lczero'])
    run('m06-checker',[PRIMARY/'.venv/bin/python','-B','scripts/check_m06.py'])
    preservation=preserve()
    dump(OUT/'summary.json',dict(tested_combined_merge=git('rev-parse','HEAD').decode().strip(),starting_main=BASE,
        source_commits=SOURCES,checks=counts,d06_labels_replayed=1000,d06_membership_and_raw_teacher_audit_passed=True,
        protected_heldout_artifacts_preserved=True,accepted_source_bytes_exact=True,teacher_settings_preserved=True,
        historical_evidence_preserved=True,bulk_audit_directory=str(BULK),all_commands_exit_zero=True,
        no_new_teacher_search=True,later_tasks_run=[],scope='D06 benchmark, D07 CPU loading/batching, E01 match protocol fixtures; no training or strength claim'))
    print('Combined verification passed',flush=True)

if __name__=='__main__':main()
