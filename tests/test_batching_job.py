"""Offline checks for guarded submission, pinned weights and source identity."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import prepare_batching_source as P
from scripts import run_batching_acceptance as A
from chess_rl.evaluation.benchmark_layouts import DEFAULT_LAYOUTS, parse_layout
from chess_rl.selfplay_execution import source_identity

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pinned(tmp_path):
    (tmp_path/'code.py').write_text('frozen source\n')
    for name in ('pyproject.toml', 'uv.lock'):
        (tmp_path/name).write_text('fixture\n')
    for name in ('left.pt','right.pt'):
        (tmp_path/name).write_text(name)
    record = tmp_path/'source.json'
    record.write_text(json.dumps(dict(schema='batching-source-v1',revision='a'*40,
                                     files={'code.py':P.sha256(tmp_path/'code.py')},
                                     execution_source=source_identity(tmp_path))))
    return tmp_path,record


def test_prepare_pins_clean_submission_host_checkout_and_never_rewrites(tmp_path, monkeypatch):
    (tmp_path/'code.py').write_text('code\n')
    for name in ('pyproject.toml', 'uv.lock'):
        (tmp_path/name).write_text('fixture\n')
    def git(command):
        if command[-1] == '--untracked-files=normal': return b''
        if command[-1] == 'HEAD': return b'a'*40+b'\n'
        if command[-1] == '-z': return b'code.py\0'
        pytest.fail(str(command))
    monkeypatch.setattr(P.subprocess,'check_output',git)
    record=tmp_path/'source.json'
    P.prepare(tmp_path,record)
    with pytest.raises(FileExistsError): P.prepare(tmp_path,record)
    monkeypatch.setattr(P.subprocess,'check_output',lambda *_:pytest.fail('compute node invoked Git'))
    assert P.verify(tmp_path,record)['revision'] == 'a'*40
    (tmp_path/'code.py').write_text('changed\n')
    with pytest.raises(ValueError,match='source changed'): P.verify(tmp_path,record)


def test_driver_checks_both_weights_before_any_model_or_output(pinned):
    root,record=pinned
    calls=[]
    with pytest.raises(ValueError,match='checkpoint checksum'):
        A.run(root,record,root/'out',Path('left.pt'),P.sha256(root/'left.pt'),Path('right.pt'),'0'*64,
              runner=lambda *a,**k:calls.append(a))
    assert not calls and not (root/'out').exists()
    A.run(root,record,root/'out',Path('left.pt'),P.sha256(root/'left.pt'),Path('right.pt'),P.sha256(root/'right.pt'),
          runner=lambda command,**kwargs:calls.append(command))
    assert len(calls)==6
    throughput=calls[-1]
    assert throughput[throughput.index('--pairs')+1]=='8'
    assert throughput[throughput.index('--simulations')+1]=='128'
    assert throughput[throughput.index('--layouts')+1:throughput.index('--pairs')]==list(DEFAULT_LAYOUTS)
    assert (root/'out/source-revision.txt').read_text().strip()=='a'*40


def test_default_layouts_fill_the_short_study_and_account_for_two_models():
    for text in DEFAULT_LAYOUTS:
        layout=parse_layout(text)
        assert layout.processes<=5 and layout.processes*layout.games_per_process<=16
        if layout.mode=='batched': assert layout.batch_size==layout.games_per_process//2
    assert parse_layout('3x32x16').batch_size==16
    assert parse_layout('3x64x32').games_per_process==64
    assert parse_layout('2x8',batch_size=4).batch_size==4
    for text in ('6x8x4','0x1','1x1x4','2x0x4','../../bad','1x2x0'):
        with pytest.raises(ValueError): parse_layout(text)


def test_batch_script_uses_pins_and_never_calls_git_on_compute_node(tmp_path):
    job=ROOT/'jobs/b200-batching/acceptance.sbatch'
    text=job.read_text()
    assert '#SBATCH --time=0-02:00:00' in text
    assert 'P=artifacts/p-1m/step-090000.pt' in text
    assert 'cf7442ee28969d7d93f5ab37a4829400f2c73efb319b6b08a1717d7adfe26dd2' in text
    assert '32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2' in text
    assert 'sbatch --export' not in text
    (tmp_path/'jobs/b200').mkdir(parents=True)
    common=(ROOT/'jobs/b200/common.sh').read_text()
    (tmp_path/'jobs/b200/common.sh').write_text(common+'\nreport_cgroup_peak() { :; }\n')
    (tmp_path/'runtime/cuda-cu128').mkdir(parents=True)
    (tmp_path/'pyproject.toml').touch()
    binary=tmp_path/'.venv-cuda-cu128/bin/python'
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n')
    binary.chmod(0o755)
    stub=tmp_path/'bin/git'
    stub.parent.mkdir()
    stub.write_text('#!/bin/bash\necho invoked > "$GIT_MARKER"\nexit 97\n')
    stub.chmod(0o755)
    env=dict(os.environ,SLURM_SUBMIT_DIR=str(tmp_path),SLURM_JOB_ID='offline-check',
             PATH=f"{stub.parent}:{os.environ['PATH']}",CALLS=str(tmp_path/'calls'),GIT_MARKER=str(tmp_path/'git-called'))
    result=subprocess.run(['bash',str(job)],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'git-called').exists()
    assert 'scripts/run_batching_acceptance.py' in (tmp_path/'calls').read_text()
