import copy
import fcntl
import subprocess
import sys
import numpy as np
import pytest
from tests.test_selfplay import cpu, fixture_checkpoint, create, core
from chess_rl import selfplay as impl
from chess_rl.selfplay import Session, audit

def test_real_process_writer_lock_rejects_without_publication(tmp_path, fixture_checkpoint):
    session=create(tmp_path,fixture_checkpoint)
    before=(session.directory/'boundary.json').read_bytes()
    script='import fcntl,sys; f=open(sys.argv[1],"a+b"); fcntl.flock(f,fcntl.LOCK_EX); print("locked",flush=True); sys.stdin.read()'
    child=subprocess.Popen([sys.executable,'-B','-c',script,str(session.directory/'.writer.lock')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='locked'
        with pytest.raises(BlockingIOError): session.run()
        assert (session.directory/'boundary.json').read_bytes()==before
    finally:
        child.communicate('',timeout=10)
    assert session.run()['game']['status']=='truncated'

@pytest.mark.parametrize('failure_point,commits',[('file-fsync',0),('replace',0),('directory-fsync',1)])
def test_atomic_syscall_failures_restore_exact_continuation(tmp_path,fixture_checkpoint,monkeypatch,failure_point,commits):
    reference=create(tmp_path,fixture_checkpoint,name='full').run()
    session=create(tmp_path,fixture_checkpoint,name='fault')
    original_fsync=impl.os.fsync
    original_replace=impl.os.replace
    calls=0
    def fsync(fd):
        nonlocal calls
        calls+=1
        if (failure_point=='file-fsync' and calls==1) or (failure_point=='directory-fsync' and calls==2):
            raise OSError('injected fsync error')
        return original_fsync(fd)
    def replace(*args):
        if failure_point=='replace': raise OSError('injected replace error')
        return original_replace(*args)
    with monkeypatch.context() as patch:
        patch.setattr(impl.os,'fsync',fsync)
        patch.setattr(impl.os,'replace',replace)
        with pytest.raises(OSError): session.run()
    restored=Session.load(session.directory)
    assert len(restored.record['positions'])==commits
    assert not list(session.directory.glob('.pending-*'))
    assert core(restored.run())==core(reference)

@pytest.mark.parametrize('field',['root_visits','visit_policy','legal_mask','rng_before','root_fen','outcome'])
def test_storage_audit_rejects_independent_tampering(tmp_path,fixture_checkpoint,field):
    session=create(tmp_path,fixture_checkpoint)
    session.run(stop_after_moves=1)
    record=copy.deepcopy(session.record)
    position=record['positions'][0]
    if field=='root_visits': position[field][position['move_id']]+=1
    elif field=='visit_policy': position[field]=[0.]*4352
    elif field=='legal_mask': position[field]=[False]*4352
    elif field=='rng_before': position[field]=np.random.PCG64(11).state
    elif field=='root_fen': position[field]='changed'
    else: record[field]='1/2-1/2'
    with pytest.raises(ValueError): audit(session.configuration,record)

def test_partial_creation_is_refused_and_stale_writer_cannot_advance(tmp_path,fixture_checkpoint):
    session=create(tmp_path,fixture_checkpoint)
    other=Session.load(session.directory)
    session.run(stop_after_moves=1)
    before=(session.directory/'boundary.json').read_bytes()
    with pytest.raises(ValueError,match='stale'): other.run()
    assert (session.directory/'boundary.json').read_bytes()==before
    incomplete=tmp_path/'incomplete'; incomplete.mkdir()
    (incomplete/'configuration.json').write_bytes((session.directory/'configuration.json').read_bytes())
    with pytest.raises(FileNotFoundError): Session.load(incomplete)
