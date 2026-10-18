"""Small benchmark plumbing checks, not a Mac performance campaign."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import torch

from chess_rl import selfplay_campaign as C
from chess_rl.adapter import make_state, terminal_result
from chess_rl.evaluation.matches import digest
from chess_rl.selfplay import Settings
from chess_rl.selfplay_batching import validate_acceptance
from chess_rl.selfplay_execution import Execution, source_identity, binding, session_runtime
from chess_rl.selfplay_benchmark import corpus, numerical_acceptance, benchmark_search
from scripts import prepare_batching_source as P
from .test_checkpoints import saved


@pytest.fixture(autouse=True)
def deterministic():
    old, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    yield
    torch.use_deterministic_algorithms(old); torch.set_num_threads(threads)


def test_corpus_has_reproducible_complete_long_histories():
    rows = corpus(5)
    assert rows == corpus(5)
    assert [len(r['moves_uci']) for r in rows] == [0, 20, 60, 120, 180]
    assert len({r['seed'] for r in rows}) == len(rows)
    for row in rows:
        state = make_state(moves_uci=row['moves_uci'])
        assert state.history_complete and terminal_result(state) is None


def test_small_acceptance_and_real_search_worker_metrics(tmp_path, saved):
    # No large-checkpoint/B8-B32 acceptance matrix; two states, B2, one repeat.
    accepted = numerical_acceptance(saved['path'], 'cpu', 2,
                                   states=[make_state(), make_state(moves_uci=['e2e4'])], repeats=1)
    assert accepted['accepted']
    execution = Execution(mode='batched', workers=1, games_per_worker=2, batch_size=2)
    rows = corpus(2)
    # Parent and spawned workers must both match the accepted runtime.
    result = benchmark_search(saved['path'], 'cpu', execution, rows, 1, accepted)
    assert result['completed_positions'] == 2 and result['completed_simulations'] == 2
    assert result['effective_active_games_per_worker'] == [2] and not result['underfilled']
    assert result['requests'] >= 2 and result['failures'] == []
    assert 0 < result['useful_batch_occupancy'] <= 1
    assert result['queue_seconds'] >= 0 and result['per_worker'][0]['memory']['process_peak_rss_bytes'] > 0
    assert 'games_per_hour' not in result and 'usable_training_positions_per_hour' not in result
    serial = benchmark_search(saved['path'], 'cpu', Execution(workers=1), rows, 1)
    assert serial['completed_positions'] == 2 and serial['completed_simulations'] == 2
    assert serial['useful_batch_occupancy'] == 1 and not serial['failures']
    plan = C.create(tmp_path / 'accepted', saved['path'], games=1, seed=1, purpose='test',
                    settings=Settings(1, 1), execution=execution, acceptance=accepted)
    for field, wrong in [('batch_size', 8), ('checkpoint_sha256', 'a'*64),
                         ('source_sha256', 'b'*64), ('accepted', False)]:
        bad = deepcopy(accepted); bad[field] = wrong
        bad['sha256'] = digest({k:v for k,v in bad.items() if k != 'sha256'})
        with pytest.raises(ValueError, match='numerical acceptance differs'):
            validate_acceptance(bad, plan)
    current = session_runtime('cpu', binding(plan)); current['threads'] += 1
    with pytest.raises(ValueError, match='numerical acceptance differs'):
        validate_acceptance(accepted, plan, current)


def test_cuda_new_campaign_requires_acceptance_before_creation(tmp_path, saved):
    target = tmp_path / 'cuda'
    with pytest.raises(ValueError, match='CUDA batching requires'):
        C.create(target, saved['path'], games=1, seed=1, purpose='test', simulations=1, device='cuda:0',
                 execution=Execution(mode='batched', workers=1, batch_size=2, games_per_worker=2))
    assert not target.exists()


def test_source_manifest_refuses_dirty_tree_and_verifies_inventory_without_git(tmp_path, monkeypatch):
    monkeypatch.setattr(P.subprocess, 'check_output', lambda *a, **k: b' M modified.py\n')
    with pytest.raises(ValueError, match='clean committed'):
        P.prepare(tmp_path, tmp_path / 'manifest.json')
    assert not (tmp_path / 'manifest.json').exists()
    (tmp_path / 'src').mkdir()
    for name in ('pyproject.toml', 'uv.lock', 'src/module.py'):
        (tmp_path / name).write_text('fixture\n')
    record = dict(schema='batching-source-v1', revision='a'*40,
                  files={name:P.sha256(tmp_path / name) for name in ('pyproject.toml', 'uv.lock', 'src/module.py')},
                  execution_source=source_identity(tmp_path))
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps(record))
    monkeypatch.setattr(P.subprocess, 'check_output', lambda *a, **k: pytest.fail('Git on compute node'))
    assert P.verify(tmp_path, path) == record
    (tmp_path / 'src/new.py').write_text('added code\n')
    with pytest.raises(ValueError, match='inventory changed'):
        P.verify(tmp_path, path)


def test_batch_job_runs_offline_without_git_or_submission(tmp_path):
    import os
    root = Path(__file__).resolve().parents[1]
    job = root / 'jobs/b200-selfplay-batching/benchmark.sbatch'
    assert subprocess.run(['bash', '-n', str(job)], capture_output=True).returncode == 0
    (tmp_path / 'jobs/b200').mkdir(parents=True)
    (tmp_path / 'jobs/b200/common.sh').write_text((root / 'jobs/b200/common.sh').read_text()
                                                + '\nreport_cgroup_peak() { :; }\n')
    (tmp_path / 'runtime/cuda-cu128').mkdir(parents=True)
    (tmp_path / 'pyproject.toml').touch()
    binary = tmp_path / '.venv-cuda-cu128/bin/python'; binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\n'); binary.chmod(0o755)
    bins = tmp_path / 'bin'; bins.mkdir()
    for name in ('git', 'sbatch'):
        stub = bins / name
        stub.write_text('#!/bin/bash\ntouch "$UNEXPECTED"\nexit 97\n'); stub.chmod(0o755)
    # Portable offline stand-in; verify the GNU timeout arguments then invoke only our fake Python.
    stub = bins / 'timeout'
    stub.write_text('#!/bin/bash\n[[ "$1" == --signal=TERM && "$2" == --kill-after=30s && "$3" == 110m ]] || exit 98\nshift 3\nexec "$@"\n')
    stub.chmod(0o755)
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_ID='offline',
               PATH=f'{bins}:{os.environ["PATH"]}', CALLS=str(tmp_path / 'calls'), UNEXPECTED=str(tmp_path / 'unexpected'))
    result = subprocess.run(['bash', str(job)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / 'unexpected').exists()
    calls = (tmp_path / 'calls').read_text()
    assert '--batch-sizes 8 16 32 --active-games-per-worker 8 16 32' in calls
    assert '--checkpoint-sha256 32045b10679e0b98a5a1dbbc08d05024b695c2efc0df16d565a3df13ffc3fbf2' in calls


def test_acceptance_rejects_checkpoint_changed_during_probe(tmp_path, saved, monkeypatch):
    from chess_rl import selfplay_benchmark as B
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(Path(saved['path']).read_bytes())
    def changed(*args, **kwargs):
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
        return dict(bit_identical=True, changed_positions=[])
    monkeypatch.setattr(B, 'arrival_probe', changed)
    with pytest.raises(ValueError, match='changed during numerical acceptance'):
        B.numerical_acceptance(checkpoint, 'cpu', 2, states=[make_state()], repeats=1)


def test_cli_wrong_checkpoint_pin_fails_before_creating_output(tmp_path, saved):
    import os
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, 'scripts/benchmark_selfplay_batching.py',
        '--checkpoint', str(saved['path']), '--checkpoint-sha256', '0'*64,
        '--output', str(tmp_path / 'should-not-exist'), '--mode', 'accept'], cwd=root,
        env=dict(os.environ, PYTHONPATH='src:.', PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'checkpoint hash differs' in result.stderr
    assert not (tmp_path / 'should-not-exist').exists()
