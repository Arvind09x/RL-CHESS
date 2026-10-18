"""Tiny self-play integration: durable boundaries, RNG, provenance and replay."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest
import torch

from chess_rl import selfplay_campaign as C
from chess_rl import selfplay as S
from chess_rl.selfplay_batching import collector_for, BoundEvaluator, play_group
from chess_rl.selfplay_execution import Execution, binding
from chess_rl.replay import ReplayBuffer
from .test_checkpoints import saved
from .test_selfplay_campaign import FIFTY

ROOT = Path(__file__).resolve().parents[1]
BATCHED = Execution(mode='batched', workers=1, games_per_worker=3, batch_size=2)


@pytest.fixture(autouse=True)
def deterministic():
    old, threads = torch.are_deterministic_algorithms_enabled(), torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(old)
    torch.set_num_threads(threads)


def create(path, saved, *, execution=BATCHED, opening=C.STANDARD_OPENING, cap=4, games=3):
    return C.create(path, saved['path'], games=games, seed=91, purpose='tiny batching test',
                    settings=S.Settings(2, cap, phases=(S.Phase(0, 1., .25), S.Phase(2, 0., .25))),
                    opening=opening, execution=execution)


def tasks(path, plan):
    return [dict(index=g['index'], directory=str(path / g['directory']), seed=g['seed'],
                 checkpoint=plan['checkpoint']['path'], opening=plan['opening'], settings=plan['settings'],
                 campaign_seed=plan['seed'], device=plan['device']) for g in plan['games']
            if not C._finished(path / g['directory'])]


def games(path, plan):
    return [S.read_game_record(path / g['directory']) for g in plan['games']]


def test_cpu_campaign_archives_and_replay_ingests_completed_games(tmp_path, saved):
    path = tmp_path / 'campaign'
    plan = create(path, saved, opening=FIFTY, cap=5)
    report = C.run(path, workers=1, log=lambda _: None)
    assert report['status_counts']['completed'] == 3
    summary = C.archive(path)
    buffer = ReplayBuffer.create(tmp_path / 'buffer.sqlite')
    receipts = [buffer.ingest(json.loads(line)) for line in (path / 'games.jsonl').read_text().splitlines()]
    assert len(receipts) == 3 and summary['positions'] == 3
    assert list(buffer.iter_samples())  # real terminal WDL labels pass existing verifier
    before = {p: p.read_bytes() for p in path.glob('games/*/*.json')}
    C.run(path, workers=1, log=lambda _: None)
    assert all(p.read_bytes() == data for p, data in before.items())
    events = C._Events(path / 'events.jsonl').read()
    metrics = next(e['collector'] for e in events if e['event'] == 'batch-worker')
    assert metrics['requests'] > 0 and metrics['peak_pending'] <= 3


def test_truncation_and_initial_terminal_preserve_filtering(tmp_path, saved):
    from chess_rl.evaluation.matches import Opening
    for name, opening, cap, expected in [
        ('cap', C.STANDARD_OPENING, 1, 'truncated'),
        ('terminal', Opening('draw', '4k3/8/8/8/8/8/8/4K3 w - - 0 1', (), True, 'setup'), 1, 'completed')]:
        path = tmp_path / name
        plan = create(path, saved, opening=opening, cap=cap, games=2)
        result = play_group(tasks(path, plan), plan, path)
        assert all(g['status'] == expected for g in result['game_results'])
        C.archive(path)
        buffer = ReplayBuffer.create(tmp_path / (name + '.sqlite'))
        for line in (path / 'games.jsonl').read_text().splitlines():
            buffer.ingest(json.loads(line))
        assert not list(buffer.iter_samples())
        if name == 'terminal':
            assert result['collector']['requests'] == 0


def test_orphan_move_retry_and_rng_are_identical_under_reversed_arrival(tmp_path, saved, monkeypatch):
    control, resumed = tmp_path / 'control', tmp_path / 'resumed'
    plan = create(control, saved)
    shutil.copytree(control, resumed)
    play_group(tasks(control, plan), plan, control)
    write = S._write
    def crash(path, value, **kwargs):
        if Path(path) == resumed / 'games/0000/boundary.json' and value.get('journal_positions') == 2:
            raise OSError('injected interruption after durable move before boundary')
        return write(path, value, **kwargs)
    monkeypatch.setattr(S, '_write', crash)
    result = play_group(list(reversed(tasks(resumed, plan))), plan, resumed)
    assert any(g['status'] == 'error' for g in result['game_results'])
    assert (resumed / 'games/0000/moves/000001.json').exists()
    assert len(S.read_game_record(resumed / 'games/0000')['positions']) == 1
    kept = {p: p.read_bytes() for p in resumed.glob('games/*/moves/*.json')}
    monkeypatch.setattr(S, '_write', write)
    play_group(tasks(resumed, plan), plan, resumed)
    assert games(control, plan) == games(resumed, plan)
    assert all(p.read_bytes() == data for p, data in kept.items())
    assert len({json.dumps(g['positions'][0]['rng_before'], sort_keys=True) for g in games(control, plan)}) == 3


def test_worker_assignment_does_not_change_game_seeds_or_rng(tmp_path, saved):
    a, b = tmp_path / 'a', tmp_path / 'b'
    plan = create(a, saved)
    shutil.copytree(a, b)
    play_group(tasks(a, plan), plan, a)
    # Same frozen B/config: emulate different grouping/order without altering any game seed.
    for task in reversed(tasks(b, plan)):
        play_group([task], plan, b)
    assert all(g["status"] in C.FINAL for g in games(a, plan))
    assert games(a, plan) == games(b, plan)


def test_rejects_incompatible_resume_before_any_game_starts(tmp_path, saved, monkeypatch):
    path = tmp_path / 'frozen'
    create(path, saved)
    for execution in (Execution(workers=1), replace(BATCHED, batch_size=4),
                      replace(BATCHED, max_wait_ms=2), replace(BATCHED, games_per_worker=2)):
        with pytest.raises(ValueError, match='execution differs'):
            C.run(path, workers=1, execution=execution)
    with pytest.raises(ValueError, match='worker count'):
        C.run(path, workers=2)
    monkeypatch.setattr('chess_rl.selfplay_execution.source_identity', lambda: {})
    with pytest.raises(ValueError, match='source differs'):
        C.run(path, workers=1)
    assert not list((path / 'games').iterdir())


def test_serial_campaign_cannot_be_switched_and_missing_fields_rejected(tmp_path, saved):
    path = tmp_path / 'serial'
    create(path, saved, execution=None)
    assert 'execution' not in C.load(path)
    with pytest.raises(ValueError, match='execution differs'):
        C.run(path, workers=1, execution=BATCHED)
    for field in BATCHED.to_dict():
        value = BATCHED.to_dict(); value.pop(field)
        with pytest.raises(ValueError, match='every frozen field'):
            Execution.from_dict(value)


def test_session_refuses_serial_loading_and_runtime_drift(tmp_path, saved):
    path = tmp_path / 'batch'
    plan = create(path, saved)
    play_group(tasks(path, plan), plan, path)
    with pytest.raises(ValueError, match='execution differs'):
        S.Session.load(path / 'games/0000')
    frozen = S._read(path / 'execution-runtime.json')
    frozen['runtime']['threads'] += 1
    S._write(path / 'execution-runtime.json', frozen, replace=True)
    with pytest.raises(ValueError, match='runtime differs'):
        play_group([], plan, path)


def test_collector_failure_discards_partial_search_and_allows_same_config_retry(tmp_path, saved, monkeypatch):
    from chess_rl.search.real_evaluator import RealEvaluator
    path, control = tmp_path / 'failure', tmp_path / 'control'
    plan = create(path, saved)
    shutil.copytree(path, control)
    original = RealEvaluator.__call__
    def fail(*args):
        raise RuntimeError('injected batch failure')
    monkeypatch.setattr(RealEvaluator, '__call__', fail)
    result = play_group(tasks(path, plan), plan, path)
    assert all(g['status'] == 'interrupted' for g in result['game_results'])
    assert all(not g['positions'] for g in games(path, plan))
    monkeypatch.setattr(RealEvaluator, '__call__', original)
    play_group(tasks(path, plan), plan, path)
    play_group(tasks(control, plan), plan, control)
    for a, b in zip(games(path, plan), games(control, plan)):
        a.pop('interruptions'); b.pop('interruptions')
        assert a == b


def test_sigkill_after_move_before_pointer_resumes_exactly(tmp_path, saved):
    path, control = tmp_path / 'kill', tmp_path / 'control'
    plan = create(path, saved, games=2)
    shutil.copytree(path, control)
    env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}{os.pathsep}{ROOT}', PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-m', 'tests.selfplay_batch_crash_worker']
    process = subprocess.Popen([*command, 'crash', str(path)], env=env, cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    import selectors
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(60), 'crash worker did not reach journal boundary'
            assert process.stdout.readline().strip() == 'DURABLE_MOVE_READY'
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    kept = {p: p.read_bytes() for p in path.glob('games/*/moves/*.json')}
    assert kept
    for target in (path, control):
        result = subprocess.run([sys.executable, 'scripts/run_selfplay_campaign.py', 'run',
                                 '--directory', str(target), '--workers', '1',
                                 '--heavy-lock', str(tmp_path / 'heavy.lock')], env=env, cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
    assert games(path, plan) == games(control, plan)
    assert all(p.read_bytes() == data for p, data in kept.items())


def test_game_setup_rejects_checkpoint_change_after_collector_load(tmp_path, saved, monkeypatch):
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(Path(saved['path']).read_bytes())
    path = tmp_path / 'changed'
    plan = create(path, dict(saved, path=checkpoint), games=1)
    original = S._checkpoint
    def changed(path, evaluator):
        checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
        return original(path, evaluator)
    monkeypatch.setattr(S, '_checkpoint', changed)
    result = play_group(tasks(path, plan), plan, path)
    assert result['game_results'][0]['status'] == 'error'
    assert 'checkpoint differs' in result['game_results'][0]['error']
    assert not games(path, plan)[0]['positions']
