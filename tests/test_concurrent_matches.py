"""Frozen execution, actual worker games, immutable stores and kill/resume."""
from copy import deepcopy
from dataclasses import replace
import json
import os
import signal
import sqlite3
import time

import pytest
pytest.importorskip('torch')
from chess_rl.evaluation import match_campaign as M
from chess_rl.evaluation.execution import Execution, execution_for
from .test_match_campaign import world, runtime, records, CAP, ROOT
from chess_rl.evaluation import development_matches as D
import subprocess
import sys


def setup_matches(world, name):
    return world['base'] / name


def plan(world, matches, runtime, *, workers=2, execution=None, count=2, simulations=2, ply_cap=CAP):
    return D.freeze(matches / 'development-match', world['root'],
                    (world['root'] / world['spec']['P']['path'],
                     world['root'] / world['spec']['candidates'][0]['path']),
                    execution or Execution(processes=workers), runtime,
                    pairs=count, simulations=simulations, ply_cap=ply_cap)


def run_cli(root, matches, *args):
    match_id = args[args.index('--match-id') + 1]
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}")
    return subprocess.Popen([sys.executable, str(ROOT / 'scripts/benchmark_matches.py'), 'run',
                             '--root', str(root), '--directory', str(matches / match_id)],
                            env=env, start_new_session=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def batched(workers=1, games=4, flush=2):
    return Execution(mode='batched', batch_size=4, games_per_process=games,
                     processes=workers, max_wait_ms=flush)


def test_frozen_execution_and_old_serial_identity(world, runtime):
    matches = setup_matches(world, 'exec-identity')
    p = plan(world, matches, runtime, workers=1, execution=batched())
    directory = matches / p['match_id']
    assert p['execution'] == batched().to_dict()
    for config in (Execution(processes=1), replace(batched(), batch_size=8),
                   replace(batched(), games_per_process=2), replace(batched(), max_wait_ms=5),
                   replace(batched(), processes=2)):
        with pytest.raises(ValueError, match='execution differs'):
            D.run(directory, world['root'], execution=config, log=lambda _: None)
    bad = dict(p, workers=2)
    with pytest.raises(ValueError, match='process count'):
        execution_for(bad)
    legacy = dict(p)
    legacy.pop('execution')
    before = deepcopy(legacy)
    assert execution_for(legacy).mode == 'serial'
    assert legacy == before
    assert M.sealed({k:v for k,v in legacy.items() if k != 'sha256'})['sha256'] != p['sha256']
    with pytest.raises(ValueError, match='blocked'):
        execution_for(dict(p, split='test'))
    with pytest.raises(ValueError, match='blocked'):
        execution_for(dict(p, protocol='production'))


@pytest.fixture(scope='module')
def concurrent_played(world, runtime):
    matches = setup_matches(world, 'concurrent-played')
    p = plan(world, matches, runtime, workers=1, execution=batched())
    directory = matches / p['match_id']
    assert D.run(directory, world['root'], log=lambda _: None)['finished']
    values = records(matches, p['match_id'])
    assert len(values) == 4
    D.run(directory, world['root'], log=lambda _: None)
    assert records(matches, p['match_id']) == values
    for j in p['schedule']:
        M.validate_game(values[j['game_id']]['record'], j, p)
    return matches, p, values


def test_full_games_changed_flush_and_process_scheduling(world, runtime, concurrent_played):
    matches0, p0, values0 = concurrent_played
    matches = setup_matches(world, 'concurrent-altered')
    p = plan(world, matches, runtime, workers=2, execution=batched(2, 2, 7))
    assert D.run(matches / p['match_id'], world['root'], log=lambda _: None)['finished']
    values = records(matches, p['match_id'])
    assert {k:v['record'] for k,v in values.items()} == {k:v['record'] for k,v in values0.items()}
    assert D.timing_report(p, values)['concurrent_games'] == 4


def test_hard_kill_preserves_rows_and_resume(world, runtime, concurrent_played):
    matches = setup_matches(world, 'concurrent-killed')
    p = plan(world, matches, runtime, workers=1, execution=batched(1, 2))
    process = run_cli(world['root'], matches, 'run', '--match-id', p['match_id'])
    database = matches / p['match_id'] / 'results.sqlite3'
    deadline = time.monotonic() + 120
    first = {}
    try:
        while time.monotonic() < deadline and not first:
            if database.exists():
                try:
                    first = records(matches, p['match_id'])
                except sqlite3.OperationalError:
                    pass
            if process.poll() is not None:
                break
            time.sleep(.01)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    assert 0 < len(first) < p['planned_games']
    D.run(matches / p['match_id'], world['root'], log=lambda _: None)
    values = records(matches, p['match_id'])
    assert all(values[k] == v for k,v in first.items())
    assert {k:v['record'] for k,v in values.items()} == {k:v['record'] for k,v in concurrent_played[2].items()}


def test_publication_failure_does_not_stop_other_games(world, runtime, monkeypatch, concurrent_played):
    from chess_rl.evaluation.concurrent_matches import collector_for, execute_games
    matches = setup_matches(world, 'publication-fail')
    p = plan(world, matches, runtime, workers=1, execution=batched())
    database = matches / p['match_id'] / 'results.sqlite3'
    M.initialize_store(database, p)
    publish = M.publish
    victim = p['schedule'][0]['game_id']
    def fail(path, job, *args):
        if job['game_id'] == victim:
            raise sqlite3.OperationalError('injected publication failure')
        return publish(path, job, *args)
    monkeypatch.setattr(M, 'publish', fail)
    # This process is already single-thread configured by test fixtures.
    players = D.load_players(p, world['root'], deferred=True)
    with collector_for(p, world['root']) as collector:
        with pytest.raises(RuntimeError, match='publication failures'):
            execute_games(p['schedule'], p, database, players, runtime, collector)
        assert not collector._games and not collector._pending
    values = M.read_results(database, p)
    assert len(values) == p['planned_games'] - 1 and victim not in values
    job = p['schedule'][1]
    existing = values[job['game_id']]
    with pytest.raises(sqlite3.IntegrityError):
        publish(database, job, p, existing['record'], existing['timing'], runtime)
    assert M.read_results(database, p) == values


def test_serial_execution_compatibility_is_normalized_without_changing_resume_identity():
    legacy = dict(workers=5, protocol='production', split='validation')
    serial = dict(legacy, execution=Execution(processes=5).to_dict())
    historical_explicit = dict(serial, execution=Execution(processes=5, implementation='match-batching-v1',
                                              flush_policy='oldest-deadline-v1', max_wait_ms=2).to_dict())
    before = deepcopy(legacy)
    assert execution_for(legacy).validation_identity() == execution_for(serial).validation_identity()
    assert execution_for(historical_explicit).validation_identity() == execution_for(serial).validation_identity()
    assert execution_for(historical_explicit) != execution_for(serial)
    assert legacy == before


def test_legacy_serial_plan_runs_without_rewriting_digest(world, runtime):
    matches = setup_matches(world, 'legacy-execution')
    p = plan(world, matches, runtime, workers=1, count=1, simulations=1, ply_cap=10)
    legacy = {k:v for k,v in p.items() if k not in ('execution','sha256')}
    legacy = M.sealed(legacy)
    path = matches / p['match_id'] / 'plan.json'
    path.write_text(json.dumps(legacy, sort_keys=True))
    original = path.read_bytes()
    assert D.run(path.parent, world['root'], log=lambda _: None)['finished']
    assert path.read_bytes() == original
    values = records(matches, p['match_id'])
    assert all(v['plan_sha256'] == legacy['sha256'] for v in values.values())
    assert D.run(path.parent, world['root'], log=lambda _: None)['finished']
    assert records(matches, p['match_id']) == values


def test_model_batch_failures_publish_immutable_failed_rows(world, runtime, monkeypatch):
    from chess_rl.evaluation.concurrent_matches import collector_for, execute_games
    matches = setup_matches(world, 'failed-rows')
    p = plan(world, matches, runtime, workers=1, execution=batched())
    database = matches / p['match_id'] / 'results.sqlite3'
    M.initialize_store(database, p)
    def broken(*args):
        raise RuntimeError('injected model failure')
    monkeypatch.setattr(D, 'load_reference', broken)
    players = D.load_players(p, world['root'], deferred=True)
    with collector_for(p, world['root']) as collector:
        results = execute_games(p['schedule'], p, database, players, runtime, collector)
    assert all(status == 'failed' for _,status in results)
    before = M.read_results(database, p)
    assert len(before) == p['planned_games']
    assert D.run(database.parent, world['root'], log=lambda _: None)['finished']
    assert M.read_results(database, p) == before


def test_execution_fields_cannot_be_implicitly_defaulted():
    raw = batched().to_dict()
    for field in raw:
        missing = dict(raw)
        missing.pop(field)
        with pytest.raises(ValueError, match='every configuration field'):
            execution_for(dict(execution=missing, workers=1, protocol='fixture', split='development'))
    with pytest.raises(ValueError, match='every configuration field'):
        execution_for(dict(execution=None, workers=1, protocol='fixture', split='development'))


def test_shared_hash_still_checks_every_checkpoint_identity(world, runtime):
    from chess_rl.evaluation.concurrent_matches import collector_for
    from chess_rl.search.batched_evaluator import CollectorUnavailable
    matches = setup_matches(world, 'same-hash-identities')
    p = plan(world, matches, runtime, workers=1, execution=batched())
    p = deepcopy(p)
    p['players'][1] = deepcopy(p['players'][0])
    p['players'][1]['checkpoint_id'] = 'incorrect-id-with-identical-bytes'
    with collector_for(p, world['root']) as collector:
        with pytest.raises(CollectorUnavailable, match='identity differs'):
            collector.wait_ready()
