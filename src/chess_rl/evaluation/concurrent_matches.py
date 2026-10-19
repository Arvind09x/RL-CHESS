"""Bounded game-thread scheduling inside one already-configured worker."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
from functools import partial
import time

from chess_rl.search.batched_evaluator import EvaluationCollector
from .execution import execution_for
from .players import SearchPlayer


def collector_for(plan, root):
    from .development_matches import load_reference, sha_file
    from pathlib import Path
    execution = execution_for(plan)
    groups = {}
    for entry in plan['players']:
        if entry['type'] == 'search':
            groups.setdefault(entry['sha256'], []).append(entry)

    def load(entries):
        for entry in entries:
            if sha_file(Path(root) / entry['path']) != entry['sha256']:
                raise ValueError('checkpoint bytes differ before grouped loading')
        reference = load_reference(entries[0], root, plan['device'])
        for entry in entries:
            if (entry['checkpoint_id'] != reference.checkpoint.checkpoint_id or
                    sha_file(Path(root) / entry['path']) != entry['sha256']):
                raise ValueError('checkpoint identity differs after grouped loading')
        if plan.get('development_benchmark'):
            from chess_rl.adapter import make_state
            from chess_rl.search.batched_evaluator import FixedBatchEvaluator
            fixed = FixedBatchEvaluator(reference, execution.batch_size)
            for _ in range(plan['development_benchmark']['warmup_calls']):
                fixed([make_state()])
        return reference
    factories = {sha: partial(load, entries) for sha, entries in groups.items()}
    return EvaluationCollector(factories, batch_size=execution.batch_size,
                               games=execution.games_per_process, max_wait_ms=execution.max_wait_ms,
                               timeout_seconds=execution.timeout_seconds, flush_policy=execution.flush_policy)


def execute_games(jobs, plan, database, players, runtime, collector):
    from . import match_campaign as M
    from .matches import play_game
    limit = execution_for(plan).games_per_process

    class IntentPlayer:
        def __init__(self, player, channel):
            self.player, self.channel = player, channel

        def metadata(self):
            return self.player.metadata()

        def choose(self, state, *, seed):
            self.channel.expect(self.player.checkpoint.sha256)
            return self.player.choose(state, seed=seed)

    def play(job):
        # Publication opens a fresh SQLite connection in this game thread.
        with collector.game(job['game_id']) as evaluate:
            bound = tuple(IntentPlayer(replace(p, evaluator=partial(evaluate, p.checkpoint.sha256)), evaluate)
                          if isinstance(p, SearchPlayer) else p for p in players)
            left, right = bound
            white, black = (left, right) if job['leg'] == 0 else (right, left)
            started_utc, started = M._now(), time.monotonic()
            record = play_game(M.opening(job['opening']), white, black, pair_id=job['pair_id'],
                               leg=job['leg'], game_seed=job['game_seed'], ply_cap=plan['ply_cap'])
            timing = dict(elapsed_seconds=time.monotonic() - started, started_utc=started_utc,
                          finished_utc=M._now(),
                          plies_played=len(record['moves_uci']) - len(job['opening']['moves_uci']))
            M.publish(database, job, plan, record, timing, runtime)
            return job['game_id'], record['status']

    completed, errors = [], []
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=limit, thread_name_prefix='match-game') as pool:
        active = {pool.submit(play, job) for job in (next(iterator, None) for _ in range(limit)) if job is not None}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    completed.append(future.result())
                except Exception as exc:
                    # A failed publication is not retried; allow unrelated games
                    # to commit before reporting the worker error to coordinator.
                    errors.append(exc)
                job = next(iterator, None)
                if job is not None:
                    active.add(pool.submit(play, job))
    if errors:
        raise RuntimeError(f'{len(errors)} game scheduling/publication failures') from errors[0]
    return completed
