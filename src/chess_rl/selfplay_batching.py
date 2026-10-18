"""One frozen model per worker, bounded independent game threads, unchanged Sessions."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import resource
import sys
import time

from chess_rl.search.batched_evaluator import EvaluationCollector
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.selfplay import _read, _write
from chess_rl.selfplay_execution import Execution, binding, session_runtime


def validate_acceptance(record, plan, current_runtime=None):
    from chess_rl.evaluation.matches import digest
    content = {k: v for k, v in record.items() if k != "sha256"}
    execution = Execution.from_dict(plan["execution"])
    if (record.get("sha256") != digest(content)
            or record.get("schema") != "selfplay-numerical-acceptance-v1"
            or record.get("accepted") is not True
            or record.get("checkpoint_sha256") != plan["checkpoint"]["sha256"]
            or record.get("batch_size") != execution.batch_size
            or record.get("source_sha256") != plan["execution_source"]["sha256"]
            or record.get("runtime", {}).get("device") != plan["device"]
            or (current_runtime is not None and record["runtime"] != current_runtime)):
        raise ValueError("numerical acceptance differs from checkpoint/source/batch/runtime")


def freeze_runtime(directory, plan):
    current = session_runtime(plan["device"], binding(plan))
    if "numerical_acceptance" in plan:
        validate_acceptance(plan["numerical_acceptance"], plan, current)
    path = Path(directory) / "execution-runtime.json"
    record = dict(runtime=current, binding=binding(plan))
    try:
        _write(path, record)
    except FileExistsError:
        if _read(path) != record:
            raise ValueError("campaign execution runtime differs from frozen runtime")
    return current


class BoundEvaluator:
    def __init__(self, channel, checkpoint, key):
        self.channel, self.checkpoint, self.key = channel, checkpoint, key

    def __call__(self, states):
        return self.channel(self.key, states)


@contextmanager
def collector_for(plan):
    execution = Execution.from_dict(plan["execution"])
    key = plan["checkpoint"]["sha256"]
    loaded = {}

    def factory():
        from chess_rl.selfplay_campaign import _sha256
        path = plan["checkpoint"]["path"]
        if _sha256(path) != key:
            raise ValueError("checkpoint bytes changed before loading")
        reference = RealEvaluator(path, device=plan["device"])
        if _sha256(path) != key:
            raise ValueError("checkpoint bytes changed during loading")
        loaded["checkpoint"] = reference.checkpoint
        return reference

    with EvaluationCollector({key: factory}, batch_size=execution.batch_size,
                             games=execution.games_per_worker, max_wait_ms=execution.max_wait_ms,
                             timeout_seconds=execution.timeout_seconds,
                             flush_policy=execution.flush_policy) as collector:
        collector.wait_ready()
        yield collector, loaded["checkpoint"]


def bounded_games(jobs, limit, play):
    """Only K live futures/threads; sleeping Events/Conditions release the GIL.

    Python tree work still shares a GIL inside a worker. Separate worker processes
    supply CPU parallelism; K is inference overlap, not K CPU cores.
    """
    iterator = iter(jobs)
    with ThreadPoolExecutor(max_workers=limit, thread_name_prefix="selfplay-game") as pool:
        active = {pool.submit(play, job) for job in (next(iterator, None) for _ in range(limit))
                  if job is not None}
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                job = next(iterator, None)
                if job is not None:
                    active.add(pool.submit(play, job))


def memory():
    import torch
    return dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None)


def play_group(tasks, plan, directory):
    from chess_rl.selfplay_campaign import _play
    started = time.monotonic()
    current = freeze_runtime(directory, plan)
    execution = Execution.from_dict(plan["execution"])
    with collector_for(plan) as (collector, checkpoint):
        def play(task):
            with collector.game(task["index"]) as channel:
                channel.expect(plan["checkpoint"]["sha256"])
                evaluator = BoundEvaluator(channel, checkpoint, plan["checkpoint"]["sha256"])
                return _play(task, evaluator=evaluator, execution=binding(plan))
        results = list(bounded_games(tasks, execution.games_per_worker, play))
        metrics = deepcopy(collector.metrics)
    return dict(game_results=results, collector=metrics, pid=os.getpid(),
                runtime=current, memory=memory(), seconds=time.monotonic() - started)
