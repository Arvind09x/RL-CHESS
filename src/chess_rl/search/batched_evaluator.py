"""Experimental fixed-shape match inference; RealEvaluator owns all numerics."""
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import threading
import time

from chess_rl.adapter import make_state, terminal_result


class FixedBatchEvaluator:
    """Pad actual nonterminal model inputs to B, discard padding results.

    RealEvaluator retains terminal, encoding, mask, finite, policy, WDL and
    float32 ownership validation. Padding is not proof of reproducibility.
    Only the collector constructs/invokes the owned reference evaluator.
    """
    def __init__(self, reference, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.reference, self.batch_size = reference, batch_size
        self.padding = make_state()

    def __call__(self, states):
        results = [terminal_result(s) for s in states]
        pending = [(i, s) for i, (s, r) in enumerate(zip(states, results)) if r is None]
        if len(pending) > self.batch_size:
            raise ValueError("too many nonterminal inputs for fixed batch")
        if pending:
            inputs = [s for _, s in pending] + [self.padding] * (self.batch_size - len(pending))
            outputs = self.reference(inputs)
            if len(outputs) != self.batch_size:
                raise ValueError("reference returned wrong batch length")
            for (index, _), output in zip(pending, outputs):
                results[index] = output
        return results


class CollectorUnavailable(RuntimeError):
    pass


@dataclass
class _Request:
    identity: tuple
    checkpoint: str
    state: object
    submitted: float
    ready: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: BaseException | None = None


class _GameChannel:
    def __init__(self, collector, game_id, token):
        self.collector, self.game_id, self.token = collector, game_id, token

    def expect(self, checkpoint):
        self.collector.expect(self.game_id, self.token, checkpoint)

    def __call__(self, checkpoint, states):
        return self.collector.evaluate(self.game_id, self.token, checkpoint, states)


class EvaluationCollector:
    """One model-owning thread; at most K games, one request per game.

    Ready queues flush at B, when all registered games are waiting, or at
    the oldest deadline. Oldest ready queue wins; no cross-model batching. Model/delivery failure poisons the collector and
    releases all waiters. close releases waiters even if a native call hangs;
    the daemon owner cannot publish late results. Discard that worker process.
    """
    def __init__(self, factories, *, batch_size, games, max_wait_ms=20.0, timeout_seconds=120.0,
                 flush_policy="all-waiting-or-deadline-v1"):
        if not factories or any(not callable(f) for f in factories.values()):
            raise ValueError("checkpoint factories required")
        if any(not isinstance(k, str) or len(k) != 64 or any(c not in '0123456789abcdef' for c in k)
               for k in factories):
            raise ValueError("checkpoint keys must be SHA-256 identities")
        if any(type(x) is not int or x < 1 for x in (batch_size, games)):
            raise ValueError("batch size and games must be positive integers")
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in (max_wait_ms, timeout_seconds)):
            raise ValueError("finite positive flush wait and timeout required")
        if flush_policy not in ("oldest-deadline-v1", "all-waiting-or-deadline-v1"):
            raise ValueError("unsupported flush policy")
        self.flush_policy = flush_policy
        self.factories = dict(factories)
        self.batch_size, self.capacity = batch_size, games
        self.max_wait, self.timeout = max_wait_ms / 1000, timeout_seconds
        self._condition = threading.Condition()
        self._games, self._pending = {}, {}
        self._queues = defaultdict(deque)
        self._closed, self._error = False, None
        self.metrics = dict(batches=0, requests=0, padded_slots=0, queue_seconds=0.0,
                            inference_seconds=0.0, peak_pending=0, per_checkpoint={},
                            flush_counts=dict(full=0, all_waiting=0, deadline=0))
        self._startup = threading.Event()
        self._thread = threading.Thread(target=self._collect, name="match-model-owner", daemon=True)
        self._thread.start()

    @contextmanager
    def game(self, game_id):
        token = object()
        with self._condition:
            if game_id in self._games or len(self._games) >= self.capacity:
                raise ValueError("duplicate game or collector game capacity exceeded")
            self._games[game_id] = [token, 0, None]
            self._condition.notify_all()
        try:
            yield _GameChannel(self, game_id, token)
        finally:
            with self._condition:
                registration = self._games.get(game_id)
                if registration is not None and registration[0] is token:
                    self._games.pop(game_id)
                    request = self._pending.pop(game_id, None)
                    if request is not None:
                        self._settle(request, error=CollectorUnavailable("game unregistered"))
                        self._queues[request.checkpoint] = deque(
                            r for r in self._queues[request.checkpoint] if r is not request)
                    self._condition.notify_all()

    def expect(self, game_id, token, checkpoint):
        """Register the model for the coming turn *before* CPU tree work."""
        with self._condition:
            registration = self._games.get(game_id)
            if registration is None or registration[0] is not token or checkpoint not in self.factories:
                raise ValueError("unregistered game or unknown checkpoint")
            if game_id in self._pending:
                raise ValueError("game already has an outstanding evaluation")
            registration[2] = checkpoint
            self._condition.notify_all()

    def evaluate(self, game_id, token, checkpoint, states):
        if len(states) != 1:
            raise ValueError("one outstanding leaf per game is required")
        state = states[0].copy()  # Owned history with full adapter validation.
        with self._condition:
            registration = self._games.get(game_id)
            if registration is None or registration[0] is not token:
                raise ValueError("unregistered game")
            if game_id in self._pending:
                raise ValueError("game already has an outstanding evaluation")
            if checkpoint not in self.factories:
                raise ValueError("unknown checkpoint")
            if self._closed:
                raise CollectorUnavailable(str(self._error or "collector closed"))
            registration[2] = checkpoint
            registration[1] += 1
            request = _Request((game_id, token, registration[1]), checkpoint, state, time.monotonic())
            self._pending[game_id] = request
            self._queues[checkpoint].append(request)
            self.metrics['peak_pending'] = max(self.metrics['peak_pending'], len(self._pending))
            self._condition.notify_all()
        if not request.ready.wait(self.timeout):
            self._fail(CollectorUnavailable("inference request timed out; discard worker"))
        with self._condition:
            if self._pending.get(game_id) is request:
                self._pending.pop(game_id)
                self._condition.notify_all()
            if request.error is not None:
                raise CollectorUnavailable(str(request.error)) from request.error
            return [request.result]

    @staticmethod
    def _settle(request, result=None, error=None):
        if not request.ready.is_set():
            request.result, request.error = result, error
            request.ready.set()

    def _deliver(self, request, result):
        request.result = result

    def _fail(self, exc):
        with self._condition:
            self._closed, self._error = True, exc
            for request in self._pending.values():
                self._settle(request, error=exc)
            self._queues.clear()
            self._condition.notify_all()

    def _collect(self):
        try:
            evaluators = {k: FixedBatchEvaluator(f(), self.batch_size) for k, f in self.factories.items()}
            self._startup.set()
            while True:
                with self._condition:
                    while not self._closed and not any(self._queues.values()):
                        self._condition.wait()
                    if self._closed:
                        return
                    for queue in self._queues.values():
                        while queue and queue[0].ready.is_set():
                            queue.popleft()
                    available = [(q[0].submitted, k) for k, q in self._queues.items() if q]
                    if not available:
                        continue
                    now = time.monotonic()
                    ready = []
                    for submitted, key in available:
                        queue = self._queues[key]
                        expected = [game_id for game_id, registration in self._games.items() if registration[2] == key]
                        all_waiting = (self.flush_policy == "all-waiting-or-deadline-v1" and bool(expected)
                                       and all(game_id in self._pending and not self._pending[game_id].ready.is_set()
                                               for game_id in expected))
                        reason = ("full" if len(queue) >= self.batch_size else
                                  "all_waiting" if all_waiting else
                                  "deadline" if submitted + self.max_wait <= now else None)
                        if reason is not None:
                            ready.append((submitted, key, reason))
                    if not ready:
                        deadline = min(submitted + self.max_wait for submitted, _ in available)
                        self._condition.wait(max(0.0, deadline - now))
                        continue
                    _, checkpoint, reason = min(ready)
                    queue = self._queues[checkpoint]
                    self.metrics['flush_counts'][reason] += 1
                    batch = [queue.popleft() for _ in range(min(self.batch_size, len(queue)))]
                    now = time.monotonic()
                    self.metrics['queue_seconds'] += sum(now - r.submitted for r in batch)
                started = time.monotonic()
                outputs = evaluators[checkpoint]([r.state for r in batch])
                if len(outputs) != len(batch):
                    raise ValueError("result count differs from request count")
                with self._condition:
                    if self._closed:
                        return
                    self.metrics['batches'] += 1
                    self.metrics['requests'] += len(batch)
                    self.metrics['padded_slots'] += self.batch_size - sum(terminal_result(r.state) is None for r in batch)
                    self.metrics['inference_seconds'] += time.monotonic() - started
                    counts = self.metrics['per_checkpoint']
                    counts[checkpoint] = counts.get(checkpoint, 0) + len(batch)
                    for request, result in zip(batch, outputs):
                        self._deliver(request, result)
                    # Deliver atomically: no waiter consumes a partially delivered batch.
                    for request in batch:
                        self._settle(request, result=request.result)
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._startup.set()
            self._fail(self._error or CollectorUnavailable("collector stopped"))

    def wait_ready(self):
        if not self._startup.wait(self.timeout):
            self._fail(CollectorUnavailable("model startup timed out"))
        with self._condition:
            if self._closed:
                raise CollectorUnavailable(str(self._error))

    def close(self):
        self._fail(CollectorUnavailable("collector closed"))
        self._thread.join(timeout=min(self.timeout, 1.0))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
