"""Collector lifecycle/routing and strict fixed-batch numerical acceptance."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from chess_rl.adapter import Evaluation, TerminalResult, legal_mask, make_state
from chess_rl.search.batched_evaluator import CollectorUnavailable, EvaluationCollector, FixedBatchEvaluator
from chess_rl.search.real_evaluator import RealEvaluator
from chess_rl.search.puct import Node
from chess_rl.search.runner import search
from .test_checkpoints import saved
from .test_evaluator import nonterminal_cases, terminal_cases, preserved


@pytest.fixture(scope='module', autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def identical(a, b):
    return (a.priors.tobytes() == b.priors.tobytes() and a.wdl.tobytes() == b.wdl.tobytes()
            and np.float64(a.value).tobytes() == np.float64(b.value).tobytes())


def test_reference_agreement_special_moves_histories_and_search(saved):
    reference = RealEvaluator(saved['path'])
    fixed = FixedBatchEvaluator(reference, 4)
    for state in nonterminal_cases():
        before = preserved(state)
        a, = reference([state])
        b, = fixed([state])
        np.testing.assert_allclose(a.priors, b.priors, atol=1e-5, rtol=0)
        np.testing.assert_allclose(a.wdl, b.wdl, atol=1e-5, rtol=0)
        assert abs(a.value - b.value) <= 1e-5
        assert b.priors.dtype == b.wdl.dtype == np.float32
        assert b.priors.flags.owndata and b.wdl.flags.owndata
        assert search(Node(state), 3, evaluator=reference).move_id == search(Node(state), 3, evaluator=fixed).move_id
        assert preserved(state) == before


def test_terminals_filter_before_actual_fixed_network_padding(saved, monkeypatch):
    reference = RealEvaluator(saved['path'])
    sizes = []
    forward = reference.checkpoint.model.forward
    def checked(features):
        sizes.append(len(features))
        return forward(features)
    monkeypatch.setattr(reference.checkpoint.model, 'forward', checked)
    fixed = FixedBatchEvaluator(reference, 4)
    terminal = terminal_cases()[0][0]
    assert fixed([]) == []
    assert isinstance(fixed([terminal])[0], TerminalResult)
    results = fixed([terminal, make_state(), terminal])
    assert sizes == [4]
    assert isinstance(results[1], Evaluation)
    with pytest.raises(ValueError, match='too many'):
        fixed([make_state()] * 5)


def numerical_probe(reference, batch_size=4):
    """Return failures as evidence; never loosen tolerance or round outputs."""
    fixed = FixedBatchEvaluator(reference, batch_size)
    cases = nonterminal_cases()
    failures, comparisons = [], 0
    for position, state in enumerate(cases):
        baseline, = fixed([state])
        for repeat in range(2):
            for count in range(1, batch_size + 1):
                for slot in range(count):
                    inputs = [cases[(position + repeat + i + 1) % len(cases)] for i in range(count)]
                    inputs[slot] = state
                    result = fixed(inputs)[slot]
                    comparisons += 1
                    if not identical(baseline, result):
                        failures.append(dict(position=position, slot=slot, real_requests=count, repeat=repeat,
                                             prior_max_abs=float(np.max(np.abs(baseline.priors-result.priors))),
                                             wdl_max_abs=float(np.max(np.abs(baseline.wdl-result.wdl)))))
    return dict(bit_identical=not failures, comparisons=comparisons, failures=failures)


def test_fixed_batch_scheduling_independence_is_measured(saved):
    report = numerical_probe(RealEvaluator(saved['path']))
    # CPU requires exact returned bytes. CUDA has a separate acceptance job.
    assert report['comparisons'] == 260
    assert report['bit_identical'], report['failures'][:3]


class Tagged:
    def __init__(self, tag, seen, fail=False, gate=None):
        self.tag, self.seen, self.fail, self.gate = tag, seen, fail, gate
        self.owner = threading.get_ident()
    def __call__(self, states):
        assert threading.get_ident() == self.owner
        self.seen.append((self.tag, len(states)))
        if self.gate:
            self.gate.wait(2)
        if self.fail:
            raise RuntimeError('broken model batch')
        return [(self.tag, s.board.fen()) for s in states]


def test_routing_two_checkpoints_partial_batches_completion_and_snapshot():
    seen, keys = [], ['a'*64, 'b'*64]
    with EvaluationCollector({k: lambda k=k: Tagged(k, seen) for k in keys}, batch_size=4, games=6) as collector:
        def game(index):
            state = make_state(moves_uci=['e2e4'] if index % 2 else [])
            expected = state.board.fen()
            with collector.game(str(index)) as evaluate:
                for _ in range(index % 3 + 1):
                    result, = evaluate(keys[index % 2], [state])
                    assert result == (keys[index % 2], expected)
            return index
        with ThreadPoolExecutor(6) as pool:
            assert sorted(pool.map(game, range(6))) == list(range(6))
        assert not collector._games and not collector._pending
        assert collector.metrics['peak_pending'] > 1
        assert {tag for tag, _ in seen} == set(keys)
        assert all(size == 4 for _, size in seen)


@pytest.mark.parametrize('stage', ['load', 'inference', 'delivery'])
def test_failure_releases_all_waiters_and_rejects_future_requests(stage, monkeypatch):
    seen, key = [], 'a'*64
    start = threading.Event()
    def factory():
        start.wait(2)
        if stage == 'load':
            raise RuntimeError('broken loader')
        return Tagged(key, seen, fail=stage == 'inference')
    with EvaluationCollector({key: factory}, batch_size=4, games=4, timeout_seconds=3) as collector:
        if stage == 'delivery':
            def broken(*args):
                raise RuntimeError('broken delivery')
            monkeypatch.setattr(collector, '_deliver', broken)
        def game(i):
            with collector.game(str(i)) as evaluate:
                with pytest.raises(CollectorUnavailable, match='broken'):
                    evaluate(key, [make_state()])
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(game, i) for i in range(4)]
            start.set()
            for f in futures:
                f.result(timeout=5)
        with collector.game('later') as evaluate:
            with pytest.raises(CollectorUnavailable):
                evaluate(key, [make_state()])
        assert not collector._pending


def test_submission_failure_is_local_and_shutdown_releases_native_waiter():
    seen, key, gate = [], 'a'*64, threading.Event()
    collector = EvaluationCollector({key: lambda: Tagged(key, seen, gate=gate)}, batch_size=2, games=2)
    with collector.game('one') as evaluate, collector.game('two') as other:
        with pytest.raises(ValueError, match='unknown checkpoint'):
            evaluate('b'*64, [make_state()])
        with pytest.raises(ValueError, match='one outstanding'):
            evaluate(key, [])
        with ThreadPoolExecutor(2) as pool:
            f = pool.submit(evaluate, key, [make_state()])
            deadline = time.monotonic() + 2
            while not seen and time.monotonic() < deadline:
                time.sleep(.005)
            with pytest.raises(ValueError, match='outstanding'):
                evaluate(key, [make_state()])
            with pytest.raises(ValueError, match='capacity'):
                with collector.game('three'):
                    pass
            collector.close()
            with pytest.raises(CollectorUnavailable, match='closed'):
                f.result(timeout=2)
            gate.set()
    collector.close()


def test_near_ties_keep_reference_tie_breaking(saved, monkeypatch):
    reference = RealEvaluator(saved['path'])
    fixed = FixedBatchEvaluator(reference, 4)
    state = make_state()
    ids = np.flatnonzero(legal_mask(state))
    for difference in (0.0, 1e-6, -1e-6):
        def forward(features):
            policy = torch.zeros((len(features), 4352), dtype=features.dtype, device=features.device)
            policy[:, int(ids[1])] = difference
            return policy, torch.zeros((len(features), 3), dtype=features.dtype, device=features.device)
        monkeypatch.setattr(reference.checkpoint.model, 'forward', forward)
        a = search(Node(state), 1, evaluator=reference)
        b = search(Node(state), 1, evaluator=fixed)
        assert a.move_id == b.move_id == int(ids[1] if difference > 0 else ids[0])


def test_owned_snapshot_unregister_and_explicit_cancellation():
    seen, key, gate = [], 'a'*64, threading.Event()
    with EvaluationCollector({key: lambda: Tagged(key, seen, gate=gate)}, batch_size=2, games=1) as collector:
        context = collector.game('reused-name')
        evaluate = context.__enter__()
        state = make_state()
        original = state.board.fen()
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(evaluate, key, [state])
            deadline = time.monotonic()+2
            while not seen and time.monotonic()<deadline:
                time.sleep(.002)
            state.board.push_uci('e2e4')
            gate.set()
            assert future.result(timeout=3) == [(key, original)]
            gate.clear()
            seen.clear()
            cancelled = pool.submit(evaluate, key, [state])
            while not seen and time.monotonic()<deadline+2:
                time.sleep(.002)
            context.__exit__(None, None, None)
            with pytest.raises(CollectorUnavailable, match='unregistered'):
                cancelled.result(timeout=3)
            gate.set()
        with collector.game('reused-name') as new_evaluate:
            with pytest.raises(ValueError, match='unregistered'):
                evaluate(key, [make_state()])
            assert new_evaluate(key, [make_state()]) == [(key, original)]


def test_partial_delivery_failure_is_atomic(monkeypatch):
    key, start, seen = 'a'*64, threading.Event(), []
    def factory():
        start.wait(3)
        return Tagged(key, seen)
    with EvaluationCollector({key:factory}, batch_size=4, games=4) as collector:
        delivered = []
        def deliver(request, result):
            delivered.append(request.identity)
            if len(delivered) == 2:
                raise RuntimeError('partial delivery failure')
            request.result = result
        monkeypatch.setattr(collector, '_deliver', deliver)
        def game(i):
            with collector.game(str(i)) as evaluate:
                with pytest.raises(CollectorUnavailable, match='partial delivery'):
                    evaluate(key, [make_state()])
        with ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(game,i) for i in range(4)]
            deadline = time.monotonic()+3
            while len(collector._pending) < 4 and time.monotonic()<deadline:
                time.sleep(.002)
            start.set()
            for f in futures:
                f.result(timeout=5)


def test_model_timeout_fails_closed_and_releases_waiter():
    key, seen, gate = 'a'*64, [], threading.Event()
    with EvaluationCollector({key:lambda:Tagged(key,seen,gate=gate)}, batch_size=2, games=1,
                             timeout_seconds=.05) as collector:
        with collector.game('one') as evaluate:
            with pytest.raises(CollectorUnavailable, match='timed out'):
                evaluate(key, [make_state()])
            with pytest.raises(CollectorUnavailable, match='timed out'):
                evaluate(key, [make_state()])
        gate.set()


def test_flushes_when_all_games_for_one_checkpoint_wait_even_if_other_model_is_busy():
    keys, seen = ['a'*64, 'b'*64], []
    with EvaluationCollector({k:lambda k=k:Tagged(k,seen) for k in keys}, batch_size=4,
                             games=3, max_wait_ms=10000) as collector:
        with collector.game('a1') as first, collector.game('a2') as second, collector.game('b') as other:
            first.expect(keys[0]); second.expect(keys[0]); other.expect(keys[1])
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(channel, keys[0], [make_state()]) for channel in (first,second)]
                for future in futures:
                    assert future.result(timeout=2)[0][0] == keys[0]
            assert collector.metrics['flush_counts']['all_waiting'] >= 1
            assert not any(tag == keys[1] for tag,_ in seen)


@pytest.mark.parametrize('policy,expected', [('all-waiting-or-deadline-v1','all_waiting'),
                                           ('oldest-deadline-v1','deadline')])
def test_legacy_flush_policy_remains_explicit(policy, expected):
    key, seen = 'a'*64, []
    with EvaluationCollector({key:lambda:Tagged(key,seen)}, batch_size=4, games=1,
                             max_wait_ms=5, flush_policy=policy) as collector:
        with collector.game('one') as evaluate:
            evaluate.expect(key)
            assert evaluate(key,[make_state()])[0][0] == key
        assert collector.metrics['flush_counts'][expected] == 1


def test_full_batch_flushes_with_another_game_still_producing_a_leaf():
    key, seen = 'a'*64, []
    with EvaluationCollector({key:lambda:Tagged(key,seen)}, batch_size=2, games=3,
                             max_wait_ms=10000) as collector:
        with ExitStack() as stack:
            channels = [stack.enter_context(collector.game(str(i))) for i in range(3)]
            for channel in channels:
                channel.expect(key)
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(channel,key,[make_state()]) for channel in channels[:2]]
                for f in futures:
                    f.result(timeout=2)
            assert collector.metrics['flush_counts']['full'] == 1


def test_deadline_and_unregister_both_release_a_partial_checkpoint_batch():
    key, seen = 'a'*64, []
    with EvaluationCollector({key:lambda:Tagged(key,seen)}, batch_size=4, games=2,
                             max_wait_ms=10) as collector:
        with collector.game('one') as first, collector.game('two') as other:
            first.expect(key); other.expect(key)
            first(key,[make_state()])
            assert collector.metrics['flush_counts']['deadline'] == 1
    with EvaluationCollector({key:lambda:Tagged(key,seen)}, batch_size=4, games=2,
                             max_wait_ms=10000) as collector:
        with collector.game('one') as first:
            context = collector.game('two')
            other = context.__enter__()
            first.expect(key); other.expect(key)
            with ThreadPoolExecutor(1) as pool:
                f = pool.submit(first,key,[make_state()])
                context.__exit__(None,None,None)
                f.result(timeout=2)
            assert collector.metrics['flush_counts']['all_waiting'] == 1
