"""Independent numerical fit diagnostics and bounded-run stopping behavior."""
import math
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from chess_rl.training.tiny_overfit import FitGate, fit_metrics, bounded_fit
from chess_rl.training.trainer import StepMetrics, LossWeights


def inputs():
    policy = torch.zeros((2, 4352))
    mask = torch.zeros_like(policy, dtype=torch.bool)
    mask[:, [10, 20]] = True
    policy[:, 0] = 1000  # illegal argmax would get both rows wrong
    policy[0, 10] = 3
    policy[1, 20] = 3
    q = torch.tensor([[.2, .3, .5], [0., 1., 0.]])
    return policy, torch.tensor([[.1, -.4, .6], [-2., 3., -1.]]), dict(
        legal_mask=mask, policy_classes=torch.tensor([10, 20]), wdl=q)


def test_metrics_match_independent_numpy_entropy_kl_and_ce():
    policy, wdl, tensors = inputs()
    metrics = fit_metrics(policy, wdl, tensors)
    logits, q = wdl.double().numpy(), tensors['wdl'].double().numpy()
    log_p = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    log_q = np.zeros_like(q)
    np.log(q, out=log_q, where=q > 0)
    entropy = -(q * log_q).sum(axis=1)
    kl = (q * (log_q - log_p)).sum(axis=1)
    assert metrics['legal_best_move_accuracy'] == 1
    assert metrics['predicted_legal_move_ids'] == [10, 20]
    assert metrics['teacher_entropy_floor'] == pytest.approx(entropy.mean(), abs=1e-12)
    assert metrics['teacher_wdl_kl_mean'] == pytest.approx(kl.mean(), abs=1e-12)
    assert metrics['per_sample_kl'] == pytest.approx(kl, abs=1e-12)
    assert metrics['wdl_cross_entropy'] == pytest.approx(-(q * log_p).sum(axis=1).mean(), abs=2e-6)
    assert metrics['entropy_identity_error'] <= 2e-6
    assert metrics['policy_ce'] == pytest.approx(math.log1p(math.exp(-3)), abs=1e-7)


def test_teacher_match_has_nonzero_entropy_floor_and_zero_kl():
    policy, _, tensors = inputs()
    tensors['wdl'] = torch.tensor([[.25, .25, .5], [.5, .25, .25]])
    metrics = fit_metrics(policy, tensors['wdl'].log(), tensors)
    assert metrics['wdl_cross_entropy'] > 1
    assert abs(metrics['teacher_wdl_kl_mean']) < 1e-12
    assert FitGate().passes(metrics)


def test_tie_break_uses_smallest_legal_id():
    policy, wdl, tensors = inputs()
    policy[:, 10] = policy[:, 20] = 0
    metrics = fit_metrics(policy, wdl, tensors)
    assert metrics['predicted_legal_move_ids'] == [10, 10]
    assert metrics['legal_best_move_accuracy'] == .5


@pytest.mark.parametrize('field', ['legal_best_move_accuracy', 'teacher_wdl_kl_mean',
    'teacher_wdl_kl_max', 'teacher_wdl_kl_min', 'entropy_identity_error'])
def test_each_gate_requirement_is_mandatory(field):
    m = dict(legal_best_move_accuracy=1., teacher_wdl_kl_mean=.01, teacher_wdl_kl_max=.03,
             teacher_wdl_kl_min=0., entropy_identity_error=2e-6)
    assert FitGate().passes(m)
    m[field] = {'legal_best_move_accuracy': .999, 'teacher_wdl_kl_mean': .010001,
        'teacher_wdl_kl_max': .030001, 'teacher_wdl_kl_min': -3e-6,
        'entropy_identity_error': 3e-6}[field]
    assert not FitGate().passes(m)
    m[field] = float('nan')
    assert not FitGate().passes(m)


@pytest.mark.parametrize('kwargs', [{'step_cap': 0}, {'step_cap': True}, {'mean_kl_max': -1},
    {'mean_kl_max': float('nan')}, {'accuracy_min': 0}, {'accuracy_min': 1.1},
    {'max_kl_max': True}, {'entropy_identity_atol': 0}])
def test_invalid_preexecution_gate(kwargs):
    with pytest.raises(ValueError): FitGate(**kwargs)


def test_nonfinite_logits_and_illegal_teacher_fail():
    policy, wdl, tensors = inputs()
    policy[0, 0] = float('nan')
    with pytest.raises(ValueError): fit_metrics(policy, wdl, tensors)
    policy[0, 0] = 1000
    tensors['policy_classes'][0] = 0
    with pytest.raises(ValueError): fit_metrics(policy, wdl, tensors)


@pytest.mark.parametrize('passing_at,expected_status,expected_step',
    [(None, 'failed_step_cap', 3), (1, 'passed', 1), (3, 'passed', 3)])
def test_cap_retained_or_first_pass_stops(monkeypatch, passing_at, expected_status, expected_step):
    import chess_rl.training.tiny_overfit as fit
    import chess_rl.training.trainer as trainer
    updates = []
    def update(*args, **kwargs):
        updates.append(True)
        return StepMetrics(8, 1., 1., 2.)
    def metrics(*args):
        passing = len(updates) == passing_at
        return dict(legal_best_move_accuracy=1. if passing else .5, teacher_wdl_kl_mean=.001,
                    teacher_wdl_kl_max=.002, teacher_wdl_kl_min=0., entropy_identity_error=0.)
    monkeypatch.setattr(trainer, 'supervised_step', update)
    monkeypatch.setattr(fit, 'measure', metrics)
    rows = []
    gate = FitGate(step_cap=3)
    frozen = asdict(gate)
    result = bounded_fit(None, None, None, weights=LossWeights(), gate=gate, record=rows.append)
    assert result['status'] == expected_status and result['step'] == expected_step
    assert len(updates) == expected_step and len(rows) == expected_step + 1
    assert rows[0]['step'] == 0 and rows[0]['update_losses'] is None
    assert asdict(gate) == frozen
