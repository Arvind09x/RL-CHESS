"""Grouped AdamW, clipping, warmup/cosine schedule and their checkpoint continuation."""
from dataclasses import asdict, replace
import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.model import ChessNetwork
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (CheckpointError, DataOrder, ModelConfig, Progress,
    load_inference, restore_training, save_checkpoint)
from chess_rl.training.mixed import validate_optimizer_settings
from chess_rl.training.mixed_state import MixedSettings
from chess_rl.training.optimizers import (OptimizerSettings, ScheduleSettings, build_optimizer,
    clip_gradients, describe, gradient_norm, parameter_groups, set_learning_rate)
from chess_rl.training.trainer import LossWeights, StepMetrics, supervised_step
from .test_supervised import fixture_bundle
from .test_training import owned_batch


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_parameter_groups_cover_every_parameter_once_with_sourced_membership():
    model = ChessNetwork()
    decay, no_decay = parameter_groups(model)
    named = dict(model.named_parameters())
    assert sorted(decay + no_decay) == sorted(named) and not set(decay) & set(no_decay)
    assert all(named[n].ndim >= 2 for n in decay)
    assert all(named[n].ndim < 2 or "norm" in n.lower() or n.endswith("square_embeddings")
               or n.endswith((".relative.q", ".relative.k", ".relative.v")) for n in no_decay)
    assert "backbone.square_embeddings" in no_decay
    assert any(n.endswith(".relative.q") for n in no_decay)
    assert any(n.endswith("in_proj_weight") for n in decay)
    assert all(n.endswith(("bias", "weight")) or n in no_decay for n in named)
    assert sum(named[n].numel() for n in decay) > sum(named[n].numel() for n in no_decay)
    assert parameter_groups(model) == (decay, no_decay)  # deterministic order
    with pytest.raises(ValueError, match="ChessNetwork"):
        parameter_groups(torch.nn.Linear(2, 2))


def test_build_optimizer_is_grouped_decoupled_adamw_with_determinism_flags():
    model = ChessNetwork()
    settings = OptimizerSettings(learning_rate=2e-4, weight_decay=0.05)
    optimizer = build_optimizer(model, settings)
    assert type(optimizer) is torch.optim.AdamW
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.05, 0.0]
    assert all(g["lr"] == 2e-4 and g["betas"] == (0.9, 0.999) and g["eps"] == 1e-8
               and g["foreach"] is False and g["fused"] is False and g["decoupled_weight_decay"]
               for g in optimizer.param_groups)
    supplied = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert sorted(supplied) == sorted(id(p) for p in model.parameters())
    record = describe(settings, ScheduleSettings(warmup_steps=3, total_steps=10))
    assert record["kind"] == "AdamW" and record["weight_decay"] == 0.05 and record["schedule"]["total_steps"] == 10
    with pytest.raises(ValueError):
        OptimizerSettings(weight_decay=-1)
    with pytest.raises(ValueError):
        OptimizerSettings(max_gradient_norm=0)
    with pytest.raises(ValueError, match="OptimizerSettings"):
        build_optimizer(model, dict(lr=1e-4))


def test_schedule_warmup_cosine_floor_and_validation():
    schedule = ScheduleSettings(warmup_steps=4, total_steps=12, final_fraction=0.1)
    values = [schedule.multiplier(step) for step in range(14)]
    assert values[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert values[4] == pytest.approx(1.0) and values[8] == pytest.approx(0.55)
    assert values[12] == pytest.approx(0.1) and values[13] == pytest.approx(0.1)  # clamped past total
    assert all(values[i] >= values[i + 1] for i in range(4, 13))
    assert ScheduleSettings(warmup_steps=0, total_steps=1).multiplier(0) == pytest.approx(1.0)
    assert ScheduleSettings(warmup_steps=5, total_steps=5).multiplier(7) == 1.0
    with pytest.raises(ValueError):
        ScheduleSettings(warmup_steps=-1, total_steps=5)
    with pytest.raises(ValueError):
        schedule.multiplier(-1)
    optimizer = build_optimizer(ChessNetwork(), OptimizerSettings())
    set_learning_rate(optimizer, 3e-5)
    assert all(g["lr"] == 3e-5 for g in optimizer.param_groups)
    with pytest.raises(ValueError):
        set_learning_rate(optimizer, 0.0)


def test_clipping_scales_to_the_bound_and_reports_the_unclipped_norm():
    torch.manual_seed(5)
    model = ChessNetwork()
    optimizer = build_optimizer(model, OptimizerSettings())
    batch = owned_batch()
    metrics = supervised_step(model, optimizer, batch, max_gradient_norm=None)
    assert isinstance(metrics, StepMetrics) and metrics.gradient_norm > 0 and metrics.clipped is False
    unclipped = metrics.gradient_norm
    # Recompute on a fresh copy with a bound below the observed norm.
    torch.manual_seed(5)
    model = ChessNetwork()
    optimizer = build_optimizer(model, OptimizerSettings())
    bound = unclipped / 2
    parameters = list(model.parameters())
    # Direct unit check of the clip itself.
    model.zero_grad()
    for p in parameters:
        p.grad = torch.full_like(p, 0.01)
    before = gradient_norm(parameters)
    norm, clipped = clip_gradients(parameters, before / 4)
    assert norm == pytest.approx(before) and clipped
    assert gradient_norm(parameters) == pytest.approx(before / 4, rel=1e-4)
    norm, clipped = clip_gradients(parameters, None)
    assert not clipped and gradient_norm(parameters) == pytest.approx(before / 4, rel=1e-4)
    model.zero_grad(set_to_none=True)
    metrics = supervised_step(model, optimizer, batch, max_gradient_norm=bound)
    assert metrics.gradient_norm == pytest.approx(unclipped, rel=1e-6) and metrics.clipped is True
    assert gradient_norm(parameters) == pytest.approx(bound, rel=1e-4)
    with pytest.raises(ValueError, match="max_gradient_norm"):
        supervised_step(model, optimizer, batch, max_gradient_norm=-1)


def _train(model, optimizer, dataset, order, steps, schedule, peak, clip):
    for step in range(steps):
        set_learning_rate(optimizer, peak * schedule.multiplier(step))
        indices = order.next_indices()
        supervised_step(model, optimizer, dataset.batch(indices), max_gradient_norm=clip)
        order = replace(order, cursor=order.cursor + len(indices))
        if not order.next_indices():
            order = replace(order, cursor=0)
    return order


def test_adamw_checkpoint_round_trip_continues_bitwise(tmp_path):
    dataset = open_dataset(**fixture_bundle(tmp_path), split="train")
    config = ModelConfig(d_policy=32, d_value=48, dropout=.27)
    schedule, peak, clip = ScheduleSettings(warmup_steps=2, total_steps=6, final_fraction=0.2), 1e-3, 0.5
    settings = OptimizerSettings(learning_rate=peak, weight_decay=0.03, max_gradient_norm=clip)
    weights = LossWeights()
    provenance = {"optimizer": describe(settings, schedule)}

    def start():
        random.seed(11); np.random.seed(12); torch.manual_seed(13)
        model = config.build()
        optimizer = build_optimizer(model, settings)
        order = DataOrder("fixture-source", DataOrder.reference_hash(dataset), (2, 0, 1), 0, 2, False, 0)
        return model, optimizer, order

    # Uninterrupted reference: four updates with the step-indexed schedule.
    model, optimizer, order = start()
    _train(model, optimizer, dataset, order, 4, schedule, peak, clip)
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    # Interrupted: two updates, save, restore, two more.
    model, optimizer, order = start()
    order = _train(model, optimizer, dataset, order, 2, schedule, peak, clip)
    path = tmp_path / "adamw-step-2.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, checkpoint_id="adamw-2", provenance=provenance,
                    weights=weights, progress=Progress(2, 0, 3), data_order=order)
    payload = torch.load(path, weights_only=False)
    recorded = payload["training"]["optimizer"]
    assert recorded["kind"] == "AdamW"
    assert [g["options"]["weight_decay"] for g in recorded["groups"]] == [0.03, 0.0]
    assert all(g["options"]["decoupled_weight_decay"] for g in recorded["groups"])
    assert recorded["groups"][0]["options"]["lr"] == pytest.approx(peak * schedule.multiplier(1))
    restored = restore_training(path, expected_config=config, expected_provenance=provenance, dataset=dataset,
        source_id="fixture-source", expected_weights=weights,
        expected_optimizer={k: v for k, v in recorded.items() if k != "state"})
    assert type(restored.optimizer) is torch.optim.AdamW
    assert [g["weight_decay"] for g in restored.optimizer.param_groups] == [0.03, 0.0]
    assert all(s["step"].item() == 2 for s in restored.optimizer.state.values())
    order = restored.data_order
    for step in (2, 3):
        set_learning_rate(restored.optimizer, peak * schedule.multiplier(step))
        indices = order.next_indices()
        supervised_step(restored.model, restored.optimizer, dataset.batch(indices), max_gradient_norm=clip)
        order = replace(order, cursor=order.cursor + len(indices))
        if not order.next_indices():
            order = replace(order, cursor=0)
    assert all(torch.equal(v, reference[k]) for k, v in restored.model.state_dict().items())
    # Inference loading of an AdamW checkpoint needs no optimizer knowledge.
    assert load_inference(path).checkpoint_id == "adamw-2"
    # A different recorded decay is rejected by the explicit optimizer expectation.
    expected = {k: v for k, v in recorded.items() if k != "state"}
    expected["groups"][0]["options"]["weight_decay"] = 0.01
    with pytest.raises(CheckpointError, match="optimizer configuration"):
        restore_training(path, expected_config=config, expected_provenance=provenance, dataset=dataset,
            source_id="fixture-source", expected_weights=weights, expected_optimizer=expected)


def test_optimizer_kind_and_decoupled_flag_must_agree(tmp_path):
    dataset = open_dataset(**fixture_bundle(tmp_path), split="train")
    config = ModelConfig(d_policy=32, d_value=48, dropout=.27)
    torch.manual_seed(1)
    model = config.build()
    order = DataOrder("fixture-source", DataOrder.reference_hash(dataset), (2, 0, 1), 0, 2, False, 0)
    common = dict(checkpoint_id="kind", provenance={"a": 1}, weights=LossWeights(), progress=Progress(0, 0, 0),
                  data_order=order)
    # Plain Adam keeps loading as before.
    adam = tmp_path / "adam.pt"
    save_checkpoint(adam, model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False), **common)
    assert torch.load(adam, weights_only=False)["training"]["optimizer"]["kind"] == "Adam"
    restored = restore_training(adam, expected_config=config, expected_provenance={"a": 1}, dataset=dataset,
                                source_id="fixture-source")
    assert type(restored.optimizer) is torch.optim.Adam
    # Tampering the kind without the decoupled flag, or vice versa, is rejected.
    payload = torch.load(adam, weights_only=False)
    payload["training"]["optimizer"]["kind"] = "AdamW"
    bad = tmp_path / "bad-kind.pt"
    torch.save(payload, bad)
    with pytest.raises(CheckpointError, match="decoupled"):
        load_inference(bad)
    payload = torch.load(adam, weights_only=False)
    payload["training"]["optimizer"]["kind"] = "SGD"
    torch.save(payload, tmp_path / "sgd.pt")
    with pytest.raises(CheckpointError, match="unsupported optimizer"):
        load_inference(tmp_path / "sgd.pt")
    with pytest.raises(CheckpointError, match="Adam or AdamW"):
        save_checkpoint(tmp_path / "sgd-live.pt", model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.1), **common)


def test_mixed_validator_accepts_both_kinds_and_rejects_coupled_decay():
    model = ChessNetwork()
    settings = MixedSettings(learning_rate=1e-5)
    from chess_rl.training.checkpoints import _optimizer_payload
    adamw = build_optimizer(model, OptimizerSettings(learning_rate=1e-5, weight_decay=0.01))
    validate_optimizer_settings(_optimizer_payload(model, adamw), settings)
    adam = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)
    validate_optimizer_settings(_optimizer_payload(model, adam), settings)
    coupled = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01, foreach=False, fused=False)
    with pytest.raises(ValueError, match="weight_decay"):
        validate_optimizer_settings(_optimizer_payload(model, coupled), settings)
    wrong_lr = build_optimizer(model, OptimizerSettings(learning_rate=1e-4))
    with pytest.raises(ValueError, match="lr"):
        validate_optimizer_settings(_optimizer_payload(model, wrong_lr), settings)
