"""T02 reload, CPU dropout continuation, immutable publication and bad inputs."""
from copy import deepcopy
from dataclasses import asdict, replace
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (CheckpointError, DataOrder, ModelConfig, Progress,
    load_inference, restore_training, save_checkpoint)
from chess_rl.training.trainer import LossWeights, supervised_step
from .test_supervised import fixture_bundle


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def rng_snapshot():
    n = np.random.get_state()
    return random.getstate(), (n[0], n[1].copy(), *n[2:]), torch.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0] and a[1][2:] == b[1][2:]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    assert torch.equal(a[2], b[2])


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_tree_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree_equal(x, y)
    else: assert a == b


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoint")
    dataset = open_dataset(**fixture_bundle(root), split="train")
    config = ModelConfig(d_policy=32, d_value=48, dropout=.27)
    random.seed(101); np.random.seed(102); torch.manual_seed(103)
    # Preserve cached Gaussian draws as well as the integer state.
    random.gauss(0, 1); np.random.normal()
    model = config.build()
    # Different settings and reversed named membership in the second group.
    params = list(model.parameters())
    optimizer = torch.optim.Adam([dict(params=params[:30], lr=.0002),
        dict(params=list(reversed(params[30:])), lr=.0003, amsgrad=True)],
        betas=(.8, .95), eps=1e-7, weight_decay=.02, foreach=False, fused=False)
    weights = LossWeights(.7, 1.3)
    order = DataOrder("fixture-source", DataOrder.reference_hash(dataset), (2, 0, 1), 0, 2, False, 0)
    batch = next(dataset.batches(2, order=order.order))
    supervised_step(model, optimizer, batch, weights=weights)
    order = replace(order, cursor=2)
    path = root / "saved.pt"
    progress = Progress(1, 0, 2)
    provenance = {"source_commit": "fixture", "data": {"teacher": "soft-root-wdl", "seed": 103}}
    kwargs = dict(model=model, optimizer=optimizer, checkpoint_id="test-step-1", provenance=provenance,
                  weights=weights, progress=progress, data_order=order)
    before = rng_snapshot()
    save_checkpoint(path, **kwargs)
    assert_rng_equal(before, rng_snapshot())
    return dict(path=path, dataset=dataset, config=config, kwargs=kwargs)


def restore(saved, path=None, **changes):
    args = dict(expected_config=saved["config"], expected_provenance=saved["kwargs"]["provenance"],
        dataset=saved["dataset"], source_id="fixture-source", expected_weights=saved["kwargs"]["weights"])
    args.update(changes)
    return restore_training(path or saved["path"], **args)


def test_portable_resume_policy_is_explicit_and_keeps_numeric_runtime_checks(saved, tmp_path):
    payload = torch.load(saved["path"], weights_only=False)
    payload["training"]["runtime"]["platform"] = "different OS description"
    path = tmp_path / "portable.pt"
    torch.save(payload, path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="runtime"):
        restore(saved, path)
    assert_rng_equal(before, rng_snapshot())
    with pytest.warns(RuntimeWarning, match="bitwise continuation is unverified"):
        restored = restore(saved, path, runtime_policy="compatible")
    assert_tree_equal(restored.model.state_dict(), saved["kwargs"]["model"].state_dict())
    event = restored.provenance["resume_runtime_history"][-1]
    assert event["policy"] == "compatible" and event["differences"] == ["platform"]
    subsequent = tmp_path / "subsequent.pt"
    # Even a caller supplying its original provenance must retain the opt-in.
    save_checkpoint(subsequent, model=restored.model, optimizer=restored.optimizer,
        checkpoint_id="after-portable", provenance=saved["kwargs"]["provenance"],
        weights=restored.weights, progress=restored.progress, data_order=restored.data_order)
    assert load_inference(subsequent).provenance == restored.provenance
    payload["training"]["runtime"]["threads"] += 1
    torch.save(payload, path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="runtime threads"):
        restore(saved, path, runtime_policy="compatible")
    assert_rng_equal(before, rng_snapshot())


@pytest.mark.parametrize("field,value", [
    ("python", "3.13.0"), ("numpy", "0.0.0"), ("torch", "0.0.0"),
    ("machine", "other-architecture"), ("threads", 999), ("interop_threads", 999),
    ("default_dtype", "torch.float16"), ("matmul_precision", "medium"),
    ("deterministic", "toggle"), ("deterministic_warn_only", "toggle"), ("mkldnn", "toggle")])
def test_compatible_resume_rejects_unsupported_runtime_differences(saved, tmp_path, field, value):
    payload = torch.load(saved["path"], weights_only=True)
    runtime = payload["training"]["runtime"]
    runtime[field] = not runtime[field] if value == "toggle" else value
    path = tmp_path / "unsupported.pt"
    torch.save(payload, path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="runtime"):
        restore(saved, path, runtime_policy="compatible")
    assert_rng_equal(before, rng_snapshot())


def test_compatible_python_patch_change_records_opt_in_even_on_same_runtime(saved, tmp_path):
    payload = torch.load(saved["path"], weights_only=True)
    version = payload["training"]["runtime"]["python"].split(".")
    version[-1] = str(int(version[-1]) + 1)
    payload["training"]["runtime"]["python"] = ".".join(version)
    path = tmp_path / "python-patch.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="runtime"):
        restore(saved, path)
    with pytest.warns(RuntimeWarning):
        restored = restore(saved, path, runtime_policy="compatible")
    assert restored.provenance["resume_runtime_history"][-1]["differences"] == ["python"]
    same = restore(saved, runtime_policy="compatible")
    assert same.provenance["resume_runtime_history"][-1]["differences"] == []


def test_compatible_resume_keeps_backend_contract(saved, tmp_path):
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="model configuration"):
        restore(saved, runtime_policy="compatible", expected_config=replace(saved["config"], device="cuda:0"))
    assert_rng_equal(before, rng_snapshot())
    payload = torch.load(saved["path"], weights_only=True)
    payload["model_config"]["device"] = "cuda:0"
    path = tmp_path / "backend-mismatch.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="version/backend"):
        restore(saved, path, runtime_policy="compatible")
    assert_rng_equal(before, rng_snapshot())


def test_inference_predictions_metadata_and_rng_unchanged(saved):
    before = rng_snapshot()
    loaded = load_inference(saved["path"], expected_config=saved["config"])
    assert_rng_equal(before, rng_snapshot())
    assert not loaded.model.training
    assert loaded.checkpoint_id == "test-step-1" and loaded.provenance == saved["kwargs"]["provenance"]
    assert loaded.config == saved["config"]
    original = saved["config"].build()
    original.load_state_dict(saved["kwargs"]["model"].state_dict())
    original.eval()
    features = next(saved["dataset"].batches(2)).to_torch()["features"]
    with torch.inference_mode():
        a, b = original(features), loaded.model(features)
    assert_tree_equal(a, b)
    bank = loaded.model.backbone.bank
    assert all(layer.self_attn.gab.bank is bank for layer in loaded.model.backbone.layers)


def test_next_dropout_update_gradients_adam_and_random_draws_are_exact(saved):
    first = restore(saved)
    order = first.data_order
    assert order.next_indices() == (1,)  # retain the partial next D07 batch
    next_batch = list(saved["dataset"].batches(order.batch_size, order=order.order))[1]
    assert next_batch.references == (saved["dataset"][1].reference,)
    draws = (random.random(), random.gauss(0, 1), np.random.random(), np.random.normal())
    metrics = supervised_step(first.model, first.optimizer, next_batch, weights=first.weights)
    rng = rng_snapshot()
    random.seed(999); np.random.seed(999); torch.manual_seed(999)
    second = restore(saved)
    assert second.progress == Progress(1, 0, 2) and second.model.training
    assert draws == (random.random(), random.gauss(0, 1), np.random.random(), np.random.normal())
    assert supervised_step(second.model, second.optimizer, next_batch, weights=second.weights) == metrics
    assert_tree_equal(first.model.state_dict(), second.model.state_dict())
    assert_tree_equal(first.optimizer.state_dict(), second.optimizer.state_dict())
    for p, q in zip(first.model.parameters(), second.model.parameters()): assert torch.equal(p.grad, q.grad)
    assert_rng_equal(rng, rng_snapshot())
    assert sum(p is second.model.backbone.bank.weight for g in second.optimizer.param_groups for p in g["params"]) == 1


def test_initial_float64_eval_mode_and_empty_adam_state(saved, tmp_path):
    config = ModelConfig(d_policy=19, d_value=23, dropout=0, dtype="float64")
    model = config.build().eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=.004)
    path = tmp_path / "initial.pt"
    order = replace(saved["kwargs"]["data_order"], cursor=0, drop_last=True)
    assert replace(order, cursor=2).next_indices() == ()
    save_checkpoint(path, model=model, optimizer=optimizer, checkpoint_id="initial",
        provenance=saved["kwargs"]["provenance"], weights=LossWeights(), progress=Progress(0, 0, 0), data_order=order)
    loaded = restore(saved, path, expected_config=config, expected_weights=LossWeights())
    assert not loaded.model.training and not loaded.optimizer.state
    assert next(loaded.model.parameters()).dtype == torch.float64 and loaded.config == config


def test_immutable_save_does_not_change_file_or_rng(saved):
    before_bytes = saved["path"].read_bytes()
    before_rng = rng_snapshot()
    with pytest.raises(CheckpointError, match="exist"):
        save_checkpoint(saved["path"], **saved["kwargs"])
    assert saved["path"].read_bytes() == before_bytes
    assert_rng_equal(before_rng, rng_snapshot())
    assert not list(saved["path"].parent.glob(".checkpoint-*"))


@pytest.mark.parametrize("bad", [
    "version", "extra", "encoding", "wdl", "ending", "relative", "dropout", "width", "dtype",
    "bool_width", "model_missing", "shape", "nonfinite", "shared_alias", "provenance",
    "adam_kind", "adam_lr", "adam_options", "adam_names", "adam_duplicate", "adam_ids",
    "adam_missing", "adam_step", "adam_shape", "adam_nan", "adam_negative", "adam_max",
    "weights", "progress", "order", "cursor", "epoch", "source", "order_version", "heldout",
    "rng_torch", "rng_numpy", "rng_python", "runtime_schema"])
def test_malformed_checkpoints_fail_clearly_without_installing_rng(saved, tmp_path, bad):
    payload = torch.load(saved["path"], weights_only=True)
    train = payload["training"]
    adam = train["optimizer"]
    state = adam["state"]["state"][0]
    model_state = payload["model_state"]
    name = next(iter(model_state))
    if bad == "version": payload["version"] = "future"
    elif bad == "extra": payload["unexpected"] = True
    elif bad == "encoding": payload["contracts"]["encoding_version"] = "old"
    elif bad == "wdl": payload["contracts"]["wdl_order"] = ("loss", "draw", "win")
    elif bad == "ending": payload["contracts"]["ending_policy"] = "claim-ahead"
    elif bad == "relative": payload["model_config"]["relative_enabled"] = False
    elif bad == "dropout": payload["model_config"]["dropout"] = float("nan")
    elif bad == "width": payload["model_config"]["d_policy"] = 0
    elif bad == "dtype": payload["model_config"]["dtype"] = "bfloat16"
    elif bad == "bool_width": payload["model_config"]["d_value"] = True
    elif bad == "model_missing": del model_state[name]
    elif bad == "shape": model_state[name] = torch.zeros(1)
    elif bad == "nonfinite": model_state[name].flatten()[0] = torch.nan
    elif bad == "shared_alias":
        key = "backbone.layers.0.self_attn.gab.bank.weight"
        model_state[key] = model_state[key].clone() + 1
    elif bad == "provenance": payload["provenance"] = {"bad": float("inf")}
    elif bad == "adam_kind": adam["kind"] = "SGD"
    elif bad == "adam_lr": adam["defaults"]["lr"] = -1
    elif bad == "adam_options": adam["groups"][0]["options"]["eps"] = 1e-5
    elif bad == "adam_names": adam["groups"][0]["names"][0] = "unknown"
    elif bad == "adam_duplicate": adam["groups"][0]["names"][0] = adam["groups"][0]["names"][1]
    elif bad == "adam_ids": adam["state"]["param_groups"][0]["params"][0] = True
    elif bad == "adam_missing": del adam["state"]["state"][0]
    elif bad == "adam_step": state["step"] += 1
    elif bad == "adam_shape": state["exp_avg"] = torch.zeros(1)
    elif bad == "adam_nan": state["exp_avg"].flatten()[0] = torch.nan
    elif bad == "adam_negative": state["exp_avg_sq"].flatten()[0] = -1
    elif bad == "adam_max":
        i = adam["state"]["param_groups"][1]["params"][0]
        adam["state"]["state"][i]["max_exp_avg_sq"][:] = 0
    elif bad == "weights": train["weights"]["policy"] = 0
    elif bad == "progress": train["progress"]["step"] = True
    elif bad == "order": train["data_order"]["order"] = (0, 0, 2)
    elif bad == "cursor": train["data_order"]["cursor"] = 1
    elif bad == "epoch": train["data_order"]["epoch"] = 2
    elif bad == "source": train["data_order"]["source_id"] = ""
    elif bad == "order_version": train["data_order"]["version"] = "unknown"
    elif bad == "heldout": train["data_order"]["split"] = "test"
    elif bad == "rng_torch": train["rng"]["torch"] = torch.zeros(3, dtype=torch.uint8)
    elif bad == "rng_numpy": train["rng"]["numpy"]["position"] = -1
    elif bad == "rng_python": train["rng"]["python"] = (99, (), None)
    elif bad == "runtime_schema": del train["runtime"]["threads"]
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="checkpoint"):
        load_inference(path)
    assert_rng_equal(before, rng_snapshot())
    with pytest.raises(CheckpointError, match="checkpoint"):
        restore(saved, path)
    assert_rng_equal(before, rng_snapshot())


@pytest.mark.parametrize("change", ["config", "provenance", "source", "membership", "weights", "optimizer", "runtime"])
def test_incompatible_expected_training_configuration_fails(saved, tmp_path, change):
    path = saved["path"]
    args = {}
    if change == "config": args["expected_config"] = replace(saved["config"], dropout=.1)
    elif change == "provenance": args["expected_provenance"] = {"source_commit": "other"}
    elif change == "source": args["source_id"] = "other"
    elif change == "weights": args["expected_weights"] = LossWeights()
    elif change == "optimizer": args["expected_optimizer"] = {"kind": "SGD"}
    else:
        payload = torch.load(path, weights_only=True)
        if change == "membership": payload["training"]["data_order"]["references_sha256"] = "0" * 64
        else: payload["training"]["runtime"]["threads"] += 1
        path = tmp_path / "incompatible.pt"
        torch.save(payload, path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError, match="incompatible"):
        restore(saved, path, **args)
    assert_rng_equal(before, rng_snapshot())
    if change == "runtime": assert not load_inference(path).model.training


@pytest.mark.parametrize("content", [b"", b"truncated", b"PK\x03\x04broken"])
def test_unreadable_checkpoint_is_actionable(tmp_path, content):
    path = tmp_path / "broken.pt"
    path.write_bytes(content)
    with pytest.raises(CheckpointError, match="cannot read checkpoint"):
        load_inference(path)


@pytest.mark.parametrize("bad", ["diagnostic", "sgd", "optimizer_duplicate", "geometry", "mixed_mode", "dropout", "norm"])
def test_unsupported_save_settings_fail(saved, tmp_path, bad):
    kwargs = dict(saved["kwargs"])
    model = saved["config"].build()
    model.load_state_dict(kwargs["model"].state_dict())
    optimizer = torch.optim.Adam(model.parameters())
    kwargs.update(model=model, optimizer=optimizer, progress=Progress(0, 0, 0),
                  data_order=replace(kwargs["data_order"], cursor=0))
    if bad == "diagnostic": model.backbone.layers[0].self_attn.relative = None
    elif bad == "sgd": kwargs["optimizer"] = torch.optim.SGD(model.parameters(), lr=.1)
    elif bad == "optimizer_duplicate": optimizer.param_groups[0]["params"].append(model.backbone.bank.weight)
    elif bad == "geometry": model.backbone.layers[0].self_attn.relative.index[0, 0] = 0
    elif bad == "mixed_mode": model.backbone.layers[0].eval()
    elif bad == "dropout": model.backbone.layers[0].dropout.p = .5
    elif bad == "norm": model.backbone.norm.eps = .1
    before = rng_snapshot()
    path = tmp_path / "unsupported.pt"
    with pytest.raises(CheckpointError): save_checkpoint(path, **kwargs)
    assert_rng_equal(before, rng_snapshot())
    assert not path.exists()


def test_data_order_validates_each_permutation_once_not_per_cursor_advance(monkeypatch):
    """Advancing the cursor via replace() must not re-sort the permutation (it cost 0.36 s/step at 1M)."""
    import chess_rl.training.checkpoints as C
    from dataclasses import replace
    calls = []
    real_sorted = sorted
    monkeypatch.setattr(C, "sorted", lambda seq: calls.append(len(seq)) or real_sorted(seq), raising=False)
    order = tuple(range(640))
    first = C.DataOrder("source", "0" * 64, order, 0, 32, False, 0)
    assert calls == [640]
    advanced = first
    for _ in range(5):
        advanced = replace(advanced, cursor=advanced.cursor + 32)
    assert calls == [640] and advanced.cursor == 160
    # an equal but distinct tuple object is validated again; a tampered one is rejected
    C.DataOrder("source", "0" * 64, tuple(range(640)), 0, 32, False, 0)
    assert calls == [640, 640]
    with pytest.raises(ValueError, match="complete permutation"):
        C.DataOrder("source", "0" * 64, order[:-1] + (0,), 0, 32, False, 0)
