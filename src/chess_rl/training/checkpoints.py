"""T02 shared, versioned M06 checkpoints at completed optimizer-step boundaries.

CPU float32/float64 with standard Adam or grouped AdamW are supported. Legacy v1 retains D07
single-source ordering. Mixed v2 replaces that cursor with both source cursors,
identities, membership, dedicated sampling RNG and frozen R03 settings.
Single cuda:0 float32 has separately versioned supervised and mixed formats.
No scheduler, accumulation, worker prefetch or AMP state is implied.
Inference never installs saved training RNG or mixed sampling state.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from chess_rl.contracts import AdapterConfig
from chess_rl.model import ChessNetwork
from chess_rl.supervised import SupervisedDataset
from .trainer import LossWeights

if TYPE_CHECKING:
    from .mixed_state import MixedState

CHECKPOINT_VERSION = "m06-checkpoint-v1"
MIXED_CHECKPOINT_VERSION = "m06-checkpoint-v2-mixed"
CUDA_CHECKPOINT_VERSION = "m06-checkpoint-cuda-v2"
CUDA_MIXED_CHECKPOINT_VERSION = "m06-checkpoint-cuda-v3-mixed"
MODEL_VERSION = "m06-gab-relative-qkv-v1"
ORDER_VERSION = "d07-explicit-order-v1"


class CheckpointError(ValueError):
    """Malformed checkpoint or unsupported/incompatible resumption contract."""


def _require(condition, message):
    if not condition:
        raise CheckpointError(message)


def _fields(value, fields, label):
    _require(isinstance(value, dict) and set(value) == set(fields), f"{label} schema mismatch")


def _json(value):
    # Strict primitive JSON also distinguishes true from 1 in comparisons.
    def validate(item):
        if type(item) is dict:
            _require(all(type(k) is str for k in item), "metadata keys must be strings")
            for v in item.values(): validate(v)
        elif type(item) in (list, tuple):
            for v in item: validate(v)
        else:
            _require(type(item) in (str, int, float, bool, type(None)), "metadata must be primitive JSON")
    validate(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(actual, expected, label):
    _require(_json(actual) == _json(expected), f"incompatible {label}")


@dataclass(frozen=True)
class ModelConfig:
    d_policy: int = 128
    d_value: int = 128
    dropout: float = 0.1
    relative_enabled: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    architecture: str = MODEL_VERSION

    def __post_init__(self):
        _require(all(type(v) is int and v > 0 for v in (self.d_policy, self.d_value)),
                 "model head widths must be positive integers")
        _require(type(self.dropout) in (int, float) and math.isfinite(self.dropout)
                 and 0 <= self.dropout <= 1, "invalid model dropout")
        _require(self.relative_enabled is True and self.architecture == MODEL_VERSION,
                 "only accepted combined M06 architecture is supported")
        _require((self.device == "cpu" and self.dtype in ("float32", "float64"))
                 or (self.device == "cuda:0" and self.dtype == "float32"),
                 "checkpoint backend must be CPU float32/float64 or single cuda:0 float32")

    def build(self):
        # Construction must not consume caller's dropout stream (also on failure).
        with torch.random.fork_rng(devices=[]):
            model = ChessNetwork(d_policy=self.d_policy, d_value=self.d_value,
                dropout=self.dropout, relative_enabled=True, device="cpu", dtype=getattr(torch, self.dtype))
        if self.device != "cpu":
            _cuda_available()
            model.to(self.device)  # copying consumes no CUDA random numbers
        return model


def _cuda_available():
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "CUDA requires exactly one visible device; set CUDA_VISIBLE_DEVICES before launch")


def _cpu_tree(value):
    """Snapshot tensor storage on CPU without aliases to live model/Adam state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def cuda_runtime_config():
    """Strict same-device/runtime resume fingerprint; no global settings changed."""
    _cuda_available()
    p = torch.cuda.get_device_properties(0)
    return dict(cuda=str(torch.version.cuda), cudnn=torch.backends.cudnn.version(),
        name=p.name, capability=list(torch.cuda.get_device_capability(0)),
        total_memory=p.total_memory, visible_devices=1,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""))


def _validate_cuda_runtime(runtime):
    schema = dict(cuda=str, cudnn=int, name=str, capability=list, total_memory=int,
        visible_devices=int, matmul_tf32=bool, cudnn_tf32=bool, cudnn_benchmark=bool,
        cudnn_deterministic=bool, cublas_workspace_config=str)
    _fields(runtime, schema, "CUDA runtime")
    _require(all(type(runtime[k]) is t for k, t in schema.items()), "invalid CUDA runtime types")
    _require(runtime["visible_devices"] == 1 and runtime["total_memory"] > 0
             and len(runtime["capability"]) == 2
             and all(type(v) is int and v >= 0 for v in runtime["capability"]),
             "invalid CUDA runtime device")
    _require(not runtime["matmul_tf32"] and not runtime["cudnn_tf32"],
             "CUDA float32 contract requires TF32 disabled")
    _require(not runtime["cudnn_benchmark"] and runtime["cudnn_deterministic"]
             and runtime["cublas_workspace_config"] == ":4096:8",
             "CUDA resume requires deterministic cuDNN and CUBLAS_WORKSPACE_CONFIG=:4096:8")


@dataclass(frozen=True)
class Progress:
    step: int
    epoch: int
    samples_seen: int

    def __post_init__(self):
        _require(all(type(v) is int and v >= 0 for v in (self.step, self.epoch, self.samples_seen)),
                 "progress must contain nonnegative integer counts")


# Permutations already verified, keyed by tuple identity; the stored tuple keeps the id live.
_validated_orders: dict = {}


@dataclass(frozen=True)
class DataOrder:
    """Current D07 permutation; cursor counts already consumed sample indices.

    source_id identifies the label/source bytes; references_sha256 binds the
    split's source-order membership. Caller starts a new explicit order at epoch
    boundaries. Global Python/NumPy/Torch RNG covers future order generation.
    """
    source_id: str
    references_sha256: str
    order: tuple[int, ...]
    cursor: int
    batch_size: int
    drop_last: bool
    epoch: int
    split: str = "train"
    version: str = ORDER_VERSION

    def __post_init__(self):
        _require(type(self.source_id) is str and bool(self.source_id), "data source identity is required")
        _require(type(self.references_sha256) is str and len(self.references_sha256) == 64
                 and all(c in "0123456789abcdef" for c in self.references_sha256), "invalid reference hash")
        _require(type(self.order) is tuple and len(self.order) > 0, "data order must be a complete permutation")
        # The permutation check is O(n log n); the training loop rebuilds this record every step
        # with the same tuple object and a new cursor, so each distinct tuple is checked once.
        if _validated_orders.get(id(self.order)) is not self.order:
            _require(all(type(i) is int for i in self.order)
                     and sorted(self.order) == list(range(len(self.order))), "data order must be a complete permutation")
            if len(_validated_orders) >= 8:
                _validated_orders.pop(next(iter(_validated_orders)))
            _validated_orders[id(self.order)] = self.order
        _require(type(self.batch_size) is int and self.batch_size > 0 and type(self.drop_last) is bool,
                 "invalid data batch settings")
        _require(type(self.cursor) is int and 0 <= self.cursor <= len(self.order)
                 and (self.cursor % self.batch_size == 0 or self.cursor == len(self.order)),
                 "data cursor must be at a completed batch boundary")
        _require(type(self.epoch) is int and self.epoch >= 0 and self.split == "train"
                 and self.version == ORDER_VERSION, "unsupported data split/order version/epoch")

    @staticmethod
    def reference_hash(dataset):
        _require(isinstance(dataset, SupervisedDataset) and dataset.split == "train", "D07 train dataset required")
        # D07 owns source-order references; hashing them requires no encoding.
        references = [dataset.reference(i) for i in range(len(dataset))]
        return hashlib.sha256(_json(references).encode()).hexdigest()

    def validate_dataset(self, dataset, *, source_id):
        _same(source_id, self.source_id, "data source identity")
        _require(len(dataset) == len(self.order), "incompatible data membership size")
        _same(self.reference_hash(dataset), self.references_sha256, "data source ordering/membership")

    def next_indices(self):
        group = self.order[self.cursor:self.cursor + self.batch_size]
        return () if self.drop_last and len(group) < self.batch_size else group


@dataclass(frozen=True)
class InferenceCheckpoint:
    model: ChessNetwork
    config: ModelConfig
    checkpoint_id: str
    provenance: dict
    contracts: dict


@dataclass(frozen=True)
class TrainingCheckpoint:
    model: ChessNetwork
    optimizer: torch.optim.Optimizer
    config: ModelConfig
    checkpoint_id: str
    provenance: dict
    weights: LossWeights
    progress: Progress
    data_order: DataOrder | None
    mixed_state: MixedState | None = None


def runtime_config():
    """Resume rejects a different runtime; caller explicitly configures it first."""
    return dict(python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
        platform=platform.platform(), machine=platform.machine(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        default_dtype=str(torch.get_default_dtype()), matmul_precision=torch.get_float32_matmul_precision(),
        mkldnn=torch.backends.mkldnn.enabled)


def _model_config(model):
    _require(type(model) is ChessNetwork, "checkpoint requires ChessNetwork")
    parameters = list(model.parameters())
    _require(all(p.device == parameters[0].device and p.dtype == parameters[0].dtype and p.requires_grad
                 for p in parameters), "model requires uniform device/dtype and trainable parameters")
    _require(all(m.training == model.training for m in model.modules()), "mixed model train/eval modes unsupported")
    bank = model.backbone.bank
    _require(all(layer.self_attn.gab.bank is bank and layer.self_attn.relative is not None
                 for layer in model.backbone.layers), "combined M06 shared GAB identity required")
    rates = [m.p for m in model.modules() if isinstance(m, nn.Dropout)]
    rates += [layer.self_attn.mha.dropout for layer in model.backbone.layers]
    _require(len(set(rates)) == 1, "inconsistent model dropout configuration")
    config = ModelConfig(d_policy=model.policy_head.d_policy, d_value=model.wdl_head.d_value,
                         dropout=rates[0], dtype=str(parameters[0].dtype).removeprefix("torch."),
                         device=str(parameters[0].device))
    template = config.build()
    _require(repr(model) == repr(template), "model settings differ from accepted M06 construction")
    buffers = dict(template.named_buffers())
    _require(all(name in buffers and torch.equal(value, buffers[name]) for name, value in model.named_buffers()),
             "model geometry differs from accepted M06 construction")
    return config


def _model_state(model, state):
    expected = model.state_dict()
    _fields(state, expected, "model state")
    for name, target in expected.items():
        tensor = state[name]
        _require(type(tensor) is torch.Tensor and tensor.device.type == "cpu"
                 and tensor.shape == target.shape and tensor.dtype == target.dtype
                 and not tensor.requires_grad and bool(torch.isfinite(tensor).all()),
                 f"invalid model tensor {name}: shape/dtype/finite values")
    # state_dict contains a path per layer for the same shared bank. Reject
    # contradictory copies instead of silently accepting the last assignment.
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        previous = aliases.setdefault(id(parameter), name)
        _require(torch.equal(state[name], state[previous]), f"contradictory shared parameter {name}")
    for name, buffer in model.named_buffers():
        # M06's nonpersistent geometry is reconstructed, never loaded as weights.
        _require(buffer.device.type == "cpu", f"invalid model buffer device: {name}")


OPTIMIZER_CLASSES = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}


def _adam_options(options, kind="Adam"):
    """Adam ('Adam', coupled L2 or none) and AdamW ('AdamW', decoupled decay) share one schema."""
    fields = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach",
              "capturable", "differentiable", "fused", "decoupled_weight_decay"}
    _fields(options, fields, "Adam configuration")
    for name in ("lr", "eps", "weight_decay"):
        v = options[name]
        _require(type(v) in (int, float) and math.isfinite(v) and v >= 0
                 and (name != "eps" or v > 0), f"invalid Adam {name}")
    betas = options["betas"]
    _require(type(betas) in (list, tuple) and len(betas) == 2
             and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < 1 for v in betas),
             "invalid Adam betas")
    for name in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"):
        _require(type(options[name]) is bool, f"invalid Adam {name}")
    for name in ("foreach", "fused"):
        _require(options[name] is None or type(options[name]) is bool, f"invalid Adam {name}")
    _require(not options["capturable"] and not options["differentiable"] and options["fused"] is not True,
             "only nonfused noncapturable Adam/AdamW is supported")
    _require(kind in OPTIMIZER_CLASSES and options["decoupled_weight_decay"] == (kind == "AdamW"),
             "decoupled weight decay must match the optimizer kind")


def _optimizer_kind(optimizer):
    for kind, cls in OPTIMIZER_CLASSES.items():
        if type(optimizer) is cls:
            return kind
    raise CheckpointError("checkpoint supports standard Adam or AdamW only")


def _optimizer_payload(model, optimizer):
    kind = _optimizer_kind(optimizer)
    names = {id(p): n for n, p in model.named_parameters()}
    supplied = [p for g in optimizer.param_groups for p in g["params"]]
    _require(len(supplied) == len(names) and {id(p) for p in supplied} == set(names),
             "optimizer must own every unique model parameter exactly once")
    groups = []
    for group in optimizer.param_groups:
        options = {k: v for k, v in group.items() if k != "params"}
        _adam_options(options, kind)
        groups.append(dict(names=[names[id(p)] for p in group["params"]], options=options))
    defaults = dict(optimizer.defaults)
    _adam_options(defaults, kind)
    return dict(kind=kind, defaults=defaults, groups=groups, state=optimizer.state_dict())


def _restore_optimizer(model, payload, progress):
    _fields(payload, ("kind", "defaults", "groups", "state"), "optimizer")
    kind = payload["kind"]
    _require(kind in OPTIMIZER_CLASSES, "unsupported optimizer version/configuration")
    _adam_options(payload["defaults"], kind)
    groups = payload["groups"]
    _require(type(groups) is list and bool(groups), "invalid optimizer groups")
    named = dict(model.named_parameters())
    flattened = []
    actual_groups = []
    for group in groups:
        _fields(group, ("names", "options"), "optimizer group")
        _adam_options(group["options"], kind)
        names = group["names"]
        _require(type(names) is list and bool(names) and all(type(n) is str and n in named for n in names),
                 "invalid optimizer parameter names")
        flattened.extend(names)
        actual_groups.append(dict(params=[named[n] for n in names], **group["options"]))
    _require(len(flattened) == len(named) and set(flattened) == set(named),
             "optimizer parameter membership/duplicates mismatch")
    saved = payload["state"]
    _fields(saved, ("param_groups", "state"), "Adam state")
    _require(type(saved["param_groups"]) is list and len(saved["param_groups"]) == len(groups),
             "Adam state group count mismatch")
    ids, param_options = [], {}
    for group, state_group in zip(groups, saved["param_groups"]):
        _fields(state_group, set(group["options"]) | {"params"}, "Adam state group")
        _same({k: v for k, v in state_group.items() if k != "params"}, group["options"], "Adam state settings")
        indices = state_group["params"]
        _require(type(indices) is list and len(indices) == len(group["names"])
                 and all(type(i) is int and i >= 0 for i in indices), "invalid Adam parameter IDs")
        ids.extend(indices)
        for index, name in zip(indices, group["names"]):
            param_options[index] = (named[name], group["options"])
    _require(ids == list(range(len(named))), "Adam parameter IDs/order mismatch")
    states = saved["state"]
    _require(type(states) is dict and all(type(k) is int for k in states)
             and set(states) == (set(ids) if progress.step else set()), "missing/extra Adam state")
    for index, state in states.items():
        param, options = param_options[index]
        fields = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if options["amsgrad"] else set())
        _fields(state, fields, "Adam per-parameter state")
        step = state["step"]
        _require(type(step) is torch.Tensor and step.device.type == "cpu" and step.shape == ()
                 and step.dtype in (torch.float32, torch.float64) and not step.requires_grad
                 and bool(torch.isfinite(step)) and step.item() == progress.step, "Adam step/progress mismatch")
        for name in fields - {"step"}:
            tensor = state[name]
            _require(type(tensor) is torch.Tensor and tensor.device.type == "cpu" and not tensor.requires_grad
                     and tensor.shape == param.shape and tensor.dtype == param.dtype
                     and bool(torch.isfinite(tensor).all()), f"invalid Adam {name} tensor")
            if name != "exp_avg":
                _require(bool((tensor >= 0).all()), f"negative Adam {name}")
        if options["amsgrad"]:
            _require(bool((state["max_exp_avg_sq"] >= state["exp_avg_sq"]).all()), "invalid Adam AMSGrad maximum")
    defaults = dict(payload["defaults"])
    if kind == "AdamW":
        defaults.pop("decoupled_weight_decay")  # AdamW fixes it True and rejects the keyword
    optimizer = OPTIMIZER_CLASSES[kind](actual_groups, **defaults)
    optimizer.load_state_dict(saved)
    return optimizer


def _rng_state():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
        numpy=dict(algorithm=algorithm, keys=torch.tensor(keys.astype(np.int64)),
                   position=position, has_gauss=has_gauss, cached=cached))


def _validate_rng(rng, *, cuda=False):
    _fields(rng, ("python", "numpy", "torch", "cuda") if cuda else ("python", "numpy", "torch"), "random states")
    if cuda:
        _fields(rng["cuda"], ("device", "state"), "CUDA RNG")
        t = rng["cuda"]["state"]
        _require(rng["cuda"]["device"] == "cuda:0" and type(t) is torch.Tensor
                 and t.device.type == "cpu" and t.dtype == torch.uint8
                 and t.ndim == 1 and t.numel() > 0 and not t.requires_grad,
                 "invalid CUDA RNG state")
    p = rng["python"]
    _require(type(p) is tuple and len(p) == 3 and type(p[0]) is int and p[0] == 3
             and type(p[1]) is tuple and len(p[1]) == 625
             and all(type(v) is int and 0 <= v < 2**32 for v in p[1][:-1])
             and type(p[1][-1]) is int and 0 <= p[1][-1] <= 624
             and (p[2] is None or type(p[2]) is float and math.isfinite(p[2])), "invalid Python RNG state/version")
    random.Random().setstate(p)
    n = rng["numpy"]
    _fields(n, ("algorithm", "keys", "position", "has_gauss", "cached"), "NumPy RNG")
    _require(n["algorithm"] == "MT19937" and type(n["keys"]) is torch.Tensor
             and n["keys"].dtype == torch.int64 and n["keys"].shape == (624,)
             and bool(((n["keys"] >= 0) & (n["keys"] <= 2**32 - 1)).all())
             and type(n["position"]) is int and 0 <= n["position"] <= 624
             and type(n["has_gauss"]) is int and n["has_gauss"] in (0, 1)
             and type(n["cached"]) is float and math.isfinite(n["cached"]), "invalid NumPy RNG state")
    numpy_state = (n["algorithm"], n["keys"].numpy().astype(np.uint32), n["position"], n["has_gauss"], n["cached"])
    np.random.RandomState().set_state(numpy_state)
    t = rng["torch"]
    _require(type(t) is torch.Tensor and t.dtype == torch.uint8 and t.ndim == 1
             and t.shape == torch.get_rng_state().shape, "invalid Torch CPU RNG state")
    torch.Generator(device="cpu").set_state(t)
    return numpy_state


def _read_payload(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return payload
    except Exception as error:
        raise CheckpointError(f"cannot read checkpoint {path}: {error}") from error


def _validate_payload(payload, expected_config=None):
    _fields(payload, ("version", "checkpoint_id", "contracts", "model_config", "model_state",
                     "model_training", "provenance", "training"), "checkpoint")
    _require(payload["version"] in (CHECKPOINT_VERSION, MIXED_CHECKPOINT_VERSION, CUDA_CHECKPOINT_VERSION, CUDA_MIXED_CHECKPOINT_VERSION), "incompatible checkpoint version")
    cuda = payload["version"] in (CUDA_CHECKPOINT_VERSION, CUDA_MIXED_CHECKPOINT_VERSION)
    _require(type(payload["checkpoint_id"]) is str and bool(payload["checkpoint_id"]), "checkpoint ID required")
    _same(payload["contracts"], asdict(AdapterConfig()), "encoding/vocabulary/WDL order/ending policy")
    _fields(payload["model_config"], ModelConfig.__dataclass_fields__, "model configuration")
    config = ModelConfig(**payload["model_config"])
    _require((config.device == "cuda:0") == cuda, "checkpoint version/backend mismatch")
    if expected_config is not None:
        _require(type(expected_config) is ModelConfig, "expected_config must be ModelConfig")
        _same(asdict(config), asdict(expected_config), "model configuration")
    _require(type(payload["model_training"]) is bool, "invalid model training mode")
    _require(type(payload["provenance"]) is dict and bool(payload["provenance"]), "provenance must be nonempty JSON")
    _json(payload["provenance"])
    # Validate all weights and optimizer tensors on CPU, including CUDA files.
    # This keeps CUDA checkpoints readable by CPU-only inference hosts.
    model = replace(config, device="cpu").build()
    _model_state(model, payload["model_state"])
    model.load_state_dict(payload["model_state"], strict=True)  # assign=False retains GAB identity
    training = payload["training"]
    mixed = payload["version"] in (MIXED_CHECKPOINT_VERSION, CUDA_MIXED_CHECKPOINT_VERSION)
    fields = {"optimizer", "weights", "progress", "mixed_state" if mixed else "data_order", "rng", "runtime"}
    _fields(training, fields | {"cuda_runtime"} if cuda else fields, "training state")
    if cuda:
        _validate_cuda_runtime(training["cuda_runtime"])
        for options in [training["optimizer"]["defaults"], *[g["options"] for g in training["optimizer"]["groups"]]]:
            _require(options["foreach"] is False and options["fused"] is False,
                     "CUDA Adam requires explicit foreach=False and fused=False")
    _fields(training["progress"], Progress.__dataclass_fields__, "progress")
    progress = Progress(**training["progress"])
    _fields(training["weights"], LossWeights.__dataclass_fields__, "loss weights")
    weights = LossWeights(**training["weights"])
    if mixed:
        from .mixed_state import MixedState
        from .mixed import validate_optimizer_settings
        order = MixedState.from_dict(training["mixed_state"])
        _require(progress.step == order.updates and progress.epoch == 0
                 and progress.samples_seen == order.supervised.samples_seen + order.selfplay.samples_seen,
                 "mixed checkpoint progress mismatch")
        _same(asdict(weights), asdict(order.settings.weights), "mixed loss weights")
        validate_optimizer_settings(training["optimizer"], order.settings)
    else:
        _fields(training["data_order"], DataOrder.__dataclass_fields__, "data order")
        order = DataOrder(**training["data_order"])
        _require(progress.epoch == order.epoch, "data epoch/progress mismatch")
        _require(progress.samples_seen >= order.cursor, "data cursor/progress mismatch")
        _require(progress.step != 0 or progress.samples_seen == order.cursor == 0, "initial progress mismatch")
    _fields(training["runtime"], runtime_config(), "runtime")
    _json(training["runtime"])
    for key, current in runtime_config().items():
        _require(type(training["runtime"][key]) is type(current), f"invalid runtime field {key}")
    _require(training["runtime"]["threads"] > 0 and training["runtime"]["interop_threads"] > 0,
             "invalid CPU runtime thread counts")
    if cuda:
        _require(training["runtime"]["deterministic"] and not training["runtime"]["deterministic_warn_only"]
                 and training["runtime"]["default_dtype"] == "torch.float32"
                 and training["runtime"]["matmul_precision"] == "highest",
                 "CUDA resume requires strict deterministic float32 runtime")
    numpy_state = _validate_rng(training["rng"], cuda=cuda)
    optimizer = _restore_optimizer(model, training["optimizer"], progress)
    return model, optimizer, config, weights, progress, order, numpy_state


def save_checkpoint(path, *, model, optimizer, checkpoint_id, provenance, weights, progress,
                    data_order=None, mixed_state=None):
    """Publish an immutable file atomically; refuse an existing destination.

    Caller supplies a completed-update boundary and exact data cursor, with no
    pending accumulated gradient, scheduler or prefetched data. Supply exactly
    one legacy DataOrder or mixed MixedState. Saved gradients are unnecessary:
    both training steps clear them before the next backward.
    """
    temporary = None
    try:
        from .mixed_state import MixedState
        _require(type(weights) is LossWeights and type(progress) is Progress,
                 "save requires LossWeights and Progress")
        _require((type(data_order) is DataOrder and mixed_state is None)
                 or (data_order is None and type(mixed_state) is MixedState),
                 "save requires exactly one DataOrder or MixedState")
        config = _model_config(model)
        history = getattr(model, "_resume_runtime_history", None)
        if history:
            provenance = {**provenance, "resume_runtime_history": json.loads(_json(history))}
        cuda = config.device == "cuda:0"
        if cuda:
            _cuda_available()
            torch.cuda.synchronize(0)
            for parameter, values in optimizer.state.items():
                for name, tensor in values.items():
                    _require(tensor.device == (torch.device("cpu") if name == "step" else parameter.device),
                             f"invalid live Adam {name} placement")
        version = ((CUDA_CHECKPOINT_VERSION if mixed_state is None else CUDA_MIXED_CHECKPOINT_VERSION)
                   if cuda else (CHECKPOINT_VERSION if mixed_state is None else MIXED_CHECKPOINT_VERSION))
        payload = dict(version=version, checkpoint_id=checkpoint_id,
            contracts=asdict(AdapterConfig()), model_config=asdict(config), model_state=model.state_dict(),
            model_training=model.training, provenance=provenance,
            training=dict(optimizer=_optimizer_payload(model, optimizer), weights=asdict(weights),
                          progress=asdict(progress), rng=_rng_state(), runtime=runtime_config()))
        payload["training"]["data_order" if mixed_state is None else "mixed_state"] = asdict(
            data_order if mixed_state is None else mixed_state)
        if cuda:
            payload["training"]["rng"]["cuda"] = dict(device="cuda:0", state=torch.cuda.get_rng_state(0))
            payload["training"]["cuda_runtime"] = cuda_runtime_config()
            payload = _cpu_tree(payload)
        _validate_payload(payload)
        path = Path(path)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # atomic no-clobber publication on supported local filesystems
        with_path = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(with_path)
        finally: os.close(with_path)
    except Exception as error:
        if isinstance(error, CheckpointError): raise
        raise CheckpointError(f"cannot save checkpoint {path}: {error}") from error
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def load_inference(path, *, expected_config=None, device="cpu"):
    """Return an eval model and identity metadata; leave all caller RNG untouched.

    Validate the entire common format, including training fields, without
    installing training state or requiring the caller's runtime to match.
    """
    try:
        payload = _read_payload(path)
        model, _, config, *_ = _validate_payload(payload, expected_config)
        config = replace(config, device=device)
        if device != "cpu":
            _cuda_available()
            model.to(device)
        model.eval()
        return InferenceCheckpoint(model, config, payload["checkpoint_id"], payload["provenance"], payload["contracts"])
    except Exception as error:
        raise CheckpointError(f"cannot load inference checkpoint {path}: {error}") from error


def restore_training(path, *, expected_config, expected_provenance, dataset=None, source_id=None,
                     expected_weights=None, expected_optimizer=None, mixed_sources=None,
                     expected_mixed_settings=None, runtime_policy="exact"):
    """Validate expectations first, then return model/Adam and install saved RNG.

    All objects are reconstructed; no caller model/optimizer is partly mutated.
    V1 requires D07 dataset/source expectations. V2 requires reopened MixedSources
    and explicit expected MixedSettings; D07-only resume is rejected.
    Caller configures the recorded execution runtime before restoration. Failures leave
    Python, NumPy and Torch RNG unchanged. Success restores mode and RNG last.
    """
    try:
        _require(type(expected_config) is ModelConfig, "training requires explicit expected ModelConfig")
        payload = _read_payload(path)
        _same(payload.get("provenance"), expected_provenance, "training provenance")
        model, optimizer, config, weights, progress, order, numpy_state = _validate_payload(payload, expected_config)
        if expected_weights is not None:
            _require(type(expected_weights) is LossWeights, "expected_weights must be LossWeights")
            _same(asdict(weights), asdict(expected_weights), "loss weights")
        if expected_optimizer is not None:
            _same({k: v for k, v in payload["training"]["optimizer"].items() if k != "state"},
                  expected_optimizer, "optimizer configuration")
        _require(runtime_policy in ("exact", "compatible"), "unknown resume runtime policy")
        saved_runtime, current_runtime = payload["training"]["runtime"], runtime_config()
        if runtime_policy == "exact":
            _same(saved_runtime, current_runtime, "CPU resumption runtime")
        else:
            # Explicit portability mode tolerates OS descriptions and Python
            # patch versions only. Backend, numeric libraries, machine and all
            # deterministic settings still match. Exact continuation is unclaimed.
            for name, value in saved_runtime.items():
                if name == "platform":
                    continue
                actual = current_runtime[name]
                if name == "python":
                    value, actual = value.split(".")[:2], actual.split(".")[:2]
                _same(value, actual, f"resumption runtime {name}")
            if saved_runtime != current_runtime:
                warnings.warn("compatible resume across OS/Python patch changes; bitwise continuation is unverified", RuntimeWarning)
        from .mixed_state import MixedState, MixedSettings
        if type(order) is MixedState:
            _require(dataset is None and source_id is None and type(expected_mixed_settings) is MixedSettings,
                     "mixed resume requires explicit mixed settings and both sources; D07-only cursor is insufficient")
            _same(asdict(order.settings), asdict(expected_mixed_settings), "mixed settings")
            order.validate_sources(mixed_sources)
        else:
            _require(mixed_sources is None and expected_mixed_settings is None,
                     "D07 checkpoint cannot supply mixed resume state")
            order.validate_dataset(dataset, source_id=source_id)
        if config.device == "cuda:0":
            _same(payload["training"]["cuda_runtime"], cuda_runtime_config(), "CUDA resumption runtime")
            state = payload["training"]["rng"]["cuda"]["state"]
            torch.Generator(device="cuda:0").set_state(state)
            model.to("cuda:0")
            optimizer = _restore_optimizer(model, payload["training"]["optimizer"], progress)
            for parameter, values in optimizer.state.items():
                _require(values["step"].device.type == "cpu", "Adam step must stay on CPU")
                _require(all(t.device == parameter.device for n, t in values.items() if n != "step"),
                         "Adam moments must follow model device")
        model.train(payload["model_training"])
        provenance = json.loads(_json(payload["provenance"]))
        history = provenance.get("resume_runtime_history", [])
        if runtime_policy == "compatible":
            history.append(dict(policy="compatible", checkpoint_id=payload["checkpoint_id"],
                saved_runtime=saved_runtime, restored_runtime=current_runtime,
                differences=[k for k in saved_runtime if saved_runtime[k] != current_runtime[k]],
                bitwise_continuation_verified=False))
            provenance["resume_runtime_history"] = history
        model._resume_runtime_history = history
        if config.device == "cuda:0":
            torch.cuda.set_rng_state(state, 0)
        random.setstate(payload["training"]["rng"]["python"])
        np.random.set_state(numpy_state)
        torch.set_rng_state(payload["training"]["rng"]["torch"])
        return TrainingCheckpoint(model, optimizer, config, payload["checkpoint_id"], provenance,
                                  weights, progress, order if type(order) is DataOrder else None,
                                  order if type(order) is MixedState else None)
    except Exception as error:
        raise CheckpointError(f"cannot restore training checkpoint {path}: {error}") from error
