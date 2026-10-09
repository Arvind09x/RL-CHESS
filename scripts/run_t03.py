"""Freeze, execute once, then verify T03 in a fresh process. No held-out data."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (ModelConfig, DataOrder, Progress,
    save_checkpoint, load_inference, restore_training, runtime_config)
from chess_rl.training.trainer import LossWeights
from chess_rl.training.tiny_overfit import FitGate, measure, bounded_fit

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'runs/t03/frozen-plan.json'
BASE = '4fa48d4'


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def write_new(path, data):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def dataset():
    bundle = ROOT / 'runs/d05/tiny-final'
    assert all(json.loads(line)['split'] == 'train' for line in (bundle / 'labels.jsonl').read_text().splitlines())
    return open_dataset(bundle=bundle,
        manifest_path=ROOT / 'data/manifests/elite-2025-01-d04-v1.json',
        games_path=Path('/private/tmp/rl-chess-d03-prepared/games.jsonl'),
        split_audit_path=Path('/private/tmp/rl-chess-d03-prepared/split-audit.json'),
        prepared=Path('/private/tmp/rl-chess-d04-prepared'),
        teacher_evidence=ROOT / 'runs/d01/stockfish-provenance.json', split='train')


def tracked_inputs():
    names = ['data/manifests/elite-2025-01-d04-v1.json', 'runs/d01/stockfish-provenance.json',
        *['runs/d05/tiny-final/' + n for n in ('labels.jsonl', 'selected.jsonl', 'attempts.jsonl',
            'exclusions.jsonl', 'settings.json', 'summary.json')]]
    paths = [ROOT / n for n in names]
    paths += [Path('/private/tmp/rl-chess-d03-prepared') / n for n in ('games.jsonl', 'split-audit.json')]
    paths += [Path('/private/tmp/rl-chess-d04-prepared') / n for n in
        ('samples.jsonl', 'overlap-audit.json', 'heldout-games.jsonl', 'protected-heldout.jsonl', 'exclusions.jsonl')]
    return {str(p): identity(p) for p in paths}


def source_hashes():
    paths = [*sorted((ROOT / 'src').rglob('*.py')), ROOT / 'scripts/run_t03.py',
             ROOT / 'tests/test_tiny_overfit.py', ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    return {str(p.relative_to(ROOT)): identity(p) for p in paths}


def tensor_hashes(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape),
            sha256=hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    if isinstance(value, dict): return {str(k): tensor_hashes(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [tensor_hashes(v) for v in value]
    return value


def arrays(batch):
    return {name: hashlib.sha256(getattr(batch, name).tobytes()).hexdigest()
            for name in ('features', 'legal_mask', 'policy_classes', 'wdl')}


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def freeze(directory):
    data = dataset()
    batch = next(data.batches(8))
    assert len(data) == 12 and len(batch.references) == 8 and batch.split == 'train'
    plan = dict(task='T03', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git', 'rev-parse', BASE], cwd=ROOT, text=True).strip(),
        reviewed_dependencies=dict(T02='27731e32d8aa64d852b439945f30e7238e87a824',
            integrated_main=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()),
        selection='first eight D07 source-order D05 training labels; repeat exactly this full batch',
        references=batch.references, policy_classes=batch.policy_classes.tolist(), teacher_wdl=batch.wdl.tolist(),
        batch_array_sha256=arrays(batch), split='train', seed=20261009, batch_size=8,
        source_dataset_size=12, unused_training_labels=4, model=asdict(ModelConfig(dropout=0.0)),
        optimizer=dict(lr=0.0003, betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
            amsgrad=False, foreach=False, fused=False), loss_weights=asdict(LossWeights()), gate=asdict(FitGate()),
        losses='T01 mean legal teacher-class CE + mean soft root-side WDL CE; weights 1:1',
        metrics='legal argmax accuracy; KL(teacher || model), nats; WDL CE versus mean teacher entropy',
        evaluation='same fixed training batch at baseline and after every update; first passing update stops',
        schedule='constant Adam; no scheduler, clipping, accumulation, augmentation or shuffle',
        checkpoint_role='T03 plumbing only; never immutable pretrained P',
        runtime=runtime_config(), inputs=tracked_inputs(), tested_sources=source_hashes(),
        external_directory=str(directory.resolve()), reload_tolerance=0,
        failure_policy='retain final curves/checkpoint and report failed_step_cap; never revise thresholds after execution')
    write_new(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), identity=identity(PLAN), references=plan['references'], gate=plan['gate']), indent=2))


def checked_plan():
    plan = json.loads(PLAN.read_text())
    assert plan['inputs'] == tracked_inputs(), 'accepted input bytes drifted'
    assert plan['tested_sources'] == source_hashes(), 'frozen source bytes drifted'
    assert plan['runtime'] == runtime_config(), 'frozen runtime drifted'
    assert plan['split'] == 'train' and plan['model']['dropout'] == 0
    data = dataset()
    batch = next(data.batches(plan['batch_size']))
    assert list(map(list, batch.references)) == plan['references']
    assert arrays(batch) == plan['batch_array_sha256']
    return plan, data, batch


def run():
    plan, data, batch = checked_plan()
    directory = Path(plan['external_directory'])
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / 'execution-start.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        frozen_plan_identity=identity(PLAN), frozen_plan=plan))
    random.seed(plan['seed']); np.random.seed(plan['seed']); torch.manual_seed(plan['seed'])
    model = ModelConfig(**plan['model']).build()
    assert sum(p.numel() for p in model.parameters()) == 5049411
    assert all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
    options = plan['optimizer'] | {'betas': tuple(plan['optimizer']['betas'])}
    optimizer = torch.optim.Adam(model.parameters(), **options)
    weights = LossWeights(**plan['loss_weights'])
    started = time.monotonic()
    with (directory / 'learning-curves.jsonl').open('x') as handle:
        def record(row):
            row['elapsed_seconds'] = time.monotonic() - started
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n'); handle.flush()
            if row['step'] % 25 == 0 or FitGate(**plan['gate']).passes(row['metrics']):
                print(json.dumps(row), flush=True)
        result = bounded_fit(model, optimizer, batch, weights=weights, gate=FitGate(**plan['gate']), record=record)
    assert arrays(batch) == plan['batch_array_sha256']
    step = result['step']
    source_id = identity(ROOT / 'runs/d05/tiny-final/labels.jsonl')['sha256']
    # Bind the complete D07 source to T02. Only the first eight indices have
    # been used; repeated-batch scheduling is frozen in provenance, not epochs.
    order = DataOrder(source_id, DataOrder.reference_hash(data), tuple(range(len(data))), 8, 8, False, 0)
    progress = Progress(step, 0, step * 8)
    prov = dict(task='T03', role=plan['checkpoint_role'], frozen_plan_sha256=identity(PLAN)['sha256'],
        frozen_plan=plan, fixed_batch_repetitions=step)
    checkpoint = directory / 'final.pt'
    save_checkpoint(checkpoint, model=model, optimizer=optimizer, checkpoint_id=f'T03-plumbing-step-{step}',
        provenance=prov, weights=weights, progress=progress, data_order=order)
    expected = dict(metrics=measure(model, batch), model=tensor_hashes(model.state_dict()),
        optimizer=tensor_hashes(optimizer.state_dict()), progress=asdict(progress),
        data_order=asdict(order), weights=asdict(weights), provenance=prov,
        config=asdict(ModelConfig(**plan['model'])), checkpoint_identity=identity(checkpoint))
    write_new(directory / 'expected-reload.json', expected)
    write_new(directory / 'result.json', result | dict(frozen_plan_identity=identity(PLAN),
        runtime=runtime_config(), training_seconds=time.monotonic() - started, checkpoint_identity=identity(checkpoint),
        batch_unchanged=True, accepted_inputs_unchanged=tracked_inputs() == plan['inputs'],
        parameter_count=5049411, dropout=0.0, fresh_process_reload='pending'))
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'verify'], cwd=ROOT, check=True)
    print(json.dumps(result, indent=2))


def verify():
    plan, data, batch = checked_plan()
    directory = Path(plan['external_directory'])
    expected = json.loads((directory / 'expected-reload.json').read_text())
    random.seed(997); np.random.seed(998); torch.manual_seed(999)
    config = ModelConfig(**plan['model'])
    loaded = load_inference(directory / 'final.pt', expected_config=config)
    assert measure(loaded.model, batch) == expected['metrics']
    assert tensor_hashes(loaded.model.state_dict()) == expected['model']
    source_id = identity(ROOT / 'runs/d05/tiny-final/labels.jsonl')['sha256']
    restored = restore_training(directory / 'final.pt', expected_config=config,
        expected_provenance=expected['provenance'], dataset=data, source_id=source_id,
        expected_weights=LossWeights(**plan['loss_weights']))
    assert measure(restored.model, batch) == expected['metrics']
    assert tensor_hashes(restored.model.state_dict()) == expected['model']
    assert tensor_hashes(restored.optimizer.state_dict()) == expected['optimizer']
    assert json.loads(json.dumps(asdict(restored.data_order))) == expected['data_order']
    assert asdict(restored.progress) == expected['progress']
    assert asdict(restored.weights) == expected['weights']
    assert all(layer.self_attn.gab.bank is restored.model.backbone.bank for layer in restored.model.backbone.layers)
    assert identity(directory / 'final.pt') == expected['checkpoint_identity']
    report = dict(fresh_python_process=True, exact_model_tensors=True, exact_adam_state=True,
        exact_inference_and_training_metrics=True, progress_order_weights_provenance_verified=True,
        shared_gab_identity=True, tolerance=0, checkpoint=identity(directory / 'final.pt'),
        frozen_plan=identity(PLAN), final_metrics=expected['metrics'], inputs_unchanged=True,
        limits='reload correctness only; no extra optimizer update or playing-strength claim')
    write_new(directory / 'reload-verification.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run', 'verify'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    configure()
    if args.mode == 'freeze':
        if args.directory is None: parser.error('freeze requires --directory')
        freeze(args.directory)
    elif args.mode == 'run': run()
    else: verify()


if __name__ == '__main__': main()
