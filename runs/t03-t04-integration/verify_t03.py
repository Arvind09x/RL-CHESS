"""Read-only verification of historical T03 artifact in the combined source tree.

Never call the frozen runner's checked_plan, freeze, run or verify entrypoints:
its source-set identity remains bound to the original isolated T03 checkout.
"""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from scripts.run_t03 import dataset, arrays, tensor_hashes
from chess_rl.training.tiny_overfit import measure, FitGate
from chess_rl.training.checkpoints import ModelConfig
from chess_rl.search.real_evaluator import RealEvaluator
from run_checks import ROOT, OUT, CHECKPOINT, identity, save, SOURCES

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
plan=json.loads((ROOT/'runs/t03/frozen-plan.json').read_text())
expected=json.loads((ROOT/'runs/t03/expected-reload.json').read_text())
assert identity(CHECKPOINT)==expected['checkpoint_identity']
# Verify original source identity at original path, with no added T04 source-set freeze.
original=ROOT.parent/'t03-tiny-overfit'
for n,target in plan['tested_sources'].items():
    assert identity(original/n)==target
    assert identity(ROOT/n)==target
config=ModelConfig(**plan['model'])
evaluator=RealEvaluator(CHECKPOINT,expected_config=config)
loaded=evaluator.checkpoint
assert tensor_hashes(loaded.model.state_dict())==expected['model']
assert loaded.provenance==expected['provenance'] and asdict(loaded.config)==expected['config']
assert loaded.checkpoint_id=='T03-plumbing-step-26'
bank=loaded.model.backbone.bank
assert all(layer.self_attn.gab.bank is bank and layer.self_attn.relative is not None
           for layer in loaded.model.backbone.layers)
data=dataset()
batch=next(data.batches(plan['batch_size']))
assert arrays(batch)==plan['batch_array_sha256']
assert list(map(list,batch.references))==plan['references']
actual=measure(loaded.model,batch)
assert actual==expected['metrics']
curves=[json.loads(row) for row in (ROOT/'runs/t03/learning-curves.jsonl').read_text().splitlines()]
gate=FitGate(**plan['gate'])
assert [row['step'] for row in curves]==list(range(27))
assert [row['step'] for row in curves if gate.passes(row['metrics'])]==[26]
assert curves[-1]['metrics']==actual
save(OUT/'t03-checkpoint-identity-metrics.json',dict(checkpoint_path=str(CHECKPOINT),
    checkpoint_identity=identity(CHECKPOINT),exact_all_model_tensor_hashes=True,
    model_tensor_entries=len(expected['model']),exact_final_metrics=True,metrics=actual,
    frozen_plan_identity=identity(ROOT/'runs/t03/frozen-plan.json'),
    original_frozen_source_identity_verified_at=str(original),
    first_passing_step_from_preserved_curves=26,curves_identity=identity(ROOT/'runs/t03/learning-curves.jsonl'),
    t04_shared_checkpoint_loader_used=True,shared_gab_and_relative_terms_verified=True,
    integration_source_commits=SOURCES,optimizer_updates_performed=0,
    scope='Separate read-only integration verification; original overfit and freeze untouched'))
print('Actual T03 checkpoint: all model hashes and final metrics exact; first passing step 26; zero updates')
