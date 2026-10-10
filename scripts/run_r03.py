"""Bounded R03 fixture verification only: at most three mixed CPU updates.

Run full, prefix, then resume in three independent processes. JSON receipts bind
each next batch, logits, gradients, model, Adam, sampling and global RNG state.
No games, matches, learning campaign or R04 cycle is performed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from chess_rl.replay import ReplayBuffer
from chess_rl.pgn_replay import file_identity
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (ModelConfig, Progress, _rng_state, load_inference,
    restore_training, runtime_config, save_checkpoint)
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedSources, MixedState

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    if isinstance(value,torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return dict(dtype=str(tensor.dtype),shape=list(tensor.shape),
                    sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
    if isinstance(value,dict):
        return {str(k):fingerprint(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [fingerprint(v) for v in value]
    return value


def tree_hash(value):
    return hashlib.sha256(json.dumps(fingerprint(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def oracle(logits, mask, *, classes=None, targets=None):
    """Independent float64 legal row log-normalization and CE using scalar math."""
    result = []
    for i,row in enumerate(logits):
        indices = np.flatnonzero(mask[i])
        scores = [float(row[j]) for j in indices]
        maximum = max(scores)
        normalizer = math.log(sum(math.exp(x-maximum) for x in scores))
        target = ([float(j==classes[i]) for j in indices] if classes is not None else targets[i,indices])
        result.append(-sum(float(t)*(x-maximum-normalizer) for t,x in zip(target,scores) if t))
    return sum(result)/len(result)


def default_source_config():
    return dict(dataset=dict(bundle=str(ROOT/'runs/d05/tiny-final'),
        manifest_path=str(ROOT/'data/manifests/elite-2025-01-d04-v1.json'),
        games_path='/private/tmp/rl-chess-d03-prepared/games.jsonl',
        split_audit_path='/private/tmp/rl-chess-d03-prepared/split-audit.json',
        prepared='/private/tmp/rl-chess-d04-prepared', teacher_evidence=str(ROOT/'runs/d01/stockfish-provenance.json')),
        replay=str(ROOT/'runs/r02-integration/fresh-replay.sqlite'),
        heldout_games='/private/tmp/rl-chess-d04-prepared/heldout-games.jsonl',
        protected_positions='/private/tmp/rl-chess-d04-prepared/protected-heldout.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('full','prefix','resume'),required=True)
    parser.add_argument('--source-config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refuse overwriting fixture evidence')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    settings = MixedSettings()
    source_config = json.loads(args.source_config.read_text()) if args.source_config else default_source_config()
    if not args.source_config:
        plan = json.loads((ROOT/'runs/r03/frozen-plan.json').read_text())
        for path,identity in plan['inputs'].items():
            assert file_identity(Path(path)) == identity, f'frozen source changed: {path}'
    dataset = open_dataset(**{k:Path(v) for k,v in source_config['dataset'].items()},split='train')
    sources = MixedSources(dataset,ReplayBuffer(source_config['replay']),
        heldout_games_path=source_config['heldout_games'],protected_positions_path=source_config['protected_positions'])
    # Real evidence must match the pre-execution frozen source and settings.
    if not args.source_config:
        assert dataset.source_identity['dataset_id'] == plan['supervised_source']['dataset_id']
        assert sources.replay_id == plan['replay']['export_sha256']
        assert list(sources.replay_membership) == plan['replay']['sample_ids']
        assert len(dataset) == plan['supervised_source']['positions']
        assert settings.supervised_fraction == plan['supervised_fraction']
        assert settings.learning_rate == plan['optimizer']['lr']
        assert asdict(settings.weights) == plan['loss_weights']
    config = ModelConfig()
    provenance = dict(task='R03 bounded fixture only; never pretrained P',
        supervised_source_id=sources.supervised_source_id,replay_id=sources.replay_id,
        settings=asdict(settings))
    if args.mode == 'resume':
        # Scramble all streams; successful continuation must come from saved state.
        random.seed(9); np.random.seed(8); torch.manual_seed(7)
        restored = restore_training(args.checkpoint,expected_config=config,
            expected_provenance=provenance,mixed_sources=sources,expected_mixed_settings=settings)
        model,opt,state = restored.model,restored.optimizer,restored.mixed_state
        assert restored.data_order is None and state.updates == 1
    else:
        model = config.build()
        opt = torch.optim.Adam(model.parameters(),lr=settings.learning_rate,betas=(.9,.999),eps=1e-8,
            weight_decay=0.0,amsgrad=False,foreach=False,fused=False)
        state = MixedState.initial(sources,settings,seed=20261010)
    records = []
    captured = []
    def capture(module, inputs, output):
        for tensor in output: tensor.retain_grad()
        captured[:] = output
    hook = model.register_forward_hook(capture)
    stop = 1 if args.mode == 'prefix' else 3
    while state.updates < stop:
        si,ri,candidate = state.next_selection()
        supervised,replay = sources.batches(si,ri)
        metrics,state = mixed_step(model,opt,sources,state)
        assert state == candidate
        p,w = [v.detach().numpy() for v in captured]
        sp = oracle(p[:2],supervised.legal_mask,classes=supervised.policy_classes)
        sw = oracle(w[:2],np.ones((2,3),bool),targets=supervised.wdl)
        rp = oracle(p[2:],replay.legal_mask,targets=replay.policy_distributions)
        rw = oracle(w[2:],np.ones((2,3),bool),targets=replay.wdl)
        expected = .5*(sp+sw+rp+rw)
        assert abs(metrics.total_loss-expected) < 2e-6
        for actual,target in ((metrics.supervised.policy_loss,sp),(metrics.supervised.wdl_loss,sw),
                             (metrics.selfplay.policy_loss,rp),(metrics.selfplay.wdl_loss,rw)):
            assert abs(actual-target) < 2e-6
        gradient = {name:p.grad for name,p in model.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() for g in gradient.values())
        mask = torch.tensor(np.concatenate((supervised.legal_mask,replay.legal_mask)))
        assert not torch.count_nonzero(captured[0].grad[~mask])
        assert metrics.supervised_contribution > 0 and metrics.selfplay_contribution > 0
        records.append(dict(update=state.updates,metrics=asdict(metrics),
            independent=dict(supervised_policy=sp,supervised_wdl=sw,selfplay_policy=rp,selfplay_wdl=rw,total=expected),
            supervised_indices=si,selfplay_indices=ri,supervised_references=supervised.references,
            selfplay_sample_ids=replay.sample_ids,teacher_classes=supervised.policy_classes.tolist(),
            soft_teacher_wdl=supervised.wdl.tolist(),final_selfplay_wdl=replay.wdl.tolist(),
            visit_nonzero_counts=np.count_nonzero(replay.policy_distributions,axis=1).tolist(),
            batch_sha256=tree_hash([torch.tensor(supervised.features),torch.tensor(replay.features)]),
            logits_sha256=tree_hash(captured),logit_gradients_sha256=tree_hash([v.grad for v in captured]),
            gradients_sha256=tree_hash(gradient),finite_parameter_gradients=len(gradient),
            model_sha256=tree_hash(model.state_dict()),optimizer_sha256=tree_hash(opt.state_dict()),
            state_sha256=tree_hash(asdict(state)),rng_sha256=tree_hash(_rng_state())))
    hook.remove()
    if args.mode == 'prefix':
        before_rng = tree_hash(_rng_state())
        save_checkpoint(args.checkpoint,model=model,optimizer=opt,checkpoint_id='R03-fixture-step-1',
            provenance=provenance,weights=settings.weights,progress=Progress(1,0,4),mixed_state=state)
        assert before_rng == tree_hash(_rng_state())
        # Both checkpoint versions must be consumable by the common inference API.
        loaded = load_inference(args.checkpoint,expected_config=config)
        assert tree_hash(loaded.model.state_dict()) == tree_hash(model.state_dict())
        assert before_rng == tree_hash(_rng_state())
    final = dict(model_sha256=tree_hash(model.state_dict()),optimizer_sha256=tree_hash(opt.state_dict()),
        gradients_sha256=tree_hash({name:p.grad for name,p in model.named_parameters()}),
        rng_sha256=tree_hash(_rng_state()),mixed_state=asdict(state),
        progress=asdict(Progress(state.updates,0,state.updates*4)))
    result = dict(mode=args.mode,runtime=runtime_config(),source_config=source_config,
        supervised_source_id=sources.supervised_source_id,supervised_membership_sha256=sources.supervised_membership_sha256,
        replay_id=sources.replay_id,replay_membership=sources.replay_membership,
        excluded_heldout=sources.excluded_heldout,settings=asdict(settings),steps=records,final=final)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(mode=args.mode,updates=[r['update'] for r in records],
        total_losses=[r['metrics']['total_loss'] for r in records],excluded_heldout=len(sources.excluded_heldout))))


if __name__ == '__main__': main()
