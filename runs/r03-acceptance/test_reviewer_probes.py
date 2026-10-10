"""Reviewer-authored numerical, sampling, integrity and failure probes."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from chess_rl.adapter import make_state
from chess_rl.position_sampling import rule_position
from chess_rl.replay import ReplayBuffer
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import CheckpointError, ModelConfig, Progress, DataOrder, save_checkpoint, restore_training, load_inference
from chess_rl.training.mixed import mixed_losses, mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedState, MixedSources, SourceCursor
from tests.test_checkpoints import rng_snapshot, assert_rng_equal, assert_tree_equal
from tests.test_selfplay import cpu, fixture_checkpoint, create
from tests.test_mixed_update import sources, optimizer, open_sources, save_mixed
from tests.test_supervised import fixture_bundle
from tests.test_replay import played, BLACK_WIN


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('ns,nr',[(1,3),(3,1),(2,2)])
@pytest.mark.parametrize('weights',[(1.,1.),(.37,1.91)])
def test_full_mixed_loss_gradient_and_finite_difference(dtype,ns,nr,weights):
    settings = MixedSettings(batch_size=ns+nr,supervised_batch_size=ns,
        supervised_fraction=ns/(ns+nr),policy_weight=weights[0],wdl_weight=weights[1])
    generator=np.random.default_rng(385)
    b=ns+nr
    p=generator.normal(size=(b,4352)); p[:,100:]=1e5
    w=generator.normal(size=(b,3))
    mask=np.zeros((b,4352),bool)
    mask[:,[4,27,51,88]]=True
    targets=np.zeros((b,4352)); targets[:ns,27]=1
    targets[ns:,4]=.2; targets[ns:,51]=.8
    wt=np.tile([.125,.375,.5],(b,1)); wt[ns:]=[0,0,1]
    expected_p=np.zeros_like(p); expected_w=np.zeros_like(w)
    def independent(scores,values,legal):
        losses=[]; gradients=np.zeros_like(scores)
        for i in range(b):
            ids=np.nonzero(legal[i])[0]
            x=scores[i,ids]; t=values[i,ids]
            shifted=x-x.max(); exps=np.array([math.exp(float(v)) for v in shifted])
            logprob=shifted-math.log(math.fsum(exps))
            losses.append(-math.fsum(float(a)*float(z) for a,z in zip(t,logprob)))
            gradients[i,ids]=(exps/math.fsum(exps)*math.fsum(t)-t)/b
        return math.fsum(losses)/b,gradients
    pl,expected_p=independent(p,targets,mask)
    wl,expected_w=independent(w,wt,np.ones((b,3),bool))
    pt=torch.tensor(p,dtype=dtype,requires_grad=True); wtensor=torch.tensor(w,dtype=dtype,requires_grad=True)
    kwargs=dict(supervised_mask=torch.tensor(mask[:ns]),classes=torch.full((ns,),27,dtype=torch.int64),
        teacher_wdl=torch.tensor(wt[:ns],dtype=dtype),selfplay_mask=torch.tensor(mask[ns:]),
        visits=torch.tensor(targets[ns:],dtype=dtype),final_wdl=torch.tensor(wt[ns:],dtype=dtype),settings=settings)
    loss,metrics=mixed_losses(pt,wtensor,**kwargs); loss.backward()
    assert loss.item()==pytest.approx(weights[0]*pl+weights[1]*wl,abs=2e-6)
    np.testing.assert_allclose(pt.grad.numpy(),weights[0]*expected_p,atol=2e-7,rtol=5e-6)
    np.testing.assert_allclose(wtensor.grad.numpy(),weights[1]*expected_w,atol=2e-7,rtol=5e-6)
    assert np.count_nonzero(pt.grad.numpy()[~mask])==0
    h=1e-5; plus=p.copy(); minus=p.copy(); plus[-1,51]+=h; minus[-1,51]-=h
    diff=(independent(plus,targets,mask)[0]-independent(minus,targets,mask)[0])/(2*h)
    assert diff==pytest.approx(expected_p[-1,51],abs=1e-9)
    assert metrics.supervised.batch_size==ns and metrics.selfplay.batch_size==nr


@pytest.mark.parametrize('sizes,counts', [((1,1),(3,5)),((3,5),(2,3)),((12,4),(2,2)),((7,2),(5,1))])
def test_sampler_exact_chunk_oracle_across_many_epochs(sizes,counts):
    ns,nr=sizes; cs,cr=counts; seed=1991
    settings=MixedSettings(batch_size=cs+cr,supervised_batch_size=cs,supervised_fraction=cs/(cs+cr))
    rng=random.Random(seed); orders=[list(range(ns)),list(range(nr))]
    for order in orders: rng.shuffle(order)
    state=MixedState(settings,'a'*64,'b'*64,'c'*64,tuple(f'{i:064x}' for i in range(nr)),
        SourceCursor(tuple(orders[0])),SourceCursor(tuple(orders[1])),rng.getstate())
    cursors=[0,0]; epochs=[0,0]; seen=[0,0]
    before=rng_snapshot()
    for _ in range(101):
        expected=[]
        for j,count in enumerate(counts):
            indices=[]
            while count:
                if cursors[j]==sizes[j]:
                    orders[j]=list(range(sizes[j])); rng.shuffle(orders[j]); cursors[j]=0; epochs[j]+=1
                take=min(count,sizes[j]-cursors[j]); indices+=orders[j][cursors[j]:cursors[j]+take]
                cursors[j]+=take; seen[j]+=take; count-=take
            expected.append(tuple(indices))
        s,r,state=state.next_selection()
        assert (s,r)==tuple(expected)
        for j,source in enumerate((state.supervised,state.selfplay)):
            assert asdict(source)==dict(order=tuple(orders[j]),cursor=cursors[j],epoch=epochs[j],samples_seen=seen[j])
        assert state.sampling_rng==rng.getstate()
        assert MixedState.from_dict(asdict(state))==state
    assert_rng_equal(before,rng_snapshot())


def test_reopened_receipt_only_drift_rejected_before_rng(sources,tmp_path,fixture_checkpoint):
    data,cfg=sources
    path=tmp_path/'boundary.pt'; args=save_mixed(path,data)
    buffer=ReplayBuffer(cfg['replay'])
    buffer.ingest(create(tmp_path,fixture_checkpoint,name='new-unfinished',cap=2,budget=1).run())
    reopened=open_sources(cfg)
    assert reopened.replay_membership==data.replay_membership and reopened.replay_id!=data.replay_id
    args['mixed_sources']=reopened
    before=rng_snapshot()
    with pytest.raises(CheckpointError,match='source identity'): restore_training(path,**args)
    assert_rng_equal(before,rng_snapshot())


def test_entire_heldout_game_blocked_even_unprotected_roots(tmp_path,fixture_checkpoint):
    args=fixture_bundle(tmp_path,specifications=[
        ('train','4k3/8/8/8/8/8/8/R3K3 w - - 0 1',[],[0],None),
        ('validation',make_state().board.fen(),BLACK_WIN,[1],None)])
    games=[json.loads(line) for line in args['games_path'].read_text().splitlines()]
    held=next(g for g in games if g['split']=='validation')
    buffer=ReplayBuffer.create(tmp_path/'held.sqlite')
    session=played(tmp_path,fixture_checkpoint,state=make_state(held['start_fen'],history_complete=True))
    buffer.ingest(session.snapshot())
    protected={json.loads(line)['rule_position'] for line in (args['prepared']/'protected-heldout.jsonl').read_text().splitlines()}
    assert sum(rule_position(make_state(s['initial_fen'],s['moves_uci'],history_complete=True).board) in protected
        for s in buffer.export()['samples'])==1
    with pytest.raises(ValueError,match='no completed non-held-out'):
        MixedSources(open_dataset(**args,split='train'),buffer,heldout_games_path=args['prepared']/'heldout-games.jsonl',
            protected_positions_path=args['prepared']/'protected-heldout.jsonl')


def test_owned_supervised_identity(sources):
    data,_=sources
    identity=data.dataset.source_identity; identity['split']='test'; identity['authoritative'].clear()
    assert data.dataset.source_identity['split']=='train'
    data.validate_dataset()


def test_optimizer_partial_failure_requires_boundary_restore(sources,tmp_path):
    data,_=sources
    path=tmp_path/'safe.pt'; args=save_mixed(path,data)
    restored=restore_training(path,**args); state=restored.mixed_state
    before=deepcopy(restored.model.state_dict()); expected_sampling=state.next_selection()
    def partial_failure(opt,a,k):
        with torch.no_grad(): next(restored.model.parameters()).add_(1.)
        raise RuntimeError('controlled partial optimizer failure')
    handle=restored.optimizer.register_step_pre_hook(partial_failure)
    with pytest.raises(RuntimeError,match='partial optimizer failure'):
        mixed_step(restored.model,restored.optimizer,data,state)
    handle.remove()
    assert state.next_selection()==expected_sampling
    assert any(not torch.equal(v,restored.model.state_dict()[k]) for k,v in before.items())
    recovered=restore_training(path,**args)
    assert_tree_equal(before,recovered.model.state_dict())
    assert recovered.mixed_state==state


def test_actual_sources_independent_parameter_gradients():
    from scripts.run_r03 import default_source_config
    data=open_sources(default_source_config())
    settings=MixedSettings(); state=MixedState.initial(data,settings,seed=20261010)
    si,ri,_=state.next_selection(); s,r=data.batches(si,ri)
    model=ModelConfig().build(); reference=ModelConfig().build(); reference.load_state_dict(model.state_dict())
    opt=optimizer(model,settings); reference_opt=optimizer(reference,settings)
    rng=torch.get_rng_state().clone()
    metrics,_=mixed_step(model,opt,data,state)
    torch.set_rng_state(rng)
    reference.train()
    p,w=reference(torch.tensor(np.concatenate((s.features,r.features))))
    terms=[]
    allmask=np.concatenate((s.legal_mask,r.legal_mask))
    for i in range(4):
        ids=torch.tensor(np.flatnonzero(allmask[i]))
        legal=p[i,ids]; norm=torch.logsumexp(legal,0)
        if i<2: policy=norm-p[i,int(s.policy_classes[i])]
        else: policy=-(torch.tensor(r.policy_distributions[i-2,ids.numpy()])*(legal-norm)).sum()
        target=torch.tensor(s.wdl[i] if i<2 else r.wdl[i-2])
        value=-(target*(w[i]-torch.logsumexp(w[i],0))).sum()
        terms.append(policy+value)
    loss=sum(terms)/4; loss.backward()
    assert loss.item()==pytest.approx(metrics.total_loss,abs=2e-6)
    maximum=0.
    for (name,a),(name2,b) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==name2
        maximum=max(maximum,float((a.grad-b.grad).abs().max()))
        torch.testing.assert_close(a.grad,b.grad,atol=5e-7,rtol=2e-4)
    assert len(list(model.parameters()))==215
    print('actual source independent parameter gradients: 215 tensors; max abs difference',maximum)
