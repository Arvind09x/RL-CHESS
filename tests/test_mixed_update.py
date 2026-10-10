"""R03 bounded CPU oracles, source gates and mixed persistence."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from chess_rl.adapter import make_state
from chess_rl.replay import ReplayBuffer
from chess_rl.selfplay import Phase
from chess_rl.supervised import open_dataset
from chess_rl.training.checkpoints import (CheckpointError, ModelConfig, Progress,
    load_inference, restore_training, save_checkpoint)
from chess_rl.training.losses import policy_distribution_cross_entropy
from chess_rl.training.mixed import mixed_step, mixed_losses
from chess_rl.training.mixed_state import MixedSettings, MixedSources, MixedState, SourceCursor
from .test_checkpoints import assert_tree_equal, rng_snapshot, assert_rng_equal
from .test_selfplay import cpu, fixture_checkpoint, create
from .test_replay import played
from .test_supervised import fixture_bundle


def optimizer(model, settings):
    return torch.optim.Adam(model.parameters(), lr=settings.learning_rate, betas=(.9, .999),
        eps=1e-8, weight_decay=0.0, amsgrad=False, foreach=False, fused=False)


def source_config(args, buffer):
    return dict(dataset={k:str(v) for k,v in args.items()}, replay=str(buffer.path),
        heldout_games=str(args['prepared']/'heldout-games.jsonl'),
        protected_positions=str(args['prepared']/'protected-heldout.jsonl'))


def open_sources(config):
    return MixedSources(open_dataset(**{k:Path(v) for k,v in config['dataset'].items()}, split='train'),
        ReplayBuffer(config['replay']), heldout_games_path=config['heldout_games'],
        protected_positions_path=config['protected_positions'])


@pytest.fixture
def sources(tmp_path, fixture_checkpoint):
    args = fixture_bundle(tmp_path)
    buffer = ReplayBuffer.create(tmp_path/'replay.sqlite')
    buffer.ingest(played(tmp_path, fixture_checkpoint).snapshot())
    # Real R01 soft visit distribution, exact final draw; <= one controlled move.
    session = create(tmp_path, fixture_checkpoint,
        make_state('4k3/8/8/8/8/8/8/R3K3 w - - 99 1', history_complete=True),
        name='soft-draw', cap=1, budget=3, phases=(Phase(0, 0., 0.),))
    buffer.ingest(session.run())
    # Unfinished source remains in receipts and contributes no samples.
    buffer.ingest(create(tmp_path, fixture_checkpoint, name='truncated', cap=1, budget=1).run())
    config = source_config(args, buffer)
    return open_sources(config), config


def oracle(logits, mask, *, classes=None, targets=None):
    """Float64 math, no production normalization or Torch CE; returns all gradients."""
    grad = np.zeros_like(logits, dtype=np.float64)
    losses = []
    for i in range(len(logits)):
        indices = np.flatnonzero(mask[i])
        row = logits[i, indices].astype(np.float64)
        shifted = row - row.max()
        z = sum(math.exp(v) for v in shifted)
        logp = shifted - math.log(z)
        p = np.exp(logp)
        if classes is not None:
            target = np.array([float(j == classes[i]) for j in indices])
        else:
            target = targets[i, indices].astype(np.float64)
        losses.append(-sum(t*l for t,l in zip(target,logp) if t))
        grad[i,indices] = (p*sum(target) - target)/len(logits)
    return sum(losses)/len(losses), grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('size', [1, 3])
def test_distribution_oracle_all_gradients_and_safe_zero(dtype, size):
    mask = torch.zeros((size,4352), dtype=torch.bool)
    mask[:,[7,65,1000]] = True
    logits = torch.full((size,4352), 1e4, dtype=dtype)
    logits[:,[7,65,1000]] = torch.tensor([.7,-1.2,2.1], dtype=dtype)
    logits.requires_grad_()
    targets = torch.zeros_like(logits)
    targets[:,7] = .25; targets[:,1000] = .75
    expected, grad = oracle(logits.detach().numpy(), mask.numpy(), targets=targets.numpy())
    loss = policy_distribution_cross_entropy(logits, mask, targets)
    loss.backward()
    assert loss.item() == pytest.approx(expected, abs=2e-6)
    np.testing.assert_allclose(logits.grad.numpy(), grad, atol=2e-7, rtol=3e-6)
    assert torch.isfinite(logits.grad).all() and not torch.count_nonzero(logits.grad[~mask])
    mask[:,[65,1000]] = False
    targets.zero_(); targets[:,7] = 1
    assert policy_distribution_cross_entropy(logits,mask,targets).item() == 0


@pytest.mark.parametrize('bad', ['negative','illegal','sum','nan','empty','shape','grad','mask_dtype','integer'])
def test_distribution_rejects_invalid_target(bad):
    logits = torch.zeros((1,4352))
    mask = torch.zeros_like(logits,dtype=torch.bool); mask[:,7] = True
    target = torch.zeros_like(logits); target[:,7] = 1
    if bad == 'negative': target[:,8] = -1
    elif bad == 'illegal': target[:,7] = 0; target[:,8] = 1
    elif bad == 'sum': target[:,7] = .5
    elif bad == 'nan': target[:,7] = torch.nan
    elif bad == 'empty': mask[:] = False
    elif bad == 'shape': target = target[:,:3]
    elif bad == 'grad': target.requires_grad_()
    elif bad == 'mask_dtype': mask = mask.float()
    elif bad == 'integer': target = target.long()
    with pytest.raises(ValueError): policy_distribution_cross_entropy(logits,mask,target)


def test_mixed_update_independent_losses_both_contributions_gradients_and_inputs(sources):
    data, _ = sources
    settings = MixedSettings(policy_weight=.7, wdl_weight=1.3)
    state = MixedState.initial(data, settings, seed=917)
    si, ri, expected_state = state.next_selection()
    s, r = data.batches(si,ri)
    original = deepcopy((s,r))
    model = ModelConfig(d_policy=23,d_value=19,dropout=.23).build()
    opt = optimizer(model,settings)
    output = []
    def capture(module, args, result):
        for tensor in result: tensor.retain_grad()
        output.extend(result)
    hook = model.register_forward_hook(capture)
    metrics, after = mixed_step(model,opt,data,state)
    hook.remove()
    assert after == expected_state and state.updates == 0
    p, w = [v.detach().numpy() for v in output]
    sp, sg = oracle(p[:2], s.legal_mask, classes=s.policy_classes)
    sw, swg = oracle(w[:2], np.ones((2,3),bool), targets=s.wdl)
    rp, rg = oracle(p[2:], r.legal_mask, targets=r.policy_distributions)
    rw, rwg = oracle(w[2:], np.ones((2,3),bool), targets=r.wdl)
    assert metrics.supervised.batch_size == metrics.selfplay.batch_size == 2
    for actual, expected in ((metrics.supervised.policy_loss,sp),(metrics.supervised.wdl_loss,sw),
                             (metrics.selfplay.policy_loss,rp),(metrics.selfplay.wdl_loss,rw),
                             (metrics.total_loss,.5*(.7*sp+1.3*sw+.7*rp+1.3*rw))):
        assert actual == pytest.approx(expected,abs=2e-6)
    np.testing.assert_allclose(output[0].grad.numpy(), .5*.7*np.concatenate((sg,rg)),atol=2e-7,rtol=4e-6)
    np.testing.assert_allclose(output[1].grad.numpy(), .5*1.3*np.concatenate((swg,rwg)),atol=2e-7,rtol=4e-6)
    assert metrics.supervised_contribution > 0 and metrics.selfplay_contribution > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for a,b in zip(original,(s,r)):
        for field in a.__dataclass_fields__:
            if isinstance(getattr(a,field),np.ndarray):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    # Contracts stay integer classes / soft teacher WDL and dense visits / exact outcomes.
    assert s.policy_classes.dtype == np.int64 and np.any((s.wdl>0)&(s.wdl<1))
    assert r.policy_distributions.shape == (2,4352) and np.isin(r.wdl,[0,1]).all()


def test_sampler_crosses_both_boundaries_deterministically_without_global_rng(sources):
    data,_ = sources
    before = rng_snapshot()
    state = MixedState.initial(data,MixedSettings(),seed=21)
    state2 = MixedState.from_dict(asdict(state))
    for i in range(8):
        a,b,next_state = state.next_selection()
        assert (a,b,next_state) == state2.next_selection()
        assert next_state.supervised.samples_seen == next_state.selfplay.samples_seen == 2*(i+1)
        state = state2 = next_state
    assert state.supervised.epoch > 0 and state.selfplay.epoch > 0
    assert_rng_equal(before,rng_snapshot())


def test_incomplete_receipts_have_no_training_membership(sources):
    data,cfg = sources
    exported = ReplayBuffer(cfg['replay']).export()
    assert exported['summary']['source_counts']['excluded-truncated'] == 1
    assert len(data.replay_membership) == len(exported['samples']) == 5
    assert all(s['source'] == 'completed-R01-self-play' for s in exported['samples'])


@pytest.mark.parametrize('bad',['soft_outcome','wrong_counts','classes_as_visits','nonfinite_logits'])
def test_distinct_mixed_target_contracts(sources,bad):
    data,_ = sources
    settings = MixedSettings()
    si,ri,_ = MixedState.initial(data,settings,seed=0).next_selection()
    s,r = data.batches(si,ri)
    p,w = torch.zeros((4,4352)),torch.zeros((4,3))
    kwargs = dict(supervised_mask=torch.tensor(s.legal_mask),classes=torch.tensor(s.policy_classes),
        teacher_wdl=torch.tensor(s.wdl),selfplay_mask=torch.tensor(r.legal_mask),
        visits=torch.tensor(r.policy_distributions),final_wdl=torch.tensor(r.wdl),settings=settings)
    if bad == 'soft_outcome': kwargs['final_wdl'][0] = torch.tensor([.2,.3,.5])
    elif bad == 'wrong_counts': kwargs['teacher_wdl'] = kwargs['teacher_wdl'][:1]
    elif bad == 'classes_as_visits': kwargs['visits'] = torch.tensor(s.policy_classes)
    elif bad == 'nonfinite_logits': p[0,0] = torch.nan
    with pytest.raises(ValueError): mixed_losses(p,w,**kwargs)


def test_distribution_extreme_finite_logits_and_overflow_refusal():
    mask = torch.zeros((1,4352),dtype=torch.bool); mask[:,[3,9]] = True
    target = torch.zeros((1,4352)); target[:,3] = 1
    logits = torch.zeros((1,4352)); logits[:,3] = 1e30; logits[:,9] = -1e30
    logits.requires_grad_()
    loss = policy_distribution_cross_entropy(logits,mask,target)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(logits.grad).all()
    with torch.no_grad(): logits[:,3] = 3e38; logits[:,9] = -3e38
    with pytest.raises(ValueError,match='nonfinite'):
        policy_distribution_cross_entropy(logits,mask,target)


@pytest.mark.parametrize('field,value', [('supervised_fraction',0),('supervised_fraction',1),
    ('supervised_fraction',.4),('supervised_fraction',True),('batch_size',1),
    ('supervised_batch_size',0),('learning_rate',0),('learning_rate',float('nan')),
    ('policy_weight',0),('schedule','unknown')])
def test_invalid_frozen_settings(field,value):
    with pytest.raises(ValueError): MixedSettings(**{field:value})


def save_mixed(path, data):
    settings = MixedSettings()
    config = ModelConfig(d_policy=23,d_value=19,dropout=.27)
    model = config.build(); opt = optimizer(model,settings)
    _,state = mixed_step(model,opt,data,MixedState.initial(data,settings,seed=10))
    provenance = {'fixture':'R03 mixed CPU'}
    save_checkpoint(path,model=model,optimizer=opt,checkpoint_id='r03-fixture',provenance=provenance,
        weights=settings.weights,progress=Progress(1,0,4),mixed_state=state)
    return dict(expected_config=config,expected_provenance=provenance,
                mixed_sources=data,expected_mixed_settings=settings)


@pytest.mark.parametrize('bad', ['supervised_cursor','replay_cursor','replay_membership','supervised_identity',
    'replay_identity','rng','settings','progress','weights','optimizer','d07_only','v1_without_mixed',
    'expected_settings','source_membership','source_identity'])
def test_mixed_resume_rejects_drift_before_rng_install(sources,tmp_path,bad):
    data,_ = sources
    path = tmp_path/'mixed.pt'
    args = save_mixed(path,data)
    payload = torch.load(path,weights_only=True)
    m = payload['training']['mixed_state']
    if bad == 'supervised_cursor': m['supervised']['cursor'] += 1
    elif bad == 'replay_cursor': m['selfplay']['cursor'] += 1
    elif bad == 'replay_membership': m['replay_membership'] = tuple(reversed(m['replay_membership']))
    elif bad == 'supervised_identity': m['supervised_source_id'] = '0'*64
    elif bad == 'replay_identity': m['replay_id'] = '0'*64
    elif bad == 'rng': m['sampling_rng'] = (3,(True,)*625,None)
    elif bad == 'settings': m['settings']['supervised_fraction'] = .25
    elif bad == 'progress': payload['training']['progress']['samples_seen'] += 1
    elif bad == 'weights': payload['training']['weights']['wdl'] = 2.
    elif bad == 'optimizer': payload['training']['optimizer']['defaults']['lr'] = 1e-4
    elif bad == 'v1_without_mixed': payload['version'] = 'm06-checkpoint-v1'
    elif bad == 'd07_only':
        args.pop('mixed_sources'); args.pop('expected_mixed_settings')
        args.update(dataset=data.dataset,source_id='old-D07-only-cursor')
    elif bad == 'expected_settings': args['expected_mixed_settings'] = MixedSettings(learning_rate=2e-5)
    elif bad == 'source_membership': data.replay_membership = tuple(reversed(data.replay_membership))
    elif bad == 'source_identity': data.replay_id = '0'*64
    torch.save(payload,path)
    before = rng_snapshot()
    with pytest.raises(CheckpointError): restore_training(path,**args)
    assert_rng_equal(before,rng_snapshot())
    if bad in ('rng','settings','progress','weights','optimizer','v1_without_mixed','supervised_cursor','replay_cursor'):
        with pytest.raises(CheckpointError): load_inference(path)


def test_mixed_inference_preserves_rng_and_legacy_cursor_not_returned(sources,tmp_path):
    path = tmp_path/'mixed.pt'; args = save_mixed(path,sources[0])
    before = rng_snapshot()
    loaded = load_inference(path)
    assert not loaded.model.training
    assert_rng_equal(before,rng_snapshot())
    restored = restore_training(path,**args)
    assert restored.data_order is None and restored.mixed_state.updates == 1


@pytest.mark.parametrize('split',['validation','test'])
def test_heldout_supervised_never_enters_sources(sources,split):
    _,cfg = sources
    data = open_dataset(**{k:Path(v) for k,v in cfg['dataset'].items()},split=split)
    with pytest.raises(ValueError,match='training split'):
        MixedSources(data,ReplayBuffer(cfg['replay']),heldout_games_path=cfg['heldout_games'],
                     protected_positions_path=cfg['protected_positions'])


def test_heldout_replay_roots_excluded_and_registry_drift_refused(tmp_path,fixture_checkpoint):
    # Protect every actual prefix of a controlled completed game as held-out.
    args = fixture_bundle(tmp_path,specifications=[
        ('train','4k3/8/8/8/8/8/8/R3K3 w - - 0 1',[],[0],None),
        ('validation',make_state().board.fen(),['f2f3','e7e5','g2g4','d8h4'],[0,1,2,3],None)])
    buffer = ReplayBuffer.create(tmp_path/'buffer.sqlite')
    session = played(tmp_path,fixture_checkpoint)
    buffer.ingest(session.snapshot())
    data = open_dataset(**args,split='train')
    with pytest.raises(ValueError,match='no completed non-held-out'):
        MixedSources(data,buffer,heldout_games_path=args['prepared']/'heldout-games.jsonl',
                     protected_positions_path=args['prepared']/'protected-heldout.jsonl')
    soft = create(tmp_path,fixture_checkpoint,
        make_state('4k3/8/8/8/8/8/8/R3K3 w - - 99 1',history_complete=True),
        name='safe',cap=1,budget=3,phases=(Phase(0,0.,0.),))
    buffer.ingest(soft.run())
    before = buffer.path.read_bytes()
    sources = MixedSources(data,buffer,heldout_games_path=args['prepared']/'heldout-games.jsonl',
                           protected_positions_path=args['prepared']/'protected-heldout.jsonl')
    assert len(sources.replay_membership) == 1 and len(sources.excluded_heldout) == 4
    assert buffer.path.read_bytes() == before
    (args['prepared']/'protected-heldout.jsonl').write_text('')
    with pytest.raises(ValueError,match='registry identity'):
        MixedSources(data,buffer,heldout_games_path=args['prepared']/'heldout-games.jsonl',
                     protected_positions_path=args['prepared']/'protected-heldout.jsonl')


@pytest.mark.parametrize('failure',['lr','gradient','output','dataset_mutation','partial_adam'])
def test_failed_update_does_not_advance_cursor_or_call_step(sources,failure):
    data,_ = sources
    state = MixedState.initial(data,MixedSettings(),seed=29)
    model = ModelConfig(d_policy=17,d_value=13).build()
    opt = optimizer(model,state.settings)
    if failure == 'lr': opt.param_groups[0]['lr'] = 1e-4
    elif failure == 'gradient': next(model.parameters()).register_hook(lambda g:g*torch.nan)
    elif failure == 'output': model.register_forward_hook(lambda m,a,o:(o[0]*torch.nan,o[1]))
    elif failure == 'dataset_mutation': data.dataset._labels[0]['best_move_id'] += 1
    elif failure == 'partial_adam': opt.state[next(model.parameters())] = {'step':torch.tensor(1.)}
    calls = []
    opt.register_step_pre_hook(lambda *args:calls.append(True))
    before = deepcopy(model.state_dict()); sampling = state.next_selection()
    with pytest.raises(ValueError): mixed_step(model,opt,data,state)
    assert calls == [] and sampling == state.next_selection()
    assert_tree_equal(before,model.state_dict())


def test_exact_cpu_fresh_process_next_update(sources,tmp_path):
    _,cfg = sources
    source_file = tmp_path/'sources.json'; source_file.write_text(json.dumps(cfg))
    checkpoint = tmp_path/'boundary.pt'
    root = Path(__file__).resolve().parents[1]
    import os
    env = dict(os.environ,PYTHONPATH='src:.',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(mode):
        output = tmp_path/(mode+'.json')
        result = subprocess.run([sys.executable,'-B','scripts/run_r03.py','--mode',mode,
            '--source-config',str(source_file),'--output',str(output),'--checkpoint',str(checkpoint)],
            cwd=root,env=env,capture_output=True,text=True,timeout=120)
        assert result.returncode == 0,result.stdout+result.stderr
        return json.loads(output.read_text())
    full = run('full'); prefix = run('prefix'); resume = run('resume')
    assert prefix['steps'] == full['steps'][:1]
    assert resume['steps'] == full['steps'][1:]
    assert resume['final'] == full['final']
    assert resume['final']['mixed_state']['supervised']['epoch'] > 0
    assert resume['final']['mixed_state']['selfplay']['epoch'] > 0
