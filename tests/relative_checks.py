"""M06 full-network checks with independent relative/backbone/head references."""
import json
from pathlib import Path
import numpy as np
import torch
from chess_rl.adapter import encode_position, make_state
from chess_rl.contracts import WDL_ORDER
from chess_rl.model import ChessNetwork
from .fixtures.combined_reference import independent_reference
from .backbone_checks import asymmetric_input, objective, TOLERANCES
from .policy_checks import component_reference
from .wdl_checks import numpy_reference
from .network_checks import compare, value_objective
from .helpers import fingerprint

def numerical_case(batch, dtype, d_policy=128, d_value=128):
    torch.manual_seed(1052026)
    model = ChessNetwork(dropout=0, dtype=dtype, d_policy=d_policy, d_value=d_value)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ('norm' in name or '.ln' in name or 'last_ln' in name) and name.endswith('weight'):
                p.uniform_(.7, 1.3)
            else:
                p.uniform_(-.04, .05)
        model.wdl_head.fc_value_hid.bias.copy_(torch.linspace(-.4, .5, d_value, dtype=dtype))
    x = asymmetric_input(batch, dtype)
    assert not x.is_contiguous()
    before = x.detach().clone()
    parameters = dict(model.named_parameters())
    saved = {n: p.detach().clone() for n, p in parameters.items()}
    ref_x = x.detach().clone().requires_grad_()
    # Deduplicated clone: exactly one bank is used throughout the independent path.
    ref = {n: p.detach().clone().requires_grad_() for n, p in parameters.items()}
    bp = {n.removeprefix('backbone.'): p for n, p in ref.items() if n.startswith('backbone.')}
    pp = {n.removeprefix('policy_head.'): p for n, p in ref.items() if n.startswith('policy_head.')}
    vp = {n.removeprefix('wdl_head.'): p.detach().double().numpy() for n, p in ref.items() if n.startswith('wdl_head.')}
    tokens, _ = independent_reference(ref_x, bp)
    policy_ref = component_reference(tokens, pp)
    value_ref, vg, pre = numpy_reference(tokens.detach().double().numpy(), vp)
    assert (pre > 0).any() and (pre < 0).any() and np.min(np.abs(pre)) > 1e-7
    outputs = model(x)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    policy, value = outputs
    assert policy.shape == (batch, 4352) and value.shape == (batch, 3)
    output_errors = {'policy': compare(policy, policy_ref, dtype), 'wdl': compare(value, value_ref, dtype)}
    # NumPy's manual WDL chain rule supplies the final-token cotangent.
    # The independent backbone propagates the sum of policy and WDL cotangents.
    ref_loss = objective(policy_ref) + (tokens * torch.as_tensor(vg['input'], dtype=dtype)).sum()
    differentiable_names = [n for n in ref if not n.startswith('wdl_head.')]
    ref_grads = torch.autograd.grad(ref_loss, (ref_x, *(ref[n] for n in differentiable_names)))
    expected = dict(zip(('input', *differentiable_names), ref_grads))
    expected.update({'wdl_head.'+n: torch.as_tensor(vg[n], dtype=dtype) for n in vp})
    actual = torch.autograd.grad(objective(policy) + value_objective(value), (x, *parameters.values()))
    errors = {n: compare(g, expected[n], dtype) for n, g in zip(('input', *parameters), actual)}
    assert all(torch.count_nonzero(g) for g in actual)
    assert torch.equal(x.detach(), before)
    assert all(torch.equal(p.detach(), saved[n]) for n, p in parameters.items())
    return {'batch': batch, 'dtype': str(dtype), 'd_policy': d_policy, 'd_value': d_value,
            'dropout': 0, 'noncontiguous_asymmetric_input': True, 'tolerance': TOLERANCES[dtype],
            'output_max_abs_errors': output_errors, 'gradient_max_abs_errors': errors,
            'gradient_max_abs_error': max(errors.values()), 'unique_parameter_tensors': len(parameters),
            'all_gradient_tensors_finite_nonzero': True, 'inputs_and_parameters_unchanged': True,
            'relu_active_and_inactive': True, 'min_abs_relu_preactivation': float(np.min(np.abs(pre)))}

def routing_and_joint_case(dtype):
    torch.manual_seed(505)
    model = ChessNetwork(dropout=0, dtype=dtype)
    x = asymmetric_input(2, dtype)
    seen = {'backbone': [], 'final_norm': [], 'policy': [], 'wdl': []}
    hooks = [model.backbone.register_forward_hook(lambda m, a, y: seen['backbone'].append(y)),
             model.backbone.norm.register_forward_hook(lambda m, a, y: seen['final_norm'].append(y)),
             model.policy_head.register_forward_pre_hook(lambda m, a: seen['policy'].append(a[0])),
             model.wdl_head.register_forward_pre_hook(lambda m, a: seen['wdl'].append(a[0]))]
    try:
        policy, value = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    assert all(len(v) == 1 for v in seen.values())
    tokens = seen['backbone'][0]
    assert tokens is seen['final_norm'][0] is seen['policy'][0] is seen['wdl'][0]
    saved_tokens = tokens.detach().clone()
    assert model.wdl_head.last_ln is not model.backbone.norm
    assert model.wdl_order is model.wdl_head.wdl_order is WDL_ORDER
    params = list(model.parameters())
    targets = (x, tokens, *params)
    gp = torch.autograd.grad(objective(policy), targets, retain_graph=True, allow_unused=True)
    gv = torch.autograd.grad(value_objective(value), targets, retain_graph=True, allow_unused=True)
    joint = torch.autograd.grad(objective(policy)+value_objective(value), targets)
    errors = [compare(j, (torch.zeros_like(j) if p is None else p)
                     + (torch.zeros_like(j) if v is None else v), dtype) for j, p, v in zip(joint, gp, gv)]
    bp_ids = {id(p) for p in model.backbone.parameters()}
    for target, p, v in zip(targets, gp, gv):
        if target is x or target is tokens or id(target) in bp_ids:
            assert p is not None and v is not None and torch.count_nonzero(p) and torch.count_nonzero(v)
    ph_ids = {id(p) for p in model.policy_head.parameters()}
    vh_ids = {id(p) for p in model.wdl_head.parameters()}
    for p, a, b in zip(params, gp[2:], gv[2:]):
        if id(p) in ph_ids:
            assert a is not None and torch.count_nonzero(a) and b is None
        if id(p) in vh_ids:
            assert b is not None and torch.count_nonzero(b) and a is None
    assert torch.equal(tokens.detach(), saved_tokens)
    generators = [b.self_attn.gab for b in model.backbone.layers]
    assert len({id(g) for g in generators}) == 8
    assert all(g.bank is model.backbone.bank for g in generators)
    private = [{id(p) for p in g.parameters() if p is not model.backbone.bank.weight} for g in generators]
    assert all(len(s) == 10 for s in private) and len(set.union(*private)) == 80
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    entries = [p for group in optimizer.param_groups for p in group['params']]
    assert len(entries) == len({id(p) for p in entries}) == 215
    assert sum(p is model.backbone.bank.weight for p in entries) == 1
    return {'dtype': str(dtype), 'backbone_evaluations': 1, 'same_encoder_final_token_object': True,
            'separate_pooled_norm': True, 'shared_wdl_order_identity': True, 'independent_generators': 8,
            'private_generator_tensor_counts': list(map(len, private)), 'optimizer_unique_tensors': 215,
            'optimizer_bank_entries': 1, 'both_branches_nonzero_through_input_tokens_all_backbone_parameters': True,
            'head_gradients_only_from_own_branch': True, 'joint_gradient_sum_max_abs_error': max(errors)}

def adapter_case(dtype):
    records = json.loads((Path(__file__).parent/'fixtures/adapter_records.json').read_text())
    states = [make_state(r['start_fen'], r['moves_uci'], history_complete=r['history_complete']) for r in records]
    before = [fingerprint(s) for s in states]
    boards = [id(s.board) for s in states]
    features = np.stack([encode_position(s).features for s in states])
    x = torch.from_numpy(features.copy()).to(dtype).requires_grad_()
    original = x.detach().clone()
    torch.manual_seed(505)
    model = ChessNetwork(dtype=dtype).eval()
    policy, value = model(x)
    assert policy.shape == (13, 4352) and value.shape == (13, 3)
    assert torch.isfinite(policy).all() and torch.isfinite(value).all()
    (objective(policy) + value_objective(value)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
               for p in (x, *model.parameters()))
    assert torch.equal(x.detach(), original)
    assert [fingerprint(s) for s in states] == before and [id(s.board) for s in states] == boards
    np.testing.assert_array_equal(features, np.stack([encode_position(s).features for s in states]))
    # Count actual allocated gradient tensors, independently of analytic count.
    return {'dtype': str(dtype), 'fixture_count': len(records), 'names': [r['name'] for r in records],
            'shapes': [list(policy.shape), list(value.shape)], 'all_outputs_and_gradients_finite_nonzero': True,
            'states_all_prefixes_provenance_boards_and_features_unchanged': True,
            'terminal_fixtures_are_tensor_probes_only': True,
            'allocated_parameter_tensor_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
            'allocated_gradient_tensor_bytes': sum(p.grad.numel()*p.grad.element_size() for p in model.parameters())}

def dropout_case():
    torch.manual_seed(505)
    model = ChessNetwork()
    for b in model.backbone.layers:
        assert b.self_attn.mha.dropout == .1
        assert all(d.p == .1 for d in (b.dropout, b.dropout1, b.dropout2))
    with torch.no_grad():
        for block in model.backbone.layers:
            for table in block.self_attn.relative.parameters():
                table.uniform_(-.05, .07)
    x = asymmetric_input(2, torch.float32)
    with torch.no_grad():
        model.train()
        assert all(m.training for m in model.modules())
        torch.manual_seed(1); a = model(x)
        torch.manual_seed(1); replay = model(x)
        torch.manual_seed(2); b = model(x)
        assert all(torch.equal(y, z) for y, z in zip(a, replay))
        assert all(not torch.equal(y, z) for y, z in zip(a, b))
        model.eval()
        assert all(not m.training for m in model.modules())
        e, repeated = model(x), model(x)
        assert all(torch.equal(y, z) for y, z in zip(e, repeated))
        assert all(not torch.equal(y, z) for y, z in zip(e, a))
        zero = ChessNetwork(dropout=0)
        assert all(torch.equal(y, z) for y, z in zip(zero.train()(x), zero.eval()(x)))
    return {'default_dropout': .1, 'all_module_train_eval_flags_propagate': True,
            'both_heads_stochastic_in_train': True, 'seed_replay_exact': True,
            'both_eval_outputs_repeat_exactly': True, 'zero_dropout_train_eval_equal': True,
            'train_max_abs_differences': [float((y-z).abs().max()) for y, z in zip(a, b)]}


def layout_case():
    from chess_rl.model.relative import RelativeQKV
    from .fixtures.relative_reference import FACTOR, expand
    model = RelativeQKV(dtype=torch.float64)
    expected = torch.tensor([[15*(i//8-j//8+7)+i%8-j%8+7 for j in range(64)] for i in range(64)])
    assert torch.equal(model.index, expected)
    assert set(expected.flatten().tolist()) == set(range(225))
    assert (FACTOR.sum(0) == 1).all()
    assert model.index[0,63] == 0 and model.index[63,0] == 224
    assert model.index[0,1] == 111 and model.index[1,0] == 113
    raw = torch.arange(256*225, dtype=torch.float64).reshape(256,225)
    logical = torch.empty(8,225,32,dtype=torch.float64)
    for h in range(8):
        for d in range(32):
            logical[h,:,d] = raw[d*8+h]
    with torch.no_grad():
        model.q.copy_(logical)
    dense = expand(logical, dense=True)
    gathered = model.q[:,model.index].permute(3,0,1,2)
    assert torch.equal(gathered,dense)
    assert not torch.equal(raw.reshape(8,32,225).permute(0,2,1),logical)
    assert not torch.equal(model.index,model.index.T)
    assert 'index' not in model.state_dict()
    return {'all_4096_indices_exact': True, '225_displacements': True,
            'depth_before_head_sentinel_exact': True, 'wrong_head_first_layout_detected': True,
            'reverse_displacement_detected': True, 'dense_factorizer_gather_exact': True}


def contraction_case(batch, dtype):
    from chess_rl.model.relative import RelativeQKV
    from .fixtures.relative_reference import contractions
    torch.manual_seed(606)
    model = RelativeQKV(dtype=dtype)
    with torch.no_grad():
        for p in model.parameters(): p.uniform_(-.09,.13)
    shape = (batch,8,64,32)
    def signal(offset, shape):
        i=torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)
        return (torch.sin(i*.023+offset)+.2*torch.cos(i*.037+offset)).to(dtype).requires_grad_()
    q,k=signal(.1,shape),signal(.7,shape)
    alpha=signal(1.3,(batch,8,64,64))
    inputs=[q,k,alpha,*model.parameters()]
    refs=[x.detach().clone().requires_grad_() for x in inputs]
    rq,rk,rv=contractions(*refs[:3],refs[3:],dense=True)
    actual=[model.logits(q,k),model.values(alpha)]
    expected=[rq+rk,rv]
    errors=[compare(a,b,dtype) for a,b in zip(actual,expected)]
    ag=torch.autograd.grad(sum(objective(a) for a in actual),inputs)
    rg=torch.autograd.grad(sum(objective(a) for a in expected),refs)
    ge=[compare(a,b,dtype) for a,b in zip(ag,rg)]
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in ag)
    return {'batch':batch,'dtype':str(dtype),'tolerance':TOLERANCES[dtype],
            'output_max_abs_errors':errors,'gradient_max_abs_errors':ge,
            'all_operand_and_table_gradients_finite_nonzero':True}


def attention_reference(x,p,dropout=0,training=False):
    from .gab_checks import functional_reference
    from .fixtures.relative_reference import contractions
    weights=p['mha.in_proj_weight'].split(256,0)
    biases=p['mha.in_proj_bias'].split(256,0)
    q,k,v=[(x@w.T+b).reshape(x.shape[0],64,8,32).permute(0,2,1,3) for w,b in zip(weights,biases)]
    gp={n.removeprefix('gab.'):v for n,v in p.items() if n.startswith('gab.')}
    bias=functional_reference(x,gp)
    tables=[p['relative.'+n] for n in ('q','k','v')]
    dummy=torch.zeros(x.shape[0],8,64,64,dtype=x.dtype)
    qr,kr,_=contractions(q,k,dummy,tables)
    scores=(q@k.transpose(-1,-2)+qr+kr)/32**.5+bias
    exp=torch.exp(scores-scores.max(-1,keepdim=True).values)
    alpha=exp/exp.sum(-1,keepdim=True)
    # Reference uses the framework dropout primitive, only for seeded mask checks.
    dropped=torch.nn.functional.dropout(alpha,p=dropout,training=training)
    _,_,vr=contractions(q,k,dropped,tables)
    out=(dropped@v+vr).transpose(1,2).reshape(x.shape[0],64,256)
    return out@p['mha.out_proj.weight'].T+p['mha.out_proj.bias']


def attention_case(batch,dtype):
    from chess_rl.model.attention import GABSelfAttention
    from chess_rl.model.gab import SharedTemplateBank
    torch.manual_seed(606)
    model=GABSelfAttention(SharedTemplateBank(dtype=dtype),dropout=0,relative_enabled=True)
    with torch.no_grad():
        for n,p in model.named_parameters():
            p.uniform_(.7,1.3) if ('ln' in n and n.endswith('weight')) else p.uniform_(-.04,.05)
    x=asymmetric_input(batch,dtype,256)
    rx=x.detach().clone().requires_grad_()
    params=dict(model.named_parameters())
    ref={n:p.detach().clone().requires_grad_() for n,p in params.items()}
    saved={n:p.detach().clone() for n,p in params.items()}
    actual=model(x);expected=attention_reference(rx,ref)
    oe=compare(actual,expected,dtype)
    ag=torch.autograd.grad(objective(actual),(x,*params.values()))
    rg=torch.autograd.grad(objective(expected),(rx,*ref.values()))
    ge={n:compare(a,b,dtype) for n,a,b in zip(('input',*params),ag,rg)}
    assert all(torch.count_nonzero(g) for g in ag)
    assert torch.equal(x,rx) and all(torch.equal(params[n],saved[n]) for n in params)
    return {'batch':batch,'dtype':str(dtype),'output_max_abs_error':oe,
            'gradient_max_abs_errors':ge,'all_gradients_finite_nonzero':True,'tolerance':TOLERANCES[dtype]}


def recovery_case(batch,dtype):
    # Constructor disabling recreates the accepted M05 structure and random stream.
    torch.manual_seed(606)
    combined=ChessNetwork(dropout=0,dtype=dtype)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(606)
    old=ChessNetwork(dropout=0,dtype=dtype,relative_enabled=False)
    assert torch.equal(rng,torch.get_rng_state())
    common=dict(old.named_parameters())
    params=dict(combined.named_parameters())
    assert len(common)==191 and len(params)==215
    assert all(torch.equal(p,params[n]) for n,p in common.items())
    tables=[p for n,p in params.items() if '.relative.' in n]
    assert len(tables)==24 and all(torch.count_nonzero(p)==0 for p in tables)
    x=asymmetric_input(batch,dtype); rx=x.detach().clone().requires_grad_()
    a=combined(x); b=old(rx)
    oe=[compare(y,z,dtype) for y,z in zip(a,b)]
    loss=lambda y:objective(y[0])+value_objective(y[1])
    ga=torch.autograd.grad(loss(a),(x,*params.values()))
    gb=torch.autograd.grad(loss(b),(rx,*common.values()))
    by_name=dict(zip(('input',*params),ga))
    ge={n:compare(by_name[n],g,dtype) for n,g in zip(('input',*common),gb)}
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in ga)
    with torch.no_grad():
        for p in tables:p.uniform_(-.05,.07)
    c=combined(x,relative_enabled=False);d=old(rx)
    assert all(torch.equal(y,z) for y,z in zip(c,d))
    gc=torch.autograd.grad(loss(c),(x,*params.values()),allow_unused=True)
    gd=torch.autograd.grad(loss(d),(rx,*common.values()))
    gradients=dict(zip(('input',*params),gc))
    assert all(torch.equal(gradients[n],g) for n,g in zip(('input',*common),gd))
    assert all(gradients[n] is None for n in params if '.relative.' in n)
    with torch.no_grad():
        for network in (combined, old):
            for block in network.backbone.layers:
                block.self_attn.mha.dropout = .1
                for dropout in (block.dropout, block.dropout1, block.dropout2):
                    dropout.p = .1
        for training in (True, False):
            combined.train(training); old.train(training)
            torch.manual_seed(117); disabled = combined(x, relative_enabled=False)
            torch.manual_seed(117); legacy = old(rx)
            assert all(torch.equal(a, b) for a, b in zip(disabled, legacy))
    return {'batch':batch,'dtype':str(dtype),'common_constructor_weights_and_rng_exact':True,
            'nonzero_table_disabled_seeded_train_and_eval_outputs_exact':True,
            'zero_table_output_max_abs_errors':oe,'zero_table_common_gradient_max_abs_error':max(ge.values()),
            'initial_all_table_gradients_finite_nonzero':True,
            'nonzero_table_disabled_outputs_and_common_gradients_exact':True,'disabled_table_gradients_absent':True}


def accounting_case(dtype):
    torch.manual_seed(606)
    model=ChessNetwork(dtype=dtype)
    # Explicit shape arithmetic including ONE bank, independent of prior totals.
    backbone=(256*128+256)+64*256+4096*32+2*256+8*(
        3*256*256+3*256+256*256+256+2*(256*256+256)+4*256
        +8*256+8+32*512+32+2*32+256*32+256+2*256)
    policy=2*256*128+4*128
    wdl=2*256+(256*128+128)+(128*3+3)
    relative=8*3*8*225*32
    total=backbone+policy+wdl+relative
    params=list(model.parameters())
    assert total==5049411==sum(p.numel() for p in params)
    assert len(params)==len({id(p) for p in params})==215
    tables=[p for n,p in model.named_parameters() if '.relative.' in n]
    assert len({p.data_ptr() for p in tables})==24
    assert all(p.shape==(8,225,32) and torch.count_nonzero(p)==0 for p in tables)
    bank=model.backbone.bank.weight
    assert sum(p is bank for p in params)==1
    assert all(b.self_attn.gab.bank.weight is bank for b in model.backbone.layers)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.no_grad():tables[0][0,0,0]=1
    assert torch.count_nonzero(tables[1])==0
    with torch.no_grad():tables[0][0,0,0]=0
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    model.to(dtype=torch.float64 if dtype==torch.float32 else torch.float32)
    assert all(b.self_attn.gab.bank.weight is model.backbone.bank.weight for b in model.backbone.layers)
    assert all(b.self_attn.relative.index.dtype==torch.int64 for b in model.backbone.layers)
    buffers = sum(b.numel()*b.element_size() for b in model.buffers())
    assert buffers == 262144
    return {'index_buffer_bytes': buffers, 'dtype':str(dtype),'backbone_without_relative':backbone,'policy':policy,'wdl':wdl,
            'relative_elements':relative,'total':total,'unique_tensors':215,'independent_tables':24,
            'shared_bank_entries':1,'parameter_bytes':total*(4 if dtype==torch.float32 else 8),
            'gradient_bytes_if_allocated':total*(4 if dtype==torch.float32 else 8),
            'dtype_move_preserves_sharing_and_index_dtype':True}


def probability_dropout_case(dtype):
    from chess_rl.model.attention import GABSelfAttention
    from chess_rl.model.gab import SharedTemplateBank
    torch.manual_seed(606)
    model=GABSelfAttention(SharedTemplateBank(dtype=dtype),dropout=.35,relative_enabled=True).train()
    with torch.no_grad():
        for p in model.relative.parameters():p.uniform_(-.2,.3)
    x=asymmetric_input(2,dtype,256)
    params=dict(model.named_parameters())
    torch.manual_seed(35);actual=model(x)
    torch.manual_seed(35);expected=attention_reference(x,params,.35,True)
    e=compare(actual,expected,dtype)
    torch.manual_seed(35);assert torch.equal(actual,model(x))
    torch.manual_seed(36);assert not torch.equal(actual,model(x))
    model.eval();assert torch.equal(model(x),model(x))
    model.mha.dropout=1;model.train()
    result=model(x)
    assert torch.equal(result,model.mha.out_proj.bias.expand_as(result))
    return {'dtype':str(dtype),'single_seeded_probability_mask_both_value_terms_max_error':e,
            'train_seed_replay_exact':True,'train_seeds_differ':True,'eval_repeat_exact':True,
            'dropout_one_removes_both_value_terms_exact':True}
