"""One recorded real D05/D07 training-batch update, with independent loss math.

Run with PYTHONPATH=src:. in the model environment. No checkpoint is saved.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from chess_rl.contracts import WDL_ORDER, ENCODING_VERSION, MOVE_VOCAB_VERSION, ENDING_POLICY
from chess_rl.model import ChessNetwork
from chess_rl.supervised import open_dataset
from chess_rl.training.trainer import LossWeights, supervised_step
from tests.fixtures.loss_reference import independent


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('runs/t01/config.json'))
    parser.add_argument('--games', type=Path, default=Path('/private/tmp/rl-chess-d03-prepared/games.jsonl'))
    parser.add_argument('--split-audit', type=Path, default=Path('/private/tmp/rl-chess-d03-prepared/split-audit.json'))
    parser.add_argument('--prepared', type=Path, default=Path('/private/tmp/rl-chess-d04-prepared'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['optimizer_steps'] == 1 and config['split'] == 'train' and config['device'] == 'cpu'
    torch.set_num_threads(config['torch_threads'])
    torch.set_num_interop_threads(config['torch_interop_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    bundle = Path(config['dataset_bundle'])
    dataset = open_dataset(bundle=bundle, manifest_path=Path('data/manifests/elite-2025-01-d04-v1.json'),
        games_path=args.games, split_audit_path=args.split_audit, prepared=args.prepared,
        teacher_evidence=Path('runs/d01/stockfish-provenance.json'), split='train')
    batch = next(dataset.batches(config['batch_size']))
    assert len(batch.references) == config['batch_size']
    arrays = {name: getattr(batch, name).copy() for name in ('features', 'legal_mask', 'policy_classes', 'wdl')}
    # Compare the tensors used by the step with the D07 conversion and prove
    # their backing memory is independent of the batch's NumPy arrays.
    captured = []
    input_check = []
    model = ChessNetwork(dropout=config['dropout'], device='cpu', dtype=torch.float32)
    assert sum(p.numel() for p in model.parameters()) == 5049411
    def capture(module, inputs, outputs):
        np.testing.assert_array_equal(inputs[0].detach().numpy(), arrays['features'])
        assert inputs[0].data_ptr() != batch.features.ctypes.data
        input_check.append(True)
        for output in outputs:
            assert torch.isfinite(output).all()
            output.retain_grad()
        captured.extend(outputs)
    hook = model.register_forward_hook(capture)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], betas=tuple(config['betas']),
        eps=config['eps'], weight_decay=config['weight_decay'], amsgrad=config['amsgrad'],
        foreach=config['foreach'], fused=config['fused'])
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    weights = LossWeights(**config['loss_weights'])
    metrics = supervised_step(model, optimizer, batch, weights=weights)
    hook.remove()
    policy_ref, policy_gradient, _ = independent(captured[0].detach().numpy(), arrays['legal_mask'], arrays['policy_classes'])
    wdl_ref, wdl_gradient, _ = independent(captured[1].detach().numpy(), np.ones((metrics.batch_size, 3), bool), targets=arrays['wdl'])
    np.testing.assert_allclose(metrics.policy_loss, policy_ref, atol=2e-6, rtol=3e-6)
    np.testing.assert_allclose(metrics.wdl_loss, wdl_ref, atol=2e-6, rtol=3e-6)
    np.testing.assert_allclose(captured[0].grad.numpy(), weights.policy * policy_gradient, atol=2e-7, rtol=3e-6)
    np.testing.assert_allclose(captured[1].grad.numpy(), weights.wdl * wdl_gradient, atol=2e-7, rtol=3e-6)
    assert torch.count_nonzero(captured[0].grad[~torch.tensor(arrays['legal_mask'])]) == 0
    checks = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.isfinite(parameter).all()
        # Independent first-step Adam formula; compare every parameter entry.
        expected = before[name] - config['learning_rate'] * gradient / (gradient.abs() + config['eps'])
        torch.testing.assert_close(parameter, expected, atol=1e-7, rtol=1e-6)
        changed = not torch.equal(before[name], parameter)
        if '.relative.' in name or name == 'backbone.bank.weight':
            assert changed and torch.count_nonzero(gradient) > 0
        checks.append(dict(name=name, changed=changed, gradient_nonzero=int(torch.count_nonzero(gradient)),
            gradient_norm=float(gradient.norm()), update_norm=float((parameter.detach() - before[name]).norm())))
    assert all(layer.self_attn.gab.bank is model.backbone.bank for layer in model.backbone.layers)
    assert sum(p is model.backbone.bank.weight for group in optimizer.param_groups for p in group['params']) == 1
    assert all(int(state['step']) == 1 for state in optimizer.state.values())
    assert len(optimizer.state) == len(before) == 215 and len(input_check) == 1
    for name, old in arrays.items(): np.testing.assert_array_equal(getattr(batch, name), old)
    # Fresh dataset accesses still reproduce the adapter-owned batch after use.
    fresh = next(dataset.batches(config['batch_size']))
    for name, old in arrays.items(): np.testing.assert_array_equal(getattr(fresh, name), old)
    assert fresh.references == batch.references and fresh.split == batch.split == 'train'
    report = dict(config=config, config_identity=identity(args.config),
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            platform=platform.platform(), machine=platform.machine(), threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), deterministic=torch.are_deterministic_algorithms_enabled()),
        contracts=dict(encoding=ENCODING_VERSION, vocabulary=MOVE_VOCAB_VERSION, ending_policy=ENDING_POLICY, wdl_order=WDL_ORDER),
        dataset_records=len(dataset), references=batch.references,
        selected_policy_classes=batch.policy_classes.tolist(), selected_wdl=batch.wdl.tolist(),
        dataset_label_identity=identity(bundle/'labels.jsonl'), metrics=asdict(metrics),
        independent_losses=dict(policy=float(policy_ref), wdl=float(wdl_ref)),
        loss_absolute_errors=dict(policy=abs(metrics.policy_loss - policy_ref), wdl=abs(metrics.wdl_loss - wdl_ref)),
        independent_output_gradients_pass=True, illegal_policy_gradients_exact_zero=True,
        batch_arrays_and_reloaded_adapter_batch_unchanged=True, unique_shared_bank=True,
        parameters=5049411, parameter_tensors=len(before), changed_parameter_tensors=sum(c['changed'] for c in checks),
        first_step_adam_formula_pass=True, parameter_checks=checks,
        limitations='one batch, one CPU float32 update; no overfit, checkpoint, evaluation, D08 or performance claim')
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('parameter_checks', 'config')}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
