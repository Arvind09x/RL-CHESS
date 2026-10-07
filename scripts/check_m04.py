"""Offline M04 WDL acceptance and optional pinned Git-object verification."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chess_rl.contracts import WDL_ORDER
from tests.wdl_checks import numerical_case, synthetic_case, constructor_case, backbone_diagnostic_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--maia3-checkout', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.__version__ == '2.8.0'
    source_check = 'not requested; independent numerical reference runs offline'
    if args.maia3_checkout:
        lock = json.loads((ROOT/'docs/upstream-lock.json').read_text())['repositories']['maia3']
        checksums = {}
        for path in ('maia3/models.py', 'maia3/uci.py'):
            data = subprocess.check_output(['git', '-C', str(args.maia3_checkout), 'show', lock['commit']+':'+path])
            digest = hashlib.sha256(data).hexdigest()
            assert digest == lock['files'][path]['sha256']
            checksums[path] = digest
            lines = data.decode().splitlines()
            if path.endswith('models.py'):
                body = '\n'.join(lines[395:397])+'\n'
                assert body in (ROOT/'tests/fixtures/wdl_reference.py').read_text()
                assert lines[322:326] == ['        self.last_ln = nn.LayerNorm(cfg.dim_vit)', '',
                    '        self.fc_value_hid = nn.Linear(cfg.dim_vit, cfg.head_hid_dim)',
                    '        self.fc_value = nn.Linear(cfg.head_hid_dim, 3)']
            else:
                assert 'loss, draw, win = torch.softmax(logits.float(), dim=-1).tolist()' in '\n'.join(lines[195:200])
        source_check = {'commit': lock['commit'], 'full_file_sha256': checksums,
                        'value_body_lines': [396, 397], 'constructor_lines': [323, 326],
                        'verbatim_value_body': True, 'upstream_interpreter_order': ['loss', 'draw', 'win'],
                        'full_upstream_module_imported': False, 'read_from_pinned_git_object': True}
    framework = {}
    for name, obj in {'Linear.__init__': torch.nn.Linear.__init__,
                      'Linear.reset_parameters': torch.nn.Linear.reset_parameters,
                      'LayerNorm.__init__': torch.nn.LayerNorm.__init__,
                      'LayerNorm.reset_parameters': torch.nn.LayerNorm.reset_parameters,
                      'LayerNorm.forward': torch.nn.LayerNorm.forward}.items():
        source = inspect.getsource(obj)
        framework[name] = {'sha256': hashlib.sha256(source.encode()).hexdigest(), 'source': source}
    report = {'task': 'M04', 'status': 'passed',
        'date': datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat(),
        'review_status': 'automated acceptance only; awaiting independent review',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'machine': platform.machine(), 'torch': torch.__version__,
                        'torch_git_version': torch.version.git_version, 'device': 'cpu',
                        'threads': torch.get_num_threads(), 'deterministic_algorithms': True,
                        'installed_distributions': dict(sorted((d.metadata['Name'], d.version)
                                                              for d in importlib.metadata.distributions()))},
        'resolved_settings': {'input': ['B', 64, 256], 'output': ['B', 3], 'pooling': 'square mean',
                              'd_value': 128, 'independent_of_d_policy': True,
                              'layernorm': {'width': 256, 'eps': 1e-5, 'elementwise_affine': True,
                                            'bias': True, 'variance': 'population / correction=0'},
                              'hidden_linear': [256, 128], 'output_linear': [128, 3],
                              'linear_biases': True, 'activation': 'ReLU', 'inplace': False,
                              'wdl_order': list(WDL_ORDER), 'perspective': 'input position side to move',
                              'parameter_elements': 33795, 'unique_parameter_tensors': 6,
                              'parameter_count_formula': '512 + (256*d_value+d_value) + (3*d_value+3) = 515+260*d_value',
                              'initializer': 'LayerNorm ones/zeros; Linear kaiming_uniform_(a=sqrt(5)), weight/bias uniform +/-1/sqrt(fan_in)',
                              'hidden_weight_bias_bound': 1/16, 'output_weight_bias_bound': 128**-.5,
                              'model_wide_reset': False, 'checkpoint_compatibility_claimed': False,
                              'softmax_value_terminal_labels_ponder_auxiliary': False},
        'synthetic': synthetic_case(), 'constructors': constructor_case(),
        'cases': [numerical_case(b, dtype, width) for width in (7, 128)
                  for dtype in (torch.float32, torch.float64) for b in (1, 2, 5)],
        'backbone_diagnostic': backbone_diagnostic_case(), 'pinned_source_check': source_check,
        'installed_framework_source': framework,
        'limitations': ['Local CPU macOS arm64 float32/float64 only; CUDA/MPS/reduced precision and other builds unvalidated',
                        'Upstream pins torch without a framework version; no upstream RNG or checkpoint equivalence',
                        '33,795 elements is WDL head only; no combined-model count, throughput or strength claim',
                        'Direct backbone/head composition and terminal fixtures are diagnostic tensor probes only',
                        'No M05 onward, assembled network, relative terms, data, training, evaluator or search',
                        'Softmax, value conversion, terminal handling and outcome labels remain consumers',
                        'No subagents or paid compute']}
    serialized = json.dumps(report, indent=2, sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
