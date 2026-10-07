"""Offline M03 policy acceptance, with optional pinned Git-object source audit."""
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
from tests.policy_checks import (numerical_case, exact_order_and_dependence_case,
                                 constructor_case, adapter_ids_case, backbone_diagnostic_case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--maia3-checkout', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source_check = 'not requested; all numerical references run offline'
    if args.maia3_checkout:
        lock = json.loads((ROOT/'docs/upstream-lock.json').read_text())['repositories']['maia3']
        data = subprocess.check_output(['git', '-C', str(args.maia3_checkout), 'show',
                                        lock['commit']+':maia3/models.py'])
        checksum = hashlib.sha256(data).hexdigest()
        assert checksum == lock['files']['maia3/models.py']['sha256']
        body = '\n'.join(data.decode().splitlines()[372:394])+'\n'
        assert body in (ROOT/'tests/fixtures/policy_reference.py').read_text()
        source_check = {'commit': lock['commit'], 'full_file_sha256': checksum,
                        'policy_body_lines': [373, 394], 'verbatim_fixture': True,
                        'read_from_pinned_git_object': True, 'full_upstream_module_imported': False}
    framework = {}
    for name, obj in {'Linear.__init__': torch.nn.Linear.__init__,
                      'Linear.reset_parameters': torch.nn.Linear.reset_parameters}.items():
        source = inspect.getsource(obj)
        framework[name] = {'sha256': hashlib.sha256(source.encode()).hexdigest(), 'source': source}
    report = {
        'task': 'M03', 'status': 'passed', 'date': datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat(),
        'review_status': 'automated acceptance only; awaiting independent review',
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'machine': platform.machine(), 'torch': torch.__version__,
                        'torch_git_version': torch.version.git_version, 'device': 'cpu',
                        'threads': torch.get_num_threads(),
                        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                        'installed_distributions': dict(sorted((d.metadata['Name'], d.version)
                                                              for d in importlib.metadata.distributions()))},
        'resolved_settings': {'input': ['B', 64, 256], 'output': ['B', 4352], 'd_policy': 128,
                              'configuration': 'validated positive integer keyword d_policy on PairPolicyHead',
                              'biases': False, 'parameter_elements': 66048, 'unique_parameter_tensors': 3,
                              'parameter_count_formula': '516*d_policy',
                              'ordinary': 'q dot k / sqrt(d_policy)',
                              'promotion': 'S[48+f,56+g] + sqrt(d_policy)*promo_bias_proj(k[56+g])[p]',
                              'ordinary_id': '64*i+j', 'promotion_id': '4096+4*(8*f+g)+p',
                              'piece_order': ['queen', 'rook', 'bishop', 'knight'],
                              'initializer': 'Linear constructor kaiming_uniform_(a=sqrt(5)); uniform +/-1/sqrt(fan_in)',
                              'origin_destination_weight_bound': 1/16,
                              'promotion_weight_bound': 128**-0.5, 'model_wide_reset': False,
                              'mask_softmax_terminal_handling': False},
        'constructors': constructor_case(),
        'cases': [numerical_case(b, dtype, width) for width in (7, 128)
                  for dtype in (torch.float32, torch.float64) for b in (1, 2, 5)],
        'exact_order_scaling_dependence': exact_order_and_dependence_case(),
        'adapter_ids': adapter_ids_case(), 'backbone_diagnostic': backbone_diagnostic_case(),
        'pinned_source_check': source_check, 'installed_framework_source': framework,
        'limitations': ['Local CPU macOS arm64 float32/float64 only; CUDA/MPS/reduced precision and other platforms unvalidated',
                        'Upstream pins only torch; no upstream framework/RNG-stream or checkpoint equivalence claimed',
                        '66,048 elements is this policy head only; no final combined-model count or performance/strength claim',
                        'No M04 onward, WDL, assembled network, relative terms, data, training or search',
                        'Backbone composition and terminal fixtures are diagnostic tensor probes only',
                        'No subagents or paid compute']}
    serialized = json.dumps(report, indent=2, sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')


if __name__ == '__main__':
    main()
