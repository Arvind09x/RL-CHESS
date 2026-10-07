"""Offline M05 intermediate GAB network joint acceptance and memory report."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from chess_rl.contracts import WDL_ORDER
from tests.network_checks import (numerical_case, routing_and_joint_case, adapter_case,
                                 dropout_case, constructor_and_memory_case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.__version__ == '2.8.0'
    report = {'task': 'M05', 'status': 'passed',
              'date': datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat(),
              'review_status': 'automated acceptance only; stop for independent review',
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'machine': platform.machine(), 'torch': torch.__version__,
                              'torch_git_version': torch.version.git_version, 'device': 'cpu',
                              'threads': torch.get_num_threads(), 'deterministic_algorithms': True,
                              'installed_distributions': dict(sorted((d.metadata['Name'], d.version)
                                                                    for d in importlib.metadata.distributions()))},
              'configuration': {'input': ['B', 64, 128], 'final_tokens': ['B', 64, 256],
                                'output_tuple': [['B', 4352], ['B', 3]], 'raw_logits': True,
                                'd_policy': 128, 'd_value': 128, 'training_dropout': .1,
                                'wdl_order': list(WDL_ORDER), 'layers': 8, 'heads': 8,
                                'width': 256, 'ffn_width': 256, 'gab_dimensions': [8, 32, 32],
                                'relative_terms': False, 'model_wide_reset': False},
              'numerical_cases': [numerical_case(batch, dtype) for dtype in (torch.float32, torch.float64)
                                  for batch in (1, 2, 5)],
              'independent_width_cases': [numerical_case(2, dtype, 7, 11) for dtype in (torch.float32, torch.float64)],
              'routing_and_joint_gradients': [routing_and_joint_case(dtype) for dtype in (torch.float32, torch.float64)],
              'adapter': [adapter_case(dtype) for dtype in (torch.float32, torch.float64)],
              'dropout': dropout_case(), 'constructors_and_tensor_memory': constructor_and_memory_case(),
              'seeds': {'numerical': 1052026, 'routing_adapter_constructor_dropout': 505,
                        'dropout_replay': 1, 'dropout_alternative': 2},
              'objectives': {'policy': 'M02 objective: sum(y*probe + .03*y**2)/sqrt(numel); probe=sin(.031*i)+.4*cos(.011*i)',
                             'wdl': 'M04 objective: sum(c*y+.03*y**2); c=.17*i-.4',
                             'combined': 'policy objective + WDL objective'},
              'references': {'backbone': 'existing tests.backbone_checks.independent_reference',
                             'policy': 'existing tests.policy_checks.component_reference',
                             'wdl': 'existing tests.wdl_checks.numpy_reference: float64 manual chain rule',
                             'combined_backward': 'NumPy WDL token cotangent plus independent policy autograd through independent backbone'},
              'limitations': ['intermediate GAB integration only; M06 required before supervised/RL training or real evaluator',
                              'CPU float32/float64 only; CUDA/MPS/reduced precision/other builds unvalidated',
                              'no throughput, checkpoint compatibility or playing-strength claim',
                              'terminal adapter fixtures are raw tensor probes, not evaluator calls',
                              'no training, optimizer step, relative terms, data, evaluator, search, subagents or paid compute']}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == 'Darwin' else peak*1024
    report['measured_process_memory'] = {
        'metric': 'resource.getrusage(RUSAGE_SELF).ru_maxrss', 'raw_value': peak,
        'raw_unit': 'bytes' if platform.system() == 'Darwin' else 'KiB', 'process_peak_rss_bytes': peak_bytes,
        'scope': 'lifetime of this checker process, including imports, references, cloned tensors, all cases and allocator retention',
        'not_model_peak_or_training_memory': True, 'not_tensor_payload_accounting': True}
    text = json.dumps(report, indent=2, sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
