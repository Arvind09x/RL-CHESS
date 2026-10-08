"""Offline CPU M06 relative/combined architecture acceptance and accounting."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tests.relative_checks import (layout_case, contraction_case, attention_case, numerical_case,
                                   recovery_case, accounting_case, probability_dropout_case,
                                   routing_and_joint_case, adapter_case, dropout_case)


def source_audit(checkout):
    lock=json.loads((ROOT/'docs/upstream-lock.json').read_text())
    pin=lock['repositories']['lczero']
    command=['git','-C',str(checkout),'show',pin['commit']+':tf/tfprocess.py']
    source=subprocess.check_output(command)
    digest=hashlib.sha256(source).hexdigest()
    assert digest==pin['files']['tf/tfprocess.py']['sha256']
    lines=source.decode().splitlines(keepends=True)
    snippets={name:''.join(lines[a-1:b]) for name,a,b in (
        ('map',83,91),('logits',285,317),('values',319,341),('attention',1960,2007))}
    assert "self.head_depth, self.head_count, 64, 64" in snippets['logits']
    assert "initializer=\"zeros\"" in snippets['logits'] and "initializer=\"zeros\"" in snippets['values']
    assert "'bhkd, dhqk->bhqk'" in snippets['logits']
    return {'command':command,'sha256':digest,'commit':pin['commit'],
            'inspected_operations':snippets,'runtime_upstream_import':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--lczero-checkout',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dtypes=(torch.float32,torch.float64)
    report={'milestone':'M06 only','status':'passed automated acceptance; stop for independent review',
            'date':datetime.now(ZoneInfo('Asia/Kolkata')).date().isoformat(),
            'environment':{'python':platform.python_version(),'platform':platform.platform(),
                           'torch':torch.__version__,'torch_git':torch.version.git_version,
                           'device':'cpu','threads':1,'deterministic_algorithms':True,
                           'installed_distributions':dict(sorted((d.metadata['Name'],d.version)
                                                                 for d in importlib.metadata.distributions()))},
            'seeds':{'nonzero_contractions_attention_recovery_accounting_dropout':606,
                     'combined_numerical':1052026,'joint_adapter_network_dropout':505,
                     'attention_dropout_replay':35,'attention_dropout_alternative':36},
            'tolerances':{'float32':{'atol':3e-5,'rtol':5e-4},'float64':{'atol':2e-11,'rtol':2e-10}},
            'objectives':{'tensor':'sum(y*(sin(.031*i)+.4*cos(.011*i))+.03*y**2)/sqrt(numel)',
                          'wdl':'sum((.17*i-.4)*y+.03*y**2)', 'joint':'policy tensor objective + wdl objective'},
            'index_layout':layout_case(),
            'contractions':[contraction_case(b,d) for d in dtypes for b in (1,2,5)],
            'attention':[attention_case(b,d) for d in dtypes for b in (1,2,5)],
            'combined':[numerical_case(b,d) for d in dtypes for b in (1,2,5)],
            'independent_head_widths':[numerical_case(2,d,7,11) for d in dtypes],
            'm05_recovery':[recovery_case(b,d) for d in dtypes for b in (1,2,5)],
            'accounting':[accounting_case(d) for d in dtypes],
            'probability_dropout':[probability_dropout_case(d) for d in dtypes],
            'joint_head_routing':[routing_and_joint_case(d) for d in dtypes],
            'adapter':[adapter_case(d) for d in dtypes],
            'network_dropout':dropout_case(),
            'parameter_arithmetic':'3667011 + 8*3*8*225*32 = 5049411',
            'estimate_resolution':'6–8M was provisional; exact fixed-width shape arithmetic yields 5,049,411. No width/layer/head/FFN inflation.',
            'references':{'relative':'independent one-hot factorizer, explicit depth-before-head stack and per-square matrix products',
                          'combined':'independent M02 affine/manual norm/erf math extended by the relative reference; independent M03 component reference and M04 NumPy manual WDL gradients',
                          'production_math_calls':False},
            'limitations':['CPU float32/float64 local checks only; CUDA/MPS/reduced precision/other builds unvalidated',
                           'upstream TensorFlow runtime not executed; source equations/layout ported to independent PyTorch math',
                           'no upstream RNG/framework or checkpoint equivalence claim',
                           'terminal fixtures are raw tensor probes, not evaluator calls',
                           'no performance, training-memory, playing-strength or superiority claim',
                           'no data/training/checkpoints/evaluator/search/GPU/paid compute/subagents']}
    if args.lczero_checkout:report['pinned_source_audit']=source_audit(args.lczero_checkout)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['process_memory']={'metric':'resource.getrusage(RUSAGE_SELF).ru_maxrss','raw':rss,
                              'native_unit':'bytes' if platform.system()=='Darwin' else 'KiB',
                              'lifetime_peak_rss_bytes':rss if platform.system()=='Darwin' else rss*1024,
                              'scope':'checker process including imports, all cases, references, clones, graphs and allocator retention; not isolated model peak or tensor payload'}
    text=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print(text)

if __name__=='__main__':main()
