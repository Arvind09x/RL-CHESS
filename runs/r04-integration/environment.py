"""Bind the actual isolated imports and pinned runtimes."""
import importlib.metadata as metadata,importlib.util,json,platform,sys
from pathlib import Path
import chess_rl
from checks import OUT,ROOT,identity,save
result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
    isolated_package=str(Path(chess_rl.__file__).resolve()),
    dependencies={n:metadata.version(n) for n in ('numpy','python-chess','chess','pytest')})
assert Path(chess_rl.__file__).resolve().is_relative_to(ROOT/'src')
assert sys.version_info[:2]==(3,12)
assert result['dependencies']['numpy']=='2.5.3'
assert result['dependencies']['python-chess']=='1.999'
has_torch=importlib.util.find_spec('torch') is not None
if has_torch:
    import torch
    result.update(torch=torch.__version__,cuda_runtime=torch.version.cuda)
    assert torch.__version__=='2.8.0'
save(OUT/('environment-model.json' if has_torch else 'environment-adapter.json'),result)
print(json.dumps(result,indent=2))

