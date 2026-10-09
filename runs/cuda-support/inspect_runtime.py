import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

versions = {n:metadata.version(n) for n in ("numpy", "python-chess", "chess", "pytest")}
assert versions == dict(numpy="2.5.3", **{"python-chess":"1.999", "chess":"1.11.2", "pytest":"9.1.1"})
assert platform.python_version() == "3.12.12"
report = dict(python=platform.python_version(), platform=platform.platform(),
    interpreter=sys.executable, versions=versions, torch_available=importlib.util.find_spec("torch") is not None,
    distributions={d.metadata["Name"]:d.version for d in metadata.distributions()})
if report["torch_available"]:
    import torch
    assert str(torch.__version__) == "2.8.0"
    report.update(torch=str(torch.__version__), cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), visible_cuda_devices=torch.cuda.device_count())
print(json.dumps(report, indent=2, sort_keys=True))
