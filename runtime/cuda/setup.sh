#!/usr/bin/env bash
# Run from the isolated repository on an authorized Linux x86_64 CUDA host.
set -euo pipefail
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
test "$(python3.12 -c 'import platform; print(platform.python_version())')" = 3.12.12
test "$(uv --version | cut -d ' ' -f 2)" = 0.12.10
test ! -e .venv-cuda-cu126
uv venv --python "$(command -v python3.12)" .venv-cuda-cu126
uv pip sync --python .venv-cuda-cu126/bin/python --require-hashes \
  --torch-backend cu126 --build-constraints runtime/cuda/build-constraints.txt \
  runtime/cuda/requirements.lock
uv pip check --python .venv-cuda-cu126/bin/python
.venv-cuda-cu126/bin/python -B -c 'import torch, chess, numpy, platform; assert platform.python_version() == "3.12.12"; assert torch.__version__ == "2.8.0+cu126"; assert torch.version.cuda == "12.6"; assert chess.__version__ == "1.11.2"; assert numpy.__version__ == "2.5.3"; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
