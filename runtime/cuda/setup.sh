#!/usr/bin/env bash
# Pinned user-space environment; does not alter the college machine's driver.
set -euo pipefail
cd "$(dirname "$0")/../.."
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
command -v uv >/dev/null
if test -e .venv-cuda-cu126; then
  echo '.venv-cuda-cu126 already exists; use it or choose a fresh checkout' >&2
  exit 1
fi
uv venv --python 3.12.12 .venv-cuda-cu126
uv pip sync --python .venv-cuda-cu126/bin/python --require-hashes \
  --torch-backend cu126 --build-constraints runtime/cuda/build-constraints.txt \
  runtime/cuda/requirements.lock
uv pip install --python .venv-cuda-cu126/bin/python --no-deps -e .
uv pip check --python .venv-cuda-cu126/bin/python
.venv-cuda-cu126/bin/python -B -c 'import torch, chess, numpy; assert torch.__version__ == "2.8.0+cu126"; assert torch.version.cuda == "12.6"; assert chess.__version__ == "1.11.2"; assert numpy.__version__ == "2.5.3"; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
