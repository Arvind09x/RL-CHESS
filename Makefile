.PHONY: test test-adapter gpu-preflight gpu-test

test:
	uv run --locked --extra model pytest

test-adapter:
	uv run --locked pytest

gpu-preflight:
	.venv-cuda-cu126/bin/python scripts/gpu_preflight.py

gpu-test:
	.venv-cuda-cu126/bin/python scripts/check_cuda_support.py --device cuda:0 --directory artifacts/gpu-smoke
	.venv-cuda-cu126/bin/python -m pytest -m cuda tests/test_cuda_pipeline.py
