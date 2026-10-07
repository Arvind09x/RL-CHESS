# M01 compact GAB acceptance — 7 October 2026

M01 ports only the compact GAB operation and shared template bank. The contract
is `[B,64,256] → [B,8,64,64]`, with d1=8, d2=32, d3=32. Destination varies
fastest. Biased square projection precedes square-first flattening; biased
board projection precedes GELU and LayerNorm(32); biased coefficient projection
precedes GELU and one LayerNorm(256), then head reshape and bank contraction.
There is no mean pooling, per-head normalization or extra scaling.

The source is CSSLab/maia3 at `1e13597c42d4858b7cfd7cfdae01e297263364b2`,
`maia3/models.py`: MHA constructor 73–90, `_sq_bias` 95–118, shared bank
304–306 and encoder sharing 253–271. Authors: Daniel Monroe, George Eilender,
Philip Chalmers, Zhenwei Tang, Ashton Anderson / UofT CSSLab, *Chessformer:
A Unified Architecture for Chess Modeling*, ICLR 2026. This local port isolates
the compact generator from attention and requires an explicit bank.
[Port map](upstream-sources.md), [source lock](upstream-lock.json) and retained
[AGPL-3.0 text](notices/maia3-AGPL-3.0.txt) supply attribution/notices. The
frozen test method preserves the pinned `_sq_bias` verbatim with a local wrapper.

## Tested environment and constructor defaults

PyTorch **2.8.0**, build commit `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`,
Python 3.12.12, macOS 14.5 arm64; CPU execution, one thread and deterministic
algorithms for the numerical report. No GPU or paid compute was used. The
supported/tested M01 environment is this CPU build; Linux/Windows CPU artifacts
are locked but untested. PyTorch lists this release in its
[official installation archive](https://pytorch.org/get-started/previous-versions/).
`pyproject.toml` pins the optional model extra and `uv.lock` resolves exact
versions/artifact hashes, including the tested macOS cp312 arm64 wheel
SHA-256 `619c2869db3ada2c0105487ba21b5008defcc472d23f8b80ed91ac4a380283b0`.

Resolved PyTorch dependencies in the tested environment: filelock 4.0.12,
fsspec 2026.9.0, Jinja2 3.1.6, MarkupSafe 3.0.4, mpmath 1.3.0, networkx 3.7,
setuptools 84.0.0, sympy 1.14.0, typing-extensions 4.16.0. Adapter/runtime pins
remain NumPy 2.5.3, python-chess 1.999 / chess 1.11.2; pytest 9.1.1.
The report records all installed distributions and Torch's build configuration.

Resolved from the installed framework source and tested constructors:

| Constructor | Retained behavior |
|---|---|
| Linear 256→8, 512→32, 32→256 | All biases enabled. Weights use `kaiming_uniform_(a=sqrt(5))`, equivalent to uniform ±`1/sqrt(fan_in)`; biases use the same bound. No model-wide reset. |
| LayerNorm(32), LayerNorm(256) | eps=1e-5, elementwise_affine=True, bias=True, initial weight=1, bias=0, biased variance. The second norm spans all 256 coefficients jointly before head reshape. |
| GELU | `approximate='none'` (erf form). |
| Shared bank [4096,32] | One `nn.Parameter`, Xavier **normal**, gain=1, fan_in=32 and fan_out=4096, theoretical standard deviation `sqrt(2/4128)`. |
| Device/dtype | New bank defaults to CPU float32; generator constructors inherit its device/dtype. Validation additionally uses float64. |

Upstream pins no PyTorch version. These are resolved local defaults, not a claim
of equivalent upstream framework build or original RNG stream. M02's other
constructors and the full model parameter count remain later work.

## Numerical and sharing acceptance

The actual [numerical report](../runs/m01/numerical-acceptance.json) records 12
cases: batches 1/2/5 × float32/float64 × two independent reference paths. Inputs
are asymmetric and noncontiguous. All model parameters are made nonzero,
including projection biases and both norms' affine weights/biases. The
nonuniform scalar objective checks input gradients and all 11 parameter tensors.
There are no reference calls into production math helpers.

One path invokes the preserved pinned method through a separate minimal wrapper.
Its output and gradient errors are exactly zero in all cases. Acceptance also
verified the full upstream file checksum and exact method bytes from the pinned
Git object, without importing the full Maia project. The second path uses
explicit per-square concatenation, functional affine projections, erf GELU,
manual variance normalization and separate per-head matrix products.

| Dtype | atol / rtol | Independent maximum output error | Independent maximum gradient error |
|---|---|---|---|
| float32 | 2e-5 / 2e-4 | 1.0728836059570312e-6 | 9.238719940185547e-7 |
| float64 | 1e-11 / 1e-10 | 1.5543122344752192e-15 | 1.7763568394002505e-15 |

All outputs/gradients are finite; all input and parameter gradients are nonzero.
Off-diagonal checks distinguish origin/destination transposition. Tests also
verify constructor defaults and seeded exact Xavier-normal bank initialization.

Two generators have identical `bank.weight` parameter identity, with ten
separate projection/norm parameter tensors each. A `ModuleList.parameters()`
optimizer has 21 unique parameter tensors (186,064 elements), exactly one bank
entry. Each generator contributes a nonzero bank gradient; combined backward
matches the sum of individual bank gradients with maximum absolute error zero
in both dtypes. A single SGD update matches its expected result (float32 maximum
error 7.450580596923828e-9, float64 zero). This is an optimizer semantics check,
not a training run. Use the enclosing module's parameter iterator rather than
concatenating per-generator parameter lists. Shared modules may expose repeated
state-dictionary paths; loading requires construction with the same bank.

## Commands and saved results

[Command records](../runs/m01/commands.json) link each command to saved raw
output. Actual runs used the installed interpreters, requiring no network:

```sh
.venv/bin/python scripts/check_m01.py --maia3-checkout /private/tmp/rl-chess-f03-maia3 --output runs/m01/numerical-acceptance.json
.venv/bin/python -m pytest --junitxml=runs/m01/pytest-model.xml
.venv-acceptance/bin/python -m pytest --junitxml=runs/m01/pytest-adapter-only.xml
.venv/bin/python scripts/check_adapter.py
.venv-acceptance/bin/python scripts/check_adapter.py
uv pip check --python .venv/bin/python
uv pip check --python .venv-acceptance/bin/python
.venv/bin/python scripts/check_upstream.py --maia3-checkout /private/tmp/rl-chess-f03-maia3 --lczero-checkout /private/tmp/rl-chess-f03-lczero
git diff --check
```

Results: **139 passed** with the model extra; **119 passed, one optional model
module skipped** without Torch. All 13 adapter replay fixtures matched in both
environments and in the subprocess regression tests. Both dependency checks
passed; F03 source-check coverage/notices passed unchanged. The adapter-only
interpreter confirmed Torch was neither installed nor imported, including after
`UV_PROJECT_ENVIRONMENT=.venv-acceptance uv sync --locked --offline` using
the updated lock.
[Planning preservation](../runs/m01/original-plan-preservation.json) verifies the
original files against the foundation hashes and unchanged repository
`interface.md` / `plan1.md`. Historical F03 and adapter evidence is preserved.

Ordinary tests and `scripts/check_m01.py` run offline with no checkout argument;
the checkout argument is only an optional source-byte audit. Setup downloads
locked dependencies once. Default `uv sync --locked` omits the model extra;
use `--extra model` for model checks, including `uv run`, to retain Torch.

M01 is complete after these numerical passes and stops for review. M02,
attention integration, policy/WDL heads, relative terms, data annotation,
training and search remain unimplemented. CUDA, MPS, reduced precision and
other platform builds are unvalidated. No pretrained checkpoint compatibility
or full-model count is claimed. No subagents were launched.


## Independent AI review acceptance — 7 October 2026

The user supplied acceptance of M01 commit
`66eea36ba74ba811298ce8347a5153f562c474be` following independent AI review.
The reviewer inspected the implementation and reran 139 model tests, 119
adapter-only tests with one expected skip, all 12 numerical cases, sharing
checks, both 13-fixture adapter replays, pinned-source checks and dependency
checks. Original planning hashes and the clean working tree were verified;
no blocking findings were identified. This is AI review acceptance, not a
human teammate's review. It supersedes the historical pending-review status
above without changing its evidence. `runs/m01/` is preserved byte-for-byte.
See [structured review record](../runs/m02/m01-review-acceptance.json) and
[M02 backbone acceptance](m02-backbone.md). Only M02 was authorized next.
