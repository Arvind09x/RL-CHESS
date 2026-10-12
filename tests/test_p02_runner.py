"""P02 diagnostics, with no production dataset or pilot execution."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from scripts.train_supervised import gradient_norm, validate
from .test_training import owned_batch
from chess_rl.model import ChessNetwork
from chess_rl.training.trainer import supervised_step
from chess_rl.training.curves import write_learning_curves


def test_gradient_norm_is_global_unclipped_and_counts_shared_bank_once():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(123)
        model = ChessNetwork()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False, fused=False)
        supervised_step(model, optimizer, owned_batch())
        gradients = [p.grad.detach().numpy().astype(np.float64) for p in model.parameters()]
        expected = np.sqrt(sum(np.sum(g * g) for g in gradients))
        copies = [p.grad.clone() for p in model.parameters()]
        assert gradient_norm(model) == pytest.approx(expected, rel=1e-12)
        assert all(torch.equal(p.grad, old) for p, old in zip(model.parameters(), copies))
        next(model.parameters()).grad.flatten()[0] = torch.inf
        with pytest.raises(ValueError, match="finite"):
            gradient_norm(model)
        optimizer.zero_grad(set_to_none=True)
        with pytest.raises(ValueError, match="every"):
            gradient_norm(model)
    finally:
        torch.set_num_threads(previous)


def test_baseline_validation_preserves_model_rng_mode_and_never_updates():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = ChessNetwork()
        model.train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        batch = owned_batch()
        class Validation:
            def batches(self, batch_size):
                yield batch
        result = validate(model, Validation(), 2)
        assert result["positions"] == 2 and 0 <= result["policy_accuracy"] <= 1
        assert result["total_loss"] == pytest.approx(result["policy_loss"] + result["wdl_loss"])
        assert model.training and torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("threads", ["0", "-2"])
def test_cli_rejects_invalid_threads_before_opening_data(tmp_path, threads):
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, "scripts/train_supervised.py", "--device", "cpu",
        "--data-config", str(tmp_path / "missing.json"), "--output", str(output), "--steps", "1",
        "--threads", threads], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "threads" in result.stderr
    assert not output.exists()


def test_curves_keep_sparse_validation_measurements(tmp_path):
    import csv
    import json
    from xml.etree import ElementTree
    validation = dict(total_loss=3., policy_loss=2., wdl_loss=1., policy_accuracy=.25, positions=4)
    train = dict(batch_size=2, total_loss=2.5, policy_loss=1.5, wdl_loss=1., gradient_norm=5.)
    rows = [dict(step=10, epoch=0, event="resumed_baseline", elapsed_seconds=0, validation=validation),
            dict(step=11, epoch=0, event="update", elapsed_seconds=1, train=train),
            dict(step=12, epoch=0, event="update", elapsed_seconds=2, train=train, validation=validation)]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    write_learning_curves(tmp_path)
    with (tmp_path / "learning-curves.csv").open() as handle:
        values = list(csv.DictReader(handle))
    assert values[1]["validation_total_loss"] == ""  # no invented validation measurement
    assert values[2]["validation_total_loss"] == "3.0"
    assert "Resumed baseline at step 10" in (tmp_path / "learning-curves.svg").read_text()
    plot = ElementTree.parse(tmp_path / "learning-curves.svg")
    assert all("nan" not in line.get("points") for line in plot.findall(".//{http://www.w3.org/2000/svg}polyline"))
