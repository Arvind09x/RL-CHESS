"""CUDA mixed checkpoints have CPU schema coverage and a real hardware gate."""
import copy

import pytest

torch = pytest.importorskip("torch")
from chess_rl.training.checkpoints import (CUDA_MIXED_CHECKPOINT_VERSION, ModelConfig, Progress,
    load_inference, restore_training, save_checkpoint)
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedSettings, MixedState
from .test_mixed_update import sources, optimizer, save_mixed
from .test_selfplay import cpu, fixture_checkpoint, opening
from .test_cuda_support import cuda_payload
from .test_checkpoints import assert_tree_equal
from chess_rl.selfplay import Session, Settings


def test_cuda_mixed_schema_reads_on_cpu(sources, tmp_path):
    data, _ = sources
    path = tmp_path / "cpu.pt"
    save_mixed(path, data)
    payload = cuda_payload({"path": path})
    payload["version"] = CUDA_MIXED_CHECKPOINT_VERSION
    torch.save(payload, tmp_path / "cuda.pt")
    loaded = load_inference(tmp_path / "cuda.pt")
    assert loaded.config.device == "cpu"
    assert_tree_equal(loaded.model.state_dict(), payload["model_state"])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() != 1, reason="requires one visible CUDA GPU")
def test_real_cuda_mixed_reload_resume_and_selfplay(sources, tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    data, _ = sources
    settings = MixedSettings()
    config = ModelConfig(d_policy=7, d_value=5, dropout=.2, device="cuda:0")
    model = config.build()
    opt = optimizer(model, settings)
    _, state = mixed_step(model, opt, data, MixedState.initial(data, settings, seed=7))
    path = tmp_path / "mixed.pt"
    save_checkpoint(path, model=model, optimizer=opt, checkpoint_id="gpu-fixture", provenance={"test":"mixed"},
                    weights=settings.weights, progress=Progress(1, 0, 4), mixed_state=state)
    _, expected_state = mixed_step(model, opt, data, state)
    expected_model = copy.deepcopy(model.state_dict())
    expected_opt = copy.deepcopy(opt.state_dict())
    torch.manual_seed(999)
    loaded = restore_training(path, expected_config=config, expected_provenance={"test":"mixed"},
                              mixed_sources=data, expected_mixed_settings=settings)
    _, actual_state = mixed_step(loaded.model, loaded.optimizer, data, loaded.mixed_state)
    assert actual_state == expected_state
    for name, tensor in loaded.model.state_dict().items():
        torch.testing.assert_close(tensor, expected_model[name], atol=1e-6, rtol=1e-5)
    for key, values in loaded.optimizer.state_dict()["state"].items():
        for name, tensor in values.items():
            torch.testing.assert_close(tensor, expected_opt["state"][key][name], atol=1e-6, rtol=1e-5)
    # The sources fixture used a dummy evaluator only for its source records.
    from chess_rl.search.real_evaluator import RealEvaluator
    monkeypatch.setattr("chess_rl.selfplay.RealEvaluator", RealEvaluator)
    session = Session.create(tmp_path / "gpu-game", path, opening(), settings=Settings(2, 2),
                             seed=7, opening_selection_seed=8, device="cuda:0", journal=True)
    session.run(stop_after_moves=1)
    result = Session.load(session.directory).run()
    assert result["game"]["status"] == "truncated"
    assert result["configuration"]["runtime"]["device"] == "cuda:0"
