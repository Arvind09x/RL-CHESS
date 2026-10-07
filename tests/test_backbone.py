"""M02 optional model checks; adapter-only environments skip this module."""
import pytest

torch = pytest.importorskip("torch")
from chess_rl.model import EncoderBackbone
from .backbone_checks import numerical_case, sharing_case, dropout_case, adapter_case, constructor_case


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_independent_stages_outputs_and_all_gradients(batch, dtype):
    numerical_case(batch, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eight_generators_shared_bank_and_optimizer(dtype):
    sharing_case(dtype)


def test_training_and_evaluation_dropout():
    dropout_case()


def test_real_adapter_batches_preserve_state_and_history():
    adapter_case()


def test_resolved_constructors_and_initialization():
    constructor_case()


@pytest.mark.parametrize("shape", [(64, 128), (1, 63, 128), (1, 64, 256), (1, 8, 64, 128)])
def test_invalid_input_shapes(shape):
    with pytest.raises(ValueError, match="expects"):
        EncoderBackbone()(torch.zeros(shape))


@pytest.mark.parametrize("dropout", [-0.1, 1.1, float("nan")])
def test_invalid_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        EncoderBackbone(dropout=dropout)
