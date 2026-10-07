"""Optional M05 assembly acceptance; Torch-free adapter installs skip."""
import pytest

torch = pytest.importorskip('torch')
from chess_rl.model import ChessNetwork
from .network_checks import (numerical_case, routing_and_joint_case, adapter_case,
                             dropout_case, constructor_and_memory_case)


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('batch', [1, 2, 5])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_joint_independent_outputs_input_all_parameter_gradients(batch, dtype):
    numerical_case(batch, dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_independent_head_widths(dtype):
    numerical_case(2, dtype, 7, 11)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_one_backbone_same_final_tokens_joint_gradients_unique_optimizer(dtype):
    routing_and_joint_case(dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_real_adapter_batch_preservation(dtype):
    adapter_case(dtype)


def test_dropout_propagation():
    dropout_case()


def test_direct_component_parity_count_memory_dtype():
    constructor_and_memory_case()


@pytest.mark.parametrize('shape', [(64, 128), (1, 63, 128), (1, 64, 256), (1, 1, 64, 128)])
def test_reject_invalid_feature_shape(shape):
    with pytest.raises(ValueError, match='expects'):
        ChessNetwork()(torch.zeros(shape))


@pytest.mark.parametrize('field', ['d_policy', 'd_value'])
@pytest.mark.parametrize('width', [0, -1, True, False, 1.5, '128', None])
def test_head_width_validation_propagates(field, width):
    with pytest.raises(ValueError, match=field):
        ChessNetwork(**{field: width})


@pytest.mark.parametrize('dropout', [-.1, 1.1])
def test_backbone_dropout_validation_propagates(dropout):
    with pytest.raises(ValueError, match='dropout'):
        ChessNetwork(dropout=dropout)


def test_reject_empty_tensor_batch():
    with pytest.raises(ValueError, match='nonempty batch'):
        ChessNetwork().eval()(torch.empty(0, 64, 128))
