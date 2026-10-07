"""Optional M03 head acceptance; adapter-only environments skip this module."""
import pytest

torch = pytest.importorskip('torch')
from chess_rl.model import PairPolicyHead
from .policy_checks import (numerical_case, exact_order_and_dependence_case,
                            constructor_case, adapter_ids_case, backbone_diagnostic_case)


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('batch', [1, 2, 5])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('d_policy', [7, 128])
def test_reference_scalars_outputs_and_all_gradients(batch, dtype, d_policy):
    numerical_case(batch, dtype, d_policy)


def test_every_slot_exact_order_scaling_and_destination_dependence():
    exact_order_and_dependence_case()


def test_bias_free_resolved_constructor_initialization():
    constructor_case()


def test_known_ids_through_adapter_apis():
    adapter_ids_case()


def test_accepted_backbone_contextual_tokens_diagnostic():
    backbone_diagnostic_case()


@pytest.mark.parametrize('width', [0, -1, True, False, 1.5, 128.0, '128', None, float('nan')])
def test_invalid_policy_width(width):
    with pytest.raises(ValueError, match='d_policy'):
        PairPolicyHead(d_policy=width)


@pytest.mark.parametrize('shape', [(64, 256), (1, 63, 256), (1, 64, 128), (1, 1, 64, 256)])
def test_invalid_contextual_token_shape(shape):
    with pytest.raises(ValueError, match='expects'):
        PairPolicyHead()(torch.zeros(shape))


def test_empty_tensor_batch_retains_shape():
    assert PairPolicyHead()(torch.empty(0, 64, 256)).shape == (0, 4352)
