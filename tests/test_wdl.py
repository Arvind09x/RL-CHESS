"""Optional M04 acceptance; adapter-only installations skip this module."""
import pytest

torch = pytest.importorskip('torch')
from chess_rl.model import WDLHead
from .wdl_checks import numerical_case, synthetic_case, constructor_case, backbone_diagnostic_case


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('batch', [1, 2, 5])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('d_value', [7, 128])
def test_numpy_outputs_output_input_all_parameter_gradients_and_permutation(batch, dtype, d_value):
    numerical_case(batch, dtype, d_value)


def test_known_synthetic_pool_norm_relu_contract_rows():
    synthetic_case()


def test_resolved_constructor_initialization_and_settings():
    constructor_case()


def test_real_adapter_backbone_diagnostic_preserves_history():
    backbone_diagnostic_case()


@pytest.mark.parametrize('width', [0, -1, True, False, 1.5, 128.0, '128', None, float('nan')])
def test_invalid_value_width(width):
    with pytest.raises(ValueError, match='d_value'):
        WDLHead(d_value=width)


@pytest.mark.parametrize('shape', [(64, 256), (1, 63, 256), (1, 64, 128), (1, 1, 64, 256)])
def test_invalid_final_token_shape(shape):
    with pytest.raises(ValueError, match='expects'):
        WDLHead()(torch.zeros(shape))


def test_empty_tensor_batch():
    assert WDLHead()(torch.empty(0, 64, 256)).shape == (0, 3)
