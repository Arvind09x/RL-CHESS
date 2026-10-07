"""M01 optional model checks; adapter-only environments skip this module."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from chess_rl.model.gab import CompactGAB, SharedTemplateBank
from .gab_checks import numerical_case, sharing_case, functional_reference, pinned_reference


@pytest.mark.parametrize("batch", [1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reference", [functional_reference, pinned_reference])
def test_outputs_and_all_gradients(batch, dtype, reference):
    numerical_case(batch, dtype, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_parameter_and_gradient_accumulation(dtype):
    sharing_case(dtype)


def test_bank_initializer_and_constructor_defaults():
    torch.manual_seed(11)
    bank = SharedTemplateBank()
    torch.manual_seed(11)
    expected = torch.empty(4096, 32)
    nn.init.xavier_normal_(expected)
    torch.testing.assert_close(bank.weight, expected, rtol=0, atol=0)
    model = CompactGAB(bank)
    assert bank.weight.device.type == "cpu" and bank.weight.dtype == torch.float32
    for layer in (model.sm1, model.sm2, model.sm3):
        bound = layer.in_features ** -0.5
        assert layer.bias is not None
        assert layer.weight.abs().max() <= bound and layer.bias.abs().max() <= bound
    for norm, shape in ((model.ln1, (32,)), (model.ln2, (256,))):
        assert norm.normalized_shape == shape and norm.eps == 1e-5
        assert norm.elementwise_affine
        assert torch.all(norm.weight == 1) and torch.all(norm.bias == 0)
    assert model.sm_act.approximate == "none"


@pytest.mark.parametrize("shape", [(64, 256), (1, 63, 256), (1, 64, 128), (1, 8, 64, 256)])
def test_input_shape_rejected(shape):
    with pytest.raises(ValueError, match="expects"):
        CompactGAB(SharedTemplateBank())(torch.zeros(shape))


def test_explicit_bank_required():
    with pytest.raises(TypeError, match="SharedTemplateBank"):
        CompactGAB(nn.Parameter(torch.empty(4096, 32)))
