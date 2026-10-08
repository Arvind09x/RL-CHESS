"""Optional M06 CPU acceptance; adapter-only installs skip without Torch."""
import pytest
torch=pytest.importorskip('torch')
from .relative_checks import (layout_case, contraction_case, attention_case, numerical_case,
                             recovery_case, accounting_case, probability_dropout_case,
                             routing_and_joint_case, adapter_case, dropout_case)

@pytest.fixture(scope='module',autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_index_and_depth_before_head_layout():layout_case()

@pytest.mark.parametrize('batch',[1,2,5])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('check',[contraction_case,attention_case,numerical_case,recovery_case])
def test_independent_outputs_inputs_and_all_parameters(batch,dtype,check):check(batch,dtype)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('check',[accounting_case,probability_dropout_case,routing_and_joint_case,adapter_case])
def test_ownership_memory_dropout_heads_and_adapter(dtype,check):check(dtype)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_head_widths(dtype):numerical_case(2,dtype,7,11)

def test_full_network_train_eval_dropout():dropout_case()
