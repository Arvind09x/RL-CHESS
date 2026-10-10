"""Explicitly measure the RNG boundary of a rejected gradient."""
from copy import deepcopy
import torch
import pytest
from chess_rl.training.checkpoints import ModelConfig
from chess_rl.training.mixed import mixed_step
from chess_rl.training.mixed_state import MixedState, MixedSettings
from tests.test_selfplay import cpu, fixture_checkpoint
from tests.test_mixed_update import sources, optimizer
from tests.test_checkpoints import assert_tree_equal

def test_rejected_gradient_preserves_model_cursor_but_consumes_dropout_rng(sources):
    data,_=sources; settings=MixedSettings()
    state=MixedState.initial(data,settings,seed=77)
    model=ModelConfig(dropout=.1).build(); opt=optimizer(model,settings)
    before=deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
    candidate=state.next_selection(); calls=[]
    handle=next(model.parameters()).register_hook(lambda g:g*torch.nan)
    opt.register_step_pre_hook(lambda *args:calls.append(True))
    with pytest.raises(ValueError,match='finite mixed gradient'): mixed_step(model,opt,data,state)
    handle.remove()
    assert_tree_equal(before,model.state_dict())
    assert calls==[] and state.next_selection()==candidate and len(opt.state)==0
    assert not torch.equal(rng,torch.get_rng_state())
