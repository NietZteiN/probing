"""Regression for the actual Gemma float16-cache overflow."""
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.mark.heavy
def test_forced_cache_preserves_large_finite_activations():
    import torch
    from cueconf.runner import forced_pass
    class Model:
        device = 'cpu'
        def __call__(self, input_ids, attention_mask, output_hidden_states):
            h=torch.full((*input_ids.shape,1),100000.,dtype=torch.bfloat16)
            return SimpleNamespace(hidden_states=(h,), logits=torch.zeros((*input_ids.shape,3)))
    args=(SimpleNamespace(pad_token_id=0),Model(),[[1,2]],[[0]],[[]],1)
    with pytest.raises(ValueError,match='non-finite cached hidden states'):
        list(forced_pass(*args,storage_dtype=np.float16))
    hidden,_=next(forced_pass(*args,storage_dtype=np.float32))
    assert hidden.dtype == np.float32
    assert np.isfinite(hidden).all()
    assert hidden[0,0,0] > np.finfo(np.float16).max

@pytest.mark.heavy
def test_nonfinite_probe_inputs_do_not_become_chance_scores():
    import torch
    from cueconf.probes import BatchedProbes, LinearProbe, evaluate
    probe=BatchedProbes(1,2,[0],'cpu')
    with pytest.raises(ValueError,match='non-finite training cache'):
        probe.fit_sgd(torch.full((1,2,2),float('inf')),torch.tensor([0,1]),epochs=1)
    linear=LinearProbe(2,0,'cpu')
    with pytest.raises(ValueError,match='non-finite evaluation readout'):
        evaluate(linear,torch.full((2,2),float('inf')),np.array([0,1]),np.array([2,3]))
