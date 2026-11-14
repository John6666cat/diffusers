# Experimental pre-encoded reference-token I2I; not a full VAE/Qwen/GPU test.
import pytest
import torch
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel


def _model():
    return Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        num_attention_heads=2, attention_head_dim=16, joint_attention_dim=32,
        timestep_channels=32, axes_dims_rope=(4, 4, 4, 4), flow_map=False,
    ).eval()


def _config(allow=False):
    return SpectrumCacheConfig(num_inference_steps=4, warmup_steps=2,
        forecast_step_indices=(2,), degree=1,
        coordinate_policy="runtime_horizon_normalized",
        allow_flux3_image_conditioned=allow)


def _fields(refs=1):
    assert refs in (1, 2)
    return dict(hidden_states=torch.randn(1, 4, 8), encoder_hidden_states=torch.randn(1, 3, 32),
                img_ids=torch.zeros(4, 4), txt_ids=torch.zeros(3, 4),
                image_cond_hidden_states=torch.randn(1, 2 * refs, 8),
                image_cond_ids=torch.zeros(2 * refs, 4))


def _steps(model, fields, *, label='cond'):
    outs = []
    for i in range(4):
        with model.cache_context(label, step_index=i, num_inference_steps=4, timestep=i):
            outs.append(model(**fields, timestep=torch.tensor([1.-0.2*i])).sample)
    return outs


def _state(model, name='cond'):
    hook = model._diffusers_hook.get_hook('spectrum_cache_denoiser')
    return hook.state_manager._state_cache[name]


@pytest.mark.parametrize('refs', [1, 2])
@torch.no_grad()
def test_default_i2i_fully_bypasses_and_matches_no_cache(refs):
    torch.manual_seed(97)
    model = _model()
    fields = _fields(refs)
    expected = [model(**fields, timestep=torch.tensor([1.-0.2*i])).sample for i in range(4)]
    model.enable_cache(_config(allow=False))
    actual = _steps(model, fields)
    assert all(torch.equal(a,b) for a,b in zip(actual, expected))
    state = _state(model)
    assert state.guard_latched
    assert state.summary()['prediction_call_count'] == 0
    assert state.summary()['full_call_count'] == 4
    model.disable_cache()


@pytest.mark.parametrize('refs', [1, 2])
@torch.no_grad()
def test_optin_preencoded_references_predict_separately_per_cfg_slot(refs):
    torch.manual_seed(97)
    model = _model()
    fields = _fields(refs)
    negative = dict(fields, encoder_hidden_states=torch.randn_like(fields['encoder_hidden_states']))
    block = model.single_transformer_blocks[0]
    calls = [0]
    original = block.forward

    def count(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    block.forward = count
    model.enable_cache(_config(allow=True))
    for i in range(4):
        for slot, branch in [('cond',fields),('uncond',negative)]:
            with model.cache_context(slot, step_index=i, num_inference_steps=4, timestep=i):
                y=model(**branch,timestep=torch.tensor([1.-0.2*i])).sample
                assert torch.isfinite(y).all()
    assert calls[0] == 6  # 8 calls, 1 prediction per CFG slot
    assert _state(model,'cond').summary()['prediction_call_count'] == 1
    assert _state(model,'uncond').summary()['prediction_call_count'] == 1
    assert not _state(model,'cond').guard_latched
    assert not _state(model,'uncond').guard_latched
    model.disable_cache()


@pytest.mark.parametrize('key', ['image_cond_hidden_states', 'image_cond_ids'])
@torch.no_grad()
def test_reference_identity_change_midrun_sticky_bypass(key):
    model=_model()
    fields=_fields(2)
    model.enable_cache(_config(allow=True))
    for i in range(4):
        current=dict(fields)
        if i >= 1:
            current[key]=fields[key].clone()
        with model.cache_context('cond',step_index=i,num_inference_steps=4,timestep=i):
            y=model(**current,timestep=torch.tensor([1.-0.2*i])).sample
            assert torch.isfinite(y).all()
    state=_state(model)
    assert state.guard_latched
    assert state.summary()['prediction_call_count']==0
    model.disable_cache()


@torch.no_grad()
def test_optin_sticky_state_cleared_between_calls():
    model=_model()
    fields=_fields()
    model.enable_cache(_config(allow=True))
    _steps(model, fields)
    assert _state(model).summary()['prediction_call_count']==1
    model._reset_stateful_cache()
    assert not model._diffusers_hook.get_hook('spectrum_cache_denoiser').state_manager._state_cache
    changed=_fields()
    _steps(model, changed)
    assert _state(model).summary()['prediction_call_count']==1
    model.disable_cache()


def test_distilled_still_rejected_explicitly():
    model=Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        num_attention_heads=2, attention_head_dim=16, joint_attention_dim=32,
        timestep_channels=32, axes_dims_rope=(4, 4, 4, 4), flow_map=True,
    ).eval()
    with pytest.raises(ValueError,match='distilled requires explicit opt-in'):
        model.enable_cache(_config(allow=True))
    assert not model.is_cache_enabled


@pytest.mark.parametrize('key', ['image_cond_hidden_states', 'image_cond_ids'])
@torch.no_grad()
def test_reference_inplace_mutation_midrun_sticky_bypass(key):
    model=_model()
    fields=_fields(2)
    model.enable_cache(_config(allow=True))
    for i in range(4):
        if i == 1:
            fields[key].add_(0.1)  # Same storage pointer, different tensor version.
        with model.cache_context('cond',step_index=i,num_inference_steps=4,timestep=i):
            y=model(**fields,timestep=torch.tensor([1.-0.2*i])).sample
            assert torch.isfinite(y).all()
    state=_state(model)
    assert state.guard_latched
    assert state.summary()['prediction_call_count']==0
    model.disable_cache()
