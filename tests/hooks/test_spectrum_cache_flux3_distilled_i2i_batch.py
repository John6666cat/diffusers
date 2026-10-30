# Isolated CPU tests: 4-step distilled flow-map + pre-encoded I2I references. No external weights.
import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.denoise import Flux3DistilledDenoiseStep
from diffusers.modular_pipelines.flux3.modular_pipeline import Flux3ModularPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def tiny():
    return Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        num_attention_heads=2, attention_head_dim=16, joint_attention_dim=32,
        timestep_channels=32, axes_dims_rope=(4, 4, 4, 4), flow_map=True,
    ).eval()


def fields(refs=1):
    return dict(hidden_states=torch.randn(1,4,8), encoder_hidden_states=torch.randn(1,3,32),
        img_ids=torch.zeros(4,4), txt_ids=torch.zeros(3,4),
        image_cond_hidden_states=torch.randn(1,2*refs,8), image_cond_ids=torch.zeros(2*refs,4))


def config(allow=False):
    return SpectrumCacheConfig(num_inference_steps=4, warmup_steps=2,
        forecast_step_indices=(2,), degree=1, blend_w=0.35,
        coordinate_policy='runtime_horizon_normalized', allow_flux3_distilled=True,
        allow_flux3_distilled_image_conditioned=allow)


def step(model, state, i):
    with model.cache_context('distilled', step_index=i, num_inference_steps=4, timestep=i):
        return model(**state,timestep=torch.tensor([1.-0.25*i]),
            target_timestep=torch.tensor([.75-.25*i])).sample


def state(model):
    hook=model._diffusers_hook.get_hook('spectrum_cache_denoiser')
    return hook.state_manager._state_cache['distilled']


@pytest.mark.parametrize('n', [1,2])
@torch.no_grad()
def test_default_off_exact_full_compute(n):
    torch.manual_seed(101)
    m=tiny(); f=fields(n)
    baseline=[m(**f,timestep=torch.tensor([1.-.25*i]),
        target_timestep=torch.tensor([.75-.25*i])).sample for i in range(4)]
    m.enable_cache(config(False))
    actual=[step(m,f,i) for i in range(4)]
    assert all(torch.equal(a,b) for a,b in zip(actual,baseline))
    assert state(m).guard_latched
    assert state(m).summary()['prediction_call_count']==0
    m.disable_cache()


@pytest.mark.parametrize('n', [1,2])
@torch.no_grad()
def test_optin_uses_one_forecast_and_recovers(n):
    torch.manual_seed(101)
    m=tiny(); f=fields(n)
    reference=[m(**f,timestep=torch.tensor([1.-.25*i]),
        target_timestep=torch.tensor([.75-.25*i])).sample for i in range(4)]
    m.enable_cache(config(True))
    outputs=[step(m,f,i) for i in range(4)]
    s=state(m).summary()
    assert all(torch.isfinite(o).all() and o.shape==reference[i].shape for i,o in enumerate(outputs))
    assert s['prediction_call_count']==1 and s['full_call_count']==3
    assert s['logical_prediction_steps_used']==[2]
    assert not s['guard_latched'] and not s['predict_failures']
    m.disable_cache()
    assert all(torch.equal(m(**f,timestep=torch.tensor([1.-.25*i]),
        target_timestep=torch.tensor([.75-.25*i])).sample,reference[i]) for i in range(4))


@pytest.mark.parametrize('key', ['image_cond_hidden_states','image_cond_ids'])
@torch.no_grad()
def test_reference_replacement_latches(key):
    m=tiny(); f=fields(2);m.enable_cache(config(True))
    for i in range(4):
        g=dict(f)
        if i>=1:g[key]=f[key].clone()
        assert torch.isfinite(step(m,g,i)).all()
    assert state(m).guard_latched
    assert state(m).summary()['prediction_call_count']==0
    m.disable_cache()


@pytest.mark.parametrize('key', ['image_cond_hidden_states','image_cond_ids'])
@torch.no_grad()
def test_reference_mutation_latches(key):
    m=tiny();f=fields(2);m.enable_cache(config(True))
    for i in range(4):
        if i==1:f[key].add_(0.1)
        assert torch.isfinite(step(m,f,i)).all()
    assert state(m).guard_latched
    assert state(m).summary()['prediction_call_count']==0
    m.disable_cache()


@torch.no_grad()
def test_reset_and_injected_prediction_failure():
    m=tiny();f=fields(1);m.enable_cache(config(True))
    step(m,f,0);step(m,f,1)
    state(m).predict=lambda: (_ for _ in ()).throw(RuntimeError('injected'))
    assert torch.isfinite(step(m,f,2)).all()
    assert state(m).guard_latched
    m._reset_stateful_cache()
    assert not m._diffusers_hook.get_hook('spectrum_cache_denoiser').state_manager._state_cache
    for i in range(4):step(m,f,i)
    assert state(m).summary()['prediction_call_count']==1
    m.disable_cache()


def test_distilled_requires_primary_optin():
    m=tiny(); c=config(True)
    c.allow_flux3_distilled=False
    with pytest.raises(ValueError,match='requires explicit opt-in'):
        m.enable_cache(c)
