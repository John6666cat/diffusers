# Distilled research-only; no Model Hub downloads or GPU required.
import torch
import pytest
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


def inputs():
    torch.manual_seed(44)
    return dict(hidden_states=torch.randn(1, 4, 8),
                encoder_hidden_states=torch.randn(1, 3, 32),
                img_ids=torch.zeros(4, 4), txt_ids=torch.zeros(3, 4))


def config(predictions=(2,)):
    return SpectrumCacheConfig(num_inference_steps=4, warmup_steps=2,
        forecast_step_indices=predictions, degree=1, blend_w=0.35,
        coordinate_policy="runtime_horizon_normalized", allow_flux3_distilled=True)


def call(model, fields, step, context=True):
    kwargs = dict(fields)
    kwargs.update(timestep=torch.tensor([1.0-0.25*step]),
                  target_timestep=torch.tensor([1.0-0.25*(step+1)]))
    if context:
        with model.cache_context("distilled", step_index=step,
                                 num_inference_steps=4, timestep=step):
            return model(**kwargs).sample
    return model(**kwargs).sample


def state(model):
    hook=model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    return hook.state_manager._state_cache["distilled"]


@torch.no_grad()
def test_default_off_and_no_context_exact():
    model=tiny(); fields=inputs()
    with pytest.raises(ValueError, match="explicit opt-in"):
        model.enable_cache(SpectrumCacheConfig(num_inference_steps=4,
            coordinate_policy="runtime_horizon_normalized"))
    before=call(model,fields,0,False)
    model.enable_cache(config())
    assert torch.equal(before,call(model,fields,0,False))
    model.disable_cache()
    assert torch.equal(before,call(model,fields,0,False))


@torch.no_grad()
def test_distilled_target_step_output_head_exact_all_real():
    model=tiny(); fields=inputs()
    reference=[call(model,fields,i,False) for i in range(4)]
    model.enable_cache(config(()))
    actual=[call(model,fields,i) for i in range(4)]
    assert all(torch.equal(a,b) for a,b in zip(reference,actual))
    assert state(model).summary()["prediction_call_count"]==0
    model.disable_cache()


@torch.no_grad()
def test_single_lane_skip_blocks_and_target_modulation(monkeypatch):
    model=tiny(); fields=inputs(); calls=[]
    block=model.single_transformer_blocks[0]
    forward=block.forward
    def traced(*args,**kwargs):
        calls.append(1)
        return forward(*args,**kwargs)
    monkeypatch.setattr(block,"forward",traced)
    model.enable_cache(config())
    results=[call(model,fields,i) for i in range(4)]
    assert all(torch.isfinite(x).all() for x in results)
    assert len(calls)==3
    summary=state(model).summary()
    assert summary["prediction_call_count"]==1
    assert summary["full_call_count"]==3
    assert summary["logical_prediction_steps_used"]==[2]
    assert not summary["guard_latched"]
    assert set(model._diffusers_hook.get_hook("spectrum_cache_denoiser").state_manager._state_cache)=={"distilled"}
    model.disable_cache()


@torch.no_grad()
def test_prompt_change_falls_back():
    model=tiny(); fields=inputs(); model.enable_cache(config())
    call(model,fields,0)
    changed=dict(fields); changed["encoder_hidden_states"]=fields["encoder_hidden_states"].clone()
    for i in range(1,4):call(model,changed,i)
    assert state(model).guard_latched
    assert state(model).summary()["prediction_call_count"]==0
    model.disable_cache()


@torch.no_grad()
def test_image_conditioning_not_yet_approved():
    model=tiny(); fields=inputs(); model.enable_cache(config())
    ref=dict(fields,image_cond_hidden_states=torch.randn(1,2,8),
             image_cond_ids=torch.zeros(2,4))
    for i in range(4):assert torch.isfinite(call(model,ref,i)).all()
    assert state(model).guard_latched
    assert state(model).summary()["prediction_call_count"]==0
    model.disable_cache()


@torch.no_grad()
def test_prediction_exception_latches_and_recovers():
    model=tiny(); fields=inputs(); model.enable_cache(config())
    call(model,fields,0);call(model,fields,1)
    def fail():raise RuntimeError("injected")
    state(model).predict=fail
    assert torch.isfinite(call(model,fields,2)).all()
    assert state(model).guard_latched
    assert state(model).summary()["prediction_call_count"]==0
    model.disable_cache()


@torch.no_grad()
def test_reset_clears_history_after_interruption():
    model=tiny(); fields=inputs(); model.enable_cache(config())
    first=[call(model,fields,i) for i in range(4)]
    model._reset_stateful_cache()
    second=[call(model,fields,i) for i in range(4)]
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    assert state(model).summary()["prediction_call_count"]==1
    model.disable_cache()


def make_pipe():
    model=tiny()
    pipe=Flux3ModularPipeline(blocks=Flux3DistilledDenoiseStep())
    pipe.register_components(transformer=model, scheduler=FlowMatchEulerDiscreteScheduler())
    return pipe


def run_pipe(pipe, latents, fields):
    pipe.scheduler.set_timesteps(4, device="cpu")
    return pipe(latents=latents.clone(),denoiser_input_fields=fields,
        timesteps=pipe.scheduler.timesteps,num_inference_steps=4,output="latents")


@torch.no_grad()
def test_real_distilled_modular_loop_single_lane_and_repeat():
    torch.manual_seed(76)
    pipe=make_pipe();model=pipe.transformer
    latents=torch.randn(1,4,8)
    fields=dict(prompt_embeds=torch.randn(1,3,32),txt_ids=torch.zeros(3,4),img_ids=torch.zeros(4,4))
    baseline=run_pipe(pipe,latents,fields)
    model.enable_cache(config())
    candidate=run_pipe(pipe,latents,fields)
    summary=state(model).summary()
    assert torch.isfinite(candidate).all() and candidate.shape==baseline.shape
    assert summary["prediction_call_count"]==1
    assert summary["full_call_count"]==3
    assert not summary["guard_latched"]
    model.disable_cache()
    assert torch.equal(run_pipe(pipe,latents,fields),baseline)
