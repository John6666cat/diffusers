"""FLUX.3 Base: actual image-conditioned core denoise path, tiny CPU-only.

Exercises pre-encoded reference latents, not VAE/Qwen/bbox or real weights.
"""
import functools
import pytest
import torch

from diffusers.guiders import ClassifierFreeGuidance
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import Flux3ImageConditionedCoreDenoiseStep
from diffusers.modular_pipelines.flux3.modular_pipeline import Flux3ModularPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def _pipe(*, allow, enabled=True):
    torch.manual_seed(82)
    model = Flux3Transformer2DModel(
        in_channels=8, cond_in_channels=8, num_layers=1, num_single_layers=1,
        num_attention_heads=2, attention_head_dim=16, joint_attention_dim=32,
        timestep_channels=32, axes_dims_rope=(4, 4, 4, 4), flow_map=False,
    ).eval()
    pipe = Flux3ModularPipeline(blocks=Flux3ImageConditionedCoreDenoiseStep())
    pipe.register_components(transformer=model, guider=ClassifierFreeGuidance(guidance_scale=3.5),
                             scheduler=FlowMatchEulerDiscreteScheduler())
    if enabled:
        model.enable_cache(SpectrumCacheConfig(num_inference_steps=4, warmup_steps=2,
                           forecast_step_indices=(2,), degree=1,
                           coordinate_policy="runtime_horizon_normalized",
                           allow_flux3_image_conditioned=allow))
    return pipe


def _inputs(*, refs=1, seed=70):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return dict(
            prompt_embeds=torch.randn(1, 3, 32),
            negative_prompt_embeds=torch.randn(1, 3, 32),
            image_cond_latents=[torch.randn(1, 8, 2, 2) for _ in range(refs)],
            latents=torch.randn(1, 8, 2, 2),
            height=32, width=32, num_inference_steps=4,
            output="latents",
        )


def _run(pipe, inputs):
    return pipe(**inputs)


def _states(pipe):
    hook = pipe.transformer._diffusers_hook.get_hook("spectrum_cache_denoiser")
    return hook.state_manager._state_cache


@pytest.mark.parametrize("refs", [1, 2])
@torch.no_grad()
def test_actual_image_conditioned_core_default_bypass_matches_no_hook(refs):
    data = _inputs(refs=refs)
    default = _pipe(allow=False)
    off = _pipe(allow=False, enabled=False)
    expected = _run(off, data)
    actual = _run(default, data)
    assert torch.equal(actual, expected)
    assert set(_states(default)) == {"pred_cond", "pred_uncond"}
    for state in _states(default).values():
        assert state.guard_latched
        assert state.summary()["prediction_call_count"] == 0
        assert state.summary()["full_call_count"] == 4


@pytest.mark.parametrize("refs", [1, 2])
@torch.no_grad()
def test_actual_core_optin_forecast_and_reference_forwarded(refs, monkeypatch):
    pipe = _pipe(allow=True)
    data = _inputs(refs=refs)
    original = pipe.transformer.forward
    ref_ids = []
    ref_tokens = []

    @functools.wraps(original)
    def recorded(*args, **kwargs):
        ref_ids.append(kwargs["image_cond_ids"])
        ref_tokens.append(kwargs["image_cond_hidden_states"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipe.transformer, "forward", recorded)
    result = _run(pipe, data)
    assert result.shape == (1, 8, 2, 2) and torch.isfinite(result).all()
    assert len(ref_ids) == 8
    assert all(x is ref_ids[0] for x in ref_ids)
    assert all(x is ref_tokens[0] for x in ref_tokens)
    assert ref_ids[0].shape == (4 * refs, 4)
    assert ref_tokens[0].shape == (1, 4 * refs, 8)
    for j in range(refs):
        assert torch.equal(ref_ids[0][4*j:4*(j+1), 3], torch.full((4,), -(1+j)))
    assert set(_states(pipe)) == {"pred_cond", "pred_uncond"}
    for state in _states(pipe).values():
        assert not state.guard_latched
        assert state.summary()["prediction_call_count"] == 1
        assert state.summary()["full_call_count"] == 3


@pytest.mark.parametrize("refs", [1, 2])
@torch.no_grad()
def test_second_generation_different_references_no_history_leak(refs):
    pipe = _pipe(allow=True)
    first_inputs = _inputs(refs=refs, seed=70)
    second_inputs = _inputs(refs=refs, seed=71)
    first = _run(pipe, first_inputs)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())
    second = _run(pipe, second_inputs)
    fresh = _run(_pipe(allow=True), second_inputs)
    assert torch.equal(second, fresh)
    assert not torch.equal(first, second)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())
    assert all(s.summary()["full_call_count"] == 3 for s in _states(pipe).values())


@pytest.mark.parametrize("refs", [1, 2])
@torch.no_grad()
def test_actual_core_failure_then_recovery(refs, monkeypatch):
    pipe = _pipe(allow=True)
    inputs = _inputs(refs=refs)
    baseline = _run(pipe, inputs)
    layer = pipe.transformer.single_transformer_blocks[0]
    original = layer.forward
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected image-condition interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(layer, "forward", interrupted)
    with pytest.raises(RuntimeError, match="injected image-condition interruption"):
        _run(pipe, inputs)
    monkeypatch.setattr(layer, "forward", original)
    recovered = _run(pipe, inputs)
    assert torch.equal(baseline, recovered)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())


@torch.no_grad()
def test_mutated_reference_latents_are_rebuilt_next_call():
    pipe = _pipe(allow=True)
    inputs = _inputs(refs=2)
    _run(pipe, inputs)
    inputs["image_cond_latents"][0].add_(1)
    second = _run(pipe, inputs)
    expected = _run(_pipe(allow=True), inputs)
    assert torch.equal(second, expected)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())
