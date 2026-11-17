import math
from dataclasses import replace

import pytest
import torch

from diffusers import FluxTransformer2DModel, SpectrumCacheConfig
from diffusers.hooks.spectrum_cache import SpectrumSchedule, _SPECTRUM_DENOISER_HOOK


FULL_STEPS = (0, 1, 2, 3, 4, 6, 11, 15, 19, 24, 27, 30, 33, 37, 40, 44, 47)


def make_model(guidance_embeds=True):
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        guidance_embeds=guidance_embeds,
        axes_dims_rope=(4, 4, 8),
    ).eval()


def test_spectrum_flux1_dev_profile_is_fixed_to_distributed_17_full_steps():
    config = SpectrumCacheConfig.for_flux1_dev()
    forecasts = tuple(step for step in range(50) if step not in FULL_STEPS)

    assert config.num_inference_steps == 50
    assert (config.warmup_steps, config.window_size, config.flex_window) == (5, 2.0, 3.0)
    assert (config.degree, config.ridge_lambda, config.blend_w) == (4, 0.1, 0.5)
    assert (config.history_limit, config.predictor_backend) == (100, "dense")
    assert (config.coordinate_policy, config.coordinate_max) == ("legacy_fixed_max", 50.0)
    assert (config.tail_actual_steps, config.max_consecutive_forecast_steps) == (0, None)
    assert config.forecast_step_indices == forecasts
    schedule = SpectrumSchedule(config)
    assert tuple(step for step in range(50) if schedule.decide(step)) == FULL_STEPS
    assert len(forecasts) == 33

    assert SpectrumCacheConfig().forecast_step_indices is None
    for steps in (4, 20, 49, 51):
        with pytest.raises(ValueError, match="50 inference steps"):
            SpectrumCacheConfig.for_flux1_dev(steps)


def test_spectrum_flux1_dev_profile_rejects_non_guidance_flux_transformer():
    with pytest.raises(ValueError, match="guidance_embeds=True"):
        make_model(guidance_embeds=False).enable_cache(SpectrumCacheConfig.for_flux1_dev())


@torch.no_grad()
def test_spectrum_flux1_dev_context_forecasts_and_restores_original_forward():
    torch.manual_seed(47)
    model = make_model()
    latents = torch.randn(1, 4, 8)
    condition = {
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "pooled_projections": torch.randn(1, 16),
        "img_ids": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        "txt_ids": torch.zeros(3, 3),
        "guidance": torch.tensor([3.5]),
        "joint_attention_kwargs": {},
        "return_dict": False,
    }

    def call(step, *, with_context):
        inputs = dict(condition)
        inputs["hidden_states"] = latents + step / 300
        inputs["timestep"] = torch.tensor([1.0 - step / 50])
        if not with_context:
            return model(**inputs)[0]
        with model.cache_context("cond", step_index=step, num_inference_steps=50):
            return model(**inputs)[0]

    assert not model.is_cache_enabled
    baseline = call(0, with_context=False)
    model.enable_cache(SpectrumCacheConfig.for_flux1_dev())
    try:
        assert torch.equal(call(0, with_context=False), baseline)

        for step in range(50):
            assert torch.isfinite(call(step, with_context=True)).all()

        state = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK).state_manager._state_cache["cond"]
        summary = state.summary()
        assert summary["compute_steps"] == list(FULL_STEPS)
        assert summary["full_call_count"] == 17
        assert summary["prediction_call_count"] == 33
        assert not summary["guard_latched"]
        assert summary["predict_failures"] == []
    finally:
        model.disable_cache()

    assert not model.is_cache_enabled
    assert torch.equal(call(0, with_context=False), baseline)


@pytest.mark.parametrize("steps", (40, 49, 50, 51, 60))
def test_spectrum_flux1_variable_n_schedule_is_explicit_and_bounded(steps):
    config = SpectrumCacheConfig.for_flux1_dev_variable(steps)
    old = SpectrumCacheConfig.for_flux1_dev()
    if steps == 50:
        assert config == old
    else:
        assert config.allow_flux1_variable_n is True
        assert config.coordinate_policy == "runtime_horizon_normalized"
        assert config.tail_actual_steps == 2
        assert config.denoiser_call_topology == "single"
        assert (config.warmup_steps, config.history_limit, config.degree) == (5, 100, 4)
        assert (config.window_size, config.flex_window, config.ridge_lambda, config.blend_w) == (2, 3, .1, .5)

    schedule = SpectrumSchedule(config)
    real = [step for step in range(steps) if schedule.decide(step)]
    forecasts = [step for step in range(steps) if step not in real]
    assert len(real) + len(forecasts) == steps
    assert tuple(forecasts) == config.forecast_step_indices
    if steps == 50:
        assert tuple(real) == FULL_STEPS
    else:
        assert set(range(5)).issubset(real)
        assert set(range(steps - 2, steps)).issubset(real)
        assert len(real) >= math.ceil(17 * steps / 50)
        assert all(right - left - 1 <= 4 for left, right in zip(real, real[1:]))


@pytest.mark.parametrize("steps", (True, False, 0, 39, 61, 50.0, "50"))
def test_spectrum_flux1_variable_n_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError, match="integer in"):
        SpectrumCacheConfig.for_flux1_dev_variable(steps)


def test_spectrum_flux1_variable_n_rejects_unqualified_backends_and_bypass_configs():
    with pytest.raises(ValueError, match="dense or fp32_weighted"):
        SpectrumCacheConfig.for_flux1_dev_variable(49, predictor_backend="chunked")

    model = make_model()
    with pytest.raises(ValueError, match="50 inference steps and legacy 50-coordinate mapping"):
        model.enable_cache(SpectrumCacheConfig(num_inference_steps=49))

    config = SpectrumCacheConfig.for_flux1_dev_variable(49)
    for altered in (
        replace(config, allow_flux1_variable_n=False),
        replace(config, coordinate_policy="legacy_fixed_max"),
        replace(config, forecast_step_indices=config.forecast_step_indices[1:]),
        replace(config, history_limit=5),
        replace(config, predictor_backend="chunked"),
    ):
        message = "variable-N research opt-in" if altered.allow_flux1_variable_n else "50 inference steps"
        with pytest.raises(ValueError, match=message):
            model.enable_cache(altered)

    model.enable_cache(config)
    assert model.is_cache_enabled
    model.disable_cache()
    assert not model.is_cache_enabled


@pytest.mark.parametrize("steps", (49, 51))
@torch.no_grad()
def test_spectrum_flux1_variable_n_native_context_and_cleanup(steps):
    torch.manual_seed(47)
    model = make_model()
    latents = torch.randn(1, 4, 8)
    condition = {
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "pooled_projections": torch.randn(1, 16),
        "img_ids": torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.]]),
        "txt_ids": torch.zeros(3, 3),
        "guidance": torch.tensor([3.5]),
        "joint_attention_kwargs": {},
        "return_dict": False,
    }

    def call(step, *, label=None, reported_n=None):
        kwargs = dict(condition)
        kwargs["hidden_states"] = latents + step / 300
        kwargs["timestep"] = torch.tensor([1.0 - step / steps])
        if label is None:
            return model(**kwargs)[0]
        with model.cache_context(label, step_index=step, num_inference_steps=reported_n or steps):
            return model(**kwargs)[0]

    baseline = call(0)
    model.enable_cache(SpectrumCacheConfig.for_flux1_dev_variable(steps))
    try:
        assert torch.equal(call(0), baseline)
        for step in range(steps):
            assert torch.isfinite(call(step, label="cond")).all()
        denoiser = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)
        state = denoiser.state_manager._state_cache["cond"]
        result = state.summary()
        assert not result["guard_latched"]
        assert result["predict_failures"] == []
        assert result["full_call_count"] + result["prediction_call_count"] == steps
        assert result["forecast_steps"] == list(state.config.forecast_step_indices)

        # Native-N mismatch must latch full-compute only for this lane.
        call(0, label="uncond", reported_n=steps + 1)
        call(1, label="uncond")
        uncond = denoiser.state_manager._state_cache["uncond"]
        assert uncond.guard_latched
        assert uncond.summary()["prediction_call_count"] == 0
        assert not state.guard_latched
        # Beginning a fresh run at step zero clears any stale lane state.
        call(0, label="uncond")
        assert not uncond.guard_latched
    finally:
        model.disable_cache()
    assert torch.equal(call(0), baseline)


def test_spectrum_flux1_variable_n_fp32_is_separate_opt_in_arm():
    base = SpectrumCacheConfig.for_flux1_dev_variable(51)
    fp32 = SpectrumCacheConfig.for_flux1_dev_variable(51, predictor_backend="fp32_weighted")
    assert base.predictor_backend == "dense"
    assert fp32.predictor_backend == "fp32_weighted"
    assert fp32.forecast_step_indices == base.forecast_step_indices
    assert SpectrumCacheConfig.for_flux1_dev_variable(50, predictor_backend="fp32_weighted") == replace(
        SpectrumCacheConfig.for_flux1_dev(), predictor_backend="fp32_weighted"
    )
