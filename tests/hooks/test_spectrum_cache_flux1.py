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
