# coding=utf-8
import torch

from diffusers import SpectrumCacheConfig, UNet2DConditionModel
from diffusers.hooks.spectrum_cache import _SPECTRUM_DENOISER_HOOK, _SPECTRUM_UNET_FEATURE_HOOK


def make_tiny_unet():
    model = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=4,
    )
    model.eval()
    return model


def make_inputs():
    generator = torch.Generator().manual_seed(0)
    return {
        "sample": torch.randn((2, 4, 16, 16), generator=generator),
        "timestep": torch.tensor(10),
        "encoder_hidden_states": torch.randn((2, 5, 16), generator=generator),
    }


@torch.no_grad()
def test_spectrum_unet_enable_forecast_disable():
    model = make_tiny_unet()
    inputs = make_inputs()
    config = SpectrumCacheConfig(num_inference_steps=2, warmup_steps=1, degree=1)

    model.enable_cache(config)
    assert model.is_cache_enabled
    assert model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK) is not None
    assert model.conv_norm_out._diffusers_hook.get_hook(_SPECTRUM_UNET_FEATURE_HOOK) is not None

    with model.cache_context("unet"):
        first = model(**inputs).sample
        changed = dict(inputs)
        changed["sample"] = inputs["sample"] + 0.2
        forecast = model(**changed).sample

    state = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK).state_manager._state_cache["unet"]
    assert state.compute_steps == [0]
    assert state.forecast_steps == [1]
    assert state.forecaster.features[0].shape[1:] == (16, 16, 16)
    assert torch.isfinite(first).all()
    assert torch.isfinite(forecast).all()

    model.disable_cache()
    assert not model.is_cache_enabled
    assert model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK) is None
    assert model.conv_norm_out._diffusers_hook.get_hook(_SPECTRUM_UNET_FEATURE_HOOK) is None


@torch.no_grad()
def test_spectrum_unet_control_residual_path_fails_closed_without_advancing_schedule():
    model = make_tiny_unet()
    inputs = make_inputs()
    config = SpectrumCacheConfig(num_inference_steps=2, warmup_steps=1, degree=1)
    model.enable_cache(config)

    # A non-None adapter/control residual argument is enough to exercise the root fail-closed gate.
    # We use an empty tuple so the base UNet remains valid and performs a normal full forward.
    controlled = dict(inputs)
    controlled["down_intrablock_additional_residuals"] = ()

    with model.cache_context("control"):
        cached_model = model(**controlled).sample

    state = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK).state_manager._state_cache["control"]
    assert state.compute_steps == []
    assert state.forecast_steps == []
    assert state.forecaster.steps == []
    assert state.forecaster.features == []

    model.disable_cache()
    uncached_model = model(**controlled).sample
    assert torch.allclose(cached_model, uncached_model, atol=1e-5)


@torch.no_grad()
def test_spectrum_unet_conditioning_opt_ins_are_independent_narrow_and_default_off():
    model = make_tiny_unet()

    classic_pair = {
        "down_block_additional_residuals": (),
        "mid_block_additional_residual": torch.zeros(1),
    }
    t2i_kwargs = {"down_intrablock_additional_residuals": ()}
    ip_kwargs = {"added_cond_kwargs": {"image_embeds": torch.zeros(1)}}

    default_config = SpectrumCacheConfig(num_inference_steps=2, warmup_steps=1, degree=1)
    assert default_config.allow_unet_controlnet_residuals is False
    assert default_config.allow_unet_t2i_adapter_residuals is False
    assert default_config.allow_unet_ip_adapter_image_embeds is False
    model.enable_cache(default_config)
    root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)

    with model.cache_context("default-controlnet-gate"):
        assert root_hook._should_bypass((), classic_pair)
    model._reset_stateful_cache()
    with model.cache_context("default-t2i-gate"):
        assert root_hook._should_bypass((), t2i_kwargs)
    model._reset_stateful_cache()
    with model.cache_context("default-ip-gate"):
        assert root_hook._should_bypass((), ip_kwargs)

    model.disable_cache()

    opt_in_config = SpectrumCacheConfig(
        num_inference_steps=2,
        warmup_steps=1,
        degree=1,
        allow_unet_controlnet_residuals=True,
        allow_unet_t2i_adapter_residuals=True,
        allow_unet_ip_adapter_image_embeds=True,
    )
    model.enable_cache(opt_in_config)
    root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)

    with model.cache_context("opt-in-controlnet-gate"):
        assert not root_hook._should_bypass((), classic_pair)
    model._reset_stateful_cache()

    with model.cache_context("opt-in-t2i-gate"):
        assert not root_hook._should_bypass((), t2i_kwargs)
        # T2I residuals may disappear later in the same trajectory; an allowed
        # T2I run must remain eligible rather than requiring a residual every call.
        assert not root_hook._should_bypass((), {})
    model._reset_stateful_cache()

    with model.cache_context("opt-in-ip-gate"):
        assert not root_hook._should_bypass((), ip_kwargs)
    model._reset_stateful_cache()

    with model.cache_context("partial-controlnet-remains-closed"):
        assert root_hook._should_bypass((), {"down_block_additional_residuals": ()})
    model._reset_stateful_cache()

    with model.cache_context("mixed-conditioning-remains-closed"):
        mixed = dict(classic_pair)
        mixed["down_intrablock_additional_residuals"] = ()
        assert root_hook._should_bypass((), mixed)
    model._reset_stateful_cache()

    with model.cache_context("nondefault-peft-remains-closed"):
        peft_kwargs = {"cross_attention_kwargs": {"scale": 0.5}}
        assert root_hook._should_bypass((), peft_kwargs)


@torch.no_grad()
def test_spectrum_unet_unsupported_conditioning_latches_failclosed_for_rest_of_run_and_resets():
    model = make_tiny_unet()
    config = SpectrumCacheConfig(num_inference_steps=4, warmup_steps=1, degree=1)
    model.enable_cache(config)
    root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)

    with model.cache_context("sticky-unsupported"):
        assert root_hook._should_bypass((), {"down_intrablock_additional_residuals": ()})
        state = root_hook.state_manager.get_state()
        assert state.bypass_latched is True
        assert root_hook._should_bypass((), {})

    model._reset_stateful_cache()
    # cache_context clears the current StateManager context on exit. Re-enter
    # the same context name after reset to prove the latched state was cleared.
    with model.cache_context("sticky-unsupported"):
        state = root_hook.state_manager.get_state()
        assert state.bypass_latched is False
        assert not root_hook._should_bypass((), {})



@torch.no_grad()
def test_spectrum_unet_dynamic_sample_shape_change_latches_failclosed():
    model = make_tiny_unet()
    config = SpectrumCacheConfig(num_inference_steps=4, warmup_steps=1, degree=1)
    model.enable_cache(config)
    root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)

    with model.cache_context("dynamic-shape"):
        first = {"sample": torch.zeros((2, 4, 16, 16))}
        second = {"sample": torch.zeros((1, 4, 16, 16))}
        assert not root_hook._should_bypass((), first)
        assert root_hook._should_bypass((), second)
        state = root_hook.state_manager.get_state()
        assert state.bypass_latched is True
        assert root_hook._should_bypass((), first)

    model.disable_cache()
