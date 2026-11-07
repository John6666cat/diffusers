
import torch

from diffusers import Flux2Transformer2DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule


def test_spectrum_flux2_klein_base4b_p35_preset_matches_owner_gpu_profile():
    cfg = SpectrumCacheConfig.for_flux2_klein_base4b()
    expected_forecasts = (
        6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
        22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36,
        37, 38, 39, 40, 42, 43, 44, 45,
    )
    assert cfg.num_inference_steps == 50
    assert cfg.warmup_steps == 0
    assert (cfg.window_size, cfg.flex_window) == (2.0, 0.0)
    assert (cfg.degree, cfg.ridge_lambda, cfg.blend_w, cfg.history_limit) == (2, 0.1, 0.5, 8)
    assert (cfg.coordinate_policy, cfg.coordinate_max, cfg.tail_actual_steps) == (
        "legacy_fixed_max", 50.0, 0,
    )
    assert cfg.forecast_step_indices == expected_forecasts
    schedule = SpectrumSchedule(cfg)
    assert tuple(step for step in range(50) if not schedule.decide(step)) == expected_forecasts
    assert len(expected_forecasts) == 35
    assert sum(step not in expected_forecasts for step in range(50)) == 15


def test_spectrum_flux2_klein_base4b_p35_is_explicit_and_50_step_only():
    assert SpectrumCacheConfig().forecast_step_indices is None
    for steps in (4, 20, 49, 51):
        try:
            SpectrumCacheConfig.for_flux2_klein_base4b(steps)
        except ValueError as error:
            assert "50 inference steps" in str(error)
        else:
            raise AssertionError(f"Unexpected P35 preset approval at {steps} steps")


def test_spectrum_flux2_klein_base4b_p35_does_not_enable_distilled_route():
    model = make_model(guidance_embeds=True)
    try:
        model.enable_cache(SpectrumCacheConfig.for_flux2_klein_base4b())
    except ValueError as error:
        assert "non-distilled Klein Base route" in str(error)
    else:
        raise AssertionError("P35 preset must not override the existing distilled-model guard")


def make_model(guidance_embeds=False):
    return Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        timestep_guidance_channels=16,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=guidance_embeds,
    ).eval()


def make_inputs(image_tokens=4):
    return dict(
        hidden_states=torch.randn(1, image_tokens, 4),
        encoder_hidden_states=torch.randn(1, 3, 12),
        timestep=torch.tensor([0.5]),
        img_ids=torch.zeros(image_tokens, 4),
        txt_ids=torch.zeros(3, 4),
        guidance=None,
        return_dict=True,
    )


def make_config():
    return SpectrumCacheConfig(
        num_inference_steps=6,
        warmup_steps=2,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=20,
        coordinate_max=50.0,
        tail_actual_steps=1,
    )


@torch.no_grad()
def test_spectrum_flux2_standard_context_skips_root_body_and_disables_cleanly():
    model = make_model()
    cfg = make_config()

    block_calls = {"double": 0, "single": 0}
    h1 = model.transformer_blocks[0].register_forward_hook(
        lambda *args: block_calls.__setitem__("double", block_calls["double"] + 1)
    )
    h2 = model.single_transformer_blocks[0].register_forward_hook(
        lambda *args: block_calls.__setitem__("single", block_calls["single"] + 1)
    )
    try:
        model.enable_cache(cfg)
        for step in range(6):
            inputs = make_inputs()
            inputs["timestep"] = torch.tensor([1.0 - step / 6.0])
            with model.cache_context("cond"):
                output = model(**inputs).sample
            assert torch.isfinite(output).all()

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        state = root.state_manager._state_cache["cond"]
        summary = state.summary()

        assert summary["compute_steps"] == [0, 1, 3, 5]
        assert summary["forecast_steps"] == [2, 4]
        assert summary["prediction_call_count"] == 2
        assert summary["full_call_count"] == 4
        assert not summary["guard_latched"]
        assert summary["predict_failures"] == []
        assert block_calls == {"double": 4, "single": 4}

        model.disable_cache()
        assert not model.is_cache_enabled
        assert model._diffusers_hook.get_hook("spectrum_cache_denoiser") is None
    finally:
        h1.remove()
        h2.remove()


@torch.no_grad()
def test_spectrum_flux2_cache_contexts_partition_cond_uncond_history():
    model = make_model()
    model.enable_cache(make_config())
    for step in range(6):
        for label in ("cond", "uncond"):
            inputs = make_inputs()
            inputs["timestep"] = torch.tensor([1.0 - step / 6.0])
            with model.cache_context(label):
                output = model(**inputs).sample
            assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond", "uncond"}
    for label in ("cond", "uncond"):
        summary = root.state_manager._state_cache[label].summary()
        assert summary["prediction_call_count"] == 2
        assert summary["compute_steps"] == [0, 1, 3, 5]
        assert summary["forecast_steps"] == [2, 4]
        assert not summary["guard_latched"]


@torch.no_grad()
def test_spectrum_flux2_signature_change_latches_sticky_failclosed():
    model = make_model()
    model.enable_cache(make_config())

    for step in range(2):
        inputs = make_inputs(4)
        with model.cache_context("cond"):
            model(**inputs)

    changed = make_inputs(5)
    with model.cache_context("cond"):
        model(**changed)

    for _ in range(3):
        inputs = make_inputs(5)
        with model.cache_context("cond"):
            model(**inputs)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["guard_latched_at"] == 2
    assert any("signature changed" in x["reason"] for x in summary["guard_reasons"])
    assert summary["prediction_call_count"] == 0


@torch.no_grad()
def test_spectrum_flux2_kv_route_latches_failclosed_without_forecast():
    model = make_model()
    model.enable_cache(make_config())

    # Use num_ref_tokens alone to exercise the semantic guard while leaving the source forward valid.
    for _ in range(6):
        inputs = make_inputs()
        inputs["num_ref_tokens"] = 1
        with model.cache_context("cond"):
            model(**inputs)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["guard_latched_at"] == 0
    assert summary["prediction_call_count"] == 0
    assert any("KV/reference-token route" in x["reason"] for x in summary["guard_reasons"])


def test_spectrum_flux2_rejects_guidance_distilled_model_at_enable():
    model = make_model(guidance_embeds=True)
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "non-distilled Klein Base route" in str(error)
    else:
        raise AssertionError("Expected guidance-distilled Flux2 model to be rejected.")
