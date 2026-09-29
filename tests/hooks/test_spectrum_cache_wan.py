import torch

from diffusers import WanTransformer3DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


WAN21_13B_CONFIG_SIGNATURE = {
    "num_layers": 30,
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "ffn_dim": 8960,
    "text_dim": 4096,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": (1, 2, 2),
}


def make_model(*, image_dim=None):
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=256,
        ffn_dim=32,
        num_layers=3,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        image_dim=image_dim,
        rope_max_seq_len=32,
    ).eval()
    # Unit tests exercise the adapter mechanics on a tiny module graph. The runtime
    # adapter itself remains fail-closed to the exact qualified 1.3B config signature.
    model.register_to_config(**WAN21_13B_CONFIG_SIGNATURE)
    return model


def make_unqualified_model():
    return WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=256,
        ffn_dim=32,
        num_layers=3,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=32,
    ).eval()


def make_inputs(*, frames=2):
    return {
        "hidden_states": torch.randn(1, 4, frames, 8, 8),
        "encoder_hidden_states": torch.randn(1, 12, 16),
        "timestep": torch.tensor([1.0]),
        "return_dict": True,
    }


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
def test_spectrum_wan21_13b_standard_context_skips_block_body_and_disables_cleanly():
    model = make_model()
    cfg = make_config()
    block_calls = {"head_attn": 0, "tail_ffn": 0}
    h1 = model.blocks[0].attn1.register_forward_hook(
        lambda *args: block_calls.__setitem__("head_attn", block_calls["head_attn"] + 1)
    )
    h2 = model.blocks[-1].ffn.register_forward_hook(
        lambda *args: block_calls.__setitem__("tail_ffn", block_calls["tail_ffn"] + 1)
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
        assert block_calls == {"head_attn": 4, "tail_ffn": 4}

        model.disable_cache()
        assert not model.is_cache_enabled
        assert model._diffusers_hook.get_hook("spectrum_cache_denoiser") is None
    finally:
        h1.remove()
        h2.remove()


@torch.no_grad()
def test_spectrum_wan21_13b_cache_contexts_partition_cond_uncond_history():
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
def test_spectrum_wan21_13b_attention_kwargs_latch_sticky_failclosed():
    model = make_model()
    model.enable_cache(make_config())

    for step in range(6):
        inputs = make_inputs()
        inputs["attention_kwargs"] = {"scale": 1.0} if step == 0 else None
        with model.cache_context("cond"):
            output = model(**inputs).sample
        assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["guard_latched_at"] == 0
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 6
    assert any("attention_kwargs" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_wan21_13b_autograd_latches_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    with model.cache_context("cond"):
        output = model(**make_inputs()).sample
    assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 1
    assert any("autograd" in item["reason"] for item in summary["guard_reasons"])


@torch.no_grad()
def test_spectrum_wan21_13b_training_mode_latches_failclosed_even_without_autograd():
    model = make_model().train()
    model.enable_cache(make_config())
    with model.cache_context("cond"):
        output = model(**make_inputs()).sample
    assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 1
    assert any("autograd/training" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_wan_rejects_unqualified_architecture_at_enable():
    model = make_unqualified_model()
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "Wan2.1 T2V 1.3B transformer architecture" in str(error)
    else:
        raise AssertionError("Expected non-1.3B Wan architecture to be rejected.")


def test_spectrum_wan_rejects_image_conditioned_architecture_at_enable():
    model = make_model(image_dim=8)
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "text-only Wan2.1 T2V 1.3B route" in str(error)
    else:
        raise AssertionError("Expected image-conditioned Wan architecture to be rejected.")



def test_spectrum_wan21_13b_variable_factory_preserves_measured_schedule_geometry_and_bounds():
    expected = {
        40: (9, 8, (9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30)),
        45: (10, 9, (10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34)),
        50: (11, 10, (11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38)),
        55: (12, 11, (12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42)),
        60: (13, 12, (13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 46)),
    }
    for steps, (warmup, tail, forecast) in expected.items():
        config = SpectrumCacheConfig.for_wan21_13b_variable(steps)
        assert config.num_inference_steps == steps
        assert config.warmup_steps == warmup
        assert config.tail_actual_steps == tail
        assert config.forecast_step_indices == forecast
        assert config.coordinate_policy == "runtime_horizon_normalized"
        assert config.coordinate_max == 50.0
        assert config.history_limit == 100
        assert config.predictor_backend == "dense"

    for steps in (39, 61):
        try:
            SpectrumCacheConfig.for_wan21_13b_variable(steps)
        except ValueError as error:
            assert "40 <= num_inference_steps <= 60" in str(error)
        else:
            raise AssertionError(f"expected bounded-range rejection for {steps}")


@torch.no_grad()
def test_spectrum_wan21_13b_variable_factory_uses_native_cond_uncond_runtime_provenance():
    num_inference_steps = 40
    model = make_model()
    config = SpectrumCacheConfig.for_wan21_13b_variable(num_inference_steps)
    model.enable_cache(config)

    for step in range(num_inference_steps):
        for label in ("cond", "uncond"):
            inputs = make_inputs()
            inputs["timestep"] = torch.tensor([1.0 - step / num_inference_steps])
            with model.cache_context(
                label,
                step_index=step,
                num_inference_steps=num_inference_steps,
            ):
                output = model(**inputs).sample
            assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond", "uncond"}
    for label in ("cond", "uncond"):
        summary = root.state_manager._state_cache[label].summary()
        coordinate = summary["coordinate"]
        assert coordinate["coordinate_policy"] == "runtime_horizon_normalized"
        assert coordinate["provenance_source"] == "cache_context"
        assert coordinate["runtime_num_inference_steps"] == num_inference_steps
        assert coordinate["last_logical_step_index"] == num_inference_steps - 1
        assert coordinate["failure_latched"] is False
        assert summary["guard_latched"] is False
        assert summary["prediction_failure_latched"] is False
        assert summary["prediction_call_count"] == len(config.forecast_step_indices)
