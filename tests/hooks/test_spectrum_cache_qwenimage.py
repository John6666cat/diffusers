import torch

from diffusers import QwenImageTransformer2DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


QWEN_IMAGE_2512_CONFIG_SIGNATURE = {
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 60,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "guidance_embeds": False,
    "axes_dims_rope": (16, 56, 56),
    "zero_cond_t": False,
}


def make_model(*, zero_cond_t=False):
    model = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=60,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(4, 2, 2),
        zero_cond_t=zero_cond_t,
    ).eval()
    # Tiny mechanics graph; the adapter itself remains fail-closed to the exact qualified 2512 architecture.
    signature = dict(QWEN_IMAGE_2512_CONFIG_SIGNATURE)
    signature["zero_cond_t"] = zero_cond_t
    model.register_to_config(**signature)
    return model


def make_unqualified_model():
    return QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(4, 2, 2),
        zero_cond_t=False,
    ).eval()


def make_inputs(step=0, *, attention_kwargs=None, additional_t_cond=None):
    return {
        "hidden_states": torch.randn(1, 16, 4),
        "encoder_hidden_states": torch.randn(1, 4, 8),
        "encoder_hidden_states_mask": None,
        "timestep": torch.tensor([1.0 - step / 20.0]),
        "img_shapes": [[(1, 4, 4)]],
        "guidance": None,
        "attention_kwargs": attention_kwargs,
        "controlnet_block_samples": None,
        "additional_t_cond": additional_t_cond,
        "return_dict": True,
    }


def make_config():
    return SpectrumCacheConfig(
        num_inference_steps=20,
        warmup_steps=6,
        window_size=2.0,
        flex_window=0.25,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=20.0,
        tail_actual_steps=4,
    )


def assert_finite_sample(output):
    assert torch.isfinite(output.sample).all()


@torch.no_grad()
def test_spectrum_qwenimage_2512_exact_selected6_schedule_and_block_accounting():
    model = make_model()
    calls = [0 for _ in model.transformer_blocks]
    hooks = []
    for i, block in enumerate(model.transformer_blocks):
        hooks.append(block.attn.register_forward_hook(lambda *args, i=i: calls.__setitem__(i, calls[i] + 1)))
    try:
        model.enable_cache(make_config())
        for step in range(20):
            with model.cache_context("cond"):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 18, 19]
        assert summary["forecast_steps"] == [6, 8, 10, 12, 14, 15]
        assert summary["prediction_call_count"] == 6
        assert summary["full_call_count"] == 14
        assert not summary["guard_latched"]
        assert calls == [14] * 60
        assert sum(calls) == 840

        model.disable_cache()
        assert not model.is_cache_enabled
        assert model._diffusers_hook.get_hook("spectrum_cache_denoiser") is None
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_spectrum_qwenimage_2512_contexts_partition_cond_uncond_history():
    model = make_model()
    model.enable_cache(make_config())
    for step in range(20):
        for label in ("cond", "uncond"):
            with model.cache_context(label):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond", "uncond"}
    for label in ("cond", "uncond"):
        summary = root.state_manager._state_cache[label].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 18, 19]
        assert summary["forecast_steps"] == [6, 8, 10, 12, 14, 15]
        assert summary["prediction_call_count"] == 6
        assert summary["full_call_count"] == 14
        assert not summary["guard_latched"]


@torch.no_grad()
def test_spectrum_qwenimage_2512_shape_change_latches_sticky_failclosed():
    model = make_model()
    model.enable_cache(make_config())

    with model.cache_context("cond"):
        output = model(**make_inputs(0))
    assert_finite_sample(output)

    changed = make_inputs(1)
    changed["encoder_hidden_states"] = torch.randn(1, 5, 8)
    with model.cache_context("cond"):
        output = model(**changed)
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 2
    assert any("signature changed" in item["reason"] for item in summary["guard_reasons"])


@torch.no_grad()
def test_spectrum_qwenimage_rejects_edit_zero_cond_t_at_enable():
    model = make_model(zero_cond_t=True)
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "Qwen-Image-2512 standard T2I transformer architecture" in str(error)
        assert "zero_cond_t" in str(error)
    else:
        raise AssertionError("Expected Qwen-Image Edit-style zero_cond_t architecture to be rejected.")


def test_spectrum_qwenimage_autograd_latches_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    with model.cache_context("cond"):
        output = model(**make_inputs(0))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 1
    assert any("autograd/training" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_qwenimage_rejects_unqualified_architecture_at_enable():
    model = make_unqualified_model()
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "Qwen-Image-2512 standard T2I transformer architecture" in str(error)
    else:
        raise AssertionError("Expected unqualified Qwen-Image architecture to be rejected.")
