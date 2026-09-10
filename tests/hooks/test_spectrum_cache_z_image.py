import torch

from diffusers import ZImageTransformer2DModel
from diffusers.hooks._helpers import TransformerBlockRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


ZIMAGE_TURBO_CONFIG_SIGNATURE = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": 16,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "cap_feat_dim": 2560,
    "siglip_feat_dim": None,
}


def make_model():
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=32,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=16,
        siglip_feat_dim=None,
        axes_dims=[4, 2, 2],
        axes_lens=[1024, 512, 512],
    ).eval()
    # Tiny mechanics graph; adapter itself is fail-closed to the exact Turbo config.
    model.register_to_config(**ZIMAGE_TURBO_CONFIG_SIGNATURE)
    return model


def make_unqualified_model():
    return ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=32,
        n_layers=3,
        n_refiner_layers=2,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=16,
        siglip_feat_dim=None,
        axes_dims=[4, 2, 2],
        axes_lens=[1024, 512, 512],
    ).eval()


def make_inputs(step=0, *, controlnet_block_samples=None):
    return {
        "x": [torch.randn(4, 1, 8, 8)],
        "t": torch.tensor([1.0 - step / 8.0]),
        "cap_feats": [torch.randn(12, 16)],
        "return_dict": True,
        "controlnet_block_samples": controlnet_block_samples,
        "siglip_feats": None,
        "image_noise_mask": None,
        "patch_size": 2,
        "f_patch_size": 1,
    }


def make_config():
    return SpectrumCacheConfig(
        num_inference_steps=8,
        warmup_steps=4,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=20,
        coordinate_max=8.0,
        tail_actual_steps=0,
    )


def assert_finite_sample(output):
    assert isinstance(output.sample, list)
    assert output.sample
    assert all(torch.isfinite(item).all() for item in output.sample)


def test_zimage_block_registry_uses_x_as_hidden_state_argument():
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    metadata = TransformerBlockRegistry.get(ZImageTransformerBlock)
    assert metadata.hidden_states_argument_name == "x"


@torch.no_grad()
def test_spectrum_zimage_turbo_standard_context_exact_schedule_and_block_accounting():
    model = make_model()
    calls = [0 for _ in model.layers]
    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(
            layer.attention.register_forward_hook(
                lambda *args, i=i: calls.__setitem__(i, calls[i] + 1)
            )
        )
    try:
        model.enable_cache(make_config())
        for step in range(8):
            with model.cache_context("cond"):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 5, 7]
        assert summary["forecast_steps"] == [4, 6]
        assert summary["prediction_call_count"] == 2
        assert summary["full_call_count"] == 6
        assert not summary["guard_latched"]
        assert calls == [6] * 30
        assert sum(calls) == 180

        model.disable_cache()
        assert not model.is_cache_enabled
        assert model._diffusers_hook.get_hook("spectrum_cache_denoiser") is None
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_spectrum_zimage_turbo_contexts_partition_history():
    model = make_model()
    model.enable_cache(make_config())
    for step in range(8):
        for label in ("cond", "alternate"):
            with model.cache_context(label):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond", "alternate"}
    for label in ("cond", "alternate"):
        summary = root.state_manager._state_cache[label].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 5, 7]
        assert summary["forecast_steps"] == [4, 6]
        assert summary["prediction_call_count"] == 2
        assert not summary["guard_latched"]


@torch.no_grad()
def test_spectrum_zimage_controlnet_route_latches_sticky_failclosed():
    model = make_model()
    calls = {"head": 0}
    hook = model.layers[0].attention.register_forward_hook(
        lambda *args: calls.__setitem__("head", calls["head"] + 1)
    )
    try:
        model.enable_cache(make_config())
        for step in range(8):
            with model.cache_context("cond"):
                output = model(**make_inputs(step, controlnet_block_samples={}))
            assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["guard_latched"]
        assert summary["guard_latched_at"] == 0
        assert summary["prediction_call_count"] == 0
        assert summary["full_call_count"] == 8
        assert calls["head"] == 8
        assert any("ControlNet" in item["reason"] for item in summary["guard_reasons"])
    finally:
        hook.remove()


def test_spectrum_zimage_autograd_latches_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    with model.cache_context("cond"):
        output = model(**make_inputs())
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 1
    assert any("autograd/training" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_zimage_rejects_unqualified_architecture_at_enable():
    model = make_unqualified_model()
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "Z-Image Turbo standard T2I transformer architecture" in str(error)
    else:
        raise AssertionError("Expected non-Turbo Z-Image architecture to be rejected.")
