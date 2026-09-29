import torch

from diffusers import ZImageTransformer2DModel
from diffusers.hooks._helpers import TransformerBlockRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


ZIMAGE_STANDARD_T2I_CONFIG_SIGNATURE = {
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
    # Tiny mechanics graph; adapter itself is fail-closed to the exact qualified standard T2I Base/Turbo config.
    model.register_to_config(**ZIMAGE_STANDARD_T2I_CONFIG_SIGNATURE)
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


def make_inputs(step=0, *, total_steps=8, controlnet_block_samples=None):
    return {
        "x": [torch.randn(4, 1, 8, 8)],
        "t": torch.tensor([1.0 - step / float(total_steps)]),
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


def make_base_config():
    return SpectrumCacheConfig(
        num_inference_steps=28,
        warmup_steps=9,
        window_size=2.0,
        flex_window=0.0,
        degree=3,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=28,
        coordinate_max=28.0,
        tail_actual_steps=4,
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
def test_spectrum_zimage_base_standard_context_exact_28step_schedule_and_block_accounting():
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
        model.enable_cache(make_base_config())
        for step in range(28):
            with model.cache_context("cond"):
                output = model(**make_inputs(step, total_steps=28))
            assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 27]
        assert summary["forecast_steps"] == [9, 11, 13, 15, 17, 19, 21, 23]
        assert summary["prediction_call_count"] == 8
        assert summary["full_call_count"] == 20
        assert not summary["guard_latched"]
        assert calls == [20] * 30
        assert sum(calls) == 600
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
        assert "Z-Image standard T2I Base/Turbo transformer architecture" in str(error)
    else:
        raise AssertionError("Expected unqualified Z-Image architecture to be rejected.")



def make_runtime_index_config(num_inference_steps):
    return SpectrumCacheConfig(
        num_inference_steps=num_inference_steps,
        warmup_steps=3,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=16,
        coordinate_policy="runtime_index_normalized",
        coordinate_max=50.0,
        tail_actual_steps=1,
    )


def make_batched_inputs(step=0, *, total_steps=8, batch_size=2):
    return {
        "x": [torch.randn(4, 1, 8, 8) for _ in range(batch_size)],
        "t": torch.full((batch_size,), 1.0 - step / float(total_steps)),
        "cap_feats": [torch.randn(12, 16) for _ in range(batch_size)],
        "return_dict": True,
        "controlnet_block_samples": None,
        "siglip_feats": None,
        "image_noise_mask": None,
        "patch_size": 2,
        "f_patch_size": 1,
    }


@torch.no_grad()
def test_spectrum_zimage_runtime_index_cond_uncond_uses_native_cache_context_provenance():
    num_inference_steps = 9
    model = make_model()
    model.enable_cache(make_runtime_index_config(num_inference_steps))
    for step in range(num_inference_steps):
        with model.cache_context("cond_uncond", step_index=step, num_inference_steps=num_inference_steps):
            output = model(**make_batched_inputs(step, total_steps=num_inference_steps, batch_size=2))
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond_uncond"}
    summary = root.state_manager._state_cache["cond_uncond"].summary()
    coordinate = summary["coordinate"]
    assert coordinate["coordinate_policy"] == "runtime_index_normalized"
    assert coordinate["provenance_source"] == "cache_context"
    assert coordinate["runtime_num_inference_steps"] == num_inference_steps
    assert coordinate["last_logical_step_index"] == num_inference_steps - 1
    assert coordinate["failure_latched"] is False
    assert summary["prediction_call_count"] > 0


@torch.no_grad()
def test_spectrum_zimage_same_context_fails_closed_if_cfg_batch_shape_changes_midrun():
    num_inference_steps = 6
    model = make_model()
    model.enable_cache(make_runtime_index_config(num_inference_steps))

    for step in range(num_inference_steps):
        batch_size = 2 if step < 3 else 1
        with model.cache_context("cond_uncond", step_index=step, num_inference_steps=num_inference_steps):
            output = model(**make_batched_inputs(step, total_steps=num_inference_steps, batch_size=batch_size))
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond_uncond"].summary()
    assert summary["guard_latched"]
    assert any("input signature changed" in item["reason"] for item in summary["guard_reasons"])
    assert summary["coordinate"]["failure_latched"] is False


def test_spectrum_zimage_base_variable_factory_preserves_measured_schedule_geometry_and_bounds():
    expected = {
        20: (6, 3, (6, 8, 10, 12, 14, 16)),
        24: (8, 3, (8, 10, 12, 14, 16, 18, 20)),
        28: (9, 4, (9, 11, 13, 15, 17, 19, 21, 23)),
        32: (10, 5, (10, 12, 14, 16, 18, 20, 22, 24, 26)),
        36: (12, 5, (12, 14, 16, 18, 20, 22, 24, 26, 28, 30)),
        40: (13, 6, (13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33)),
    }
    for steps, (warmup, tail, forecast) in expected.items():
        config = SpectrumCacheConfig.for_zimage_base_variable(steps)
        assert config.num_inference_steps == steps
        assert config.warmup_steps == warmup
        assert config.tail_actual_steps == tail
        assert config.forecast_step_indices == forecast
        assert config.coordinate_policy == "runtime_horizon_normalized"
        assert config.coordinate_max == 28.0
        assert config.history_limit == 28
        assert config.predictor_backend == "dense"
        assert config.degree == 3
        assert config.ridge_lambda == 0.1
        assert config.blend_w == 0.5

    for steps in (19, 41):
        try:
            SpectrumCacheConfig.for_zimage_base_variable(steps)
        except ValueError as error:
            assert "20 <= num_inference_steps <= 40" in str(error)
        else:
            raise AssertionError(f"expected bounded-range rejection for {steps}")


@torch.no_grad()
def test_spectrum_zimage_base_variable_factory_uses_native_cond_uncond_runtime_provenance():
    num_inference_steps = 20
    model = make_model()
    config = SpectrumCacheConfig.for_zimage_base_variable(num_inference_steps)
    model.enable_cache(config)

    for step in range(num_inference_steps):
        with model.cache_context(
            "cond_uncond",
            step_index=step,
            num_inference_steps=num_inference_steps,
        ):
            output = model(**make_batched_inputs(step, total_steps=num_inference_steps, batch_size=2))
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond_uncond"}
    summary = root.state_manager._state_cache["cond_uncond"].summary()
    coordinate = summary["coordinate"]
    assert coordinate["coordinate_policy"] == "runtime_horizon_normalized"
    assert coordinate["provenance_source"] == "cache_context"
    assert coordinate["runtime_num_inference_steps"] == num_inference_steps
    assert coordinate["last_logical_step_index"] == num_inference_steps - 1
    assert coordinate["failure_latched"] is False
    assert summary["guard_latched"] is False
    assert summary["prediction_failure_latched"] is False
    assert summary["prediction_call_count"] == len(config.forecast_step_indices)
