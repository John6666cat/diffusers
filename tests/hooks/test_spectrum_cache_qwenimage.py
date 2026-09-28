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


def make_inputs(step=0, *, edit=False, references=1, attention_kwargs=None, additional_t_cond=None):
    if edit:
        img_shapes = [[(1, 4, 4)] + [(1, 4, 4)] * references]
        hidden_tokens = 16 * (1 + references)
    else:
        img_shapes = [[(1, 4, 4)]]
        hidden_tokens = 16
    return {
        "hidden_states": torch.randn(1, hidden_tokens, 4),
        "encoder_hidden_states": torch.randn(1, 4, 8),
        "encoder_hidden_states_mask": None,
        "timestep": torch.tensor([1.0 - step / 20.0]),
        "img_shapes": img_shapes,
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
def test_spectrum_qwenimage_edit2511_exact_selected6_schedule_and_block_accounting():
    model = make_model(zero_cond_t=True)
    calls = [0 for _ in model.transformer_blocks]
    hooks = []
    for i, block in enumerate(model.transformer_blocks):
        hooks.append(block.attn.register_forward_hook(lambda *args, i=i: calls.__setitem__(i, calls[i] + 1)))
    try:
        model.enable_cache(make_config())
        for step in range(20):
            with model.cache_context("cond"):
                output = model(**make_inputs(step, edit=True))
            assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        assert root.route == "edit"
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 18, 19]
        assert summary["forecast_steps"] == [6, 8, 10, 12, 14, 15]
        assert summary["prediction_call_count"] == 6
        assert summary["full_call_count"] == 14
        assert not summary["guard_latched"]
        assert calls == [14] * 60
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_spectrum_qwenimage_edit2511_requires_exactly_one_reference_latent():
    model = make_model(zero_cond_t=True)
    model.enable_cache(make_config())

    with model.cache_context("cond"):
        output = model(**make_inputs(0, edit=True, references=0))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 1
    assert any("exactly one reference latent" in item["reason"] for item in summary["guard_reasons"])


@torch.no_grad()
def test_spectrum_qwenimage_edit2511_rejects_multiple_reference_latents_failclosed():
    model = make_model(zero_cond_t=True)
    model.enable_cache(make_config())

    with model.cache_context("cond"):
        output = model(**make_inputs(0, edit=True, references=2))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 1
    assert any("exactly one reference latent" in item["reason"] for item in summary["guard_reasons"])


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
        assert "Qwen-Image-2512 T2I and Qwen-Image-Edit-2511 transformer architectures" in str(error)
    else:
        raise AssertionError("Expected unqualified Qwen-Image architecture to be rejected.")



def make_runtime_index_config(num_inference_steps):
    return SpectrumCacheConfig(
        num_inference_steps=num_inference_steps,
        warmup_steps=3,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=12,
        coordinate_policy="runtime_index_normalized",
        coordinate_max=50.0,
        tail_actual_steps=1,
    )


@torch.no_grad()
def test_spectrum_qwenimage_t2i_runtime_index_uses_native_cache_context_provenance():
    num_inference_steps = 8
    model = make_model()
    model.enable_cache(make_runtime_index_config(num_inference_steps))
    for step in range(num_inference_steps):
        with model.cache_context("cond", step_index=step, num_inference_steps=num_inference_steps):
            output = model(**make_inputs(step))
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    coordinate = summary["coordinate"]
    assert coordinate["coordinate_policy"] == "runtime_index_normalized"
    assert coordinate["provenance_source"] == "cache_context"
    assert coordinate["runtime_num_inference_steps"] == num_inference_steps
    assert coordinate["last_logical_step_index"] == num_inference_steps - 1
    assert coordinate["failure_latched"] is False
    assert summary["prediction_call_count"] > 0


@torch.no_grad()
def test_spectrum_qwenimage_edit_runtime_index_uses_native_cache_context_provenance():
    num_inference_steps = 8
    model = make_model(zero_cond_t=True)
    model.enable_cache(make_runtime_index_config(num_inference_steps))
    for step in range(num_inference_steps):
        with model.cache_context("cond", step_index=step, num_inference_steps=num_inference_steps):
            output = model(**make_inputs(step, edit=True, references=1))
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert root.route == "edit"
    summary = root.state_manager._state_cache["cond"].summary()
    coordinate = summary["coordinate"]
    assert coordinate["provenance_source"] == "cache_context"
    assert coordinate["runtime_num_inference_steps"] == num_inference_steps
    assert coordinate["last_logical_step_index"] == num_inference_steps - 1
    assert coordinate["failure_latched"] is False
    assert summary["prediction_call_count"] > 0


@torch.no_grad()
def test_spectrum_qwenimage_runtime_index_cond_uncond_have_independent_provenance():
    num_inference_steps = 8
    model = make_model()
    model.enable_cache(make_runtime_index_config(num_inference_steps))
    for step in range(num_inference_steps):
        for label in ("cond", "uncond"):
            with model.cache_context(label, step_index=step, num_inference_steps=num_inference_steps):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond", "uncond"}
    for label in ("cond", "uncond"):
        coordinate = root.state_manager._state_cache[label].summary()["coordinate"]
        assert coordinate["provenance_source"] == "cache_context"
        assert coordinate["runtime_num_inference_steps"] == num_inference_steps
        assert coordinate["last_logical_step_index"] == num_inference_steps - 1
        assert coordinate["failure_latched"] is False



def make_runtime_horizon_config(num_inference_steps):
    return SpectrumCacheConfig(
        num_inference_steps=num_inference_steps,
        warmup_steps=3,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=12,
        coordinate_policy="runtime_horizon_normalized",
        coordinate_max=50.0,
        tail_actual_steps=1,
    )


@torch.no_grad()
def test_spectrum_qwenimage_t2i_runtime_horizon_uses_native_cache_context_provenance():
    num_inference_steps = 8
    model = make_model()
    model.enable_cache(make_runtime_horizon_config(num_inference_steps))
    for step in range(num_inference_steps):
        with model.cache_context("cond", step_index=step, num_inference_steps=num_inference_steps):
            output = model(**make_inputs(step))
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    coordinate = summary["coordinate"]
    assert coordinate["coordinate_policy"] == "runtime_horizon_normalized"
    assert coordinate["provenance_source"] == "cache_context"
    assert coordinate["runtime_num_inference_steps"] == num_inference_steps
    assert coordinate["last_logical_step_index"] == num_inference_steps - 1
    assert coordinate["failure_latched"] is False
    assert summary["prediction_call_count"] > 0


@torch.no_grad()
def test_spectrum_qwenimage_runtime_horizon_missing_provenance_fails_closed():
    model = make_model()
    model.enable_cache(make_runtime_horizon_config(8))
    with model.cache_context("cond"):
        output = model(**make_inputs(0))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    coordinate = summary["coordinate"]
    assert coordinate["failure_latched"] is True
    assert coordinate["failures"]
    assert summary["prediction_call_count"] == 0


def test_spectrum_qwenimage_variable_factory_preserves_20_step_selected6_identity():
    from diffusers.hooks.spectrum_cache import SpectrumSchedule

    config = SpectrumCacheConfig.for_qwen_image_variable(20)
    assert config.warmup_steps == 6
    assert config.tail_actual_steps == 4
    assert config.window_size == 2.0
    assert config.flex_window == 0.25
    assert config.degree == 4
    assert config.ridge_lambda == 0.1
    assert config.blend_w == 0.5
    assert config.history_limit == 8
    assert config.coordinate_policy == "runtime_horizon_normalized"
    assert config.coordinate_max == 20.0

    schedule = SpectrumSchedule(config)
    forecast = [step for step in range(20) if not schedule.decide(step)]
    assert forecast == [6, 8, 10, 12, 14, 15]


def test_spectrum_qwenimage_variable_factory_scales_protected_regions_and_bounds_range():
    expected = {16: (5, 3), 20: (6, 4), 24: (7, 5), 30: (9, 6), 40: (12, 8)}
    for num_steps, (warmup, tail) in expected.items():
        config = SpectrumCacheConfig.for_qwen_image_variable(num_steps)
        assert config.num_inference_steps == num_steps
        assert config.warmup_steps == warmup
        assert config.tail_actual_steps == tail
        assert config.coordinate_policy == "runtime_horizon_normalized"

    for num_steps in (15, 41):
        try:
            SpectrumCacheConfig.for_qwen_image_variable(num_steps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected bounded-range rejection for {num_steps}")
