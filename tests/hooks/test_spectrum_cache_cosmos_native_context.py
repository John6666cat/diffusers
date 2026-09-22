import pytest
import torch

from diffusers import CosmosTransformer3DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


def make_model():
    return CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=2,
        mlp_ratio=2,
        text_embed_dim=16,
        adaln_lora_dim=4,
        max_size=(4, 32, 32),
        patch_size=(1, 2, 2),
        rope_scale=(2.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type="learnable",
    ).eval()


def make_inputs(seed=0):
    generator = torch.Generator("cpu").manual_seed(seed)
    return {
        "hidden_states": torch.randn((1, 4, 1, 16, 16), generator=generator),
        "timestep": torch.tensor([0.5]),
        "encoder_hidden_states": torch.randn((1, 12, 16), generator=generator),
        "padding_mask": torch.zeros(1, 1, 16, 16),
        "return_dict": False,
    }


def native_config():
    return SpectrumCacheConfig(
        num_inference_steps=4,
        warmup_steps=2,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=4.0,
        tail_actual_steps=0,
    )


def manager(model):
    hook = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    return hook.state_manager


@torch.no_grad()
def test_native_context_runs_without_legacy_callback_and_accepts_guider_names():
    model = make_model()
    model.enable_cache(native_config())

    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    for step in range(4):
        for label in ("pred_cond", "pred_uncond"):
            with model.cache_context(
                label,
                step_index=step,
                num_inference_steps=4,
                timestep=torch.tensor([float(step)]),
                sigma=1.0 - step * 0.2,
            ):
                output = model(**make_inputs(seed=100 + step))[0]
                assert torch.isfinite(output).all()

    mgr = manager(model)
    assert set(mgr._state_cache) == {"pred_cond", "pred_uncond"}
    for label in ("pred_cond", "pred_uncond"):
        state = mgr._state_cache[label]
        summary = state.summary()
        assert summary["logical_prediction_steps_used"] == [2]
        assert summary["prediction_call_count"] == 1
        assert summary["full_call_count"] == 3
        assert state.current_label == label

    assert calls["count"] == 6
    handle.remove()
    model.disable_cache()


@torch.no_grad()
def test_native_context_state_is_partitioned_by_arbitrary_nonempty_identifier():
    model = make_model()
    model.enable_cache(native_config())

    for label in ("pred_cond", "pred_aux"):
        with model.cache_context(
            label,
            step_index=0,
            num_inference_steps=4,
            timestep=torch.tensor([0.0]),
            sigma=1.0,
        ):
            model(**make_inputs(seed=5))

    assert set(manager(model)._state_cache) == {"pred_cond", "pred_aux"}
    model.disable_cache()


@torch.no_grad()
def test_incomplete_native_context_without_callback_fails_explicitly():
    model = make_model()
    model.enable_cache(native_config())

    with model.cache_context("pred_cond"):
        with pytest.raises(ValueError, match="step_index.*num_inference_steps"):
            model(**make_inputs(seed=9))

    model.disable_cache()


@torch.no_grad()
def test_legacy_callback_still_works_without_native_context():
    model = make_model()
    runtime = {
        "step": 0,
        "num_inference_steps": 4,
        "num_conditions": 1,
        "label": "cond",
        "dynamic_conditioning": False,
    }
    config = native_config()
    config.cosmos_runtime_state_callback = lambda: dict(runtime)
    model.enable_cache(config)

    for step in range(4):
        runtime["step"] = step
        output = model(**make_inputs(seed=20 + step))[0]
        assert torch.isfinite(output).all()

    mgr = manager(model)
    assert set(mgr._state_cache) == {"__legacy_cosmos__"}
    summary = mgr._state_cache["__legacy_cosmos__"].summary()
    assert summary["logical_prediction_steps_used"] == [2]
    assert summary["prediction_call_count"] == 1
    model.disable_cache()
