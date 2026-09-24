# coding=utf-8

import torch

from diffusers import HunyuanVideo15Transformer3DModel
from diffusers.hooks import SpectrumCacheConfig


def _tiny_hv15():
    model = HunyuanVideo15Transformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=54,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=1,
        patch_size_t=1,
        text_embed_dim=16,
        text_embed_2_dim=8,
        image_embed_dim=12,
        rope_axes_dim=(2, 2, 4),
        target_size=640,
        task_type="t2v",
        use_meanflow=False,
    )
    model.eval()
    return model


def _inputs():
    generator = torch.Generator("cpu").manual_seed(0)
    return {
        "hidden_states": torch.randn((1, 4, 1, 8, 8), generator=generator),
        "timestep": torch.tensor([1.0]),
        "encoder_hidden_states": torch.randn((1, 6, 16), generator=generator),
        "encoder_attention_mask": torch.ones((1, 6)),
        "encoder_hidden_states_2": torch.randn((1, 4, 8), generator=generator),
        "encoder_attention_mask_2": torch.ones((1, 4)),
        "image_embeds": torch.zeros((1, 3, 12)),
        "return_dict": False,
    }


def _config(forecast_steps):
    return SpectrumCacheConfig(
        num_inference_steps=50,
        warmup_steps=5,
        window_size=2.0,
        flex_window=0.0,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=50.0,
        tail_actual_steps=0,
        forecast_step_indices=tuple(forecast_steps),
    )


def test_hunyuan_video15_spectrum_empty_schedule_is_exact_and_disable_is_clean():
    model = _tiny_hv15()
    inputs = _inputs()

    with torch.no_grad():
        baseline = model(**inputs)[0]
        model.enable_cache(_config(()))
        with model.cache_context("pred_cond"):
            cached = model(**inputs)[0]
        assert torch.equal(cached, baseline)
        model.disable_cache()
        restored = model(**inputs)[0]

    assert torch.equal(restored, baseline)


def test_hunyuan_video15_spectrum_forecast_skips_the_whole_block_stack():
    model = _tiny_hv15()
    inputs = _inputs()
    model.enable_cache(_config(()))

    from diffusers.hooks.hooks import HookRegistry
    from diffusers.hooks.spectrum_cache import _SPECTRUM_DENOISER_HOOK

    root_hook = HookRegistry.check_if_exists_or_initialize(model).hooks[_SPECTRUM_DENOISER_HOOK]
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_hook(
        lambda *_args: calls.__setitem__("count", calls["count"] + 1)
    )
    outputs = []
    try:
        with torch.no_grad():
            for step in range(3):
                inputs["timestep"] = torch.tensor([float(step + 1)])
                with model.cache_context("pred_cond"):
                    outputs.append(model(**inputs)[0])
                if step == 0:
                    state = root_hook.state_manager._state_cache["pred_cond"]
                    state.schedule._explicit_forecast_steps = frozenset({1})
    finally:
        handle.remove()
        model.disable_cache()

    assert calls["count"] == 2
    assert all(torch.isfinite(output).all() for output in outputs)
    assert [tuple(output.shape) for output in outputs] == [(1, 4, 1, 8, 8)] * 3

def test_hunyuan_video15_qualified_factory_matches_guarded_profile():
    config = SpectrumCacheConfig.for_hunyuan_video15()
    assert config.num_inference_steps == 50
    assert config.history_limit == 8
    assert config.forecast_step_indices == (20, 22, 24, 27, 32, 34, 36, 38, 40, 42)


def test_hunyuan_video15_old_h100_research_profile_is_not_promotion_profile():
    model = _tiny_hv15()
    config = _config(())
    config.history_limit = 100
    try:
        model.enable_cache(config)
    except ValueError as error:
        assert "history=8" in str(error)
    else:
        raise AssertionError("H=100 research profile must not be accepted by the promotion adapter")
