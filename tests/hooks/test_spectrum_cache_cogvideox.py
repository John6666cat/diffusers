# coding=utf-8

import torch
from torch import nn

from diffusers import CogVideoXTransformer3DModel
from diffusers.hooks import SpectrumCacheConfig


class _PassCogVideoXBlock(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, attention_kwargs=None):
        return hidden_states, encoder_hidden_states


def _tiny_cogvideox5b_shell():
    model = CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=4,
        in_channels=2,
        out_channels=2,
        time_embed_dim=8,
        ofs_embed_dim=None,
        text_embed_dim=8,
        num_layers=1,
        sample_width=4,
        sample_height=4,
        sample_frames=2,
        patch_size=2,
        patch_size_t=None,
        temporal_compression_ratio=1,
        max_text_seq_length=4,
        use_rotary_positional_embeddings=True,
        use_learned_positional_embeddings=False,
        patch_bias=True,
    )
    model.transformer_blocks = nn.ModuleList([_PassCogVideoXBlock() for _ in range(42)])
    model.register_to_config(
        num_attention_heads=48,
        attention_head_dim=64,
        in_channels=16,
        out_channels=16,
        time_embed_dim=512,
        ofs_embed_dim=None,
        text_embed_dim=4096,
        num_layers=42,
        sample_width=90,
        sample_height=60,
        sample_frames=49,
        patch_size=2,
        patch_size_t=None,
        temporal_compression_ratio=4,
        max_text_seq_length=226,
        use_rotary_positional_embeddings=True,
        use_learned_positional_embeddings=False,
        patch_bias=True,
    )
    model.eval()
    return model


def _inputs():
    generator = torch.Generator("cpu").manual_seed(0)
    return {
        "hidden_states": torch.randn((1, 2, 2, 4, 4), generator=generator),
        "encoder_hidden_states": torch.randn((1, 4, 8), generator=generator),
        "timestep": torch.tensor([1.0]),
        "timestep_cond": None,
        "ofs": None,
        "image_rotary_emb": None,
        "attention_kwargs": None,
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
        history_limit=5,
        coordinate_max=50.0,
        tail_actual_steps=5,
        forecast_step_indices=tuple(forecast_steps),
    )


def test_cogvideox5b_qualified_factory_matches_selected_profile():
    config = SpectrumCacheConfig.for_cogvideox5b_t2v()
    assert config.num_inference_steps == 50
    assert config.history_limit == 5
    assert config.tail_actual_steps == 5
    assert config.forecast_step_indices == (17, 19, 22, 24, 27, 29, 32, 34, 37, 39, 42, 44)


def test_cogvideox5b_empty_schedule_is_exact_and_disable_is_clean():
    model = _tiny_cogvideox5b_shell()
    inputs = _inputs()

    with torch.no_grad():
        baseline = model(**inputs)[0]
        model.enable_cache(_config(()))
        with model.cache_context("cond_uncond", step_index=0, num_inference_steps=50):
            cached = model(**inputs)[0]
        assert torch.equal(cached, baseline)
        model.disable_cache()
        restored = model(**inputs)[0]

    assert torch.equal(restored, baseline)


def test_cogvideox5b_forecast_skips_whole_block_stack_and_preserves_shape():
    model = _tiny_cogvideox5b_shell()
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
                with model.cache_context("cond_uncond", step_index=step, num_inference_steps=50):
                    outputs.append(model(**inputs)[0])
                if step == 0:
                    state = root_hook.state_manager._state_cache["cond_uncond"]
                    state.schedule._explicit_forecast_steps = frozenset({1})
    finally:
        handle.remove()
        model.disable_cache()

    assert calls["count"] == 2
    assert all(torch.isfinite(output).all() for output in outputs)
    assert [tuple(output.shape) for output in outputs] == [(1, 2, 2, 4, 4)] * 3


def test_cogvideox5b_wrong_context_latches_fail_closed():
    model = _tiny_cogvideox5b_shell()
    inputs = _inputs()
    model.enable_cache(_config(()))

    from diffusers.hooks.hooks import HookRegistry
    from diffusers.hooks.spectrum_cache import _SPECTRUM_DENOISER_HOOK

    root_hook = HookRegistry.check_if_exists_or_initialize(model).hooks[_SPECTRUM_DENOISER_HOOK]
    try:
        with torch.no_grad():
            for step in range(2):
                inputs["timestep"] = torch.tensor([float(step + 1)])
                with model.cache_context("pred_cond", step_index=step, num_inference_steps=50):
                    model(**inputs)
        state = root_hook.state_manager._state_cache["pred_cond"]
        assert state.guard_latched
        assert not any(record["predicted_body_used"] for record in state.records)
    finally:
        model.disable_cache()


def test_cogvideox5b_rejects_15_and_unqualified_history_profile():
    model = _tiny_cogvideox5b_shell()
    model.register_to_config(patch_size_t=2)
    try:
        model.enable_cache(_config(()))
    except ValueError as error:
        assert "pinned original 5B T2V" in str(error)
    else:
        raise AssertionError("CogVideoX 1.5-style patch_size_t must not be accepted")

    model = _tiny_cogvideox5b_shell()
    config = _config(())
    config.history_limit = 4
    try:
        model.enable_cache(config)
    except ValueError as error:
        assert "history=5" in str(error)
    else:
        raise AssertionError("unqualified H=4 profile must not be accepted")
