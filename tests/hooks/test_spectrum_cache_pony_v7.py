
import pytest
import torch

from diffusers.hooks.hooks import HookRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_cache_pony_v7 import SpectrumPonyV7State
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.transformers.auraflow_transformer_2d import AuraFlowTransformer2DModel


def pony_cfg(**overrides):
    kwargs = dict(
        num_inference_steps=30,
        warmup_steps=0,
        window_size=3.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.25,
        history_limit=8,
        coordinate_max=8.0,
        tail_actual_steps=0,
        forecast_step_indices=(20,),
    )
    kwargs.update(overrides)
    return SpectrumCacheConfig(**kwargs)


@pytest.mark.parametrize("forecast_steps", [(20,), (17, 19, 21, 23)])
def test_pony_v7_state_qualified_schedules(forecast_steps):
    state = SpectrumPonyV7State(pony_cfg(forecast_step_indices=forecast_steps))
    conditioning_id = (123, 0, (2, 256, 2048), "torch.bfloat16", "cuda:0")
    for step in range(30):
        state.prepare_call(conditioning_id)
        if step in forecast_steps:
            assert not state.should_compute
            pred = state.predict()
            assert pred.shape == (2, 8, 6)
            state.current_forecast_used = True
            state.bypassed_block_executions += 32
        else:
            assert state.should_compute
            state.record_real_feature(torch.full((2, 8, 6), float(step), dtype=torch.float32))
            state.real_block_executions += 32
        state.finish_call()

    summary = state.summary()
    assert summary["forecast_steps"] == list(forecast_steps)
    assert summary["compute_steps"] == [i for i in range(30) if i not in forecast_steps]
    assert summary["prediction_call_count"] == len(forecast_steps)
    assert summary["full_call_count"] == 30 - len(forecast_steps)
    assert summary["real_block_executions"] == (30 - len(forecast_steps)) * 32
    assert summary["bypassed_block_executions"] == len(forecast_steps) * 32
    assert not summary["guard_latched"]


def test_pony_v7_transformer_exposes_cache_api_and_tiny_architecture_fails_closed():
    assert issubclass(AuraFlowTransformer2DModel, CacheMixin)
    for attr in ("enable_cache", "disable_cache", "cache_context", "is_cache_enabled"):
        assert hasattr(AuraFlowTransformer2DModel, attr)

    model = AuraFlowTransformer2DModel(
        sample_size=32,
        patch_size=2,
        in_channels=4,
        num_mmdit_layers=1,
        num_single_dit_layers=1,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=32,
        caption_projection_dim=32,
        out_channels=4,
        pos_embed_max_size=256,
    )
    with pytest.raises(ValueError, match="qualified only for the pinned AuraFlow architecture"):
        model.enable_cache(pony_cfg())


def _tiny_qualified_auraflow():
    model = AuraFlowTransformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_mmdit_layers=4,
        num_single_dit_layers=32,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        out_channels=4,
        pos_embed_max_size=64,
    )
    # Preserve the 4+32 topology but avoid allocating Pony's full-size weights.
    # Config mutation is used only for registration/negative tests, not forward.
    model.register_to_config(
        patch_size=2,
        in_channels=4,
        num_mmdit_layers=4,
        num_single_dit_layers=32,
        attention_head_dim=256,
        num_attention_heads=12,
        joint_attention_dim=2048,
        caption_projection_dim=3072,
        out_channels=4,
    )
    return model


def test_pony_v7_rejects_unqualified_schedule():
    model = _tiny_qualified_auraflow()
    with pytest.raises(ValueError, match="qualified Pony V7 schedules"):
        model.enable_cache(pony_cfg(forecast_step_indices=(17, 19, 21)))


def test_pony_v7_disable_cache_removes_single_block_hooks():
    model = _tiny_qualified_auraflow()
    root = HookRegistry.check_if_exists_or_initialize(model)
    first = HookRegistry.check_if_exists_or_initialize(model.single_transformer_blocks[0])
    middle = HookRegistry.check_if_exists_or_initialize(model.single_transformer_blocks[1])
    last = HookRegistry.check_if_exists_or_initialize(model.single_transformer_blocks[-1])
    for _ in range(2):
        model.enable_cache(pony_cfg())
        assert root.get_hook("spectrum_cache_denoiser") is not None
        assert first.get_hook("spectrum_cache_head_block") is not None
        assert middle.get_hook("spectrum_cache_block") is not None
        assert last.get_hook("spectrum_cache_block") is not None
        model.disable_cache()
        assert root.get_hook("spectrum_cache_denoiser") is None
        assert first.get_hook("spectrum_cache_head_block") is None
        assert middle.get_hook("spectrum_cache_block") is None
        assert last.get_hook("spectrum_cache_block") is None
        assert not model.is_cache_enabled
