
import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumPonyV7State
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


def test_pony_v7_state_exact_single_skip_schedule():
    state = SpectrumPonyV7State(pony_cfg())
    conditioning_id = (123, 0, (2, 256, 2048), "torch.bfloat16", "cuda:0")
    for step in range(30):
        state.prepare_call(conditioning_id)
        if step == 20:
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
    assert summary["forecast_steps"] == [20]
    assert summary["compute_steps"] == [i for i in range(30) if i != 20]
    assert summary["prediction_call_count"] == 1
    assert summary["full_call_count"] == 29
    assert summary["real_block_executions"] == 29 * 32
    assert summary["bypassed_block_executions"] == 32
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
