import pytest
import torch

from diffusers.models.cache_utils import CacheMixin
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumNetaYumeState
from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel


FORECAST_STEPS = (15, 20, 23, 25, 27, 36)
CACHE_KWARGS = dict(
    num_inference_steps=50,
    warmup_steps=0,
    window_size=3.0,
    flex_window=0.0,
    degree=2,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=8,
    coordinate_max=50.0,
    tail_actual_steps=0,
    forecast_step_indices=FORECAST_STEPS,
)


def _tiny_lumina2(num_layers=26):
    # Tiny weights, native topology. Config is mutated after construction only to
    # exercise exact-qualified hook registration without allocating the real model.
    model = Lumina2Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        out_channels=None,
        hidden_size=48,
        num_layers=num_layers,
        num_refiner_layers=2,
        num_attention_heads=6,
        num_kv_heads=2,
        multiple_of=16,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        scaling_factor=1.0,
        axes_dim_rope=(2, 2, 4),
        axes_lens=(16, 16, 16),
        cap_feat_dim=16,
    )
    model.register_to_config(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        out_channels=None,
        hidden_size=2304,
        num_layers=26,
        num_refiner_layers=2,
        num_attention_heads=24,
        num_kv_heads=8,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        scaling_factor=1.0,
        axes_dim_rope=(32, 32, 32),
        axes_lens=(300, 512, 512),
        cap_feat_dim=2304,
    )
    return model


def test_netayume_lumina2_exposes_cache_mixin_api():
    assert issubclass(Lumina2Transformer2DModel, CacheMixin)
    for name in ("enable_cache", "disable_cache", "cache_context", "is_cache_enabled"):
        assert hasattr(Lumina2Transformer2DModel, name)


def test_netayume_dual_lane_schedule_and_shapes():
    state = SpectrumNetaYumeState(SpectrumCacheConfig(**CACHE_KWARGS))
    pos_id = ("positive", 256)
    neg_id = ("negative", 1)

    for step in range(50):
        state.prepare_call(pos_id)
        if step in FORECAST_STEPS:
            assert not state.should_compute
            predicted = state.predict()
            assert tuple(predicted.shape) == (1, 8, 6)
            state.current_forecast_used = True
        else:
            assert state.should_compute
            state.record_real_feature(torch.full((1, 8, 6), float(step)))
        state.finish_call()

        state.prepare_call(neg_id)
        if step in FORECAST_STEPS:
            assert not state.should_compute
            predicted = state.predict()
            assert tuple(predicted.shape) == (1, 5, 6)
            state.current_forecast_used = True
        else:
            assert state.should_compute
            state.record_real_feature(torch.full((1, 5, 6), float(step) + 0.25))
        state.finish_call()

    summary = state.summary()
    assert summary["guard_latched"] is False
    assert summary["forecast_steps"] == list(FORECAST_STEPS)
    assert summary["prediction_call_count"] == 12
    assert summary["full_call_count"] == 88
    assert summary["lane_prediction_call_count"] == {"positive": 6, "negative": 6}


def test_netayume_exact_profile_registers_on_tiny_topology():
    model = _tiny_lumina2()
    config = SpectrumCacheConfig(**CACHE_KWARGS)
    model.enable_cache(config)
    assert model.is_cache_enabled
    model.disable_cache()
    assert not model.is_cache_enabled


def test_netayume_rejects_unqualified_profile():
    model = _tiny_lumina2()
    kwargs = dict(CACHE_KWARGS)
    kwargs["forecast_step_indices"] = (15,)
    with pytest.raises(ValueError, match="qualified only for the exact 50-step"):
        model.enable_cache(SpectrumCacheConfig(**kwargs))
