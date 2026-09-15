import torch

from diffusers import LTXVideoTransformer3DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumLTXVideo2BState


def make_config():
    return SpectrumCacheConfig(
        num_inference_steps=40,
        window_size=2.0,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=100,
        coordinate_max=50.0,
        warmup_steps=8,
        flex_window=0.0,
        tail_actual_steps=6,
    )


@torch.no_grad()
def test_spectrum_ltxvideo2b_w8t6_state_schedule_and_prediction():
    state = SpectrumLTXVideo2BState(make_config())
    expected_forecast = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

    for step in range(40):
        state.start_step()
        assert state.step_index == step
        if state.should_compute:
            feature = torch.full((1, 4, 8), float(step), dtype=torch.float32)
            state.record_real_features(feature)
        else:
            predicted = state.predict()
            assert predicted.shape == (1, 4, 8)
            assert torch.isfinite(predicted).all()

        state.records.append(
            {
                "logical_step": step,
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": not bool(state.should_compute),
                "guard_latched": False,
                "fallback_reason": None,
            }
        )

    summary = state.summary()
    assert summary["forecast_steps"] == expected_forecast
    assert summary["prediction_call_count"] == 13
    assert summary["full_call_count"] == 27
    assert not summary["guard_latched"]


def test_spectrum_ltxvideo2b_guard_is_sticky_full_compute():
    state = SpectrumLTXVideo2BState(make_config())
    state.latch("test guard")
    for _ in range(3):
        state.start_step()
        assert state.should_compute
        assert state.guard_latched


def test_spectrum_ltxvideo2b_rejects_unqualified_architecture_at_enable():
    model = LTXVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=4,
        cross_attention_dim=8,
        num_layers=1,
        caption_channels=8,
    ).eval()

    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "historical 2B 28-block transformer architecture" in str(error)
    else:
        raise AssertionError("Expected unqualified LTX-Video architecture to be rejected.")
