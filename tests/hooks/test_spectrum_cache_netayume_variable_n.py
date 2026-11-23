"""Research-only variable-N masks for Neta Yume's paired-CFG native route."""

from dataclasses import replace

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_cache_netayume import SpectrumNetaYumeState

from tests.hooks.test_spectrum_cache_netayume import (
    AGGRESSIVE_FORECAST_STEPS,
    CACHE_KWARGS,
    CONTROL_FORECAST_STEPS,
    N30_FORECAST_STEPS,
    _tiny_lumina2,
)


@pytest.mark.parametrize("profile,mask", [
    ("p6", CONTROL_FORECAST_STEPS),
    ("p9", AGGRESSIVE_FORECAST_STEPS),
    ("p30", N30_FORECAST_STEPS),
])
def test_netayume_n50_anchor_keeps_legacy_predictor_coordinates_and_mask(profile, mask):
    actual = SpectrumCacheConfig.for_netayume_variable(50, profile=profile)
    reference = SpectrumCacheConfig(**dict(CACHE_KWARGS, forecast_step_indices=mask))
    assert actual.forecast_step_indices == reference.forecast_step_indices
    assert actual.coordinate_policy == reference.coordinate_policy
    assert actual.coordinate_max == reference.coordinate_max == 50.0
    for attr in ("degree", "window_size", "ridge_lambda", "blend_w", "history_limit", "warmup_steps", "flex_window"):
        assert getattr(actual, attr) == getattr(reference, attr)
    assert actual.allow_netayume_variable_n is True
    assert reference.allow_netayume_variable_n is False


@pytest.mark.parametrize("n,count", [(11, 0), (12, 5), (40, 25), (49, 30), (50, 30), (51, 31), (60, 35), (80, 42)])
@torch.no_grad()
def test_netayume_variable_native_state_has_two_independent_cfg_lanes(n, count):
    config = SpectrumCacheConfig.for_netayume_variable(n)
    assert len(config.forecast_step_indices) == count
    state = SpectrumNetaYumeState(config)
    for step in range(n):
        for lane, offset, feature_shape in (("positive", 0.0, (1, 8, 6)), ("negative", 0.25, (1, 5, 6))):
            state.prepare_call((lane, 42))
            assert state.step_index == step
            assert state.current_lane == lane
            assert state.should_compute == (step not in config.forecast_step_indices)
            if state.should_compute:
                state.record_real_feature(torch.full(feature_shape, float(step) + offset))
            else:
                assert state.predict().shape == feature_shape
                state.current_forecast_used = True
            state.finish_call()
    summary = state.summary()
    assert not summary["guard_latched"]
    assert summary["forecast_steps"] == list(config.forecast_step_indices)
    assert summary["prediction_call_count"] == count * 2
    assert summary["lane_prediction_call_count"] == {"positive": count, "negative": count}


@pytest.mark.parametrize("n", [40, 50, 60])
@pytest.mark.parametrize("backend", ["dense", "fp32_weighted"])
def test_netayume_variable_registers_and_cleans_up(n, backend):
    model = _tiny_lumina2()
    config = SpectrumCacheConfig.for_netayume_variable(n, predictor_backend=backend)
    for _ in range(2):
        model.enable_cache(config)
        assert model.is_cache_enabled
        model.disable_cache()
        assert not model.is_cache_enabled


@pytest.mark.parametrize("bad", [True, False, 0, 81, 50.0, "50"])
def test_netayume_variable_rejects_invalid_n(bad):
    with pytest.raises(ValueError, match="integer N"):
        SpectrumCacheConfig.for_netayume_variable(bad)


def test_netayume_variable_guard_rejects_modified_config_and_unflagged_nonanchor():
    model = _tiny_lumina2()
    config = SpectrumCacheConfig.for_netayume_variable(40)
    for bad in (
        replace(config, allow_netayume_variable_n=False),
        replace(config, coordinate_max=50.0),
        replace(config, denoiser_call_topology="auto"),
        replace(config, forecast_step_indices=config.forecast_step_indices[:-1]),
    ):
        with pytest.raises(ValueError, match="variable-N research profile"):
            model.enable_cache(bad)
    with pytest.raises(ValueError, match="dense or fp32_weighted"):
        SpectrumCacheConfig.for_netayume_variable(40, predictor_backend="chunked")
    with pytest.raises(ValueError, match="p6, p9, or p30"):
        SpectrumCacheConfig.for_netayume_variable(40, profile="p35")


def test_netayume_variable_anchor_opt_in_cannot_bypass_strict_guard():
    model = _tiny_lumina2()
    config = SpectrumCacheConfig.for_netayume_variable(50, profile="p6")
    with pytest.raises(ValueError, match="variable-N research profile"):
        model.enable_cache(replace(config, denoiser_call_topology="auto"))
