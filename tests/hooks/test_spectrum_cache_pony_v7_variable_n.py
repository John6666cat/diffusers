"""Research-only variable-N masks for Pony V7's single-forward CFG route."""

from dataclasses import replace

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule
from diffusers.hooks.spectrum_cache_pony_v7 import SpectrumPonyV7State
from tests.hooks.test_spectrum_cache_pony_v7 import _tiny_qualified_auraflow, pony_cfg


@pytest.mark.parametrize("profile,mask", [
    ("p1", (20,)),
    ("p4", (17, 19, 21, 23)),
    ("p8", (13, 15, 17, 19, 21, 23, 25, 27)),
])
def test_pony_n30_anchor_keeps_legacy_predictor_coordinates_and_mask(profile, mask):
    config = SpectrumCacheConfig.for_pony_v7_variable(30, profile=profile)
    legacy = pony_cfg(forecast_step_indices=mask)
    assert config.forecast_step_indices == legacy.forecast_step_indices
    assert config.coordinate_policy == legacy.coordinate_policy
    assert config.coordinate_max == legacy.coordinate_max == 8.0
    assert config.blend_w == legacy.blend_w == 0.25
    assert config.degree == legacy.degree == 2
    assert config.allow_pony_v7_variable_n is True
    assert legacy.allow_pony_v7_variable_n is False


@pytest.mark.parametrize("n,count", [(11, 0), (12, 3), (24, 6), (27, 8), (30, 8), (33, 10), (36, 10), (40, 11), (80, 16)])
@torch.no_grad()
def test_pony_variable_native_state_retains_combined_cfg_feature(n, count):
    config = SpectrumCacheConfig.for_pony_v7_variable(n)
    assert len(config.forecast_step_indices) == count
    assert config.coordinate_max == pytest.approx(8 * n / 30)
    schedule = SpectrumSchedule(config)
    assert [i for i in range(n) if not schedule.decide(i)] == list(config.forecast_step_indices)
    state = SpectrumPonyV7State(config)
    for step in range(n):
        state.prepare_call(("negative+positive", 3))
        assert state.should_compute == (step not in config.forecast_step_indices)
        if state.should_compute:
            state.record_real_feature(torch.full((2, 8, 6), float(step)))
        else:
            assert state.predict().shape == (2, 8, 6)
            state.current_forecast_used = True
        state.finish_call()
    summary = state.summary()
    assert not summary["guard_latched"]
    assert summary["prediction_call_count"] == count
    assert summary["forecast_steps"] == list(config.forecast_step_indices)


@pytest.mark.parametrize("n", [24, 30, 36, 40])
@pytest.mark.parametrize("backend", ["dense", "fp32_weighted"])
def test_pony_variable_registers_and_cleans_up(n, backend):
    model = _tiny_qualified_auraflow()
    config = SpectrumCacheConfig.for_pony_v7_variable(n, predictor_backend=backend)
    for _ in range(2):
        model.enable_cache(config)
        assert model.is_cache_enabled
        model.disable_cache()
        assert not model.is_cache_enabled


@pytest.mark.parametrize("bad", [True, False, 0, 81, 30.0, "30"])
def test_pony_variable_rejects_invalid_n(bad):
    with pytest.raises(ValueError, match="integer N"):
        SpectrumCacheConfig.for_pony_v7_variable(bad)


def test_pony_variable_guard_rejects_modified_config_and_old_p12_negative():
    model = _tiny_qualified_auraflow()
    config = SpectrumCacheConfig.for_pony_v7_variable(36)
    for bad in (
        replace(config, allow_pony_v7_variable_n=False),
        replace(config, coordinate_max=8.0),
        replace(config, denoiser_call_topology="auto"),
        replace(config, blend_w=0.5),
        replace(config, forecast_step_indices=config.forecast_step_indices[:-1]),
    ):
        with pytest.raises(ValueError, match="variable-N research profile"):
            model.enable_cache(bad)
    with pytest.raises(ValueError, match="qualified Pony V7 schedules"):
        model.enable_cache(pony_cfg(forecast_step_indices=(5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27)))
    with pytest.raises(ValueError, match="dense or fp32_weighted"):
        SpectrumCacheConfig.for_pony_v7_variable(36, predictor_backend="chunked")


def test_pony_variable_anchor_opt_in_cannot_bypass_strict_guard():
    model = _tiny_qualified_auraflow()
    config = SpectrumCacheConfig.for_pony_v7_variable(30, profile="p1")
    with pytest.raises(ValueError, match="variable-N research profile"):
        model.enable_cache(replace(config, denoiser_call_topology="auto"))
