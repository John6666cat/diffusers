"""STAGED, research-only SD3.5 Large/Medium variable-N CPU gates."""

from dataclasses import replace

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_sd35 import SD35SpectrumGate
from tests.hooks.test_spectrum_cache_sd35 import P13, _run_gate


@pytest.mark.parametrize("factory", [
    SpectrumCacheConfig.for_sd35_large_variable,
    SpectrumCacheConfig.for_sd35_medium_variable,
])
def test_sd35_variable_preserves_legacy_p13_anchor(factory):
    candidate = factory(28)
    legacy = SpectrumCacheConfig.for_sd35_large()
    assert replace(candidate, allow_sd35_variable_n=False) == legacy
    assert candidate.forecast_step_indices == P13
    assert candidate.allow_sd35_variable_n is True


@pytest.mark.parametrize("n", [20, 24, 28, 32, 36, 40])
def test_sd35_variable_mask_invariants_and_family_parity(n):
    large = SpectrumCacheConfig.for_sd35_large_variable(n)
    medium = SpectrumCacheConfig.for_sd35_medium_variable(n)
    assert large == medium
    assert large.coordinate_policy == "runtime_index_normalized"
    assert large.predictor_backend == "chunked"
    assert large.degree == 2 and large.ridge_lambda == 0.1 and large.blend_w == 0.5
    assert large.num_inference_steps == n
    assert large.warmup_steps >= 4 and large.tail_actual_steps >= 2
    assert len(large.forecast_step_indices) > 0
    assert all(large.warmup_steps <= step < n - large.tail_actual_steps for step in large.forecast_step_indices)
    assert tuple(sorted(set(large.forecast_step_indices))) == large.forecast_step_indices
    assert all(
        not {step, step + 1, step + 2}.issubset(large.forecast_step_indices)
        for step in range(n - 2)
    )


@pytest.mark.parametrize("n", [20, 24, 28, 32, 36, 40])
@pytest.mark.parametrize("factory", [
    SpectrumCacheConfig.for_sd35_large_variable,
    SpectrumCacheConfig.for_sd35_medium_variable,
])
def test_sd35_variable_native_cpu_forecasts_expected_calls(n, factory):
    config = factory(n)
    assert _run_gate(config, steps=n) == {
        "real": n - len(config.forecast_step_indices),
        "forecast": len(config.forecast_step_indices),
        "fallback": 0,
    }


@pytest.mark.parametrize("n", [20, 28, 32, 40])
def test_sd35_variable_horizon_drift_fails_closed(n):
    config = SpectrumCacheConfig.for_sd35_large_variable(n)
    counts = _run_gate(config, steps=n + 1)
    assert counts == {"real": n + 1, "forecast": 0, "fallback": 0}


@pytest.mark.parametrize("bad", [True, False, 19, 41, 28.0, "28", None])
def test_sd35_variable_rejects_invalid_n(bad):
    with pytest.raises(ValueError, match="integer N"):
        SpectrumCacheConfig.for_sd35_large_variable(bad)


def test_sd35_variable_mutated_staged_profile_is_rejected():
    config = SpectrumCacheConfig.for_sd35_medium_variable(32)
    for bad in (
        replace(config, forecast_step_indices=config.forecast_step_indices[:-1]),
        replace(config, coordinate_policy="legacy_fixed_max"),
        replace(config, blend_w=0.6),
        replace(config, predictor_backend="fp32_weighted"),
        replace(config, tail_actual_steps=2),
    ):
        with pytest.raises(ValueError, match="exact staged profile"):
            SD35SpectrumGate(bad)


def test_sd35_variable_invalid_prediction_falls_back_to_real():
    config = SpectrumCacheConfig.for_sd35_medium_variable(28)
    counts = _run_gate(config, steps=28, inject_failure=True)
    assert counts == {"real": 28, "forecast": 0, "fallback": 1}


def test_sd35_variable_default_off_and_legacy_factories_stay_unchanged():
    large = SpectrumCacheConfig.for_sd35_large()
    medium = SpectrumCacheConfig.for_sd35_medium()
    assert large == medium
    assert not large.allow_sd35_variable_n
    assert large.forecast_step_indices == P13
    assert not SpectrumCacheConfig().allow_sd35_variable_n
