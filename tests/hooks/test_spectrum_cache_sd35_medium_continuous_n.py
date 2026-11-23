"""CPU guard and schedule checks for a separate SD3.5 Medium integer-N route.

This tests mechanism, not quality at unobserved conditions or production suitability.
"""
from dataclasses import replace

import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_sd35 import SD35SpectrumGate
from tests.hooks.test_spectrum_cache_sd35 import _run_gate


MEDIUM = SpectrumCacheConfig.for_sd35_medium_continuous_n
QUALIFIED_SCHEDULES = {
    12: (6, 8),
    16: (6, 8, 10, 12),
    28: (5, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24),
    48: (9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43),
}


@pytest.mark.parametrize("n", [1, 2, 3, 8, 9, 10, 11, 12, 13, 16, 19, 20, 24, 28, 32, 40, 48, 50, 60, 80, 81, 100])
def test_medium_continuous_n_accepts_positive_integers_with_bounded_forecast(n):
    config = MEDIUM(n)
    assert config.num_inference_steps == n
    assert config.allow_sd35_medium_continuous_n
    assert config.allow_sd35_variable_n
    assert not config.allow_sd35_large_relayout_n
    assert config.coordinate_policy == "runtime_index_normalized"
    assert config.predictor_backend == "chunked"
    assert 1 <= config.tail_actual_steps <= n
    if n in QUALIFIED_SCHEDULES:
        expected = QUALIFIED_SCHEDULES[n]
    else:
        first = max(6, (5*n + 27)//28)
        expected = tuple(range(first, n-config.tail_actual_steps, 2)) if 12 <= n <= 80 else ()
    assert config.forecast_step_indices == expected
    assert all(config.warmup_steps <= x < n - config.tail_actual_steps
               for x in config.forecast_step_indices)
    assert SD35SpectrumGate(config).forecast_steps == frozenset(config.forecast_step_indices)


@pytest.mark.parametrize("n", [1, 8, 12, 16, 20, 28, 32, 48, 49, 60, 80, 100])
def test_medium_continuous_n_cpu_real_forecast_and_drift(n):
    config = MEDIUM(n)
    p = len(config.forecast_step_indices)
    assert _run_gate(config, steps=n) == {"real": n-p, "forecast": p, "fallback": 0}
    assert _run_gate(config, steps=n+1) == {"real": n+1, "forecast": 0, "fallback": 0}


def test_medium_continuous_n_preserves_old_source_and_off_default():
    old = SpectrumCacheConfig.for_sd35_medium_variable(28)
    new = MEDIUM(28)
    assert replace(new, allow_sd35_medium_continuous_n=False) == old
    assert SpectrumCacheConfig.for_sd35_medium().forecast_step_indices == QUALIFIED_SCHEDULES[28]
    assert not SpectrumCacheConfig().allow_sd35_medium_continuous_n
    assert not SpectrumCacheConfig().allow_sd35_large_relayout_n


@pytest.mark.parametrize("bad", [None, True, False, 0, -1, 1.0, 12.0, "28"])
def test_medium_continuous_n_rejects_nonpositive_or_nonnumeric(bad):
    with pytest.raises(ValueError, match="positive integer N"):
        MEDIUM(bad)


def test_medium_continuous_n_invalid_mutations_are_rejected():
    cfg = MEDIUM(48)
    for bad in (
        replace(cfg, forecast_step_indices=cfg.forecast_step_indices[:-1]),
        replace(cfg, coordinate_policy="legacy_fixed_max"),
        replace(cfg, blend_w=0.6),
        replace(cfg, allow_sd35_medium_continuous_n=False),
        replace(cfg, allow_sd35_large_relayout_n=True),
    ):
        with pytest.raises(ValueError):
            SD35SpectrumGate(bad)
    # The overlapping tail is rejected at config construction, before Gate.
    with pytest.raises(ValueError, match="forecast_step_indices cannot overlap tail_actual_steps"):
        replace(cfg, tail_actual_steps=cfg.tail_actual_steps + 1)


def test_medium_continuous_n_bad_prediction_fails_closed():
    cfg = MEDIUM(48)
    assert _run_gate(cfg, steps=48, inject_failure=True) == {
        "real": 48, "forecast": 0, "fallback": 1,
    }
