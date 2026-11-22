"""Limited Large relayout Source profile: schedule, exact opt-in and old-route isolation."""

from dataclasses import replace

import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_sd35 import SD35SpectrumGate
from tests.hooks.test_spectrum_cache_sd35 import _run_gate


RELAYOUT_EXPECTED = {
    20: (7, 9, 11, 13, 15, 17),
    24: (7, 9, 11, 13, 15, 17, 19, 21),
    28: (7, 9, 11, 13, 17, 19, 21, 23, 25),
    32: (7, 9, 11, 13, 15, 19, 21, 23, 25, 27),
    40: (9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 31, 33, 35),
    48: (10, 12, 14, 16, 18, 20, 22, 24, 28, 30, 32, 34, 36, 38, 40, 42),
}


@pytest.mark.parametrize("n,indices", RELAYOUT_EXPECTED.items())
def test_exact_staged_relayout_masks(n, indices):
    config = SpectrumCacheConfig.for_sd35_large_relayout(n)
    assert config.forecast_step_indices == indices
    assert config.allow_sd35_large_relayout_n and config.allow_sd35_variable_n
    assert config.predictor_backend == "chunked"
    assert config.coordinate_policy == "runtime_index_normalized"
    assert len(indices) == len(set(indices))
    assert all(b-a >= 2 for a,b in zip(indices,indices[1:]))
    assert all(config.warmup_steps <= i < n-config.tail_actual_steps for i in indices)
    assert SD35SpectrumGate(config).forecast_steps == frozenset(indices)


@pytest.mark.parametrize("n,indices", RELAYOUT_EXPECTED.items())
def test_exact_cpu_calls_and_drift_fallback(n, indices):
    config = SpectrumCacheConfig.for_sd35_large_relayout(n)
    assert _run_gate(config, steps=n) == {
        "real": n-len(indices), "forecast": len(indices), "fallback": 0,
    }
    assert _run_gate(config, steps=n+1) == {"real": n+1, "forecast": 0, "fallback": 0}


@pytest.mark.parametrize("n", [28, 48])
def test_failed_prediction_uses_full_compute(n):
    config = SpectrumCacheConfig.for_sd35_large_relayout(n)
    assert _run_gate(config, steps=n, inject_failure=True) == {"real": n, "forecast": 0, "fallback": 1}


@pytest.mark.parametrize("bad", [True, False, 0, 9, 10, 12, 16, 19, 21, 25, 36, 41, 47, 50, 60, 72, 100, 28.0, "28", None])
def test_unqualified_n_rejected(bad):
    with pytest.raises(ValueError, match="Owner-reviewed integer N"):
        SpectrumCacheConfig.for_sd35_large_relayout(bad)


def test_mutated_profile_rejected_and_other_factories_unchanged():
    conf = SpectrumCacheConfig.for_sd35_large_relayout(28)
    for bad in (
        replace(conf, forecast_step_indices=conf.forecast_step_indices[:-1]),
        replace(conf, allow_sd35_large_relayout_n=False),
        replace(conf, allow_sd35_variable_n=False),
        replace(conf, blend_w=0.6),
        replace(conf, predictor_backend="dense"),
    ):
        with pytest.raises(ValueError):
            SD35SpectrumGate(bad)
    # The overlapping tail is rejected at config construction, before Gate.
    with pytest.raises(ValueError, match="forecast_step_indices cannot overlap tail_actual_steps"):
        replace(conf, tail_actual_steps=conf.tail_actual_steps + 1)
    assert not SpectrumCacheConfig().allow_sd35_large_relayout_n
    assert not SpectrumCacheConfig.for_sd35_large().allow_sd35_large_relayout_n
    assert not SpectrumCacheConfig.for_sd35_medium().allow_sd35_large_relayout_n
    assert SpectrumCacheConfig.for_sd35_large().forecast_step_indices == (
        5, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24,
    )
    for n in (20, 24, 28, 32, 40):
        assert not SpectrumCacheConfig.for_sd35_medium_variable(n).allow_sd35_large_relayout_n
        assert not SpectrumCacheConfig.for_sd35_large_variable(n).allow_sd35_large_relayout_n
