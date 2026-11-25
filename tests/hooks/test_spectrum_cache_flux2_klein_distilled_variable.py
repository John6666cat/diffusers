import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule

from .test_spectrum_cache_flux2 import make_inputs, make_model


def test_klein_distilled_variable_configs():
    configs = []
    for n in range(1, 81):
        config = SpectrumCacheConfig.for_flux2_klein_distilled4b_variable(n)
        schedule = SpectrumSchedule(config)
        steps = tuple(i for i in range(n) if not schedule.decide(i))
        assert steps == config.forecast_step_indices
        assert tuple(sorted(set(steps))) == steps
        assert all(2 <= i < n - 1 for i in steps)
        assert all(b - a >= 2 for a, b in zip(steps, steps[1:]))
        if n <= 4 or n == 12:
            assert steps == ()
        configs.append(asdict(config))
    encoded = json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == "8f3fd385a5d3e1b09ccd215c856418ad34c7a8d0447626b641f40a8e3e99078d"
    assert SpectrumCacheConfig().forecast_step_indices is None


@pytest.mark.parametrize("n", [0, -1, 81, 12.0, True, "12", None])
def test_klein_distilled_variable_invalid_n(n):
    with pytest.raises(ValueError):
        SpectrumCacheConfig.for_flux2_klein_distilled4b_variable(n)


@pytest.mark.parametrize("n", [4, 6, 10, 12, 14, 40, 80])
@torch.no_grad()
def test_klein_distilled_variable_native_cpu(n):
    model = make_model()
    config = SpectrumCacheConfig.for_flux2_klein_distilled4b_variable(n)
    assert not model.is_cache_enabled
    try:
        model.enable_cache(config)
        for step in range(n):
            inputs = make_inputs()
            inputs["timestep"] = torch.tensor([1.0 - step / n])
            with model.cache_context("cond"):
                assert torch.isfinite(model(**inputs).sample).all()
        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["forecast_steps"] == list(config.forecast_step_indices)
        assert summary["prediction_call_count"] == len(config.forecast_step_indices)
        assert summary["full_call_count"] == n - len(config.forecast_step_indices)
        assert not summary["guard_latched"] and not summary["predict_failures"]
    finally:
        if model.is_cache_enabled:
            model.disable_cache()
        assert not model.is_cache_enabled
