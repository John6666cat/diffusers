"""Selected Z-Image Base LATE50 and Turbo opt-in N1–80 profiles."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from diffusers.hooks.hooks import HookRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule

from .test_spectrum_cache_zimage_enable import make_model


def _digest(configs):
    return hashlib.sha256(json.dumps(configs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_selected_configs_and_legacy_exact():
    selected, legacy = [], []
    for n in range(1, 81):
        legacy.append(asdict(SpectrumCacheConfig.for_zimage_base_variable(n)))
        base = SpectrumCacheConfig.for_zimage_base_late50(n)
        turbo = SpectrumCacheConfig.for_zimage_turbo_variable(n)
        for cfg in (base, turbo):
            schedule = SpectrumSchedule(cfg)
            mask = tuple(i for i in range(n) if not schedule.decide(i))
            assert mask == cfg.forecast_step_indices
            assert all(cfg.warmup_steps <= i < n - cfg.tail_actual_steps for i in mask)
            selected.append(asdict(cfg))
        assert all(i + 1 not in base.forecast_step_indices for i in base.forecast_step_indices)
        assert all(not {i, i + 1, i + 2}.issubset(turbo.forecast_step_indices) for i in turbo.forecast_step_indices)
        if n <= 11:
            assert base.forecast_step_indices == ()
        if n <= 5:
            assert turbo.forecast_step_indices == ()
    anchor = SpectrumCacheConfig.for_zimage_turbo()
    legacy.extend([asdict(SpectrumCacheConfig.for_zimage_base_variable(28, profile="p8_spread")), asdict(anchor)])
    assert SpectrumCacheConfig.for_zimage_turbo_variable(8) == anchor
    assert _digest(selected) == "e79a6534d96b6b2571e4d2b271b61441cb58078a01ba0eee4a5907b4c0ede712"
    assert _digest(legacy) == "f8ca2381e6a4d892a342dd59aee9bb258a6d8c845be8f64d649aa08cf0398676"


@pytest.mark.parametrize("factory", (SpectrumCacheConfig.for_zimage_base_late50, SpectrumCacheConfig.for_zimage_turbo_variable))
def test_selected_invalid_n(factory):
    for n in (0, -1, 81, True, False, 12.0, "12", None):
        with pytest.raises(ValueError):
            factory(n)


@pytest.mark.parametrize(
    "route,n",
    [("base", n) for n in (1, 11, 12, 16, 28, 50, 80)] + [("turbo", n) for n in (1, 5, 6, 8, 12, 24, 40, 80)],
)
@torch.no_grad()
def test_selected_native_cpu(route, n):
    factory = SpectrumCacheConfig.for_zimage_base_late50 if route == "base" else SpectrumCacheConfig.for_zimage_turbo_variable
    cfg = factory(n)
    model = make_model()
    assert not model.is_cache_enabled
    try:
        model.enable_cache(cfg)
        for i in range(n):
            with model.cache_context("cond_uncond", step_index=i, num_inference_steps=n):
                model(x=[torch.full((1, 2, 8), 1.0 + 0.01 * i)], t=torch.tensor([i / n]),
                      cap_feats=[torch.ones(5, 8)], return_dict=False)
        summary = HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser").latest_summary
        mask = list(cfg.forecast_step_indices)
        assert summary["forecast_steps"] == mask
        assert summary["prediction_call_count"] == len(mask) and summary["full_call_count"] == n - len(mask)
        assert not summary["guard_latched"] and not summary["prediction_failure_latched"]
        assert not summary["coordinate"]["failure_latched"]
        assert all(block.calls == n - len(mask) for block in model.layers)
    finally:
        if model.is_cache_enabled:
            model.disable_cache()
    assert not model.is_cache_enabled


@pytest.mark.parametrize("n", (12, 50))
@pytest.mark.parametrize("factory", (SpectrumCacheConfig.for_zimage_base_late50, SpectrumCacheConfig.for_zimage_turbo_variable))
def test_selected_rogue_config_rejected_outside_historical_range(n, factory):
    model = make_model()
    with pytest.raises(ValueError, match="exact N1–80 research Base"):
        model.enable_cache(replace(factory(n), blend_w=0.25))
    assert not model.is_cache_enabled
