"""SD3.5 Large and Medium continuous N research; no new-N image quality is asserted."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_sd35 import SD35SpectrumGate
from tests.hooks.test_spectrum_cache_sd35 import _run_gate


LARGE_ANCHOR_DIGESTS = {
    20: "ac681e8fae5e715a1cfe85c698d4d9255f5544a4fa6a19d4615f85764eac0236",
    24: "be9e4b13d72b48ca792295e753c9a09955504b00d5a34d859634e871e7c2e74e",
    28: "c654a2ce6bb00b4d6f55093f46343c67ff858f9b44b07b585dea2f5da0cbe13b",
    32: "7d4ad84f6020fc311196019826ce9ef9f85414e4f8fc601a5c04552dcebb37b2",
    40: "68d798027e905ee811f7d3c15bcf8c15159da168fa86a7899a202f1a491affcf",
    48: "78e0d570f43a9dcaa44b9e6664407a9d17731bd36bcea9e08eb56ec4bfc5c61e",
}
MEDIUM_ANCHOR_DIGESTS = {
    12: "ff914443b8636f6d701ad130bcbcf887a6e96395e7d6b90a280265d91a3ad743",
    16: "c8c49af92b0449310dfcd4c16e33263d3aaf82a0ff90a5710534c3d41e31ec12",
    28: "a2b592a10cb88e18e2b36a9aef1dabec41aaff2a185cb2759318ffa8d7e9f9a9",
    48: "29b9e3108d18232a4c80cc5fc80acff1aa1c028b955c1bdcd478ada7eaf18ed9",
}


def _digest(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.parametrize("n,digest", LARGE_ANCHOR_DIGESTS.items())
def test_large_six_owner_relayout_anchors_unchanged(n, digest):
    assert _digest(SpectrumCacheConfig.for_sd35_large_relayout(n)) == digest


@pytest.mark.parametrize("n,digest", MEDIUM_ANCHOR_DIGESTS.items())
def test_medium_four_source_masks_unchanged(n, digest):
    assert _digest(SpectrumCacheConfig.for_sd35_medium_continuous_n(n)) == digest


@pytest.mark.parametrize("family", ("large", "medium"))
def test_all_n1_80_schedule_guards(family):
    factory = (SpectrumCacheConfig.for_sd35_large_relayout if family == "large"
               else SpectrumCacheConfig.for_sd35_medium_continuous_n)
    for n in range(1, 81):
        config = factory(n)
        indices = config.forecast_step_indices
        assert config.num_inference_steps == n
        assert indices == tuple(sorted(set(indices)))
        assert all(0 <= i < n for i in indices)
        assert config.predictor_backend == "chunked"
        assert config.coordinate_policy == "runtime_index_normalized"
        assert config.allow_sd35_variable_n
        assert config.allow_sd35_large_relayout_n == (family == "large")
        assert config.allow_sd35_medium_continuous_n == (family == "medium")
        assert all(config.warmup_steps <= i < n - config.tail_actual_steps for i in indices)
        assert all(b - a > 1 for a, b in zip(indices, indices[1:])) or n == 28 and family == "medium"
        if family == "large":
            assert (len(indices) == 0) == (n <= 19)
            assert len(indices) <= n // 3
        elif n <= 11:
            assert indices == ()
        else:
            assert indices
        if n == 1:
            assert config.tail_actual_steps == 1
        else:
            assert config.tail_actual_steps >= 2
        assert SD35SpectrumGate(config).forecast_steps == frozenset(indices)


@pytest.mark.parametrize("n", (1, 2, 11, 12, 16, 19, 20, 21, 27, 28, 29, 36, 40, 41, 48, 49, 60, 80))
@pytest.mark.parametrize("family", ("large", "medium"))
def test_real_call_counts_and_horizon_drift(n, family):
    config = (SpectrumCacheConfig.for_sd35_large_relayout(n) if family == "large"
              else SpectrumCacheConfig.for_sd35_medium_continuous_n(n))
    prediction_count = len(config.forecast_step_indices)
    assert _run_gate(config, steps=n) == {
        "real": n - prediction_count, "forecast": prediction_count, "fallback": 0
    }
    assert _run_gate(config, steps=n + 1) == {
        "real": n + 1, "forecast": 0, "fallback": 0
    }


@pytest.mark.parametrize("family,n", (("large", 21), ("large", 80), ("medium", 13), ("medium", 80)))
def test_bad_prediction_latches_real_compute(family, n):
    config = (SpectrumCacheConfig.for_sd35_large_relayout(n) if family == "large"
              else SpectrumCacheConfig.for_sd35_medium_continuous_n(n))
    assert _run_gate(config, steps=n, inject_failure=True) == {
        "real": n, "forecast": 0, "fallback": 1
    }


@pytest.mark.parametrize("family,n", (("large", 21), ("large", 80), ("medium", 13), ("medium", 80)))
def test_new_n_canonical_profile_only(family, n):
    config = (SpectrumCacheConfig.for_sd35_large_relayout(n) if family == "large"
              else SpectrumCacheConfig.for_sd35_medium_continuous_n(n))
    for bad in (
        replace(config, forecast_step_indices=config.forecast_step_indices[:-1]),
        replace(config, allow_sd35_variable_n=False),
        replace(config, blend_w=0.7),
        replace(config, predictor_backend="dense"),
    ):
        with pytest.raises(ValueError):
            SD35SpectrumGate(bad)


def test_low_n_and_beyond80_preserve_full_compute_and_legacy_isolation():
    for n in (1, 7, 11, 12, 16, 19):
        assert SpectrumCacheConfig.for_sd35_large_relayout(n).forecast_step_indices == ()
    for n in (1, 7, 11, 81, 100, 256):
        assert SpectrumCacheConfig.for_sd35_medium_continuous_n(n).forecast_step_indices == ()
    for n in (0, -1, 81, True, 28.0):
        with pytest.raises(ValueError):
            SpectrumCacheConfig.for_sd35_large_relayout(n)
    assert SpectrumCacheConfig.for_sd35_large().forecast_step_indices == (
        5, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24
    )
    assert SpectrumCacheConfig.for_sd35_large_variable(28).forecast_step_indices == (
        5, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24
    )
    assert not SpectrumCacheConfig().allow_sd35_medium_continuous_n
    assert not SpectrumCacheConfig().allow_sd35_large_relayout_n
