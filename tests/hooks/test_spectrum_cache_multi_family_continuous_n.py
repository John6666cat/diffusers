"""Bounded integer-N research factories; no image-quality qualification is inferred."""

import hashlib
import json
from dataclasses import asdict

import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule


PROFILES = (
    "flux1", "klein", "neta_p6", "neta_p9", "neta_p30", "pony_p1", "pony_p4", "pony_p8"
)
HISTORICAL_DIGESTS = {
    "flux1": "c4a2582b8fdff980b7b366e9b238e6131663fea293468f19488bb6587190c6db",
    "klein": "a44aef1a5731a3acb6bee87ba73ee71abae77c2356dcc71b54d0b48f19c8bc1f",
    "neta_p6": "a1fcaab2135563a3a54585d3f4dbe336ad6d47b00cbe82310a974568131551cd",
    "neta_p9": "55e7c50aa155a0aa18d4e30b6e35bf0a4d0916d55fa6a2d7106d3bb5db2b85f5",
    "neta_p30": "aef076639224054a2765069e8101fb61b2b0005d0bec08361556703231687ae4",
    "pony_p1": "e7f7ac410b05159c23375c774392d7fc6382acfec8cbbefc18e17c9ce87de924",
    "pony_p4": "d33645194d70f3b1f1dd6054d817ced1e563091c7bf7e00e0c17dc490cea7b67",
    "pony_p8": "23afcd942a3aa05ec0d4977d84583da16876eb38df4c7baebb8357a10b8bfb37",
}


def _build(family, n, backend="dense"):
    if family == "flux1":
        return SpectrumCacheConfig.for_flux1_dev_variable(n, predictor_backend=backend)
    if family == "klein":
        return SpectrumCacheConfig.for_flux2_klein_base4b_variable(n, predictor_backend=backend)
    if family.startswith("neta_"):
        return SpectrumCacheConfig.for_netayume_variable(
            n, profile=family.split("_", 1)[1], predictor_backend=backend
        )
    return SpectrumCacheConfig.for_pony_v7_variable(
        n, profile=family.split("_", 1)[1], predictor_backend=backend
    )


def _historical_range(family):
    return range(24, 41) if family.startswith("pony_") else range(40, 61)


@pytest.mark.parametrize("family", PROFILES)
def test_all_integer_n_and_protected_new_masks(family):
    for backend in ("dense", "fp32_weighted"):
        for n in range(1, 81):
            config = _build(family, n, backend)
            forecasts = config.forecast_step_indices
            assert config.num_inference_steps == n
            assert forecasts == tuple(sorted(set(forecasts)))
            assert all(0 <= step < n for step in forecasts)
            assert tuple(i for i in range(n) if not SpectrumSchedule(config).decide(i)) == forecasts
            if n <= 11:
                assert forecasts == ()
            elif n not in _historical_range(family):
                if family == "flux1":
                    first_real, last_real, cap = 5, 2, 4
                elif family == "klein":
                    first_real, last_real, cap = 6, 4, None
                elif family.startswith("neta_"):
                    first_real, last_real, cap = 3, 2, 4
                else:
                    first_real, last_real, cap = 3, 2, 2
                assert set(forecasts).isdisjoint(range(first_real))
                assert set(forecasts).isdisjoint(range(n - last_real, n))
                if cap is not None:
                    mask = "".join("1" if i in forecasts else "0" for i in range(n))
                    assert max(map(len, mask.split("0"))) <= cap


@pytest.mark.parametrize("family", PROFILES)
def test_historical_factory_configs_still_byte_for_byte_equivalent(family):
    # Digest pinned to every dataclass field from the source parent
    # fc5c38805c16b7b126aa56de27eaf246cc7fa8d8's immediate child 3f4793b4.
    checksum = hashlib.sha256()
    for backend in ("dense", "fp32_weighted"):
        for n in _historical_range(family):
            config = _build(family, n, backend)
            checksum.update(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode())
            checksum.update(b"\n")
    assert checksum.hexdigest() == HISTORICAL_DIGESTS[family]


@pytest.mark.parametrize("family", PROFILES)
def test_invalid_horizon_and_unsupported_backend_rejected(family):
    for bad in (True, False, 0, -1, 81, 50.0, "50", None):
        with pytest.raises((ValueError, TypeError)):
            _build(family, bad)
    with pytest.raises(ValueError):
        _build(family, 24, "chunked")


def test_historical_fixed_anchors_are_not_replaced():
    assert SpectrumCacheConfig.for_flux1_dev_variable(50) == SpectrumCacheConfig.for_flux1_dev()
    assert SpectrumCacheConfig.for_flux2_klein_base4b_variable(50) == (
        SpectrumCacheConfig.for_flux2_klein_base4b()
    )
    assert SpectrumCacheConfig.for_netayume_variable(50).allow_netayume_variable_n
    assert SpectrumCacheConfig.for_pony_v7_variable(30).allow_pony_v7_variable_n
