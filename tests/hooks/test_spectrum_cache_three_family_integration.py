"""Cross-family SPECTRUM hook lifetime checks for the staged source merge."""

import pytest

from diffusers.hooks.hooks import HookRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from tests.hooks.test_spectrum_cache_flux2 import make_config, make_model
from tests.hooks.test_spectrum_cache_netayume import CACHE_KWARGS, _tiny_lumina2
from tests.hooks.test_spectrum_cache_pony_v7 import _tiny_qualified_auraflow, pony_cfg


@pytest.mark.parametrize(
    "disable_order",
    [("netayume", "pony", "flux2"), ("pony", "netayume", "flux2")],
)
def test_staged_families_keep_separate_hook_lifetimes(disable_order):
    models = {
        "flux2": make_model(),
        "netayume": _tiny_lumina2(),
        "pony": _tiny_qualified_auraflow(),
    }
    configs = {
        "flux2": make_config(),
        "netayume": SpectrumCacheConfig(**CACHE_KWARGS),
        "pony": pony_cfg(),
    }

    for family, model in models.items():
        model.enable_cache(configs[family])

    for family in disable_order:
        model = models[family]
        model.disable_cache()
        assert not model.is_cache_enabled
        for other, remaining in models.items():
            if other in disable_order[: disable_order.index(family) + 1]:
                continue
            assert remaining.is_cache_enabled
            registry = HookRegistry.check_if_exists_or_initialize(remaining)
            assert registry.get_hook("spectrum_cache_denoiser") is not None

    assert all(not model.is_cache_enabled for model in models.values())
