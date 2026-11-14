"""Tiny CPU tests for the opt-in Z-Image Base and Turbo enable paths.

The 6B weights are not needed to exercise the 30-block hook boundary and
logical-step schedule. This follows the synthetic-model pattern used by the
Krea 2 Raw hook tests.
"""

import types
from dataclasses import replace

import pytest
import torch

from diffusers.hooks.hooks import HookRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel


MODEL_SIGNATURE = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": 16,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "cap_feat_dim": 2560,
    "siglip_feat_dim": None,
}


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 0.125


def make_model():
    model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)
    torch.nn.Module.__init__(model)
    model.register_to_config(**MODEL_SIGNATURE)
    model.layers = torch.nn.ModuleList(TinyBlock() for _ in range(30))

    def tiny_forward(self, x, t, cap_feats, **kwargs):
        output = x[0]
        for block in self.layers:
            output = block(output)
        return (output,)

    model.forward = types.MethodType(tiny_forward, model)
    model.eval()
    return model


@pytest.mark.parametrize(
    "config,forecast_steps",
    [
        (SpectrumCacheConfig.for_zimage_turbo(), (3, 5, 6)),
        (
            SpectrumCacheConfig.for_zimage_base_variable(28),
            (9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23),
        ),
    ],
    ids=("turbo-8-p3_356", "base-28-p12"),
)
def test_enable_runs_qualified_schedule_and_disable(config, forecast_steps):
    model = make_model()
    assert not model.is_cache_enabled
    model.enable_cache(config)
    assert model.is_cache_enabled
    assert HookRegistry.check_if_exists_or_initialize(model.layers[0]).get_hook("spectrum_cache_head_block")

    captions = [torch.ones(5, 8)]
    with torch.no_grad():
        for step in range(config.num_inference_steps):
            with model.cache_context("t2i", step_index=step, num_inference_steps=config.num_inference_steps):
                model(
                    x=[torch.full((1, 2, 8), 1.0 + 0.01 * step)],
                    t=torch.tensor([step / config.num_inference_steps]),
                    cap_feats=captions,
                    return_dict=False,
                )

    summary = HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser").latest_summary
    expected_actual = config.num_inference_steps - len(forecast_steps)
    assert summary["guard_latched"] is False, summary["guard_reasons"]
    assert summary["coordinate"]["failure_latched"] is False
    assert summary["prediction_failure_latched"] is False, summary["predict_failures"]
    assert summary["forecast_steps"] == list(forecast_steps)
    assert summary["prediction_call_count"] == len(forecast_steps)
    assert summary["full_call_count"] == expected_actual
    assert all(block.calls == expected_actual for block in model.layers)

    model.disable_cache()
    assert not model.is_cache_enabled
    assert HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser") is None


@pytest.mark.parametrize("steps", (20, 28, 40))
def test_base_variable_step_gate_preserved(steps):
    model = make_model()
    model.enable_cache(SpectrumCacheConfig.for_zimage_base_variable(steps))
    assert model.is_cache_enabled
    model.disable_cache()


@pytest.mark.parametrize(
    "config",
    [
        replace(SpectrumCacheConfig.for_zimage_turbo(), warmup_steps=2),
        replace(SpectrumCacheConfig.for_zimage_turbo(), forecast_step_indices=(3, 5)),
        SpectrumCacheConfig(
            num_inference_steps=9,
            coordinate_policy="runtime_horizon_normalized",
            forecast_step_indices=(3,),
        ),
    ],
    ids=("turbo-warmup-changed", "turbo-schedule-changed", "unqualified-nine-steps"),
)
def test_unqualified_short_step_profiles_rejected(config):
    model = make_model()
    with pytest.raises(ValueError, match="20–40-step Base profile or the exact 8-step Turbo"):
        model.enable_cache(config)
    assert not model.is_cache_enabled


def test_turbo_nonstandard_architecture_rejected():
    model = make_model()
    model.register_to_config(n_layers=29)
    with pytest.raises(ValueError, match="standard 30-block T2I architecture"):
        model.enable_cache(SpectrumCacheConfig.for_zimage_turbo())
    assert not model.is_cache_enabled
