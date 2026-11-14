"""Tiny CPU mechanics for the opt-in Krea 2 Raw research Source adapter.

The real 28-block model is intentionally not constructed: a synthetic instance
checks paired contexts, per-lane accounting, bypass and block interception.
"""

import types

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel


OLD_TEN = (7, 10, 21, 28, 30, 32, 35, 37, 42, 44)
P32 = (7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 21, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 37, 39, 40, 41, 42, 43, 44, 45, 47, 49)
MODEL_SIGNATURE = {
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": (32, 48, 48),
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, temb=None, image_rotary_emb=None, attention_mask=None):
        self.calls += 1
        return hidden_states + 0.125


def make_model():
    model = Krea2Transformer2DModel.__new__(Krea2Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.register_to_config(**MODEL_SIGNATURE)
    model.transformer_blocks = torch.nn.ModuleList(TinyBlock() for _ in range(28))

    def tiny_forward(
        self, hidden_states, encoder_hidden_states, timestep, position_ids,
        encoder_attention_mask=None, attention_kwargs=None, return_dict=True,
    ):
        result = hidden_states
        for block in self.transformer_blocks:
            result = block(result)
        return (result,)

    model.forward = types.MethodType(tiny_forward, model)
    model.eval()
    return model


def config(forecasts=OLD_TEN):
    return SpectrumCacheConfig(
        num_inference_steps=52,
        forecast_step_indices=forecasts,
        denoiser_call_topology="paired_cfg",
        coordinate_policy="runtime_horizon_normalized",
        degree=2,
        history_limit=8,
    )


@pytest.mark.parametrize("forecasts", (OLD_TEN, P32), ids=("old10", "p32"))
def test_paired_cfg_forecast_counts_and_lane_separation(forecasts):
    model = make_model()
    model.enable_cache(config(forecasts))
    # Conditioning embeds must retain tensor identity across all steps.
    positive = torch.ones(1, 3, 12, 4)
    negative = torch.full((1, 3, 12, 4), 2.0)
    ids = torch.zeros(5, 3)
    with torch.no_grad():
        for step in range(52):
            for embed in (positive, negative):
                with model.cache_context("krea", step_index=step, num_inference_steps=52):
                    model(
                        hidden_states=torch.ones(1, 2, 4) + step,
                        encoder_hidden_states=embed,
                        timestep=torch.ones(1),
                        position_ids=ids,
                        return_dict=False,
                    )

    hook = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    report = hook.latest_summary
    assert report["guard_latched"] is False, report["guard_reasons"]
    assert report["forecast_steps"] == list(forecasts)
    assert report["prediction_call_count"] == 2 * len(forecasts)
    assert report["lane_prediction_call_count"] == {"positive": len(forecasts), "negative": len(forecasts)}
    assert report["full_call_count"] == 2 * (52 - len(forecasts))
    assert sum(block.calls for block in model.transformer_blocks) == 28 * 2 * (52 - len(forecasts))
    model.disable_cache()


def test_forecast_off_matches_tiny_reference():
    model = make_model()
    sample = torch.ones(1, 2, 4)
    args = dict(
        hidden_states=sample, encoder_hidden_states=torch.ones(1, 3, 12, 4),
        timestep=torch.ones(1), position_ids=torch.zeros(5, 3), return_dict=False,
    )
    with torch.no_grad():
        reference = model(**args)[0]
    model.enable_cache(config(()))
    with torch.no_grad(), model.cache_context("krea", step_index=0, num_inference_steps=52):
        result = model(**args)[0]
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    model.disable_cache()


def test_wrong_native_step_fails_closed():
    model = make_model()
    model.enable_cache(config())
    with torch.no_grad(), model.cache_context("krea", step_index=1, num_inference_steps=52):
        model(
            hidden_states=torch.ones(1, 2, 4),
            encoder_hidden_states=torch.ones(1, 3, 12, 4),
            timestep=torch.ones(1), position_ids=torch.zeros(5, 3), return_dict=False,
        )
    report = model._diffusers_hook.get_hook("spectrum_cache_denoiser").latest_summary
    assert report["guard_latched"] is True
    assert report["prediction_call_count"] == 0
    assert "context" in report["guard_reasons"][0]["reason"]
    model.disable_cache()
