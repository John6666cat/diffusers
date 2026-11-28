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


def test_variable_factory_keeps_historical_n52_p32_unchanged():
    # Exact research/Source recipe recorded after the Owner's P32 visual review.
    # Matching just the forecast mask would miss predictor-parameter drift.
    owner_p32 = SpectrumCacheConfig(
        num_inference_steps=52,
        denoiser_call_topology="paired_cfg",
        forecast_step_indices=P32,
        predictor_backend="dense",
        coordinate_policy="runtime_horizon_normalized",
        coordinate_max=100.0,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        warmup_steps=0,
        window_size=2.0,
        flex_window=0.0,
        tail_actual_steps=0,
    )
    assert SpectrumCacheConfig.for_krea2_raw_variable(52) == owner_p32


@pytest.mark.parametrize("num_steps", range(40, 61))
def test_variable_n_schedule_cpu_policy(num_steps):
    cfg = SpectrumCacheConfig.for_krea2_raw_variable(num_steps)
    forecasts = cfg.forecast_step_indices
    assert forecasts == tuple(sorted(set(forecasts)))
    assert len(forecasts) > 0
    assert forecasts[0] >= 7
    assert forecasts[-1] < num_steps - (0 if num_steps == 52 else 2)
    assert cfg.denoiser_call_topology == "paired_cfg"
    assert cfg.coordinate_policy == "runtime_horizon_normalized"
    assert cfg.predictor_backend == "dense"
    assert cfg.allow_krea2_raw_variable_n == (num_steps != 52)
    if num_steps != 52:
        assert len(forecasts) <= num_steps - (num_steps * 20 + 51) // 52
        binary = "".join("1" if step in forecasts else "0" for step in range(num_steps))
        assert max(map(len, binary.split("0"))) <= 7


@pytest.mark.parametrize("num_steps", (10, 20, 39, 40, 45, 50, 55, 60, 61, 80))
def test_variable_n_real_paired_cfg_cpu_mechanics(num_steps):
    model = make_model()
    cfg = SpectrumCacheConfig.for_krea2_raw_variable(num_steps)
    model.enable_cache(cfg)
    positive = torch.ones(1, 3, 12, 4)
    negative = torch.full((1, 3, 12, 4), 2.0)
    ids = torch.zeros(5, 3)
    with torch.no_grad():
        for step in range(num_steps):
            for embed in (positive, negative):
                with model.cache_context("krea", step_index=step, num_inference_steps=num_steps):
                    model(
                        hidden_states=torch.ones(1, 2, 4) + step,
                        encoder_hidden_states=embed,
                        timestep=torch.ones(1), position_ids=ids, return_dict=False,
                    )

    report = model._diffusers_hook.get_hook("spectrum_cache_denoiser").latest_summary
    expected = cfg.forecast_step_indices
    assert report["guard_latched"] is False, report["guard_reasons"]
    assert report["forecast_steps"] == list(expected)
    assert report["prediction_call_count"] == 2 * len(expected)
    assert report["lane_prediction_call_count"] == {"positive": len(expected), "negative": len(expected)}
    assert report["full_call_count"] == 2 * (num_steps - len(expected))
    assert sum(block.calls for block in model.transformer_blocks) == 28 * 2 * (num_steps - len(expected))
    model.disable_cache()


@pytest.mark.parametrize("invalid", (-1, 0, 81, 52.0, True))
def test_variable_n_rejects_invalid_horizon(invalid):
    with pytest.raises(ValueError, match="integer"):
        SpectrumCacheConfig.for_krea2_raw_variable(invalid)


def test_variable_n_requires_exact_opt_in_and_canonical_guard():
    from dataclasses import replace

    canonical = SpectrumCacheConfig.for_krea2_raw_variable(40)
    for cfg in (
        replace(canonical, allow_krea2_raw_variable_n=False),
        replace(canonical, forecast_step_indices=(7, 8)),
        replace(canonical, predictor_backend="fp32_weighted"),
        replace(canonical, degree=3),
        SpectrumCacheConfig(
            num_inference_steps=40, forecast_step_indices=(7, 8),
            denoiser_call_topology="paired_cfg", coordinate_policy="runtime_horizon_normalized",
        ),
    ):
        model = make_model()
        with pytest.raises(ValueError, match="canonical opt-in"):
            model.enable_cache(cfg)


@pytest.mark.parametrize("num_steps", range(1, 81))
def test_continuous_n_cpu_schedule_guards(num_steps):
    cfg = SpectrumCacheConfig.for_krea2_raw_variable(num_steps)
    assert cfg.num_inference_steps == num_steps
    assert cfg.denoiser_call_topology == "paired_cfg"
    assert cfg.coordinate_policy == "runtime_horizon_normalized"
    assert cfg.allow_krea2_raw_variable_n == (num_steps != 52)
    assert cfg.forecast_step_indices == tuple(sorted(set(cfg.forecast_step_indices)))
    forecasts = set(cfg.forecast_step_indices)
    if num_steps <= 11:
        assert not forecasts  # Not enough history; full real compute.
    else:
        assert not forecasts.intersection(range(7))
        assert not forecasts.intersection(range(num_steps - 2, num_steps))
        assert len(forecasts) > 0
        runs = "".join("1" if i in forecasts else "0" for i in range(num_steps))
        assert max(map(len, runs.split("0"))) <= 7
    if num_steps != 52:
        assert cfg.tail_actual_steps == min(2, num_steps)


@pytest.mark.parametrize("num_steps", (1, 9, 10, 11))
def test_continuous_n_all_real_cpu_mechanics(num_steps):
    model = make_model()
    model.enable_cache(SpectrumCacheConfig.for_krea2_raw_variable(num_steps))
    positive = torch.ones(1, 3, 12, 4)
    negative = torch.full((1, 3, 12, 4), 2.0)
    ids = torch.zeros(5, 3)
    with torch.no_grad():
        for step in range(num_steps):
            for embed in (positive, negative):
                with model.cache_context("krea", step_index=step, num_inference_steps=num_steps):
                    model(
                        hidden_states=torch.ones(1, 2, 4) + step,
                        encoder_hidden_states=embed,
                        timestep=torch.ones(1), position_ids=ids, return_dict=False,
                    )
    report = model._diffusers_hook.get_hook("spectrum_cache_denoiser").latest_summary
    assert report["prediction_call_count"] == 0
    assert report["full_call_count"] == 2 * num_steps
    assert sum(block.calls for block in model.transformer_blocks) == 28 * 2 * num_steps
    model.disable_cache()


@pytest.mark.parametrize(
    "num_steps, forecasts",
    ((8, (3, 5, 6)), (9, (3, 5, 7)), (10, (3, 5, 7, 8)), (11, (3, 5, 7, 9)), (12, (4, 6, 8, 9))),
)
def test_kroma_turbo_opd_native_single_call(num_steps, forecasts):
    from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline

    model = make_model()
    model.register_parameter("probe_device_anchor", torch.nn.Parameter(torch.zeros(1)))
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400
    )
    pipe = Krea2Pipeline(
        scheduler=scheduler, vae=None, text_encoder=None, tokenizer=None, transformer=model, is_distilled=True
    )
    pipe.set_progress_bar_config(disable=True)
    positive = torch.ones(1, 3, 12, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    latents = torch.zeros(1, 4, 64)
    cfg = SpectrumCacheConfig.for_kroma_turbo_opd(num_steps)
    assert cfg.forecast_step_indices == forecasts

    def sample(guidance=0.0):
        return pipe(
            prompt_embeds=positive, prompt_embeds_mask=mask,
            negative_prompt_embeds=torch.full_like(positive, 2.0), negative_prompt_embeds_mask=mask,
            latents=latents.clone(), height=32, width=32, num_inference_steps=num_steps,
            guidance_scale=guidance, output_type="latent",
        ).images

    with torch.no_grad():
        reference = sample()
        before = sum(block.calls for block in model.transformer_blocks)
        model.enable_cache(cfg)
        try:
            output = sample()
            report = model._diffusers_hook.get_hook("spectrum_cache_denoiser").latest_summary
            assert not report["guard_latched"], report["guard_reasons"]
            assert report["forecast_steps"] == list(forecasts)
            assert report["lane_prediction_call_count"] == {"positive": len(forecasts), "negative": 0}
            assert report["full_call_count"] == num_steps - len(forecasts)
            assert sum(block.calls for block in model.transformer_blocks) - before == 28 * (num_steps - len(forecasts))
            assert torch.isfinite(output).all()
        finally:
            model.disable_cache()
        torch.testing.assert_close(reference, sample(), rtol=0, atol=0)
        if num_steps == 8:
            model.enable_cache(cfg)
            try:
                sample(guidance=0.5)
                report = model._diffusers_hook.get_hook("spectrum_cache_denoiser").latest_summary
                assert report["guard_latched"] and report["prediction_call_count"] == 0
            finally:
                model.disable_cache()
    assert not model.is_cache_enabled


@pytest.mark.parametrize("bad", (True, 0, 7, 13, 80, 8.0))
def test_kroma_turbo_opd_rejects_unqualified_horizon(bad):
    with pytest.raises(ValueError, match="integer"):
        SpectrumCacheConfig.for_kroma_turbo_opd(bad)


def test_kroma_turbo_opd_rejects_noncanonical_profile():
    from dataclasses import replace

    cfg = SpectrumCacheConfig.for_kroma_turbo_opd(8)
    model = make_model()
    for invalid in (replace(cfg, degree=3), replace(cfg, forecast_step_indices=(3, 4, 5))):
        with pytest.raises(ValueError, match="canonical opt-in"):
            model.enable_cache(invalid)
        assert not model.is_cache_enabled
