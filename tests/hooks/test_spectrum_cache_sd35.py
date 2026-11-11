"""CPU mechanics for the optional SD3/SD3.5 transformer-output SPECTRUM path."""

import inspect

import pytest
import torch

from diffusers import StableDiffusion3Pipeline
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.hooks.spectrum_sd35 import SD35SpectrumGate


P13 = (5, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24)


def test_sd35_pipeline_remains_default_off():
    parameter = inspect.signature(StableDiffusion3Pipeline.__call__).parameters["spectrum_cache_config"]
    assert parameter.default is None


@pytest.mark.parametrize("factory", [SpectrumCacheConfig.for_sd35_large, SpectrumCacheConfig.for_sd35_medium])
def test_sd35_p13_profile_is_explicit_and_pinned(factory):
    config = factory()
    assert config.num_inference_steps == 28
    assert config.forecast_step_indices == P13
    assert config.degree == 2
    assert config.ridge_lambda == 0.1
    assert config.blend_w == 0.5
    assert config.history_limit == 8
    assert config.predictor_backend == "chunked"
    assert config.coordinate_policy == "runtime_index_normalized"
    assert config.tail_actual_steps == 2


def _run_gate(config, *, steps=28, kwargs=None, inject_failure=False):
    gate = SD35SpectrumGate(config)
    timesteps = torch.arange(steps, dtype=torch.float32)
    latent = torch.linspace(-0.2, 0.2, 128, dtype=torch.float32).reshape(2, 16, 2, 2)
    prompt = torch.zeros((2, 2), dtype=torch.float32)
    pooled = torch.ones((2, 2), dtype=torch.float32)
    actual_calls = 0
    if inject_failure:
        gate.forecaster.predict = lambda _: torch.full_like(latent, float("nan"))
    with torch.inference_mode():
        gate.begin(timesteps=timesteps, do_classifier_free_guidance=True)
        for step, timestep in enumerate(timesteps):
            def real_forward():
                nonlocal actual_calls
                actual_calls += 1
                return latent + step / 100.0

            output = gate.predict_or_compute(
                step_index=step,
                timestep=timestep,
                model_input=latent,
                prompt_embeds=prompt,
                pooled_prompt_embeds=pooled,
                guidance_scale=5.0,
                joint_attention_kwargs=kwargs,
                skip_guidance_layers=None,
                forward=real_forward,
            )
            assert output.shape == latent.shape and torch.isfinite(output).all()
    assert actual_calls == gate.counts["real"]
    return gate.counts


@pytest.mark.parametrize("factory", [SpectrumCacheConfig.for_sd35_large, SpectrumCacheConfig.for_sd35_medium])
def test_sd35_p13_forecasts_exactly_thirteen_calls(factory):
    assert _run_gate(factory()) == {"real": 15, "forecast": 13, "fallback": 0}


def test_sd35_fail_closed_on_horizon_or_conditioning_mismatch():
    config = SpectrumCacheConfig.for_sd35_large()
    assert _run_gate(config, steps=30) == {"real": 30, "forecast": 0, "fallback": 0}
    assert _run_gate(config, kwargs={"scale": 1.0}) == {"real": 28, "forecast": 0, "fallback": 0}


def test_sd35_invalid_forecast_sticks_to_real_compute():
    counts = _run_gate(SpectrumCacheConfig.for_sd35_medium(), inject_failure=True)
    assert counts == {"real": 28, "forecast": 0, "fallback": 1}


def test_sd35_rejects_unconfigured_adaptive_forecast():
    with pytest.raises(ValueError, match="explicit forecast steps"):
        SD35SpectrumGate(SpectrumCacheConfig())
