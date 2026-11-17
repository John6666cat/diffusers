"""CPU-only regression tests for the opt-in MOSS variable-N cache."""

import pytest
import torch

from diffusers.hooks import SpectrumCacheConfig
from diffusers.hooks.spectrum_cache import SpectrumForecaster, SpectrumSchedule
from diffusers.hooks.spectrum_moss_variable_n import moss_variable_n_forecast_indices
from diffusers.pipelines.moss_image.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
)


@pytest.mark.parametrize("n, expected", [
    (18, 6), (28, 13), (30, 14), (35, 18), (40, 21),
    (50, 28), (55, 32), (60, 35),
])
def test_native_mask_exact_and_two_real_anchors(n, expected):
    config = SpectrumCacheConfig.for_moss_image_variable(n)
    mask = moss_variable_n_forecast_indices(n)
    assert config.predictor_backend == "fp32_weighted"
    assert config.forecast_step_indices == mask
    assert config.coordinate_policy == "runtime_index_normalized"
    assert len(mask) == expected
    assert min(mask) >= 8 and max(mask) < n - 2
    assert sum(i not in mask for i in range(8, n - 2)) >= 2
    schedule = SpectrumSchedule(config)
    assert tuple(i for i in range(n) if not schedule.decide(i)) == mask


def test_reference_n50_and_low_n_off_only():
    assert moss_variable_n_forecast_indices(50) == (
        8, 10, 12, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 30,
        31, 32, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 46, 47,
    )
    for n in range(1, 18):
        assert moss_variable_n_forecast_indices(n) == ()
        with pytest.raises(ValueError, match="keep cache OFF"):
            SpectrumCacheConfig.for_moss_image_variable(n)
    for invalid in (True, 35.0, "35", None):
        with pytest.raises(TypeError):
            moss_variable_n_forecast_indices(invalid)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp32_weighted_matches_independent_matmul_reference(dtype):
    config = SpectrumCacheConfig(
        num_inference_steps=50, degree=4, ridge_lambda=0.1,
        blend_w=0.55, history_limit=6,
        predictor_backend="fp32_weighted", coordinate_policy="runtime_index_normalized",
    )
    forecaster = SpectrumForecaster(config)
    for index, step in enumerate((0, 2, 4, 6, 8, 10)):
        values = torch.tensor(
            [1.5 + 0.13 * index ** 2, -0.7 + 0.23 * index,
             5.21 - 0.07 * index ** 2, 0.19 * index ** 3], dtype=dtype,
        )
        forecaster.update(step, values)

    actual = forecaster.predict(12)
    design = forecaster._history_design
    chol = forecaster._history_chol
    phi = forecaster._design(forecaster._tau(torch.tensor([12.0])))
    stacked = torch.stack([feature.float() for feature in forecaster.features])
    spectral = phi @ torch.cholesky_solve(design.T @ stacked, chol)
    slope = (12.0 - forecaster.steps[-1]) / (forecaster.steps[-1] - forecaster.steps[-2])
    local = stacked[-1] + slope * (stacked[-1] - stacked[-2])
    reference = ((1 - config.blend_w) * local + config.blend_w * spectral[0]).to(dtype)
    if dtype == torch.float32:
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=5e-5)
    else:
        torch.testing.assert_close(actual.float(), reference.float(), rtol=0.01, atol=0.04)
    assert actual.dtype == dtype and bool(torch.isfinite(actual).all())
    assert forecaster.predictor_state_bytes() < 4096


def test_generic_old_backends_unchanged_and_opt_in_is_local():
    for backend in ("dense", "chunked", "bounded"):
        config = SpectrumCacheConfig(num_inference_steps=20, predictor_backend=backend)
        assert config.predictor_backend == backend
    assert SpectrumCacheConfig().predictor_backend == "dense"


def test_tiny_native_moss_cfg_lanes_and_off_on_off():
    torch.manual_seed(13)
    transformer = MOSSImageTransformer(
        dim=32, ffn_dim=64, num_heads=4, num_layers=42,
        text_dim=16, freq_dim=8, latent_channels=128,
    ).eval()
    pipe = MOSSImagePipeline(transformer, MOSSImageScheduler())
    generator = torch.Generator(device="cpu").manual_seed(27)
    inputs = dict(
        height=32, width=32, num_inference_steps=18, guidance_scale=3.0,
        latents=torch.randn(1, 128, 2, 2, generator=generator),
        prompt_embeds=torch.randn(1, 3, 16, generator=generator),
        negative_prompt_embeds=torch.randn(1, 3, 16, generator=generator),
        output_type="latent",
    )
    assert not transformer.is_cache_enabled
    with torch.inference_mode():
        off_before = pipe(**inputs).images
        assert pipe.spectrum_moss_last_summary is None
        try:
            transformer.enable_cache(SpectrumCacheConfig.for_moss_image_variable(18))
            on = pipe(**inputs).images
            summary = pipe.spectrum_moss_last_summary
            assert set(summary) == {"cond", "uncond_skip"}
            mask = list(moss_variable_n_forecast_indices(18))
            for lane in summary.values():
                assert lane["forecast_steps"] == mask
                assert lane["forecast_calls"] == 6
                assert lane["full_calls"] == 12
                assert lane["guard_reason"] is None
                assert lane["predict_failures"] == []
                assert lane["coordinate"]["failure_latched"] is False
            assert bool(torch.isfinite(on).all())
        finally:
            if transformer.is_cache_enabled:
                transformer.disable_cache()
        off_after = pipe(**inputs).images
        assert pipe.spectrum_moss_last_summary is None
        torch.testing.assert_close(off_before, off_after, rtol=0, atol=0)
