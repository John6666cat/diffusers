"""Small no-weights contract tests for the isolated official MOSS port.

Run on the 0.43.0 fork with torch>=2.9 and transformers>=5.3 available.
These tests do not demonstrate full-model GPU performance or image quality.
"""

import torch

from diffusers.hooks import SpectrumCacheConfig
from diffusers.pipelines.moss_image.pipeline_moss_image import (
    MOSSImagePipeline,
    MOSSImageScheduler,
    MOSSImageTransformer,
    _linear_input_dtype,
)


def test_nf4_packed_linear_uses_compute_dtype_not_uint8_weight():
    linear = torch.nn.Linear(4, 4)
    assert _linear_input_dtype(linear) == torch.float32
    # No bitsandbytes import needed for this no-weights CPU dtype contract.
    linear.weight = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.uint8), requires_grad=False)
    linear.compute_dtype = torch.bfloat16
    assert _linear_input_dtype(linear) == torch.bfloat16


def _pipeline():
    torch.manual_seed(13)
    model = MOSSImageTransformer(
        dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=42,
        text_dim=16,
        freq_dim=8,
        latent_channels=128,
    ).eval()
    return MOSSImagePipeline(model, MOSSImageScheduler())


def _inputs():
    generator = torch.Generator(device="cpu").manual_seed(27)
    return dict(
        height=32,
        width=32,
        num_inference_steps=8,
        guidance_scale=3.0,
        latents=torch.randn(1, 128, 2, 2, generator=generator),
        prompt_embeds=torch.randn(1, 3, 16, generator=generator),
        negative_prompt_embeds=torch.randn(1, 3, 16, generator=generator),
        output_type="latent",
    )


def _config():
    return SpectrumCacheConfig(
        num_inference_steps=8,
        warmup_steps=3,
        forecast_step_indices=(4, 5),
        tail_actual_steps=1,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=5,
        predictor_backend="chunked",
        coordinate_policy="runtime_index_normalized",
    )


def test_off_path_and_cache_lifecycle():
    pipe = _pipeline()
    args = _inputs()
    with torch.no_grad():
        before = pipe(**args).images
        assert pipe.spectrum_moss_last_summary is None
        pipe.transformer.enable_cache(_config())
        cached = pipe(**args).images
        summary = pipe.spectrum_moss_last_summary
        assert set(summary) == {"cond", "uncond_skip"}
        assert all(lane["forecast_calls"] >= 1 for lane in summary.values())
        assert all(lane["coordinate"]["failure_latched"] is False for lane in summary.values())
        assert torch.isfinite(cached).all()
        # A new invocation must rebuild independent per-lane histories.
        cached_repeat = pipe(**args).images
        torch.testing.assert_close(cached, cached_repeat, rtol=0, atol=0)
        pipe.transformer.disable_cache()
        after = pipe(**args).images
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_direct_forward_without_native_context_is_full_compute():
    pipe = _pipeline()
    args = _inputs()
    transformer = pipe.transformer
    transformer.enable_cache(_config())
    with torch.no_grad():
        output = transformer(
            args["latents"], torch.tensor(900.0), args["prompt_embeds"],
        ).sample
    assert output.shape == args["latents"].shape
    transformer.disable_cache()


def test_wrong_step_count_fails_closed_without_forecast():
    pipe = _pipeline()
    args = _inputs()
    pipe.transformer.enable_cache(_config())
    args["num_inference_steps"] = 9
    with torch.no_grad():
        out = pipe(**args).images
    assert torch.isfinite(out).all()
    assert all(
        lane["forecast_calls"] == 0 and lane["coordinate"]["failure_latched"]
        for lane in pipe.spectrum_moss_last_summary.values()
    )
    pipe.transformer.disable_cache()


def test_batch_two_bypasses_prediction():
    pipe = _pipeline()
    args = _inputs()
    args["latents"] = args["latents"].expand(2, -1, -1, -1).clone()
    args["prompt_embeds"] = args["prompt_embeds"].expand(2, -1, -1).clone()
    args["negative_prompt_embeds"] = args["negative_prompt_embeds"].expand(2, -1, -1).clone()
    pipe.transformer.enable_cache(_config())
    with torch.no_grad():
        out = pipe(**args).images
    assert out.shape[0] == 2
    assert all(lane["forecast_calls"] == 0 for lane in pipe.spectrum_moss_last_summary.values())
    pipe.transformer.disable_cache()
