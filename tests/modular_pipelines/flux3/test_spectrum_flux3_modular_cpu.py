# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU integration of the real Flux3 modular denoise loop with precomputed text embeddings.

No model downloads, Qwen text encoder, VAE, or real weights. The optional LoRA class
is shimmed ONLY for importing the pipeline when Transformers is unavailable.
"""

import torch
import pytest

from diffusers import loaders

# A full dependency environment supplies this class normally. The test-only shim
# permits testing the *real* modular denoise loop without installing Transformers;
# it is not used by these tests and must not be used to qualify a full T2I workflow.
try:
    from diffusers.loaders import Flux3LoraLoaderMixin
except ImportError:
    loaders.Flux3LoraLoaderMixin = type("Flux3LoraLoaderMixin", (), {})

from diffusers.guiders import ClassifierFreeGuidance
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.denoise import Flux3DenoiseStep
from diffusers.modular_pipelines.flux3.modular_pipeline import Flux3ModularPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


def _pipeline(*, forecast_steps=(2,)):
    torch.manual_seed(77)
    model = Flux3Transformer2DModel(
        in_channels=8,
        cond_in_channels=8,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        timestep_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
        flow_map=False,
    ).eval()
    pipe = Flux3ModularPipeline(blocks=Flux3DenoiseStep())
    pipe.register_components(
        transformer=model,
        guider=ClassifierFreeGuidance(guidance_scale=3.5),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipe.scheduler.set_timesteps(4, device="cpu")
    model.enable_cache(
        SpectrumCacheConfig(
            num_inference_steps=4,
            warmup_steps=2,
            forecast_step_indices=forecast_steps,
            degree=1,
            coordinate_policy="runtime_horizon_normalized",
        )
    )
    return pipe


def _fields():
    torch.manual_seed(88)
    return {
        "prompt_embeds": torch.randn(1, 3, 32),
        "negative_prompt_embeds": torch.randn(1, 3, 32),
        "txt_ids": torch.zeros(3, 4),
        "negative_txt_ids": torch.zeros(3, 4),
        "img_ids": torch.zeros(4, 4),
    }


def _call(pipe, fields, *, latents=None):
    if latents is None:
        latents = torch.randn(1, 4, 8)
    # The complete Flux3AutoBlocks graph does this via Flux3SetTimestepsStep.
    pipe.scheduler.set_timesteps(4, device="cpu")
    return pipe(
        latents=latents.clone(),
        denoiser_input_fields=fields,
        timesteps=pipe.scheduler.timesteps,
        num_inference_steps=4,
        output="latents",
    )


def _states(pipe):
    hook = pipe.transformer._diffusers_hook.get_hook("spectrum_cache_denoiser")
    return hook.state_manager._state_cache


@torch.no_grad()
def test_full_modular_loop_separates_cfg_and_forecasts(monkeypatch):
    pipe = _pipeline()
    block = pipe.transformer.single_transformer_blocks[0]
    original = block.forward
    calls = []

    def traced(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(block, "forward", traced)
    out = _call(pipe, _fields(), latents=torch.randn(1, 4, 8))
    assert out.shape == (1, 4, 8) and torch.isfinite(out).all()
    assert len(calls) == 6, "eight CFG forwards minus two forecast forwards"
    assert set(_states(pipe)) == {"pred_cond", "pred_uncond"}
    for state in _states(pipe).values():
        assert state.summary()["prediction_call_count"] == 1
        assert state.summary()["full_call_count"] == 3
        assert not state.guard_latched


@torch.no_grad()
def test_two_consecutive_modular_generations_reset_state():
    pipe = _pipeline()
    fields = _fields()
    latents = torch.randn(1, 4, 8)
    first = _call(pipe, fields, latents=latents)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())
    second = _call(pipe, fields, latents=latents)
    assert torch.equal(first, second)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())
    assert all(s.summary()["full_call_count"] == 3 for s in _states(pipe).values())


@torch.no_grad()
def test_exception_inside_real_loop_does_not_contaminate_next_generation(monkeypatch):
    pipe = _pipeline()
    fields = _fields()
    latents = torch.randn(1, 4, 8)
    baseline = _call(pipe, fields, latents=latents)
    original = pipe.transformer.single_transformer_blocks[0].forward
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(pipe.transformer.single_transformer_blocks[0], "forward", fail_once)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _call(pipe, fields, latents=latents)
    monkeypatch.setattr(pipe.transformer.single_transformer_blocks[0], "forward", original)
    recovered = _call(pipe, fields, latents=latents)
    assert torch.equal(baseline, recovered)
    assert all(s.summary()["prediction_call_count"] == 1 for s in _states(pipe).values())


@torch.no_grad()
def test_modular_all_real_compute_matches_disabled_cache():
    pipe = _pipeline(forecast_steps=())
    fields = _fields()
    latents = torch.randn(1, 4, 8)
    enabled = _call(pipe, fields, latents=latents)
    assert all(s.summary()["prediction_call_count"] == 0 for s in _states(pipe).values())
    pipe.transformer.disable_cache()
    disabled = _call(pipe, fields, latents=latents)
    assert torch.equal(enabled, disabled)
