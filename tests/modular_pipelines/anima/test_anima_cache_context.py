from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from diffusers import ClassifierFreeGuidance, FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines.anima.denoise import AnimaLoopDenoiser
from diffusers.modular_pipelines.modular_pipeline import BlockState


class RecordingTransformer:
    def __init__(self):
        self.contexts = []

    @contextmanager
    def cache_context(self, name, **kwargs):
        self.contexts.append((name, kwargs))
        yield

    def __call__(self, hidden_states, **kwargs):
        return (torch.zeros_like(hidden_states),)


def make_components(num_schedule_steps=4, begin_index=0):
    scheduler = FlowMatchEulerDiscreteScheduler()
    sigmas = np.linspace(1.0, 1.0 / num_schedule_steps, num_schedule_steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    scheduler.set_begin_index(begin_index)
    return SimpleNamespace(
        guider=ClassifierFreeGuidance(guidance_scale=4.0),
        transformer=RecordingTransformer(),
        scheduler=scheduler,
    )


def make_state(active_steps, timestep):
    return BlockState(
        latent_model_input=torch.zeros(1, 16, 1, 4, 4),
        dtype=torch.float32,
        num_inference_steps=active_steps,
        timestep=timestep,
        padding_mask=torch.zeros(1, 1, 4, 4),
        prompt_embeds=torch.ones(1, 8, 16),
        negative_prompt_embeds=torch.zeros(1, 8, 16),
    )


def test_anima_t2i_cache_context_has_guider_identity_and_scheduler_coordinates():
    components = make_components(num_schedule_steps=4, begin_index=0)
    t = components.scheduler.timesteps[0]
    state = make_state(active_steps=4, timestep=t)
    block = AnimaLoopDenoiser()

    _, out_state = block(components, state, i=0, t=t)

    assert hasattr(out_state, "noise_pred")
    assert [name for name, _ in components.transformer.contexts] == ["pred_cond", "pred_uncond"]
    assert len(components.transformer.contexts) == 2

    expected_sigma = float(components.scheduler.sigmas[0])
    for _, ctx in components.transformer.contexts:
        assert ctx["step_index"] == 0
        assert ctx["num_inference_steps"] == 4
        assert torch.equal(ctx["timestep"], t)
        assert ctx["sigma"] == expected_sigma


def test_anima_img2img_cache_context_keeps_local_step_but_offsets_sigma_table():
    components = make_components(num_schedule_steps=4, begin_index=2)
    t = components.scheduler.timesteps[2]
    state = make_state(active_steps=2, timestep=t)
    block = AnimaLoopDenoiser()

    _, out_state = block(components, state, i=0, t=t)

    assert hasattr(out_state, "noise_pred")
    assert [name for name, _ in components.transformer.contexts] == ["pred_cond", "pred_uncond"]

    expected_sigma = float(components.scheduler.sigmas[2])
    for _, ctx in components.transformer.contexts:
        assert ctx["step_index"] == 0
        assert ctx["num_inference_steps"] == 2
        assert torch.equal(ctx["timestep"], t)
        assert ctx["sigma"] == expected_sigma


def test_anima_context_advances_local_step_and_sigma_together():
    components = make_components(num_schedule_steps=4, begin_index=0)
    t = components.scheduler.timesteps[1]
    state = make_state(active_steps=4, timestep=t)
    block = AnimaLoopDenoiser()

    block(components, state, i=1, t=t)

    expected_sigma = float(components.scheduler.sigmas[1])
    for _, ctx in components.transformer.contexts:
        assert ctx["step_index"] == 1
        assert ctx["num_inference_steps"] == 4
        assert torch.equal(ctx["timestep"], t)
        assert ctx["sigma"] == expected_sigma
