import pytest
import torch

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

BASE = {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
}
CLEAN = torch.tensor([[[[0.25, -0.40], [0.70, 0.10]]]], dtype=torch.float32)
NOISE = torch.tensor([[[[-0.75, 0.50], [-0.20, 0.90]]]], dtype=torch.float32)

def _cases():
    return [
        ("ddim", lambda: DDIMScheduler.from_config(BASE)),
        ("pndm", lambda: PNDMScheduler.from_config(BASE, skip_prk_steps=True)),
        ("euler", lambda: EulerDiscreteScheduler.from_config(BASE)),
        ("euler_karras", lambda: EulerDiscreteScheduler.from_config(BASE, use_karras_sigmas=True)),
        ("euler_a", lambda: EulerAncestralDiscreteScheduler.from_config(BASE)),
        ("lms", lambda: LMSDiscreteScheduler.from_config(BASE)),
        ("lms_karras", lambda: LMSDiscreteScheduler.from_config(BASE, use_karras_sigmas=True)),
        ("dpm", lambda: DPMSolverMultistepScheduler.from_config(BASE, algorithm_type="dpmsolver++", solver_order=2)),
        ("dpm_karras", lambda: DPMSolverMultistepScheduler.from_config(BASE, algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=True)),
        ("dpm_exp", lambda: DPMSolverMultistepScheduler.from_config(BASE, algorithm_type="dpmsolver++", solver_order=2, use_exponential_sigmas=True)),
        ("unipc", lambda: UniPCMultistepScheduler.from_config(BASE)),
        ("unipc_karras", lambda: UniPCMultistepScheduler.from_config(BASE, use_karras_sigmas=True)),
        ("dpm_flow", lambda: DPMSolverMultistepScheduler.from_config(BASE, algorithm_type="dpmsolver++", solver_order=2, use_flow_sigmas=True)),
        ("unipc_flow", lambda: UniPCMultistepScheduler.from_config(BASE, use_flow_sigmas=True)),
    ]

@pytest.mark.parametrize("name,factory", _cases())
@pytest.mark.parametrize("position", [0, 5, 11, 22])
def test_model_input_coefficients_match_add_noise_and_preconditioning(name, factory, position):
    scheduler=factory()
    scheduler.set_timesteps(24)
    position=min(position,len(scheduler.timesteps)-2)
    if hasattr(scheduler,"set_begin_index"):
        scheduler.set_begin_index(position)
    timestep=scheduler.timesteps[position]
    noisy=scheduler.add_noise(CLEAN,NOISE,timestep.reshape(1))
    model_input=scheduler.scale_model_input(noisy.clone(),timestep)
    signal_scale,noise_scale=scheduler.get_model_input_coefficients(timestep)
    reconstructed=signal_scale*CLEAN+noise_scale*NOISE
    assert torch.allclose(model_input,reconstructed,atol=2e-5,rtol=2e-5)
    assert torch.isfinite(torch.as_tensor(signal_scale))
    assert torch.isfinite(torch.as_tensor(noise_scale))
    assert float(torch.as_tensor(signal_scale)) >= 0
    assert float(torch.as_tensor(noise_scale)) >= 0

def test_ddim_zero_snr_endpoint():
    scheduler=DDIMScheduler.from_config(BASE,timestep_spacing="trailing",rescale_betas_zero_snr=True)
    scheduler.set_timesteps(24)
    timestep=scheduler.timesteps[0]
    signal_scale,noise_scale=scheduler.get_model_input_coefficients(timestep)
    assert float(signal_scale)==0.0
    assert float(noise_scale)==1.0
