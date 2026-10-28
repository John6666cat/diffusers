"""Internal fail-closed gate for the bounded, opt-in SDXL Euler Spectrum route.

Only the SDXL pipeline uses this helper. Model/provenance and image quality
qualification remain separate from the library's runtime geometry contract.
"""

from functools import wraps
import logging
import sys

import torch

from ...hooks.spectrum_cache import SpectrumCacheConfig


# Pin the Euler schedule behavior observed in the original L4 research route.
# A canonical comparison alone would also accept unqualified scheduler configs.
_EULER_CONFIG = {
    "num_train_timesteps": 1000,
    "trained_betas": None,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "interpolation_type": "linear",
    "use_karras_sigmas": False,
    "use_exponential_sigmas": False,
    "use_beta_sigmas": False,
    "sigma_min": None,
    "sigma_max": None,
    "timestep_spacing": "leading",
    "timestep_type": "discrete",
    "steps_offset": 1,
    "rescale_betas_zero_snr": False,
    "final_sigmas_type": "zero",
}


def _spectrum_config(unet):
    config = getattr(unet, "_cache_config", None)
    # Do not touch other cache techniques in a shared pipeline.
    return config if (isinstance(config, SpectrumCacheConfig)
                      and config.sdxl_euler_post_grid_strict) else None


def _has_unqualified_offload(unet):
    """Reject Accelerate and Diffusers group offload hooks, including child hooks.

    Spectrum cache itself uses Diffusers HookRegistry, so a generic presence
    check for ``_diffusers_hook`` would incorrectly reject every cached UNet.
    Only inspect named offload registrations; keep this guard read-only.
    """
    modules = unet.modules() if hasattr(unet, "modules") else (unet,)
    for module in modules:
        if getattr(module, "_hf_hook", None) is not None:
            return True
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        # These names come from pinned Source group_offloading.py.
        for name in ("group_offloading", "lazy_prefetch_group_offloading"):
            if registry.get_hook(name) is not None:
                return True
    return False


def _spectrum_sdxl_euler_reasons(pipe, timesteps, effective_steps, *, requested_steps, plain_request):
    """Pure selection: no activation, hook install, scheduler mutation, or reset."""
    reasons = []
    config = _spectrum_config(pipe.unet)
    if config is None:
        return ("not_spectrum_config",)
    if not plain_request:
        reasons.append("unqualified_pipeline_options")
    if type(pipe).__name__ != "StableDiffusionXLPipeline":
        reasons.append("unqualified_pipeline_type")
    if type(pipe.scheduler).__name__ != "EulerDiscreteScheduler" or (
        type(pipe.scheduler).__module__ != "diffusers.schedulers.scheduling_euler_discrete"
    ):
        reasons.append("unqualified_scheduler_type")
    if type(requested_steps) is not int or requested_steps not in (24, 32):
        reasons.append("unqualified_step_count")
    if (effective_steps != requested_steps or len(timesteps) != requested_steps
            or getattr(pipe, "_num_timesteps", None) != requested_steps):
        reasons.append("effective_timestep_mismatch")
    if getattr(pipe.unet, "peft_config", None):
        reasons.append("unet_adapter_present")
    if any(getattr(getattr(pipe, name, None), "peft_config", None)
           for name in ("text_encoder", "text_encoder_2")):
        reasons.append("text_encoder_adapter_present")
    try:
        if _has_unqualified_offload(pipe.unet):
            reasons.append("unet_offload_hook_present")
    except Exception as exc:
        reasons.append("offload_hook_inspection_failed:" + type(exc).__name__)
    try:
        p = next(pipe.unet.parameters())
        if p.dtype != torch.float16 or p.device.type != "cuda":
            reasons.append("unqualified_unet_device_or_dtype")
    except (StopIteration, AttributeError, TypeError):
        reasons.append("unet_parameter_unavailable")
    if torch.is_grad_enabled():
        reasons.append("autograd_enabled")
    try:
        if requested_steps in (24, 32) and config != SpectrumCacheConfig.for_sdxl_variable(requested_steps, strict_post_grid=True):
            reasons.append("unexpected_spectrum_profile")
        for key, expected in _EULER_CONFIG.items():
            if pipe.scheduler.config[key] != expected:
                reasons.append("euler_config_mismatch:" + key)
        reference = type(pipe.scheduler).from_config(pipe.scheduler.config)
        reference.set_timesteps(requested_steps, device=timesteps.device)
        if (not torch.equal(timesteps, pipe.scheduler.timesteps)
                or not torch.equal(timesteps, reference.timesteps)
                or not torch.equal(pipe.scheduler.sigmas, reference.sigmas)):
            reasons.append("post_grid_mismatch")
        if len(pipe.scheduler.sigmas) != requested_steps + 1:
            reasons.append("sigma_length_mismatch")
    except Exception as exc:
        reasons.append("grid_validation_failed:" + type(exc).__name__)
    return tuple(reasons)


def _spectrum_sdxl_euler_maybe_bypass(pipe, timesteps, effective_steps, *, requested_steps, plain_request):
    """An opt-in Spectrum cache is removed before any UNet call if unqualified."""
    if _spectrum_config(pipe.unet) is None:
        return ()
    try:
        reasons = _spectrum_sdxl_euler_reasons(
            pipe, timesteps, effective_steps, requested_steps=requested_steps, plain_request=plain_request
        )
    except Exception as exc:
        reasons = ("unexpected_gate_error:" + type(exc).__name__,)
    if reasons:
        try:
            pipe.unet.disable_cache()
        finally:
            pipe.unet._reset_stateful_cache()
    return reasons


def _spectrum_sdxl_euler_lifecycle(call):
    """Preserve other cache techniques; clean opt-in Spectrum even on exceptions."""
    @wraps(call)
    def wrapper(self, *args, **kwargs):
        unet = self.unet
        if _spectrum_config(unet) is None:
            return call(self, *args, **kwargs)
        try:
            unet._reset_stateful_cache()
            return call(self, *args, **kwargs)
        finally:
            # A cleanup failure must not replace the original pipeline error.
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            try:
                if _spectrum_config(unet) is not None:
                    unet.disable_cache()
            except Exception as exc:
                cleanup_error = exc
            try:
                unet._reset_stateful_cache()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    logging.getLogger(__name__).error(
                        "Spectrum state reset also failed during cleanup", exc_info=True
                    )
            if cleanup_error is not None:
                if primary_error is None:
                    raise cleanup_error
                logging.getLogger(__name__).error(
                    "Spectrum cleanup failed while preserving pipeline error",
                    exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
                )
    return wrapper
