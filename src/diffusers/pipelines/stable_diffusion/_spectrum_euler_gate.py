"""Strict, research-only SD1.5 Euler integer-N Spectrum admission and lifecycle."""

from functools import wraps
import logging
import sys

import torch

from ...hooks.spectrum_cache import SpectrumCacheConfig


# Ordinary SD1.5 Euler scaled-linear (epsilon) grid; other settings fail closed.
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


def _spectrum_sd15_config(unet):
    config = getattr(unet, "_cache_config", None)
    return config if (isinstance(config, SpectrumCacheConfig)
                      and config.sd15_euler_continuous_n_candidate) else None


def _spectrum_sd15_offload_present(unet):
    for module in unet.modules() if hasattr(unet, "modules") else (unet,):
        if getattr(module, "_hf_hook", None) is not None:
            return True
        registry = getattr(module, "_diffusers_hook", None)
        if registry is not None:
            for name in ("group_offloading", "lazy_prefetch_group_offloading"):
                if registry.get_hook(name) is not None:
                    return True
    return False


def _spectrum_sd15_euler_reasons(pipe, timesteps, effective_steps, *, requested_steps, plain_request):
    """Read-only post-grid validation; never mutates a scheduler or installs cache."""
    reasons = []
    config = _spectrum_sd15_config(pipe.unet)
    if config is None:
        return ("not_sd15_continuous_config",)
    if not plain_request:
        reasons.append("unqualified_pipeline_options")
    if type(pipe).__name__ != "StableDiffusionPipeline":
        reasons.append("unqualified_pipeline_type")
    scheduler = pipe.scheduler
    if type(scheduler).__name__ != "EulerDiscreteScheduler" or (
        type(scheduler).__module__ != "diffusers.schedulers.scheduling_euler_discrete"
    ):
        reasons.append("unqualified_scheduler_type")
    if type(requested_steps) is not int or requested_steps < 1:
        reasons.append("unqualified_step_count")
    if effective_steps != requested_steps or len(timesteps) != requested_steps or (
        getattr(pipe, "_num_timesteps", None) != requested_steps
    ):
        reasons.append("effective_timestep_mismatch")
    unet_config = getattr(pipe.unet, "config", None)
    text_config = getattr(getattr(pipe, "text_encoder", None), "config", None)
    if not (
        getattr(unet_config, "cross_attention_dim", None) == 768
        and getattr(unet_config, "sample_size", None) == 64
        and getattr(unet_config, "in_channels", None) == 4
        and getattr(unet_config, "out_channels", None) == 4
        and getattr(unet_config, "time_cond_proj_dim", None) is None
        and getattr(text_config, "hidden_size", None) == 768
    ):
        reasons.append("unqualified_sd15_model_geometry")
    if getattr(pipe.unet, "peft_config", None):
        reasons.append("unet_adapter_present")
    if getattr(getattr(pipe, "text_encoder", None), "peft_config", None):
        reasons.append("text_encoder_adapter_present")
    try:
        if _spectrum_sd15_offload_present(pipe.unet):
            reasons.append("unet_offload_hook_present")
    except Exception as exc:
        reasons.append("offload_hook_inspection_failed:" + type(exc).__name__)
    try:
        parameter = next(pipe.unet.parameters())
        if parameter.dtype != torch.float16 or parameter.device.type != "cuda":
            reasons.append("unqualified_unet_device_or_dtype")
    except (StopIteration, AttributeError, TypeError):
        reasons.append("unet_parameter_unavailable")
    if torch.is_grad_enabled():
        reasons.append("autograd_enabled")
    try:
        expected_config = (
            SpectrumCacheConfig.for_sd15_continuous_n(requested_steps)
            if type(requested_steps) is int and requested_steps >= 1 else None
        )
        if expected_config is not None and config != expected_config:
            reasons.append("unexpected_spectrum_profile")
        for key, expected in _EULER_CONFIG.items():
            if scheduler.config[key] != expected:
                reasons.append("euler_config_mismatch:" + key)
        reference = type(scheduler).from_config(scheduler.config)
        reference.set_timesteps(requested_steps, device=timesteps.device)
        if not (
            torch.equal(timesteps, scheduler.timesteps)
            and torch.equal(timesteps, reference.timesteps)
            and torch.equal(scheduler.sigmas, reference.sigmas)
        ):
            reasons.append("post_grid_mismatch")
        if len(scheduler.sigmas) != requested_steps + 1:
            reasons.append("sigma_length_mismatch")
    except Exception as exc:
        reasons.append("grid_validation_failed:" + type(exc).__name__)
    return tuple(reasons)


def _spectrum_sd15_euler_maybe_bypass(pipe, timesteps, effective_steps, *, requested_steps, plain_request):
    """Disable only the explicit candidate on bad request/grid, before the first UNet call."""
    if _spectrum_sd15_config(pipe.unet) is None:
        return ()
    try:
        reasons = _spectrum_sd15_euler_reasons(
            pipe, timesteps, effective_steps, requested_steps=requested_steps,
            plain_request=plain_request,
        )
    except Exception as exc:
        reasons = ("unexpected_gate_error:" + type(exc).__name__,)
    if reasons:
        try:
            pipe.unet.disable_cache()
        finally:
            pipe.unet._reset_stateful_cache()
    return reasons


def _spectrum_sd15_euler_lifecycle(call):
    """Always reset and discard research cache on success or failure."""
    @wraps(call)
    def wrapper(self, *args, **kwargs):
        unet = self.unet
        if _spectrum_sd15_config(unet) is None:
            return call(self, *args, **kwargs)
        try:
            unet._reset_stateful_cache()
            return call(self, *args, **kwargs)
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            try:
                if _spectrum_sd15_config(unet) is not None:
                    unet.disable_cache()
            except Exception as exc:
                cleanup_error = exc
            try:
                unet._reset_stateful_cache()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
                else:
                    logging.getLogger(__name__).error("SD1.5 state reset also failed", exc_info=True)
            if cleanup_error is not None:
                if primary_error is None:
                    raise cleanup_error
                logging.getLogger(__name__).error(
                    "SD1.5 cleanup failed while preserving original error",
                    exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
                )
    return wrapper
