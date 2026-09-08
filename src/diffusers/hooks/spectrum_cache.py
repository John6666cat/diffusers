# Copyright 2026 The HuggingFace Team. All rights reserved.
#
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

import inspect
import math
from dataclasses import dataclass
from typing import Any

import torch

from ..utils import logging
from ..utils.torch_utils import unwrap_module
from ._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)

_SPECTRUM_DENOISER_HOOK = "spectrum_cache_denoiser"
_SPECTRUM_HEAD_BLOCK_HOOK = "spectrum_cache_head_block"
_SPECTRUM_BLOCK_HOOK = "spectrum_cache_block"
_SPECTRUM_UNET_FEATURE_HOOK = "spectrum_cache_unet_feature"
_FLUX_BLOCK_GROUPS = ("transformer_blocks", "single_transformer_blocks")
_CONTROL_ARGUMENTS = ("controlnet_block_samples", "controlnet_single_block_samples")
_UNET_RESIDUAL_ARGUMENTS = (
    "down_block_additional_residuals",
    "mid_block_additional_residual",
    "down_intrablock_additional_residuals",
)


@dataclass
class SpectrumCacheConfig:
    """Configuration for SPECTRUM cache on FLUX.1 transformers.

    SPECTRUM predicts the final image-stream transformer feature on selected denoising steps and skips the expensive
    transformer blocks on those prediction steps. The initial defaults reproduce the 50-step FLUX.1 research profile
    evaluated by the implementation experiment in this fork.

    Args:
        num_inference_steps (`int`, defaults to `50`):
            Expected denoising-step count for the configured refresh schedule. The default profile has been evaluated at
            50 steps. Other values are experimental.
        warmup_steps (`int`, defaults to `5`):
            Number of initial full-compute steps used to establish feature history.
        window_size (`float`, defaults to `2.0`):
            Initial adaptive refresh-window size.
        flex_window (`float`, defaults to `0.75`):
            Amount added to the adaptive refresh window after each refresh.
        degree (`int`, defaults to `4`):
            Degree of the Chebyshev polynomial used by the spectral forecaster.
        ridge_lambda (`float`, defaults to `0.1`):
            Ridge regularization coefficient used by the forecaster fit.
        blend_w (`float`, defaults to `0.5`):
            Blend weight for the spectral prediction. `1.0` is pure spectral prediction; lower values mix in the local
            first-order extrapolation.
        history_limit (`int`, defaults to `100`):
            Maximum number of full-compute feature snapshots retained per cache context.
        coordinate_max (`float`, defaults to `50.0`):
            Maximum coordinate used to map denoising-step indices into the Chebyshev domain.
        tail_actual_steps (`int`, defaults to `0`):
            Number of final denoising steps forced to full compute. This is a generic quality guard; the default `0`
            preserves the source-faithful refresh schedule.
        allow_unet_controlnet_residuals (`bool`, defaults to `False`):
            Explicitly allow classic UNet ControlNet residual pairs (`down_block_additional_residuals` together with
            `mid_block_additional_residual`) to use SPECTRUM. The default remains fail-closed.
        allow_unet_t2i_adapter_residuals (`bool`, defaults to `False`):
            Explicitly allow T2I-Adapter intrablock residuals (`down_intrablock_additional_residuals`) to use SPECTRUM.
            The default remains fail-closed and latches bypass for the rest of the denoising run.
        allow_unet_ip_adapter_image_embeds (`bool`, defaults to `False`):
            Explicitly allow IP-Adapter image embeddings carried in `added_cond_kwargs["image_embeds"]` to use SPECTRUM.
            The default remains fail-closed. Mixed special conditioning paths remain fail-closed even when their
            individual opt-ins are enabled. Autograd and non-default PEFT scale also remain fail-closed.

    Note:
        The default 50-step profile computes transformer blocks at steps
        `[0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 31, 38, 46]`, i.e. 14 real transformer evaluations.
        It is not intended as a lossless/transparent guarantee at every resolution.
    """

    num_inference_steps: int = 50
    warmup_steps: int = 5
    window_size: float = 2.0
    flex_window: float = 0.75
    degree: int = 4
    ridge_lambda: float = 0.1
    blend_w: float = 0.5
    history_limit: int = 100
    coordinate_max: float = 50.0
    tail_actual_steps: int = 0
    allow_unet_controlnet_residuals: bool = False
    allow_unet_t2i_adapter_residuals: bool = False
    allow_unet_ip_adapter_image_embeds: bool = False

    def __post_init__(self) -> None:
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if self.flex_window < 0:
            raise ValueError("flex_window must be >= 0")
        if self.degree < 0:
            raise ValueError("degree must be >= 0")
        if self.ridge_lambda < 0:
            raise ValueError("ridge_lambda must be >= 0")
        if not 0.0 <= self.blend_w <= 1.0:
            raise ValueError("blend_w must be in [0, 1]")
        if self.history_limit < 1:
            raise ValueError("history_limit must be >= 1")
        if self.coordinate_max <= 0:
            raise ValueError("coordinate_max must be > 0")
        if self.tail_actual_steps < 0:
            raise ValueError("tail_actual_steps must be >= 0")
        if self.tail_actual_steps > self.num_inference_steps:
            raise ValueError("tail_actual_steps must be <= num_inference_steps")


class SpectrumSchedule:
    """Adaptive SPECTRUM refresh schedule using denoising-loop step indices."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.cached_run_length = 0
        self.current_window = float(self.config.window_size)

    def decide(self, step_index: int) -> bool:
        if self.config.tail_actual_steps and step_index >= self.config.num_inference_steps - self.config.tail_actual_steps:
            self.cached_run_length = 0
            return True

        should_compute = True
        if step_index >= self.config.warmup_steps:
            width = max(1, math.floor(self.current_window))
            should_compute = ((self.cached_run_length + 1) % width) == 0
            if should_compute:
                self.current_window = round(self.current_window + self.config.flex_window, 3)

        if should_compute:
            self.cached_run_length = 0
        else:
            self.cached_run_length += 1
        return should_compute


class SpectrumForecaster:
    """Chebyshev ridge predictor with a local first-order blend."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.steps: list[int] = []
        self.features: list[torch.Tensor] = []
        self._coef: torch.Tensor | None = None
        self._shape: torch.Size | None = None

    def reset(self) -> None:
        self.steps.clear()
        self.features.clear()
        self._coef = None
        self._shape = None

    def update(self, step: int, feature: torch.Tensor) -> None:
        feature = feature.detach().clone()
        if self._shape is None:
            self._shape = feature.shape
        elif feature.shape != self._shape:
            raise ValueError(f"SPECTRUM feature shape changed: {self._shape} -> {feature.shape}")

        self.steps.append(int(step))
        self.features.append(feature)
        self.steps = self.steps[-self.config.history_limit :]
        self.features = self.features[-self.config.history_limit :]
        self._coef = None

    def history_bytes(self) -> int:
        return sum(feature.numel() * feature.element_size() for feature in self.features)

    def _tau(self, steps: torch.Tensor) -> torch.Tensor:
        return (steps - self.config.coordinate_max / 2.0) * (2.0 / self.config.coordinate_max)

    def _design(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        columns = [torch.ones_like(tau)]
        if self.config.degree >= 1:
            columns.append(tau)
        for _ in range(2, self.config.degree + 1):
            columns.append(2 * tau * columns[-1] - columns[-2])
        return torch.cat(columns[: self.config.degree + 1], dim=1)

    @torch.compiler.disable
    def _fit(self) -> None:
        if self._coef is not None:
            return
        if not self.features:
            raise ValueError("SPECTRUM cannot predict before a full-compute feature has been recorded.")

        device = self.features[-1].device
        steps = torch.tensor(self.steps, device=device, dtype=torch.float32)
        design = self._design(self._tau(steps)).to(torch.float32)
        features = torch.stack([feature.reshape(-1) for feature in self.features], dim=0).to(torch.float32)

        order = design.shape[1]
        design_t = design.transpose(0, 1)
        gram = design_t @ design
        identity = torch.eye(order, device=device, dtype=torch.float32)
        gram = gram + self.config.ridge_lambda * identity
        try:
            chol = torch.linalg.cholesky(gram)
        except RuntimeError:
            jitter = 1e-6 * gram.diag().mean().clamp_min(1e-12)
            chol = torch.linalg.cholesky(gram + jitter * identity)

        self._coef = torch.cholesky_solve(design_t @ features, chol).to(self.features[-1].dtype)

    def _local_first_order(self, step: int) -> torch.Tensor:
        if len(self.features) < 2:
            return self.features[-1]

        current, previous = self.features[-1], self.features[-2]
        current_step, previous_step = self.steps[-1], self.steps[-2]
        spacing = max(float(current_step - previous_step), 1e-8)
        scale = float(step - current_step) / spacing
        return current + scale * (current - previous)

    @torch.compiler.disable
    def predict(self, step: int) -> torch.Tensor:
        self._fit()
        if self._coef is None or self._shape is None:
            raise RuntimeError("SPECTRUM forecaster fit did not produce coefficients.")

        device = self.features[-1].device
        target_step = torch.tensor([step], device=device, dtype=torch.float32)
        design = self._design(self._tau(target_step)).to(self._coef.dtype)
        predicted = (design @ self._coef).reshape(self._shape)

        if self.config.blend_w < 1.0:
            local = self._local_first_order(step)
            predicted = (1.0 - self.config.blend_w) * local + self.config.blend_w * predicted
        return predicted


class SpectrumState(BaseState):
    """Context-local mutable SPECTRUM state shared by all SPECTRUM hooks."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.forecaster = SpectrumForecaster(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.forecaster.reset()
        self.step_index = -1
        self.should_compute = True
        self.bypass = False
        self.bypass_latched = False
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.peak_history_bytes = 0

    def start_step(self) -> None:
        self.step_index += 1
        if self.step_index >= self.config.num_inference_steps:
            raise ValueError(
                f"SPECTRUM received denoising step {self.step_index}, but config expects "
                f"{self.config.num_inference_steps} steps. Reset the cache state or use a matching config."
            )

        self.should_compute = self.schedule.decide(self.step_index)
        target = self.compute_steps if self.should_compute else self.forecast_steps
        target.append(self.step_index)

    def record_real_feature(self, image_feature: torch.Tensor) -> None:
        self.forecaster.update(self.step_index, image_feature)
        self.peak_history_bytes = max(self.peak_history_bytes, self.forecaster.history_bytes())

    def predict(self) -> torch.Tensor:
        return self.forecaster.predict(self.step_index)


class SpectrumDenoiserHook(ModelHook):
    """Root state owner and fail-closed gate for unsupported FLUX conditioning paths."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._argument_indices: dict[str, int] = {}

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        names = (*_CONTROL_ARGUMENTS, "joint_attention_kwargs")
        self._argument_indices = {name: parameters.index(name) for name in names if name in parameters}
        return module

    def _get_argument(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name in kwargs:
            return kwargs[name]
        index = self._argument_indices.get(name)
        if index is not None and index < len(args):
            return args[index]
        return None

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumState = self.state_manager.get_state()

        bypass = torch.is_grad_enabled()
        if not bypass:
            bypass = any(self._get_argument(name, args, kwargs) is not None for name in _CONTROL_ARGUMENTS)
        if not bypass:
            joint_attention_kwargs = self._get_argument("joint_attention_kwargs", args, kwargs)
            bypass = isinstance(joint_attention_kwargs, dict) and "ip_adapter_image_embeds" in joint_attention_kwargs

        state.bypass = bypass
        try:
            return self.fn_ref.original_forward(*args, **kwargs)
        finally:
            state.bypass = False

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumHeadBlockHook(ModelHook):
    """Advance the schedule and inject the forecast at the first FLUX transformer block."""

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._metadata: TransformerBlockMetadata | None = None

    def initialize_hook(self, module: torch.nn.Module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)
        if self._metadata is None:
            raise RuntimeError("SPECTRUM head-block metadata is unavailable.")

        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        state.start_step()
        if state.should_compute:
            return self.fn_ref.original_forward(*args, **kwargs)

        predicted = state.predict()
        if predicted.shape != hidden_states.shape:
            raise ValueError(
                f"SPECTRUM predicted feature shape {predicted.shape} does not match FLUX image feature shape "
                f"{hidden_states.shape}."
            )
        return _pack_block_output(self._metadata, predicted, encoder_hidden_states)


class SpectrumBlockHook(ModelHook):
    """Skip middle blocks during forecast steps and record the final real feature at the tail block."""

    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.is_tail = is_tail
        self._metadata: TransformerBlockMetadata | None = None

    def initialize_hook(self, module: torch.nn.Module):
        self._metadata = TransformerBlockRegistry.get(unwrap_module(module).__class__)
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)
        if self._metadata is None:
            raise RuntimeError("SPECTRUM block metadata is unavailable.")

        if state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                hidden_states, _ = _get_block_outputs(self._metadata, output)
                # FLUX keeps encoder/image streams separate in its registered block metadata; this tail hidden state is
                # already the final image feature for the transformer stack.
                state.record_real_feature(hidden_states)
            return output

        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        return _pack_block_output(self._metadata, hidden_states, encoder_hidden_states)



class SpectrumUNetDenoiserHook(ModelHook):
    """Root SPECTRUM hook for UNet2DConditionModel / SDXL-style denoisers.

    Full-compute steps run the original UNet forward. Forecast steps skip the expensive
    down/mid/up body and execute only the source-faithful post-feature tail
    ``conv_norm_out -> conv_act -> conv_out``. Unsupported conditioning and autograd
    paths fail closed to the original forward without advancing SPECTRUM state. Classic
    ControlNet down/mid residual pairs are eligible only with the explicit config opt-in.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._argument_indices: dict[str, int] = {}
        self._module_ref: torch.nn.Module | None = None

    def initialize_hook(self, module: torch.nn.Module):
        self._module_ref = module
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        names = (
            "sample",
            *_UNET_RESIDUAL_ARGUMENTS,
            "added_cond_kwargs",
            "cross_attention_kwargs",
            "return_dict",
        )
        self._argument_indices = {name: parameters.index(name) for name in names if name in parameters}
        return module

    def _get_argument(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if name in kwargs:
            return kwargs[name]
        index = self._argument_indices.get(name)
        if index is not None and index < len(args):
            return args[index]
        return None

    def _ip_adapter_scale_signature(self) -> tuple[tuple[str, str], ...]:
        if self._module_ref is None:
            return ()
        unwrapped = unwrap_module(self._module_ref)
        processors = getattr(unwrapped, "attn_processors", {})
        return tuple(
            (name, repr(getattr(processor, "scale")))
            for name, processor in sorted(processors.items())
            if hasattr(processor, "scale")
        )

    def _should_bypass(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        state: SpectrumState = self.state_manager.get_state()
        if state.bypass_latched:
            return True

        def latch() -> bool:
            state.bypass_latched = True
            return True

        if torch.is_grad_enabled():
            return latch()

        down_residuals = self._get_argument("down_block_additional_residuals", args, kwargs)
        mid_residual = self._get_argument("mid_block_additional_residual", args, kwargs)
        intrablock_residuals = self._get_argument("down_intrablock_additional_residuals", args, kwargs)
        added_cond_kwargs = self._get_argument("added_cond_kwargs", args, kwargs)
        sample = self._get_argument("sample", args, kwargs)

        # Pipeline callbacks can change the denoiser batch/shape mid-trajectory.
        # Forecast history from the previous shape must never be reused.
        if torch.is_tensor(sample):
            sample_shape = tuple(sample.shape)
            previous_shape = getattr(state, "_unet_sample_shape", None)
            if previous_shape is None:
                state._unet_sample_shape = sample_shape
            elif previous_shape != sample_shape:
                return latch()

        has_down = down_residuals is not None
        has_mid = mid_residual is not None
        has_controlnet = has_down or has_mid
        has_t2i_adapter = intrablock_residuals is not None
        has_ip_adapter = isinstance(added_cond_kwargs, dict) and "image_embeds" in added_cond_kwargs

        # Partial/legacy ControlNet residual forms are never eligible.
        if has_controlnet and not (has_down and has_mid):
            return latch()

        # Each validated conditioning family has a separate opt-in. Keep unvalidated
        # mixed special-conditioning compositions fail-closed even if every individual
        # family was enabled independently.
        active_special_paths = int(has_controlnet) + int(has_t2i_adapter) + int(has_ip_adapter)
        if active_special_paths > 1:
            return latch()

        if has_controlnet and not state.config.allow_unet_controlnet_residuals:
            return latch()

        # T2I-Adapter can stop passing intrablock residuals partway through one
        # denoising trajectory when adapter_conditioning_factor < 1. If it is not
        # explicitly enabled, latch bypass for the whole run so SPECTRUM cannot
        # start midway after the residuals disappear.
        if has_t2i_adapter and not state.config.allow_unet_t2i_adapter_residuals:
            return latch()

        if has_ip_adapter and not state.config.allow_unet_ip_adapter_image_embeds:
            return latch()

        # IP-Adapter cutoff callbacks mutate attention-processor scales while keeping
        # image_embeds present. Forecasting across that discontinuity is unsafe.
        if has_ip_adapter:
            current_ip_scale_signature = self._ip_adapter_scale_signature()
            previous_ip_scale_signature = getattr(state, "_ip_adapter_scale_signature", None)
            if previous_ip_scale_signature is None:
                state._ip_adapter_scale_signature = current_ip_scale_signature
            elif previous_ip_scale_signature != current_ip_scale_signature:
                return latch()

        # A non-default PEFT scale is applied/unapplied inside the original UNet forward.
        # Forecasting at the root would skip that lifecycle, so keep the whole run fail-closed.
        cross_attention_kwargs = self._get_argument("cross_attention_kwargs", args, kwargs)
        if isinstance(cross_attention_kwargs, dict) and cross_attention_kwargs.get("scale", 1.0) != 1.0:
            return latch()

        return False

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        from ..models.unets.unet_2d_condition import UNet2DConditionOutput

        state: SpectrumState = self.state_manager.get_state()
        if self._should_bypass(args, kwargs):
            state.bypass = True
            try:
                return self.fn_ref.original_forward(*args, **kwargs)
            finally:
                state.bypass = False

        state.start_step()
        if state.should_compute:
            return self.fn_ref.original_forward(*args, **kwargs)

        predicted = state.predict()
        unwrapped = unwrap_module(module)

        if unwrapped.conv_norm_out is not None:
            predicted = unwrapped.conv_norm_out(predicted)
            predicted = unwrapped.conv_act(predicted)
        predicted = unwrapped.conv_out(predicted)

        return_dict = self._get_argument("return_dict", args, kwargs)
        if return_dict is False:
            return (predicted,)
        return UNet2DConditionOutput(sample=predicted)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumUNetFeatureHook(ModelHook):
    """Record the real UNet feature immediately before ``conv_norm_out``."""

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumState = self.state_manager.get_state()
        if state.bypass:
            return args, kwargs
        if state.should_compute:
            if args:
                feature = args[0]
            else:
                feature = kwargs.get("input")
            if feature is None:
                raise RuntimeError("SPECTRUM UNet feature hook could not locate conv_norm_out input.")
            state.record_real_feature(feature)
        return args, kwargs

def _get_block_inputs(
    metadata: TransformerBlockMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = metadata._get_parameter_from_args_kwargs("hidden_states", args, kwargs)
    encoder_hidden_states = None
    if metadata.return_encoder_hidden_states_index is not None:
        encoder_hidden_states = metadata._get_parameter_from_args_kwargs("encoder_hidden_states", args, kwargs)
    return hidden_states, encoder_hidden_states


def _get_block_outputs(metadata: TransformerBlockMetadata, output: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        hidden_states = output[metadata.return_hidden_states_index]
        encoder_hidden_states = None
        if metadata.return_encoder_hidden_states_index is not None:
            encoder_hidden_states = output[metadata.return_encoder_hidden_states_index]
        return hidden_states, encoder_hidden_states
    return output, None


def _pack_block_output(
    metadata: TransformerBlockMetadata, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None
) -> Any:
    if metadata.return_encoder_hidden_states_index is None:
        return hidden_states

    output = [None, None]
    output[metadata.return_hidden_states_index] = hidden_states
    output[metadata.return_encoder_hidden_states_index] = encoder_hidden_states
    return tuple(output)


def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply native-style SPECTRUM caching to supported FLUX and 2D conditional UNet denoisers.

    FLUX uses block-stack hooks. UNet/SDXL uses the official SPECTRUM boundary immediately before
    ``conv_norm_out``: real steps record that feature, while forecast steps skip the UNet body and
    execute only ``conv_norm_out -> conv_act -> conv_out``. Unsupported conditioning paths fail closed.
    """

    from ..models.transformers.transformer_flux import FluxTransformer2DModel
    from ..models.unets.unet_2d_condition import UNet2DConditionModel

    unwrapped_module = unwrap_module(module)

    if isinstance(unwrapped_module, UNet2DConditionModel):
        if unwrapped_module.conv_norm_out is None:
            raise ValueError("SPECTRUM UNet support requires conv_norm_out to expose the source-faithful feature boundary.")

        state_manager = StateManager(SpectrumState, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumUNetDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

        feature_registry = HookRegistry.check_if_exists_or_initialize(unwrapped_module.conv_norm_out)
        feature_registry.register_hook(SpectrumUNetFeatureHook(state_manager), _SPECTRUM_UNET_FEATURE_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to UNet2DConditionModel with %d expected inference steps.",
            config.num_inference_steps,
        )
        return

    if not isinstance(unwrapped_module, FluxTransformer2DModel):
        raise ValueError(
            "SpectrumCacheConfig currently supports FluxTransformer2DModel and UNet2DConditionModel, "
            f"got {type(unwrapped_module)}."
        )

    blocks: list[torch.nn.Module] = []
    for group_name in _FLUX_BLOCK_GROUPS:
        group = getattr(unwrapped_module, group_name, None)
        if isinstance(group, torch.nn.ModuleList):
            blocks.extend(group)

    if len(blocks) < 2:
        raise ValueError("SPECTRUM requires at least two FLUX transformer blocks.")

    state_manager = StateManager(SpectrumState, init_args=(config,))

    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    root_registry.register_hook(SpectrumDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

    head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
    head_registry.register_hook(SpectrumHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)

    for block in blocks[1:-1]:
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(SpectrumBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)

    tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
    tail_registry.register_hook(SpectrumBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

    logger.debug(
        "Applied SPECTRUM cache to FLUX transformer with %d blocks and %d expected inference steps.",
        len(blocks),
        config.num_inference_steps,
    )
