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

"""Qualified historical Pony V7 / AuraFlow SPECTRUM adapter.

The single-DiT forecast boundary and the qualified standard-CFG profiles
remain isolated from the focused SPECTRUM family dispatcher.
"""

import inspect
from typing import Any

import torch

from ..utils import logging
from ..utils.torch_utils import unwrap_module
from .hooks import BaseState, HookRegistry, ModelHook, StateManager
from .spectrum_cache import (
    SpectrumCacheConfig,
    SpectrumForecaster,
    SpectrumSchedule,
    _SPECTRUM_BLOCK_HOOK,
    _SPECTRUM_DENOISER_HOOK,
    _SPECTRUM_HEAD_BLOCK_HOOK,
    _spectrum_tensor_identity,
)

logger = logging.get_logger(__name__)


class SpectrumPonyV7State(BaseState):
    """Context-local state for the qualified Pony V7 AuraFlow CFG route."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.forecaster = SpectrumForecaster(self.config)
        self.step_index = -1
        self.should_compute = True
        self.bypass = False
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.conditioning_identity: Any = None
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.real_block_executions = 0
        self.bypassed_block_executions = 0
        self.peak_history_bytes = 0
        self.current_forecast_used = False

    def latch(self, reason: str) -> None:
        logical_step = max(int(self.step_index), 0)
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = logical_step
        record = {"step": logical_step, "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)

    def prepare_call(self, conditioning_identity: Any) -> None:
        self.step_index += 1
        self.current_forecast_used = False

        if self.step_index >= self.config.num_inference_steps:
            self.latch(
                f"logical step {self.step_index} outside configured inference-step range "
                f"{self.config.num_inference_steps}"
            )

        if self.conditioning_identity is None:
            self.conditioning_identity = conditioning_identity
        elif self.conditioning_identity != conditioning_identity:
            self.latch("Pony V7 conditioning identity changed within one cache context")

        if self.guard_latched:
            self.should_compute = True
            return

        decision = bool(self.schedule.decide(self.step_index))
        self.should_compute = decision
        target = self.compute_steps if decision else self.forecast_steps
        target.append(self.step_index)

    def record_real_feature(self, feature: torch.Tensor) -> None:
        self.forecaster.update(self.step_index, feature.detach())
        self.peak_history_bytes = max(self.peak_history_bytes, self.forecaster.history_bytes())

    def predict(self) -> torch.Tensor:
        return self.forecaster.predict(self.step_index)

    def finish_call(self) -> None:
        self.records.append(
            {
                "logical_step": int(self.step_index),
                "scheduled_full_compute": bool(self.should_compute),
                "predicted_body_used": bool(self.current_forecast_used),
                "guard_latched": bool(self.guard_latched),
                "fallback_reason": self.guard_reasons[-1]["reason"] if self.guard_latched else None,
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "prediction_call_count": sum(bool(r["predicted_body_used"]) for r in self.records),
            "full_call_count": sum(not bool(r["predicted_body_used"]) for r in self.records),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "real_block_executions": self.real_block_executions,
            "bypassed_block_executions": self.bypassed_block_executions,
            "total_block_slots": self.real_block_executions + self.bypassed_block_executions,
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumPonyV7DenoiserHook(ModelHook):
    """Fail-closed root adapter for the qualified Pony V7 standard CFG route."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._argument_indices: dict[str, int] = {}

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        names = ("hidden_states", "encoder_hidden_states", "timestep", "attention_kwargs", "return_dict")
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
        state: SpectrumPonyV7State = self.state_manager.get_state()
        hidden_states = self._get_argument("hidden_states", args, kwargs)
        encoder_hidden_states = self._get_argument("encoder_hidden_states", args, kwargs)
        timestep = self._get_argument("timestep", args, kwargs)
        attention_kwargs = self._get_argument("attention_kwargs", args, kwargs)

        state.prepare_call(_spectrum_tensor_identity(encoder_hidden_states))

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for Pony V7 SPECTRUM")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 4:
            state.latch("Pony V7 hidden_states must be rank 4")
        elif hidden_states.shape[0] != 2:
            state.latch("Pony V7 native candidate is qualified only for batch-2 standard CFG")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            state.latch("Pony V7 encoder_hidden_states must be rank 3")
        elif encoder_hidden_states.shape[0] != 2:
            state.latch("Pony V7 encoder_hidden_states must contain exactly negative/positive CFG lanes")
        if not torch.is_tensor(timestep) or timestep.ndim != 1 or timestep.shape[0] != 2:
            state.latch("Pony V7 timestep must be a rank-1 batch-2 tensor")
        if attention_kwargs:
            state.latch("non-empty attention kwargs / LoRA route is not qualified for Pony V7 SPECTRUM")

        state.bypass = state.guard_latched
        try:
            output = self.fn_ref.original_forward(*args, **kwargs)
        finally:
            state.bypass = False

        state.finish_call()
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class _SpectrumPonyV7SingleBlockHookBase(ModelHook):
    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._hidden_index: int | None = None

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        self._hidden_index = parameters.index("hidden_states")
        return module

    def _hidden(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        return args[self._hidden_index]


class SpectrumPonyV7HeadBlockHook(_SpectrumPonyV7SingleBlockHookBase):
    """Inject the full combined-state forecast at the first single-DiT block."""

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumPonyV7State = self.state_manager.get_state()
        hidden_states = self._hidden(args, kwargs)

        if state.bypass or state.guard_latched or state.should_compute:
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        try:
            predicted = state.predict().to(device=hidden_states.device, dtype=hidden_states.dtype)
            if tuple(predicted.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    f"predicted combined-state shape {tuple(predicted.shape)} does not match "
                    f"{tuple(hidden_states.shape)}"
                )
            if not bool(torch.isfinite(predicted).all()):
                raise ValueError("predicted combined state is non-finite")
        except Exception as error:
            state.latch(f"forecast injection failed: {type(error).__name__}: {error}")
            state.should_compute = True
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        state.current_forecast_used = True
        state.bypassed_block_executions += 1
        return predicted


class SpectrumPonyV7SingleBlockHook(_SpectrumPonyV7SingleBlockHookBase):
    """Bypass remaining single-DiT blocks or record the real final combined state."""

    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__(state_manager)
        self.is_tail = is_tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumPonyV7State = self.state_manager.get_state()
        hidden_states = self._hidden(args, kwargs)

        if state.bypass or state.guard_latched or state.should_compute:
            state.real_block_executions += 1
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail and not state.bypass:
                if not torch.is_tensor(output):
                    state.latch(f"Pony V7 final single-DiT block returned {type(output)}")
                else:
                    state.record_real_feature(output)
            return output

        state.bypassed_block_executions += 1
        return hidden_states


def apply_spectrum_cache_pony_v7(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, including the qualified Pony V7 / AuraFlow adapter."""

    from ..models.transformers.auraflow_transformer_2d import AuraFlowTransformer2DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, AuraFlowTransformer2DModel):
        raise TypeError("Pony V7 SPECTRUM adapter requires AuraFlowTransformer2DModel.")

    expected_signature = {
        "patch_size": 2,
        "in_channels": 4,
        "num_mmdit_layers": 4,
        "num_single_dit_layers": 32,
        "attention_head_dim": 256,
        "num_attention_heads": 12,
        "joint_attention_dim": 2048,
        "caption_projection_dim": 3072,
        "out_channels": 4,
    }
    observed_signature = {key: getattr(unwrapped_module.config, key) for key in expected_signature}
    if observed_signature != expected_signature:
        raise ValueError(
            "The current SPECTRUM Pony V7 adapter is qualified only for the pinned AuraFlow architecture. "
            f"Expected {expected_signature}, got {observed_signature}."
        )

    qualified_schedules = (
        (20,),
        (17, 19, 21, 23),
    )
    observed_steps = tuple(config.forecast_step_indices or ())
    qualified_steps = observed_steps if observed_steps in qualified_schedules else None
    profile_ok = (
        config.num_inference_steps == 30
        and tuple(config.forecast_step_indices or ()) == qualified_steps
        and float(config.window_size) == 3.0
        and int(config.degree) == 2
        and float(config.ridge_lambda) == 0.1
        and float(config.blend_w) == 0.25
        and int(config.history_limit) == 8
        and float(config.coordinate_max) == 8.0
        and int(config.warmup_steps) == 0
        and float(config.flex_window) == 0.0
        and int(config.tail_actual_steps) == 0
    )
    if not profile_ok:
        raise ValueError(
            'The current Pony V7 native adapter accepts only the qualified Pony V7 schedules for the exact 30-step d2/window3/blend0.25 profile.'
        )

    joint_blocks = list(unwrapped_module.joint_transformer_blocks)
    single_blocks = list(unwrapped_module.single_transformer_blocks)
    if len(joint_blocks) != 4 or len(single_blocks) != 32:
        raise ValueError("SPECTRUM Pony V7 support requires exactly 4 joint + 32 single transformer blocks.")

    state_manager = StateManager(SpectrumPonyV7State, init_args=(config,))

    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    root_registry.register_hook(SpectrumPonyV7DenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

    head_registry = HookRegistry.check_if_exists_or_initialize(single_blocks[0])
    head_registry.register_hook(SpectrumPonyV7HeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)

    for block in single_blocks[1:-1]:
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(SpectrumPonyV7SingleBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)

    tail_registry = HookRegistry.check_if_exists_or_initialize(single_blocks[-1])
    tail_registry.register_hook(
        SpectrumPonyV7SingleBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK
    )

    logger.debug(
        "Applied SPECTRUM cache to qualified Pony V7 AuraFlow transformer with %d joint and %d single blocks.",
        len(joint_blocks),
        len(single_blocks),
    )

# --- end Pony V7 / AuraFlow native SPECTRUM candidate -------------------------------------------
