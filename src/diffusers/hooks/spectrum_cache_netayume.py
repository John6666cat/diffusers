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

"""Qualified historical Neta Yume / Lumina2 SPECTRUM adapter.

Kept separate from the active-family dispatcher to make the historical
paired-CFG feature boundary and its narrow profile easy to review.
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

class SpectrumNetaYumeState(BaseState):
    """Context-local dual-lane state for the qualified Neta Yume / Lumina2 CFG route."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.forecasters = {
            "positive": SpectrumForecaster(self.config),
            "negative": SpectrumForecaster(self.config),
        }
        self.call_index = -1
        self.step_index = -1
        self.current_lane: str | None = None
        self.should_compute = True
        self.bypass = False
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.lane_identities: dict[str, Any] = {}
        self.decisions: dict[int, bool] = {}
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
        self.call_index += 1
        self.step_index = self.call_index // 2
        self.current_lane = "positive" if self.call_index % 2 == 0 else "negative"
        self.current_forecast_used = False

        if self.step_index >= self.config.num_inference_steps:
            self.latch(
                f"logical step {self.step_index} outside configured inference-step range "
                f"{self.config.num_inference_steps}"
            )

        previous = self.lane_identities.get(self.current_lane)
        if previous is None:
            self.lane_identities[self.current_lane] = conditioning_identity
        elif previous != conditioning_identity:
            self.latch(f"{self.current_lane} conditioning identity changed")

        if (
            len(self.lane_identities) == 2
            and self.lane_identities["positive"] == self.lane_identities["negative"]
        ):
            self.latch("Neta Yume CFG positive/negative conditioning identities are not distinct")

        if self.guard_latched:
            self.should_compute = True
            return

        if self.current_lane == "positive":
            if self.step_index in self.decisions:
                self.latch(f"duplicate positive CFG lane at logical step {self.step_index}")
                self.should_compute = True
                return
            decision = bool(self.schedule.decide(self.step_index))
            self.decisions[self.step_index] = decision
            target = self.compute_steps if decision else self.forecast_steps
            target.append(self.step_index)
        else:
            if self.step_index not in self.decisions:
                self.latch("Neta Yume negative CFG lane arrived before its positive lane")
                self.should_compute = True
                return
            decision = self.decisions[self.step_index]
        self.should_compute = decision

    def _forecaster(self) -> SpectrumForecaster:
        if self.current_lane is None:
            raise RuntimeError("SPECTRUM Neta Yume forecaster has no active CFG lane.")
        return self.forecasters[self.current_lane]

    def record_real_feature(self, feature: torch.Tensor) -> None:
        self._forecaster().update(self.step_index, feature)
        total = sum(forecaster.history_bytes() for forecaster in self.forecasters.values())
        self.peak_history_bytes = max(self.peak_history_bytes, total)

    def predict(self) -> torch.Tensor:
        return self._forecaster().predict(self.step_index)

    def finish_call(self) -> None:
        self.records.append(
            {
                "call_index": int(self.call_index),
                "logical_step": int(self.step_index),
                "lane": self.current_lane,
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
            "lane_prediction_call_count": {
                lane: sum(
                    bool(r["predicted_body_used"]) and r.get("lane") == lane for r in self.records
                )
                for lane in ("positive", "negative")
            },
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "real_block_executions": self.real_block_executions,
            "bypassed_block_executions": self.bypassed_block_executions,
            "total_block_slots": self.real_block_executions + self.bypassed_block_executions,
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumNetaYumeDenoiserHook(ModelHook):
    """Fail-closed root adapter for the qualified Neta Yume standard CFG route."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._argument_indices: dict[str, int] = {}

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        names = (
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "attention_kwargs",
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

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumNetaYumeState = self.state_manager.get_state()
        hidden_states = self._get_argument("hidden_states", args, kwargs)
        timestep = self._get_argument("timestep", args, kwargs)
        encoder_hidden_states = self._get_argument("encoder_hidden_states", args, kwargs)
        encoder_attention_mask = self._get_argument("encoder_attention_mask", args, kwargs)
        attention_kwargs = self._get_argument("attention_kwargs", args, kwargs)

        context = self.state_manager.context
        state.prepare_call(_spectrum_tensor_identity(encoder_hidden_states))

        # Low-level callers may omit step metadata. Native Pipeline contexts
        # supply it; any step/CFG mismatch uses real forward.
        if context.step_index is not None and (
            context.name != "cond_uncond"
            or type(context.step_index) is not int
            or context.step_index != state.step_index
            or type(context.num_inference_steps) is not int
            or context.num_inference_steps != state.config.num_inference_steps
        ):
            state.latch("Neta Yume native CFG/step cache context is not qualified")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for Neta Yume SPECTRUM")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 4:
            state.latch("Neta Yume hidden_states must be rank 4")
        elif hidden_states.shape[0] != 1:
            state.latch("Neta Yume native candidate is qualified only for batch size 1")
        if not torch.is_tensor(timestep) or timestep.ndim != 1 or timestep.shape[0] != 1:
            state.latch("Neta Yume timestep must be a rank-1 batch-size-1 tensor")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            state.latch("Neta Yume encoder_hidden_states must be rank 3")
        if not torch.is_tensor(encoder_attention_mask) or encoder_attention_mask.ndim != 2:
            state.latch("Neta Yume encoder_attention_mask must be rank 2")
        elif torch.is_tensor(encoder_hidden_states) and (
            encoder_attention_mask.shape[0] != encoder_hidden_states.shape[0]
            or encoder_attention_mask.shape[1] != encoder_hidden_states.shape[1]
        ):
            state.latch("Neta Yume encoder attention-mask shape contract changed")
        if attention_kwargs:
            state.latch("non-empty attention kwargs / LoRA route is not qualified for Neta Yume SPECTRUM")

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


class _SpectrumNetaYumeBlockHookBase(ModelHook):
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


class SpectrumNetaYumeHeadBlockHook(_SpectrumNetaYumeBlockHookBase):
    """Inject the dual-lane forecast at the first joint Lumina2 block."""

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumNetaYumeState = self.state_manager.get_state()
        hidden_states = self._hidden(args, kwargs)

        if state.bypass or state.guard_latched or state.should_compute:
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        try:
            predicted = state.predict().to(device=hidden_states.device, dtype=hidden_states.dtype)
            if tuple(predicted.shape) != tuple(hidden_states.shape):
                raise ValueError(
                    f"predicted joint-state shape {tuple(predicted.shape)} does not match {tuple(hidden_states.shape)}"
                )
            if not bool(torch.isfinite(predicted).all()):
                raise ValueError("predicted joint state is non-finite")
        except Exception as error:
            state.latch(f"forecast injection failed: {type(error).__name__}: {error}")
            state.should_compute = True
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        state.current_forecast_used = True
        state.bypassed_block_executions += 1
        return predicted


class SpectrumNetaYumeBlockHook(_SpectrumNetaYumeBlockHookBase):
    """Bypass remaining joint blocks or record the real final joint state."""

    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__(state_manager)
        self.is_tail = is_tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumNetaYumeState = self.state_manager.get_state()
        hidden_states = self._hidden(args, kwargs)

        if state.bypass or state.guard_latched or state.should_compute:
            state.real_block_executions += 1
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail and not state.bypass:
                if not torch.is_tensor(output):
                    state.latch(f"Neta Yume final joint block returned {type(output)}")
                else:
                    state.record_real_feature(output)
            return output

        state.bypassed_block_executions += 1
        return hidden_states


def apply_spectrum_cache_netayume(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, including the qualified Neta Yume / Lumina2 dual-lane adapter."""

    from ..models.transformers.transformer_lumina2 import Lumina2Transformer2DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, Lumina2Transformer2DModel):
        raise TypeError("Neta Yume SPECTRUM adapter requires Lumina2Transformer2DModel.")

    expected_signature = {
        "sample_size": 128,
        "patch_size": 2,
        "in_channels": 16,
        "out_channels": None,
        "hidden_size": 2304,
        "num_layers": 26,
        "num_refiner_layers": 2,
        "num_attention_heads": 24,
        "num_kv_heads": 8,
        "multiple_of": 256,
        "ffn_dim_multiplier": None,
        "norm_eps": 1e-5,
        "scaling_factor": 1.0,
        "cap_feat_dim": 2304,
    }
    observed_signature = {key: getattr(unwrapped_module.config, key) for key in expected_signature}
    if observed_signature != expected_signature:
        raise ValueError(
            "The current SPECTRUM Neta Yume adapter is qualified only for the pinned Lumina2 architecture. "
            f"Expected {expected_signature}, got {observed_signature}."
        )
    if tuple(unwrapped_module.config.axes_dim_rope) != (32, 32, 32):
        raise ValueError("Neta Yume SPECTRUM requires axes_dim_rope=(32,32,32).")
    if tuple(unwrapped_module.config.axes_lens) != (300, 512, 512):
        raise ValueError("Neta Yume SPECTRUM requires axes_lens=(300,512,512).")

    qualified_schedules = (
        (15, 20, 23, 25, 27, 36),
        (15, 18, 20, 22, 24, 26, 28, 30, 36),
    )
    observed_steps = tuple(config.forecast_step_indices or ())
    qualified_steps = observed_steps if observed_steps in qualified_schedules else None
    profile_ok = (
        config.num_inference_steps == 50
        and tuple(config.forecast_step_indices or ()) == qualified_steps
        and float(config.window_size) == 3.0
        and int(config.degree) == 2
        and float(config.ridge_lambda) == 0.1
        and float(config.blend_w) == 0.5
        and int(config.history_limit) == 8
        and float(config.coordinate_max) == 50.0
        and int(config.warmup_steps) == 0
        and float(config.flex_window) == 0.0
        and int(config.tail_actual_steps) == 0
    )
    if not profile_ok:
        raise ValueError(
            "The current Neta Yume native adapter accepts only the qualified Neta Yume schedules for the exact "
            "50-step d2/window3/blend0.5 profile."
        )

    blocks = list(unwrapped_module.layers)
    if len(blocks) != 26:
        raise ValueError("SPECTRUM Neta Yume support requires exactly 26 joint transformer blocks.")
    if len(unwrapped_module.context_refiner) != 2 or len(unwrapped_module.noise_refiner) != 2:
        raise ValueError("SPECTRUM Neta Yume requires the native 2+2 context/noise refiner topology.")

    state_manager = StateManager(SpectrumNetaYumeState, init_args=(config,))

    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    root_registry.register_hook(SpectrumNetaYumeDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

    head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
    head_registry.register_hook(SpectrumNetaYumeHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)

    for block in blocks[1:-1]:
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(SpectrumNetaYumeBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)

    tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
    tail_registry.register_hook(SpectrumNetaYumeBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

    logger.debug(
        "Applied SPECTRUM cache to qualified Neta Yume Lumina2 transformer with %d joint blocks and %d expected steps.",
        len(blocks),
        config.num_inference_steps,
    )

# --- end Neta Yume / Lumina2 native SPECTRUM candidate -------------------------------------------
