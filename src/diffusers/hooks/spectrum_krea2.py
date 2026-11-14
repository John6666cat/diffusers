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

"""Experimental Krea 2 Raw paired-CFG adapter (opt-in, research-only; not Product/Stable).

Both positive and negative denoiser calls use the *same* ``cache_context('krea')``
but retain independent feature histories. The first and last transformer blocks
form the body feature boundary, as in the historical 0.41 Krea adapter.
"""

from typing import Any

import torch

from .hooks import BaseState, ModelHook, StateManager
from .spectrum_cache import (
    SpectrumCacheConfig,
    SpectrumForecaster,
    SpectrumSchedule,
    _spectrum_tensor_identity,
    _spectrum_tensor_signature,
)


class SpectrumKrea2RawState(BaseState):
    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.forecasters = {lane: SpectrumForecaster(self.config) for lane in ("positive", "negative")}
        self.call_index = -1
        self.step_index = -1
        self.current_lane: str | None = None
        self.should_compute = True
        self.bypass = False
        self.guard_latched = False
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.lane_identities: dict[str, Any] = {}
        self.decisions: dict[int, bool] = {}
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.peak_history_bytes = 0

    def latch(self, reason: str) -> None:
        self.guard_latched = True
        record = {"step": max(self.step_index, 0), "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)
        self.should_compute = True

    def prepare_call(self, context: Any, conditioning_identity: Any) -> None:
        self.call_index += 1
        self.step_index, phase = divmod(self.call_index, 2)
        self.current_lane = "positive" if phase == 0 else "negative"

        if (
            context.name != "krea"
            or context.step_index != self.step_index
            or context.num_inference_steps != self.config.num_inference_steps
        ):
            self.latch("Krea native cache context / dual-lane logical-step mismatch")
        if self.step_index >= self.config.num_inference_steps:
            self.latch("Krea denoiser calls exceeded configured logical-step count")

        previous = self.lane_identities.get(self.current_lane)
        if previous is None:
            self.lane_identities[self.current_lane] = conditioning_identity
        elif previous != conditioning_identity:
            self.latch(f"{self.current_lane} conditioning identity changed")
        if (
            len(self.lane_identities) == 2
            and self.lane_identities["positive"] == self.lane_identities["negative"]
        ):
            self.latch("positive and negative conditioning identities are not distinct")

    def start_step(self) -> None:
        if self.guard_latched:
            self.should_compute = True
            return
        if self.current_lane == "positive":
            self.should_compute = bool(self.schedule.decide(self.step_index))
            self.decisions[self.step_index] = self.should_compute
            target = self.compute_steps if self.should_compute else self.forecast_steps
            target.append(self.step_index)
        elif self.step_index in self.decisions:
            self.should_compute = self.decisions[self.step_index]
        else:
            self.latch("negative lane preceded positive lane")

        if not self.should_compute and not self.forecasters[self.current_lane].features:
            self.latch("cannot forecast before this CFG lane has a full-compute feature")

    def record_real_feature(self, feature: torch.Tensor) -> None:
        self.forecasters[self.current_lane].update(self.step_index, feature)
        usage = sum(predictor.history_bytes() for predictor in self.forecasters.values())
        self.peak_history_bytes = max(self.peak_history_bytes, usage)

    def predict(self) -> torch.Tensor:
        return self.forecasters[self.current_lane].predict(self.step_index)

    def summary(self) -> dict[str, Any]:
        return {
            "guard_latched": self.guard_latched,
            "guard_reasons": list(self.guard_reasons),
            "prediction_call_count": sum(bool(row["predicted_body_used"]) for row in self.records),
            "full_call_count": sum(not row["predicted_body_used"] for row in self.records),
            "lane_prediction_call_count": {
                lane: sum(row["predicted_body_used"] and row["lane"] == lane for row in self.records)
                for lane in ("positive", "negative")
            },
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumKrea2RawDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self.latest_summary: dict[str, Any] | None = None

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        state: SpectrumKrea2RawState = self.state_manager.get_state()
        context = self.state_manager.context
        state.prepare_call(context, _spectrum_tensor_identity(encoder_hidden_states))
        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
            "timestep": _spectrum_tensor_signature(timestep),
            "position_ids": _spectrum_tensor_signature(position_ids),
            "encoder_attention_mask": _spectrum_tensor_signature(encoder_attention_mask),
        }
        if state.signature is None:
            state.signature = signature
        elif signature != state.signature:
            state.latch("Krea context-local input tensor signatures changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not supported")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            state.latch("Krea image hidden_states must be rank 3")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 4:
            state.latch("Krea encoder_hidden_states must be rank 4")
        elif encoder_hidden_states.shape[2] != 12:
            state.latch("Krea encoder_hidden_states must have 12 text layers")
        if (
            torch.is_tensor(hidden_states) and hidden_states.ndim == 3
            and torch.is_tensor(encoder_hidden_states) and encoder_hidden_states.ndim == 4
            and hidden_states.shape[0] != encoder_hidden_states.shape[0]
        ):
            state.latch("image/text batch sizes differ")
        if not torch.is_tensor(timestep) or timestep.ndim != 1:
            state.latch("Krea timestep must be rank 1")
        elif torch.is_tensor(hidden_states) and timestep.shape[0] != hidden_states.shape[0]:
            state.latch("Krea timestep batch size differs")
        if not torch.is_tensor(position_ids) or position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            state.latch("Krea position_ids shape mismatch")
        elif (
            torch.is_tensor(hidden_states) and hidden_states.ndim == 3
            and torch.is_tensor(encoder_hidden_states) and encoder_hidden_states.ndim == 4
            and position_ids.shape[0] != hidden_states.shape[1] + encoder_hidden_states.shape[1]
        ):
            state.latch("Krea position_ids length mismatch")
        if encoder_attention_mask is not None and (
            not torch.is_tensor(encoder_attention_mask)
            or encoder_attention_mask.ndim != 2
            or not torch.is_tensor(encoder_hidden_states)
            or encoder_hidden_states.ndim != 4
            or tuple(encoder_attention_mask.shape) != tuple(encoder_hidden_states.shape[:2])
        ):
            state.latch("Krea text mask shape mismatch")
        if attention_kwargs:
            state.latch("attention kwargs / LoRA route is not supported")

        kwargs = dict(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )
        bypass = state.guard_latched
        state.bypass = bypass
        try:
            result = self.fn_ref.original_forward(**kwargs)
        finally:
            state.bypass = False
        state.records.append(
            {
                "logical_step": state.step_index,
                "lane": state.current_lane,
                "scheduled_full_compute": state.step_index not in state.forecast_steps,
                "predicted_body_used": bool(not state.should_compute and not state.guard_latched),
                "guard_latched": state.guard_latched,
                "fallback_reason": state.guard_reasons[-1]["reason"] if state.guard_latched else None,
            }
        )
        self.latest_summary = state.summary()
        return result

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumKrea2RawBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, *, head: bool = False, tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.head = head
        self.tail = tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumKrea2RawState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        if self.head:
            state.start_step()
        if state.should_compute:
            result = self.fn_ref.original_forward(*args, **kwargs)
            if self.tail:
                state.record_real_feature(result)
            return result

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if self.head:
            try:
                predicted = state.predict()
                if predicted.shape != hidden_states.shape:
                    raise ValueError("Krea predicted body feature shape mismatch")
            except (RuntimeError, ValueError) as exc:
                state.latch(f"Krea forecast failure: {type(exc).__name__}: {exc}")
                return self.fn_ref.original_forward(*args, **kwargs)
            return predicted
        return hidden_states
