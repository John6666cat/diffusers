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

"""Opt-in Qwen-Image 2.1 native-KV SPECTRUM adapter.

The first (extract) call runs every block. On later cached calls, an explicit
55%-target schedule may forecast the last block's *target-only* feature.
Only the 32 transformer blocks are skipped; the transformer's output norm
and projection still run on every denoising step.
"""

from __future__ import annotations

from typing import Any

import torch

from .hooks import BaseState, ModelHook, StateManager
from .spectrum_cache import SpectrumCacheConfig, SpectrumForecaster, SpectrumSchedule


class SpectrumQwen21State(BaseState):
    """Per-cache-context history; a guard latches to real compute for the run."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.forecaster = SpectrumForecaster(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.forecaster.reset()
        self.step_index = -1
        self._cache_identity: int | None = None
        self._feature_shape: tuple[int, ...] | None = None
        self._compute = True
        self._prediction: torch.Tensor | None = None
        self.guard_reason: str | None = None
        self.records: list[dict[str, Any]] = []
        self.block_real_calls = 0
        self.block_skipped_calls = 0
        self.peak_history_bytes = 0

    def latch(self, reason: str) -> None:
        if self.guard_reason is None:
            self.guard_reason = str(reason)

    @staticmethod
    def _tensor(kwargs: dict, key: str) -> torch.Tensor:
        value = kwargs.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"expected tensor keyword argument {key}")
        return value

    def _begin(self, module: torch.nn.Module, kwargs: dict) -> None:
        self.step_index += 1
        self._prediction = None
        self._compute = True
        mode = kwargs.get("kv_cache_mode")
        expected_mode = "extract" if self.step_index == 0 else "cached"
        if self.step_index >= self.config.num_inference_steps:
            self.latch("more transformer calls than declared denoising steps")
        if mode != expected_mode:
            self.latch(f"native-KV mode changed: expected {expected_mode}, got {mode!r}")
        layer_cache = kwargs.get("layer_cache")
        if layer_cache is None:
            self.latch("missing native-KV layer cache")
        elif self._cache_identity is None:
            self._cache_identity = id(layer_cache)
        elif id(layer_cache) != self._cache_identity:
            self.latch("native-KV cache identity changed")
        if module.training or torch.is_grad_enabled():
            self.latch("training/autograd is outside the Qwen-Image 2.1 research route")

        try:
            states = self._tensor(kwargs, "hidden_states")
            target_mask = self._tensor(kwargs, "target_token_mask")
            if target_mask.dtype != torch.bool:
                raise ValueError("target_token_mask must be boolean")
            target_tokens = int(target_mask.sum().item())
            if target_tokens < 1 or target_tokens > states.shape[1]:
                raise ValueError("invalid target-token span")
            shape = (states.shape[0], target_tokens, states.shape[2])
            if self._feature_shape is None:
                self._feature_shape = shape
            elif shape != self._feature_shape:
                self.latch(f"target-only feature geometry changed: {self._feature_shape} -> {shape}")
        except (ValueError, IndexError, TypeError) as error:
            self.latch(f"feature guard: {error}")
            shape = None

        scheduled_real = self.schedule.decide(self.step_index) if self.guard_reason is None else True
        if mode == "extract":
            scheduled_real = True
        if not scheduled_real and self.forecaster.features and shape is not None:
            try:
                candidate = self.forecaster.predict(self.step_index)
                if (
                    tuple(candidate.shape) != shape
                    or candidate.dtype != states.dtype
                    or candidate.device != states.device
                    or not torch.isfinite(candidate).all().item()
                ):
                    raise ValueError("non-finite or structurally changed target feature")
                self._prediction = candidate
                self._compute = False
            except Exception as error:
                self.latch(f"predictor failure: {type(error).__name__}: {error}")
                self.forecaster.reset()

        self.records.append(
            {
                "step": self.step_index,
                "native_kv_mode": mode,
                "scheduled_real": bool(scheduled_real),
                "predicted_body_used": not self._compute,
                "guard_latched": self.guard_reason is not None,
            }
        )

    def block(self, module: torch.nn.Module, index: int, block_count: int, original, args: tuple, kwargs: dict):
        # This Source revision invokes Qwen-Image 2.1 blocks by keyword.
        if args:
            self.latch("unexpected positional transformer-block arguments")
            return original(*args, **kwargs)
        if index == 0:
            self._begin(module, kwargs)
        if self._compute:
            result = original(**kwargs)
            self.block_real_calls += 1
            if index == block_count - 1 and self.guard_reason is None:
                try:
                    target_tokens = int(self._tensor(kwargs, "target_token_mask").sum().item())
                    feature = result[:, -target_tokens:, :]
                    if not torch.isfinite(feature).all().item():
                        raise FloatingPointError("non-finite real target feature")
                    self.forecaster.update(self.step_index, feature)
                    self.peak_history_bytes = max(self.peak_history_bytes, self.forecaster.history_bytes())
                except Exception as error:
                    self.latch(f"history update failure: {type(error).__name__}: {error}")
                    self.forecaster.reset()
            return result
        self.block_skipped_calls += 1
        if index == 0:
            if self._prediction is None:
                raise RuntimeError("missing SPECTRUM target feature after forecast decision")
            return self._prediction
        return self._tensor(kwargs, "hidden_states")

    def summary(self) -> dict[str, Any]:
        return {
            "declared_steps": self.config.num_inference_steps,
            "observed_steps": len(self.records),
            "prediction_call_count": sum(x["predicted_body_used"] for x in self.records),
            "real_call_count": sum(not x["predicted_body_used"] for x in self.records),
            "block_real_calls": self.block_real_calls,
            "block_skipped_calls": self.block_skipped_calls,
            "guard_reason": self.guard_reason,
            "peak_feature_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumQwen21DenoiserHook(ModelHook):
    """Root hook: temporarily intercept blocks for one native-KV forward."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumQwen21State = self.state_manager.get_state()
        # A fresh native prefill identifies a new run even when the same
        # cache context is reused by a second pipeline invocation.
        if kwargs.get("kv_cache_mode") == "extract" and state.step_index >= 0:
            state.reset()
        # Unsupported call signatures take the original full-compute path.
        if args or kwargs.get("kv_cache") is None or kwargs.get("kv_cache_mode") not in {"extract", "cached"}:
            state.latch("missing native KV or unsupported transformer call signature")
            return self.fn_ref.original_forward(*args, **kwargs)
        if kwargs.get("attention_kwargs"):
            state.latch("nonempty attention_kwargs is not qualified")
        if module.training or torch.is_grad_enabled():
            state.latch("training/autograd is not qualified")

        blocks = list(module.transformer_blocks)
        originals: list[tuple[torch.nn.Module, Any, bool]] = []
        try:
            for index, block in enumerate(blocks):
                original = block.forward
                was_instance_attr = "forward" in vars(block)

                def wrapped(*b_args, _index=index, _original=original, **b_kwargs):
                    return state.block(module, _index, len(blocks), _original, b_args, b_kwargs)

                block.forward = wrapped
                originals.append((block, original, was_instance_attr))
            return self.fn_ref.original_forward(**kwargs)
        finally:
            for block, original, was_instance_attr in reversed(originals):
                if was_instance_attr:
                    block.forward = original
                else:
                    delattr(block, "forward")

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module
