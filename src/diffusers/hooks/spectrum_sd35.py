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

"""Optional SD3/SD3.5 transformer-output prediction during denoising.

This gate is local to a pipeline invocation. Its predictor is updated only with
actual transformer outputs; unsupported or changing conditions revert to real
compute. The caller controls whether to enable the experimental path.
"""

import math
from collections.abc import Callable
from typing import Any

import torch

from .spectrum_cache import SpectrumCacheConfig, SpectrumForecaster


class SD35SpectrumGate:
    def __init__(self, config: SpectrumCacheConfig):
        if not isinstance(config, SpectrumCacheConfig):
            raise TypeError("SD3 SPECTRUM requires a SpectrumCacheConfig")
        if config.forecast_step_indices is None or config.tail_actual_steps < 2:
            raise ValueError("SD3 SPECTRUM requires explicit forecast steps and at least two real tail steps")
        self.config = config
        self.forecast_steps = frozenset(config.forecast_step_indices)
        self.final_real_steps = config.tail_actual_steps
        self.forecaster = SpectrumForecaster(config)
        self._reset()

    def _reset(self) -> None:
        self._schedule = None
        self._cfg = None
        self._signature = None
        self._inference_conditioning_snapshot = None
        self._last_real = None
        self._last_index = -1
        self._invalidated = True
        self._refresh_cooldown = 2
        self._drift_latched = False
        self._prediction_failure_latched = False
        try:
            self.forecaster.reset()
        except Exception:
            self._prediction_failure_latched = True
        self.counts = {"real": 0, "forecast": 0, "fallback": 0}

    @staticmethod
    def _scalar(t: Any) -> float:
        if isinstance(t, torch.Tensor):
            if t.numel() != 1:
                raise ValueError("non-scalar timestep")
            t = t.detach().cpu().item()
        result = float(t)
        if not math.isfinite(result):
            raise ValueError("non-finite timestep")
        return result

    def begin(self, *, timesteps: Any, do_classifier_free_guidance: bool) -> None:
        self._reset()
        self._schedule = tuple(self._scalar(t) for t in timesteps)
        if not self._schedule:
            raise ValueError("empty schedule")
        self._cfg = bool(do_classifier_free_guidance)
        if self.config.num_inference_steps != len(self._schedule):
            # A profile for a different horizon must not predict silently.
            self._drift_latched = True

    def invalidate(self, reason: str = "external state changed") -> None:
        self._invalidated = True
        self._last_real = None
        self._signature = None
        self._inference_conditioning_snapshot = None
        self._refresh_cooldown = 2
        try:
            self.forecaster.reset()
        except Exception:
            self._prediction_failure_latched = True

    @staticmethod
    def _tensor_id(t: torch.Tensor) -> tuple:
        version = None if t.is_inference() else t._version
        return (tuple(t.shape), str(t.dtype), str(t.device), t.data_ptr(), version)

    def predict_or_compute(
        self,
        *,
        step_index: int,
        timestep: Any,
        model_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        guidance_scale: float,
        joint_attention_kwargs: dict | None,
        skip_guidance_layers: list[int] | None,
        forward: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        if self._schedule is None:
            raise RuntimeError("begin() required before inference")
        if not 0 <= step_index < len(self._schedule):
            raise ValueError("invalid step index")
        if self._scalar(timestep) != self._schedule[step_index]:
            self.invalidate("timestep drift")
            self._drift_latched = True

        signature = (
            self._cfg,
            self._tensor_id(model_input)[:3],
            self._tensor_id(prompt_embeds),
            self._tensor_id(pooled_prompt_embeds),
            float(guidance_scale),
        )
        changed = signature != self._signature or step_index != self._last_index + 1
        if not changed and self._inference_conditioning_snapshot is not None:
            # Inference tensors have no version counter; detect in-place edits.
            for current, saved in zip(
                (prompt_embeds, pooled_prompt_embeds), self._inference_conditioning_snapshot
            ):
                if saved is not None and not torch.equal(current, saved):
                    changed = True
                    break
        if changed:
            self.invalidate("signature or step discontinuity or conditioning mutation")
        self._signature = signature
        if self._inference_conditioning_snapshot is None:
            self._inference_conditioning_snapshot = tuple(
                tensor.detach().clone() if tensor.is_inference() else None
                for tensor in (prompt_embeds, pooled_prompt_embeds)
            )

        can_forecast = (
            not self._invalidated
            and self._refresh_cooldown == 0
            and self._last_real is not None
            and step_index in self.forecast_steps
            and step_index < len(self._schedule) - self.final_real_steps
            and joint_attention_kwargs is None
            and skip_guidance_layers is None
            and math.isfinite(float(guidance_scale))
            and not self._drift_latched
            and not self._prediction_failure_latched
            and not (model_input.requires_grad or prompt_embeds.requires_grad or pooled_prompt_embeds.requires_grad)
            and not torch.is_grad_enabled()
            and step_index != 0
        )
        if can_forecast:
            try:
                out = self.forecaster.predict(step_index)
                if (
                    isinstance(out, torch.Tensor)
                    and out.shape == self._last_real.shape
                    and out.dtype == self._last_real.dtype
                    and out.device == self._last_real.device
                    and bool(torch.isfinite(out).all())
                ):
                    self._last_index = step_index
                    self.counts["forecast"] += 1
                    return out
            except Exception:
                pass
            self.counts["fallback"] += 1
            self._prediction_failure_latched = True

        output = forward()
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != model_input.shape
            or not bool(torch.isfinite(output).all())
        ):
            raise RuntimeError("Invalid real SD3 transformer prediction")
        self._last_real = output.detach().clone()
        if not self._prediction_failure_latched and not self._drift_latched:
            try:
                self.forecaster.update(step_index, self._last_real)
            except Exception:
                self._prediction_failure_latched = True
                try:
                    self.forecaster.reset()
                except Exception:
                    pass
        self._refresh_cooldown = max(0, self._refresh_cooldown - 1)
        self._invalidated = False
        self._last_index = step_index
        self.counts["real"] += 1
        return output
