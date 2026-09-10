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
from collections.abc import Callable, Mapping
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
_SPECTRUM_COSMOS_FEATURE_HOOK = "spectrum_cache_cosmos_feature"
_SPECTRUM_FLUX2_FEATURE_HOOK = "spectrum_cache_flux2_feature"
_FLUX_BLOCK_GROUPS = ("transformer_blocks", "single_transformer_blocks")
_CONTROL_ARGUMENTS = ("controlnet_block_samples", "controlnet_single_block_samples")
_UNET_RESIDUAL_ARGUMENTS = (
    "down_block_additional_residuals",
    "mid_block_additional_residual",
    "down_intrablock_additional_residuals",
)


@dataclass
class SpectrumCacheConfig:
    """Configuration for SPECTRUM cache on supported denoisers.

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
        forecast_step_indices (`tuple[int, ...]`, *optional*):
            Explicit denoising-step indices to forecast. When provided, these indices replace the adaptive refresh
            schedule while preserving all forecaster settings. This is useful for route-qualified sparse schedules.
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
        cosmos_runtime_state_callback (`Callable`, *optional*):
            Runtime-state callback required for the experimental Anima / `CosmosTransformer3DModel` adapter. It must
            return a mapping containing `step`, `num_inference_steps`, `num_conditions`, and `label` (`"cond"` or
            `"uncond"`). It may also return `dynamic_conditioning=True` to force sticky fail-closed behavior for the
            full trajectory. The callback is intentionally explicit because the model does not own guider state.

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
    forecast_step_indices: tuple[int, ...] | None = None
    allow_unet_controlnet_residuals: bool = False
    allow_unet_t2i_adapter_residuals: bool = False
    allow_unet_ip_adapter_image_embeds: bool = False
    cosmos_runtime_state_callback: Callable[[], Mapping[str, Any]] | None = None

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
        if self.forecast_step_indices is not None:
            indices = tuple(int(step) for step in self.forecast_step_indices)
            if len(set(indices)) != len(indices):
                raise ValueError("forecast_step_indices must not contain duplicate step indices")
            if any(step < 0 or step >= self.num_inference_steps for step in indices):
                raise ValueError("forecast_step_indices must be within the configured denoising-step range")
            indices = tuple(sorted(indices))
            if self.tail_actual_steps and any(
                step >= self.num_inference_steps - self.tail_actual_steps for step in indices
            ):
                raise ValueError("forecast_step_indices cannot overlap tail_actual_steps")
            self.forecast_step_indices = indices
        if self.cosmos_runtime_state_callback is not None and not callable(self.cosmos_runtime_state_callback):
            raise ValueError("cosmos_runtime_state_callback must be callable when provided")


class SpectrumSchedule:
    """Adaptive SPECTRUM refresh schedule using denoising-loop step indices."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self._explicit_forecast_steps = (
            frozenset(config.forecast_step_indices) if config.forecast_step_indices is not None else None
        )
        self.reset()

    def reset(self) -> None:
        self.cached_run_length = 0
        self.current_window = float(self.config.window_size)

    def decide(self, step_index: int) -> bool:
        if self.config.tail_actual_steps and step_index >= self.config.num_inference_steps - self.config.tail_actual_steps:
            self.cached_run_length = 0
            return True

        if self._explicit_forecast_steps is not None:
            return step_index not in self._explicit_forecast_steps

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


class SpectrumWanState(SpectrumState):
    """Context-local mutable state for the qualified Wan2.1 T2V 1.3B SPECTRUM route."""

    def reset(self) -> None:
        super().reset()
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def latch(self, reason: str) -> None:
        logical_step = max(int(self.step_index) + 1, 0)
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = logical_step
        record = {"step": logical_step, "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)

    def summary(self) -> dict[str, Any]:
        return {
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "prediction_call_count": sum(record["predicted_body_used"] for record in self.records),
            "full_call_count": sum(not record["predicted_body_used"] for record in self.records),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumZImageState(SpectrumWanState):
    """Context-local mutable state for the qualified Z-Image standard T2I Base/Turbo SPECTRUM route."""


class SpectrumQwenImageState(SpectrumWanState):
    """Context-local mutable state for the qualified Qwen-Image-2512 standard T2I SPECTRUM route."""


class SpectrumKrea2State(SpectrumWanState):
    """Context-local mutable state for the qualified Krea 2 Turbo standard T2I SPECTRUM route."""


class SpectrumKrea2RawState(SpectrumWanState):
    """Context-local dual-lane state for the qualified Krea 2 Raw CFG route."""

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
        self.bypass_latched = False
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.lane_identities: dict[str, Any] = {}
        self.decisions: dict[int, bool] = {}
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.peak_history_bytes = 0

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
            self.latch("Krea 2 Raw CFG positive/negative conditioning identities are not distinct")

    def start_step(self) -> None:
        if self.current_lane is None:
            raise RuntimeError("SPECTRUM Krea 2 Raw call has no active CFG lane.")
        if self.guard_latched:
            self.should_compute = True
            return

        if self.current_lane == "positive":
            decision = bool(self.schedule.decide(self.step_index))
            self.decisions[self.step_index] = decision
            target = self.compute_steps if decision else self.forecast_steps
            target.append(self.step_index)
        else:
            if self.step_index not in self.decisions:
                self.latch("Krea 2 Raw negative CFG lane arrived before its positive lane")
                self.should_compute = True
                return
            decision = self.decisions[self.step_index]
        self.should_compute = decision

    def _forecaster(self) -> SpectrumForecaster:
        if self.current_lane is None:
            raise RuntimeError("SPECTRUM Krea 2 Raw forecaster has no active CFG lane.")
        return self.forecasters[self.current_lane]

    def record_real_feature(self, image_feature: torch.Tensor) -> None:
        self._forecaster().update(self.step_index, image_feature)
        total = sum(forecaster.history_bytes() for forecaster in self.forecasters.values())
        self.peak_history_bytes = max(self.peak_history_bytes, total)

    def predict(self) -> torch.Tensor:
        return self._forecaster().predict(self.step_index)

    def summary(self) -> dict[str, Any]:
        return {
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "prediction_call_count": sum(record["predicted_body_used"] for record in self.records),
            "full_call_count": sum(not record["predicted_body_used"] for record in self.records),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "lane_prediction_call_count": {
                lane: sum(
                    record["predicted_body_used"] and record.get("lane") == lane for record in self.records
                )
                for lane in ("positive", "negative")
            },
            "records": list(self.records),
        }


def _spectrum_tensor_signature(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_spectrum_tensor_signature(item) for item in value)
    if torch.is_tensor(value):
        return (tuple(value.shape), str(value.dtype), str(value.device))
    return type(value).__name__


def _spectrum_tensor_identity(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return (
            int(value.data_ptr()),
            int(value.storage_offset()),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )
    return type(value).__name__


class SpectrumFlux2State(BaseState):
    """Context-local mutable state for the FLUX.2 Klein SPECTRUM adapter."""

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
        self.record_real_feature_enabled = False
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.predict_failures: list[dict[str, Any]] = []
        self.peak_history_bytes = 0

    def latch(self, reason: str) -> None:
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = int(self.step_index)
        record = {"step": int(self.step_index), "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)

    def start_call(self, signature: dict[str, Any]) -> None:
        self.step_index += 1
        if self.step_index >= self.config.num_inference_steps:
            self.latch(
                f"logical step {self.step_index} outside configured inference-step range "
                f"{self.config.num_inference_steps}"
            )

        if self.signature is None:
            self.signature = signature
        elif self.signature != signature:
            self.latch("context-local input signature changed")

        if self.guard_latched:
            self.should_compute = True
            return

        self.should_compute = bool(self.schedule.decide(self.step_index))
        target = self.compute_steps if self.should_compute else self.forecast_steps
        target.append(self.step_index)

    def record_real_feature(self, feature: torch.Tensor) -> None:
        self.forecaster.update(self.step_index, feature)
        self.peak_history_bytes = max(self.peak_history_bytes, self.forecaster.history_bytes())

    def can_predict(self) -> bool:
        return bool(self.forecaster.features)

    def predict(self) -> torch.Tensor:
        return self.forecaster.predict(self.step_index)

    def summary(self) -> dict[str, Any]:
        return {
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "prediction_call_count": sum(record["predicted_body_used"] for record in self.records),
            "full_call_count": sum(not record["predicted_body_used"] for record in self.records),
            "predict_failures": list(self.predict_failures),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumCosmosState(BaseState):
    """Mutable state for the Anima / Cosmos SPECTRUM adapter."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.forecasters: dict[str, SpectrumForecaster] = {}
        self.decisions: dict[int, bool] = {}
        self.last_num_conditions: int | None = None
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.slot_signatures: dict[str, dict[str, Any]] = {}
        self.current_label: str | None = None
        self.current_step = -1
        self.should_compute = True
        self.bypass = False
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.predict_failures: list[dict[str, Any]] = []
        self.peak_history_bytes = 0

    def latch(self, step: int, reason: str) -> None:
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = int(step)
        record = {"step": int(step), "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)

    def prepare_call(
        self,
        *,
        step: int,
        label: str,
        num_conditions: int,
        dynamic_conditioning: bool,
        signature: dict[str, Any],
    ) -> None:
        self.current_step = int(step)
        self.current_label = label

        if self.current_step < 0 or self.current_step >= self.config.num_inference_steps:
            self.latch(self.current_step, "logical step outside configured inference-step range")
        if dynamic_conditioning:
            self.latch(self.current_step, "dynamic guider/conditioning schedule declared by runtime callback")
        if self.last_num_conditions is not None and num_conditions != self.last_num_conditions:
            self.latch(
                self.current_step,
                f"guider condition-count changed {self.last_num_conditions}->{num_conditions}",
            )
        self.last_num_conditions = int(num_conditions)

        previous_signature = self.slot_signatures.get(label)
        if previous_signature is None:
            self.slot_signatures[label] = signature
        elif previous_signature != signature:
            self.latch(self.current_step, f"conditioning/input signature changed for {label}")

        if self.guard_latched:
            self.should_compute = True
            return

        if self.current_step not in self.decisions:
            self.decisions[self.current_step] = bool(self.schedule.decide(self.current_step))
            target = self.compute_steps if self.decisions[self.current_step] else self.forecast_steps
            target.append(self.current_step)
        self.should_compute = self.decisions[self.current_step]

    def _forecaster(self, label: str) -> SpectrumForecaster:
        forecaster = self.forecasters.get(label)
        if forecaster is None:
            forecaster = SpectrumForecaster(self.config)
            self.forecasters[label] = forecaster
        return forecaster

    def record_real_feature(self, feature: torch.Tensor) -> None:
        if self.current_label is None:
            raise RuntimeError("SPECTRUM Cosmos feature record has no active guider label.")
        forecaster = self._forecaster(self.current_label)
        forecaster.update(self.current_step, feature)
        total = sum(item.history_bytes() for item in self.forecasters.values())
        self.peak_history_bytes = max(self.peak_history_bytes, total)

    def can_predict(self) -> bool:
        if self.current_label is None:
            return False
        return bool(self._forecaster(self.current_label).features)

    def predict(self) -> torch.Tensor:
        if self.current_label is None:
            raise RuntimeError("SPECTRUM Cosmos forecast has no active guider label.")
        return self._forecaster(self.current_label).predict(self.current_step)

    def summary(self) -> dict[str, Any]:
        logical_prediction_steps = sorted(
            {record["logical_step"] for record in self.records if record["predicted_body_used"]}
        )
        return {
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "logical_prediction_steps_used": logical_prediction_steps,
            "prediction_call_count": sum(record["predicted_body_used"] for record in self.records),
            "full_call_count": sum(not record["predicted_body_used"] for record in self.records),
            "predict_failures": list(self.predict_failures),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


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
                f"SPECTRUM predicted feature shape {predicted.shape} does not match denoiser feature shape "
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
                # The registered block metadata exposes the body feature forecasted by SPECTRUM.
                state.record_real_feature(hidden_states)
            return output

        hidden_states, encoder_hidden_states = _get_block_inputs(self._metadata, args, kwargs)
        return _pack_block_output(self._metadata, hidden_states, encoder_hidden_states)




class SpectrumWanDenoiserHook(ModelHook):
    """Fail-closed root adapter for the qualified Wan2.1 T2V 1.3B text-only route.

    The actual feature forecasting is handled by the generic head/middle/tail block hooks. This root hook owns
    context-local safety guards only. Image-conditioned routes, Wan 2.2 sequence timesteps, autograd/training,
    non-empty attention kwargs, and changing context-local structural signatures latch sticky full-compute behavior.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _record(self, state: SpectrumWanState, predicted: bool, fallback_reason: str | None = None) -> None:
        state.records.append(
            {
                "logical_step": max(int(state.step_index), 0),
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ):
        state: SpectrumWanState = self.state_manager.get_state()

        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "timestep": _spectrum_tensor_signature(timestep),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for Wan SPECTRUM")
        if encoder_hidden_states_image is not None:
            state.latch("image-conditioned Wan route is not qualified")
        if timestep is None or timestep.ndim != 1:
            state.latch("non-1D Wan timestep route is not qualified")
        if attention_kwargs:
            state.latch("non-empty Wan attention_kwargs are not qualified")

        if state.guard_latched:
            state.bypass = True
            reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else "sticky fail-closed guard"
            try:
                output = self.fn_ref.original_forward(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    return_dict=return_dict,
                    attention_kwargs=attention_kwargs,
                )
            finally:
                state.bypass = False
            self._record(state, predicted=False, fallback_reason=reason)
            return output

        output = self.fn_ref.original_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )
        self._record(state, predicted=not state.should_compute)
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumQwenImageDenoiserHook(ModelHook):
    """Fail-closed root adapter for qualified Qwen-Image-2512 T2I and Edit-2511 single-reference routes.

    The generic head/middle/tail block hooks forecast the image stream after the 60 dual-stream transformer blocks.
    Each qualified route keeps its own structural contract while guidance-distilled model inputs,
    ControlNet/additional timestep conditioning, non-empty attention kwargs, autograd/training, and changing
    context-local structural signatures latch sticky full-compute behavior.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager, route: str):
        super().__init__()
        if route not in {"t2i", "edit"}:
            raise ValueError(f"Unsupported Qwen-Image SPECTRUM route: {route}.")
        self.state_manager = state_manager
        self.route = route

    @staticmethod
    def _structure_signature(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(SpectrumQwenImageDenoiserHook._structure_signature(item) for item in value)
        if torch.is_tensor(value):
            return _spectrum_tensor_signature(value)
        return value

    @staticmethod
    def _standard_t2i_shapes(img_shapes: Any, hidden_states: torch.Tensor) -> bool:
        if not isinstance(img_shapes, (list, tuple)) or len(img_shapes) != hidden_states.shape[0]:
            return False
        for sample_shapes in img_shapes:
            if not isinstance(sample_shapes, (list, tuple)) or len(sample_shapes) != 1:
                return False
            shape = sample_shapes[0]
            if not isinstance(shape, (list, tuple)) or len(shape) != 3:
                return False
            if math.prod(int(value) for value in shape) != hidden_states.shape[1]:
                return False
        return True

    @staticmethod
    def _standard_edit_shapes(img_shapes: Any, hidden_states: torch.Tensor) -> bool:
        if not isinstance(img_shapes, (list, tuple)) or len(img_shapes) != hidden_states.shape[0]:
            return False
        for sample_shapes in img_shapes:
            if not isinstance(sample_shapes, (list, tuple)) or len(sample_shapes) != 2:
                return False
            target_shape, reference_shape = sample_shapes
            if (
                not isinstance(target_shape, (list, tuple))
                or len(target_shape) != 3
                or not isinstance(reference_shape, (list, tuple))
                or len(reference_shape) != 3
            ):
                return False
            total_tokens = math.prod(int(value) for value in target_shape) + math.prod(
                int(value) for value in reference_shape
            )
            if total_tokens != hidden_states.shape[1]:
                return False
        return True

    def _record(self, state: SpectrumQwenImageState, predicted: bool, fallback_reason: str | None = None) -> None:
        state.records.append(
            {
                "logical_step": max(int(state.step_index), 0),
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        guidance: torch.Tensor = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ):
        state: SpectrumQwenImageState = self.state_manager.get_state()

        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
            "encoder_hidden_states_mask": _spectrum_tensor_signature(encoder_hidden_states_mask),
            "timestep": _spectrum_tensor_signature(timestep),
            "img_shapes": self._structure_signature(img_shapes),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch(f"autograd/training is not qualified for Qwen-Image {self.route} SPECTRUM")
        if self.route == "t2i":
            if not self._standard_t2i_shapes(img_shapes, hidden_states):
                state.latch("non-standard or reference-image Qwen-Image T2I shape route is not qualified")
        elif not self._standard_edit_shapes(img_shapes, hidden_states):
            state.latch("Qwen-Image Edit SPECTRUM is qualified only for exactly one reference latent per sample")
        if timestep is None or not torch.is_tensor(timestep) or timestep.ndim != 1:
            state.latch("non-1D Qwen-Image timestep route is not qualified")
        if guidance is not None:
            state.latch("guidance-distilled Qwen-Image transformer input is not qualified")
        if attention_kwargs:
            state.latch("non-empty Qwen-Image attention_kwargs are not qualified")
        if controlnet_block_samples is not None:
            state.latch("ControlNet Qwen-Image route is not qualified")
        if additional_t_cond is not None:
            state.latch(f"additional timestep conditioning is not qualified for Qwen-Image {self.route} SPECTRUM")

        call_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
            "guidance": guidance,
            "attention_kwargs": attention_kwargs,
            "controlnet_block_samples": controlnet_block_samples,
            "additional_t_cond": additional_t_cond,
            "return_dict": return_dict,
        }

        if state.guard_latched:
            state.bypass = True
            reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else "sticky fail-closed guard"
            try:
                output = self.fn_ref.original_forward(**call_kwargs)
            finally:
                state.bypass = False
            self._record(state, predicted=False, fallback_reason=reason)
            return output

        output = self.fn_ref.original_forward(**call_kwargs)
        self._record(state, predicted=not state.should_compute)
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumZImageDenoiserHook(ModelHook):
    """Fail-closed root adapter for the qualified Z-Image standard T2I Base/Turbo route.

    The generic head/middle/tail block hooks forecast the unified post-refiner stream across the 30 main
    transformer blocks. Omni/nested-image, ControlNet, SigLIP, image-noise-mask, non-default patching,
    autograd/training, and changing context-local structural signatures latch sticky full-compute behavior.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    @staticmethod
    def _sequence_signature(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(_spectrum_tensor_signature(item) for item in value)
        return _spectrum_tensor_signature(value)

    def _record(self, state: SpectrumZImageState, predicted: bool, fallback_reason: str | None = None) -> None:
        state.records.append(
            {
                "logical_step": max(int(state.step_index), 0),
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )

    def new_forward(
        self,
        module: torch.nn.Module,
        x,
        t,
        cap_feats,
        return_dict: bool = True,
        controlnet_block_samples: dict[int, torch.Tensor] | None = None,
        siglip_feats=None,
        image_noise_mask=None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        state: SpectrumZImageState = self.state_manager.get_state()

        signature = {
            "x": self._sequence_signature(x),
            "t": _spectrum_tensor_signature(t),
            "cap_feats": self._sequence_signature(cap_feats),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for Z-Image standard T2I SPECTRUM")
        if not isinstance(x, (list, tuple)) or not x:
            state.latch("non-sequence Z-Image input is not qualified")
        elif isinstance(x[0], list):
            state.latch("Omni/nested-image Z-Image route is not qualified")
        if isinstance(cap_feats, (list, tuple)) and cap_feats and isinstance(cap_feats[0], list):
            state.latch("nested caption-feature Z-Image route is not qualified")
        if controlnet_block_samples is not None:
            state.latch("ControlNet Z-Image route is not qualified")
        if siglip_feats is not None:
            state.latch("SigLIP Z-Image route is not qualified")
        if image_noise_mask is not None:
            state.latch("image-noise-mask Z-Image route is not qualified")
        if int(patch_size) != 2 or int(f_patch_size) != 1:
            state.latch("non-default Z-Image patch geometry is not qualified")
        if t is None or not torch.is_tensor(t) or t.ndim != 1:
            state.latch("non-1D Z-Image timestep route is not qualified")

        call_kwargs = {
            "x": x,
            "t": t,
            "cap_feats": cap_feats,
            "return_dict": return_dict,
            "controlnet_block_samples": controlnet_block_samples,
            "siglip_feats": siglip_feats,
            "image_noise_mask": image_noise_mask,
            "patch_size": patch_size,
            "f_patch_size": f_patch_size,
        }

        if state.guard_latched:
            state.bypass = True
            reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else "sticky fail-closed guard"
            try:
                output = self.fn_ref.original_forward(**call_kwargs)
            finally:
                state.bypass = False
            self._record(state, predicted=False, fallback_reason=reason)
            return output

        output = self.fn_ref.original_forward(**call_kwargs)
        self._record(state, predicted=not state.should_compute)
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumKrea2DenoiserHook(ModelHook):
    """Fail-closed root adapter for qualified Krea 2 Turbo and Raw standard T2I routes."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _record(
        self, state: SpectrumKrea2State | SpectrumKrea2RawState, predicted: bool, fallback_reason: str | None = None
    ) -> None:
        record = {
            "logical_step": max(int(state.step_index), 0),
            "scheduled_full_compute": bool(state.should_compute),
            "predicted_body_used": bool(predicted),
            "guard_latched": bool(state.guard_latched),
            "fallback_reason": fallback_reason,
        }
        if isinstance(state, SpectrumKrea2RawState):
            record["lane"] = state.current_lane
        state.records.append(record)

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
        state: SpectrumKrea2State | SpectrumKrea2RawState = self.state_manager.get_state()

        if isinstance(state, SpectrumKrea2RawState):
            state.prepare_call(_spectrum_tensor_identity(encoder_hidden_states))

        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
            "timestep": _spectrum_tensor_signature(timestep),
            "position_ids": _spectrum_tensor_signature(position_ids),
            "encoder_attention_mask": _spectrum_tensor_signature(encoder_attention_mask),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for Krea 2 SPECTRUM")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            state.latch("Krea image hidden_states must be a rank-3 tensor")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 4:
            state.latch("Krea encoder_hidden_states must be a rank-4 tapped-hidden-state tensor")
        elif encoder_hidden_states.shape[2] != 12:
            state.latch("Krea encoder_hidden_states lost the qualified 12-layer prompt contract")
        if (
            torch.is_tensor(hidden_states)
            and hidden_states.ndim == 3
            and torch.is_tensor(encoder_hidden_states)
            and encoder_hidden_states.ndim == 4
            and hidden_states.shape[0] != encoder_hidden_states.shape[0]
        ):
            state.latch("Krea image/text batch dimensions differ")
        if timestep is None or not torch.is_tensor(timestep) or timestep.ndim != 1:
            state.latch("non-1D Krea timestep route is not qualified")
        elif (
            torch.is_tensor(hidden_states)
            and hidden_states.ndim == 3
            and timestep.shape[0] != hidden_states.shape[0]
        ):
            state.latch("Krea timestep batch dimension differs from image batch")
        if position_ids is None or not torch.is_tensor(position_ids) or position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            state.latch("Krea position_ids must have shape (sequence_length, 3)")
        elif (
            torch.is_tensor(hidden_states)
            and hidden_states.ndim == 3
            and torch.is_tensor(encoder_hidden_states)
            and encoder_hidden_states.ndim == 4
            and position_ids.shape[0] != hidden_states.shape[1] + encoder_hidden_states.shape[1]
        ):
            state.latch("Krea position_ids length does not equal text + image token length")
        if encoder_attention_mask is not None:
            if (
                not torch.is_tensor(encoder_attention_mask)
                or encoder_attention_mask.ndim != 2
                or not torch.is_tensor(encoder_hidden_states)
                or encoder_hidden_states.ndim != 4
                or tuple(encoder_attention_mask.shape)
                != (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1])
            ):
                state.latch("Krea encoder_attention_mask shape is not qualified")
        if attention_kwargs:
            state.latch("non-empty Krea attention kwargs / LoRA route is not qualified")

        call_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "position_ids": position_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "attention_kwargs": attention_kwargs,
            "return_dict": return_dict,
        }

        if state.guard_latched:
            state.bypass = True
            reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else "sticky fail-closed guard"
            try:
                output = self.fn_ref.original_forward(**call_kwargs)
            finally:
                state.bypass = False
            self._record(state, predicted=False, fallback_reason=reason)
            return output

        output = self.fn_ref.original_forward(**call_kwargs)
        self._record(state, predicted=not state.should_compute)
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumFlux2DenoiserHook(ModelHook):
    """Root adapter for standard non-distilled FLUX.2 Klein denoising.

    Cache contexts supplied by `Flux2KleinPipeline` own independent `SpectrumFlux2State` instances, so `cond` and
    `uncond` histories remain isolated without an external callback. Forecast steps bypass the expensive input/block
    path and recompute only the source-faithful `time_guidance_embed -> norm_out -> proj_out` tail.

    KV/reference-image routes, guidance-distilled transformer inputs, autograd/training, non-empty attention kwargs,
    and changing context-local structural signatures latch sticky fail-closed behavior.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _record(self, state: SpectrumFlux2State, predicted: bool, fallback_reason: str | None = None) -> None:
        state.records.append(
            {
                "logical_step": state.step_index,
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        kv_cache: Any = None,
        kv_cache_mode: str | None = None,
        num_ref_tokens: int = 0,
        ref_fixed_timestep: float = 0.0,
    ):
        from ..models.transformers.transformer_flux2 import Flux2Transformer2DModelOutput

        state: SpectrumFlux2State = self.state_manager.get_state()
        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
            "img_ids": _spectrum_tensor_signature(img_ids),
            "txt_ids": _spectrum_tensor_signature(txt_ids),
            "guidance_is_none": guidance is None,
        }
        state.start_call(signature)

        if torch.is_grad_enabled() or module.training:
            state.latch("autograd/training is not qualified for FLUX.2 SPECTRUM")
        if kv_cache_mode is not None or kv_cache is not None or int(num_ref_tokens) != 0:
            state.latch("KV/reference-token route is not qualified for FLUX.2 SPECTRUM")
        if guidance is not None:
            state.latch("guidance-distilled transformer route is not qualified for FLUX.2 SPECTRUM")
        if isinstance(joint_attention_kwargs, dict) and joint_attention_kwargs:
            state.latch("non-empty joint_attention_kwargs is not qualified for FLUX.2 SPECTRUM")

        if state.guard_latched:
            state.should_compute = True

        if state.should_compute or not state.can_predict():
            state.record_real_feature_enabled = not state.guard_latched
            try:
                output = self.fn_ref.original_forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=return_dict,
                    kv_cache=kv_cache,
                    kv_cache_mode=kv_cache_mode,
                    num_ref_tokens=num_ref_tokens,
                    ref_fixed_timestep=ref_fixed_timestep,
                )
            finally:
                state.record_real_feature_enabled = False
            self._record(state, predicted=False)
            return output

        try:
            predicted = state.predict()
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("non-finite predicted FLUX.2 body feature")

            tail_timestep = timestep.to(hidden_states.dtype) * 1000
            temb = module.time_guidance_embed(tail_timestep, None)
            output = module.proj_out(module.norm_out(predicted, temb))
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            state.predict_failures.append({"step": state.step_index, "error": reason})
            state.latch("forecast failure; sticky fail-closed")
            state.record_real_feature_enabled = False
            result = self.fn_ref.original_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                kv_cache=kv_cache,
                kv_cache_mode=kv_cache_mode,
                num_ref_tokens=num_ref_tokens,
                ref_fixed_timestep=ref_fixed_timestep,
            )
            self._record(state, predicted=False, fallback_reason=reason)
            return result

        self._record(state, predicted=True)
        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumFlux2FeatureHook(ModelHook):
    """Record the post-single-block image feature immediately before FLUX.2 `norm_out`."""

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumFlux2State = self.state_manager.get_state()
        if not state.record_real_feature_enabled:
            return args, kwargs

        feature = args[0] if args else kwargs.get("hidden_states")
        if not torch.is_tensor(feature):
            raise RuntimeError("SPECTRUM FLUX.2 feature hook could not locate norm_out input.")
        state.record_real_feature(feature)
        return args, kwargs


class SpectrumCosmosDenoiserHook(ModelHook):
    """Root adapter for Anima's `CosmosTransformer3DModel`.

    Full-compute steps execute the model's original forward. Forecast steps reproduce only the source-faithful
    pre-block preparation and `norm_out -> proj_out -> unpatchify` tail around a predicted post-block feature.
    Guider ownership is supplied explicitly by `cosmos_runtime_state_callback`; unsupported or changing conditions
    latch sticky fail-closed behavior.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _runtime_state(self, config: SpectrumCacheConfig) -> dict[str, Any]:
        callback = config.cosmos_runtime_state_callback
        if callback is None:
            raise RuntimeError("Cosmos SPECTRUM requires cosmos_runtime_state_callback.")
        runtime = callback()
        if not isinstance(runtime, Mapping):
            raise TypeError("cosmos_runtime_state_callback must return a mapping.")
        required = ("step", "num_inference_steps", "num_conditions", "label")
        missing = [name for name in required if name not in runtime]
        if missing:
            raise ValueError(f"Cosmos runtime-state callback is missing required keys: {missing}")
        label = str(runtime["label"])
        if label not in {"cond", "uncond"}:
            raise ValueError(f"Cosmos runtime-state label must be 'cond' or 'uncond', got {label!r}.")
        return dict(runtime)

    def _record(self, state: SpectrumCosmosState, predicted: bool, fallback_reason: str | None = None) -> None:
        state.records.append(
            {
                "logical_step": state.current_step,
                "guider_label": state.current_label,
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        fps: int | None = None,
        condition_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        from torchvision import transforms

        from ..models.modeling_outputs import Transformer2DModelOutput

        state: SpectrumCosmosState = self.state_manager.get_state()
        runtime = self._runtime_state(state.config)
        step = int(runtime["step"])
        num_inference_steps = int(runtime["num_inference_steps"])
        num_conditions = int(runtime["num_conditions"])
        label = str(runtime["label"])
        dynamic_conditioning = bool(runtime.get("dynamic_conditioning", False))

        if num_inference_steps != state.config.num_inference_steps:
            state.latch(
                step,
                f"runtime num_inference_steps {num_inference_steps} != configured {state.config.num_inference_steps}",
            )
        if torch.is_grad_enabled() or module.training:
            state.latch(step, "autograd/training entry")
        if block_controlnet_hidden_states is not None:
            state.latch(step, "block_controlnet_hidden_states present")
        if condition_mask is not None:
            state.latch(step, "condition_mask present")
        if fps is not None:
            state.latch(step, "video/fps path is not qualified by the Anima adapter")
        if isinstance(encoder_hidden_states, tuple):
            state.latch(step, "tuple/image-context encoder_hidden_states path is not qualified")

        signature = {
            "encoder": _spectrum_tensor_signature(encoder_hidden_states),
            "attention_mask": _spectrum_tensor_signature(attention_mask),
            "padding_mask": _spectrum_tensor_signature(padding_mask),
            "input_shape": tuple(hidden_states.shape),
        }
        state.prepare_call(
            step=step,
            label=label,
            num_conditions=num_conditions,
            dynamic_conditioning=dynamic_conditioning,
            signature=signature,
        )

        if state.guard_latched:
            state.bypass = True
            try:
                output = self.fn_ref.original_forward(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    block_controlnet_hidden_states=block_controlnet_hidden_states,
                    attention_mask=attention_mask,
                    fps=fps,
                    condition_mask=condition_mask,
                    padding_mask=padding_mask,
                    return_dict=return_dict,
                )
            finally:
                state.bypass = False
            self._record(state, predicted=False)
            return output

        if state.should_compute or not state.can_predict():
            output = self.fn_ref.original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                block_controlnet_hidden_states=block_controlnet_hidden_states,
                attention_mask=attention_mask,
                fps=fps,
                condition_mask=condition_mask,
                padding_mask=padding_mask,
                return_dict=return_dict,
            )
            self._record(state, predicted=False)
            return output

        try:
            predicted = state.predict()
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("non-finite predicted Cosmos body feature")
        except Exception as error:
            fallback_reason = f"{type(error).__name__}: {error}"
            state.predict_failures.append(
                {"step": step, "label": label, "error": fallback_reason}
            )
            state.latch(step, "forecast failure; fail closed")
            state.should_compute = True
            output = self.fn_ref.original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                block_controlnet_hidden_states=block_controlnet_hidden_states,
                attention_mask=attention_mask,
                fps=fps,
                condition_mask=condition_mask,
                padding_mask=padding_mask,
                return_dict=return_dict,
            )
            self._record(state, predicted=False, fallback_reason=fallback_reason)
            return output

        unwrapped = unwrap_module(module)
        batch_size, _, num_frames, height, width = hidden_states.shape

        if unwrapped.config.concat_padding_mask:
            if padding_mask is None:
                raise ValueError("Cosmos SPECTRUM requires padding_mask when concat_padding_mask is enabled.")
            padding_mask_resized = transforms.functional.resize(
                padding_mask,
                list(hidden_states.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [
                    hidden_states,
                    padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1),
                ],
                dim=1,
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # Preserve the source forward's preparation path even though the expensive block stack is skipped.
        unwrapped.rope(hidden_states, fps=fps)
        if unwrapped.config.extra_pos_embed_type:
            unwrapped.learnable_pos_embed(hidden_states)

        p_t, p_h, p_w = unwrapped.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        patch_hidden_states = unwrapped.patch_embed(hidden_states).flatten(1, 3)

        if timestep.ndim == 1:
            temb, embedded_timestep = unwrapped.time_embed(patch_hidden_states, timestep)
        elif timestep.ndim == 5:
            if tuple(timestep.shape) != (batch_size, 1, num_frames, 1, 1):
                raise ValueError(f"Unexpected Cosmos timestep shape {tuple(timestep.shape)}")
            flat_timestep = timestep.flatten()
            temb, embedded_timestep = unwrapped.time_embed(patch_hidden_states, flat_timestep)
            temb, embedded_timestep = (
                value.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for value in (temb, embedded_timestep)
            )
        else:
            raise ValueError(f"Unexpected Cosmos timestep ndim {timestep.ndim}")

        hidden_states = unwrapped.norm_out(predicted, embedded_timestep, temb)
        hidden_states = unwrapped.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(
            1, (post_patch_num_frames, post_patch_height, post_patch_width)
        )
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        self._record(state, predicted=True)
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumCosmosFeatureHook(ModelHook):
    """Record the real post-`transformer_blocks` feature at the `norm_out` boundary."""

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumCosmosState = self.state_manager.get_state()
        if state.bypass or not state.should_compute:
            return args, kwargs
        if args:
            feature = args[0]
        else:
            feature = kwargs.get("hidden_states")
        if feature is None:
            raise RuntimeError("SPECTRUM Cosmos feature hook could not locate norm_out input.")
        state.record_real_feature(feature)
        return args, kwargs


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
    hidden_states = metadata._get_parameter_from_args_kwargs(metadata.hidden_states_argument_name, args, kwargs)
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
    """Apply native-style SPECTRUM caching to supported FLUX/FLUX.2, Qwen-Image, Z-Image, Wan, Anima/Cosmos, and 2D UNet denoisers.

    FLUX.1, the qualified Qwen-Image-2512 standard T2I route, the qualified Z-Image standard T2I Base/Turbo route,
    the qualified Krea 2 Turbo/Raw standard T2I routes, and the qualified Wan2.1 T2V 1.3B route use block-stack
    hooks. FLUX.2 Klein predicts the
    post-single-block image feature and recomputes
    `time_guidance_embed -> norm_out -> proj_out`. Anima/Cosmos predicts the post-transformer-block feature and recomputes
    `norm_out -> proj_out -> unpatchify`. UNet/SDXL uses the official SPECTRUM boundary immediately before
    ``conv_norm_out``: real steps record that feature, while forecast steps skip the UNet body and
    execute only ``conv_norm_out -> conv_act -> conv_out``. Unsupported conditioning paths fail closed.
    """

    from ..models.transformers.transformer_cosmos import CosmosTransformer3DModel
    from ..models.transformers.transformer_flux import FluxTransformer2DModel
    from ..models.transformers.transformer_flux2 import Flux2Transformer2DModel
    from ..models.transformers.transformer_krea2 import Krea2Transformer2DModel
    from ..models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from ..models.transformers.transformer_wan import WanTransformer3DModel
    from ..models.transformers.transformer_z_image import ZImageTransformer2DModel
    from ..models.unets.unet_2d_condition import UNet2DConditionModel

    unwrapped_module = unwrap_module(module)

    if isinstance(unwrapped_module, Krea2Transformer2DModel):
        expected_signature = {
            "in_channels": 64,
            "num_layers": 28,
            "attention_head_dim": 128,
            "num_attention_heads": 48,
            "num_key_value_heads": 12,
            "intermediate_size": 16384,
            "timestep_embed_dim": 256,
            "text_hidden_dim": 2560,
            "num_text_layers": 12,
            "text_num_attention_heads": 20,
            "text_num_key_value_heads": 20,
            "text_intermediate_size": 6912,
            "num_layerwise_text_blocks": 2,
            "num_refiner_text_blocks": 2,
            "axes_dims_rope": (32, 48, 48),
            "rope_theta": 1000.0,
            "norm_eps": 1e-5,
        }
        observed_signature = {
            key: tuple(getattr(unwrapped_module.config, key))
            if key == "axes_dims_rope"
            else getattr(unwrapped_module.config, key)
            for key in expected_signature
        }
        if observed_signature != expected_signature:
            raise ValueError(
                "The current SPECTRUM Krea 2 adapter is qualified only for the Krea 2 standard T2I "
                f"transformer architecture. Expected {expected_signature}, got {observed_signature}."
            )

        blocks = list(unwrapped_module.transformer_blocks)
        if len(blocks) != 28:
            raise ValueError("SPECTRUM Krea 2 support requires exactly 28 transformer blocks.")

        if config.num_inference_steps == 52:
            if config.forecast_step_indices is None:
                raise ValueError(
                    "The qualified Krea 2 Raw 52-step CFG route requires explicit forecast_step_indices."
                )
            state_cls = SpectrumKrea2RawState
        else:
            state_cls = SpectrumKrea2State

        state_manager = StateManager(state_cls, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumKrea2DenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

        head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
        head_registry.register_hook(SpectrumHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)
        for block in blocks[1:-1]:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            registry.register_hook(SpectrumBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)
        tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
        tail_registry.register_hook(SpectrumBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to qualified Krea 2 standard T2I transformer with %d blocks "
            "and %d expected steps.",
            len(blocks),
            config.num_inference_steps,
        )
        return


    if isinstance(unwrapped_module, QwenImageTransformer2DModel):
        common_signature = {
            "patch_size": 2,
            "in_channels": 64,
            "out_channels": 16,
            "num_layers": 60,
            "attention_head_dim": 128,
            "num_attention_heads": 24,
            "joint_attention_dim": 3584,
            "guidance_embeds": False,
            "axes_dims_rope": (16, 56, 56),
        }
        observed_signature = {
            key: tuple(getattr(unwrapped_module.config, key)) if key == "axes_dims_rope" else getattr(unwrapped_module.config, key)
            for key in common_signature
        }
        if observed_signature != common_signature:
            raise ValueError(
                "The current SPECTRUM Qwen-Image adapter is qualified only for the Qwen-Image-2512 T2I and "
                f"Qwen-Image-Edit-2511 transformer architectures. Expected {common_signature}, got {observed_signature}."
            )

        zero_cond_t = bool(getattr(unwrapped_module.config, "zero_cond_t", False))
        route = "edit" if zero_cond_t else "t2i"

        blocks = list(unwrapped_module.transformer_blocks)
        if len(blocks) != 60:
            raise ValueError("SPECTRUM Qwen-Image support requires exactly 60 transformer blocks.")

        state_manager = StateManager(SpectrumQwenImageState, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumQwenImageDenoiserHook(state_manager, route=route), _SPECTRUM_DENOISER_HOOK)

        head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
        head_registry.register_hook(SpectrumHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)
        for block in blocks[1:-1]:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            registry.register_hook(SpectrumBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)
        tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
        tail_registry.register_hook(SpectrumBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to qualified Qwen-Image %s transformer with %d blocks and %d expected steps.",
            route,
            len(blocks),
            config.num_inference_steps,
        )
        return

    if isinstance(unwrapped_module, ZImageTransformer2DModel):
        expected_signature = {
            "all_patch_size": (2,),
            "all_f_patch_size": (1,),
            "in_channels": 16,
            "dim": 3840,
            "n_layers": 30,
            "n_refiner_layers": 2,
            "n_heads": 30,
            "n_kv_heads": 30,
            "cap_feat_dim": 2560,
            "siglip_feat_dim": None,
        }
        observed_signature = {
            key: tuple(getattr(unwrapped_module.config, key))
            if key in {"all_patch_size", "all_f_patch_size"}
            else getattr(unwrapped_module.config, key)
            for key in expected_signature
        }
        if observed_signature != expected_signature:
            raise ValueError(
                "The current SPECTRUM Z-Image adapter is qualified only for the Z-Image standard T2I Base/Turbo "
                f"transformer architecture. Expected {expected_signature}, got {observed_signature}."
            )

        blocks = list(unwrapped_module.layers)
        if len(blocks) != 30:
            raise ValueError("SPECTRUM Z-Image standard T2I Base/Turbo support requires exactly 30 main transformer blocks.")

        state_manager = StateManager(SpectrumZImageState, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumZImageDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

        head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
        head_registry.register_hook(SpectrumHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)
        for block in blocks[1:-1]:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            registry.register_hook(SpectrumBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)
        tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
        tail_registry.register_hook(SpectrumBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to qualified Z-Image standard T2I Base/Turbo transformer with %d main blocks "
            "and %d expected steps.",
            len(blocks),
            config.num_inference_steps,
        )
        return

    if isinstance(unwrapped_module, WanTransformer3DModel):
        expected_signature = {
            "num_layers": 30,
            "num_attention_heads": 12,
            "attention_head_dim": 128,
            "ffn_dim": 8960,
            "text_dim": 4096,
            "in_channels": 16,
            "out_channels": 16,
            "patch_size": (1, 2, 2),
        }
        observed_signature = {
            key: tuple(getattr(unwrapped_module.config, key))
            if key == "patch_size"
            else getattr(unwrapped_module.config, key)
            for key in expected_signature
        }
        if observed_signature != expected_signature:
            raise ValueError(
                "The current SPECTRUM Wan adapter is qualified only for the Wan2.1 T2V 1.3B transformer "
                f"architecture. Expected {expected_signature}, got {observed_signature}."
            )
        if getattr(unwrapped_module.config, "image_dim", None) is not None or getattr(
            unwrapped_module.config, "added_kv_proj_dim", None
        ) is not None:
            raise ValueError(
                "The current SPECTRUM Wan adapter is qualified only for the text-only Wan2.1 T2V 1.3B route."
            )

        blocks = list(unwrapped_module.blocks)
        if len(blocks) < 2:
            raise ValueError("SPECTRUM Wan support requires at least two transformer blocks.")

        state_manager = StateManager(SpectrumWanState, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumWanDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

        head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
        head_registry.register_hook(SpectrumHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)
        for block in blocks[1:-1]:
            registry = HookRegistry.check_if_exists_or_initialize(block)
            registry.register_hook(SpectrumBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)
        tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
        tail_registry.register_hook(SpectrumBlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to qualified Wan2.1 T2V 1.3B transformer with %d blocks and %d expected steps.",
            len(blocks),
            config.num_inference_steps,
        )
        return

    if isinstance(unwrapped_module, Flux2Transformer2DModel):
        if bool(getattr(unwrapped_module.config, "guidance_embeds", False)):
            raise ValueError(
                "The current SPECTRUM FLUX.2 adapter is qualified only for the non-distilled Klein Base route "
                "with guidance_embeds=False."
            )
        if getattr(unwrapped_module, "norm_out", None) is None or getattr(unwrapped_module, "proj_out", None) is None:
            raise ValueError("SPECTRUM FLUX.2 support requires norm_out and proj_out.")

        state_manager = StateManager(SpectrumFlux2State, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumFlux2DenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

        feature_registry = HookRegistry.check_if_exists_or_initialize(unwrapped_module.norm_out)
        feature_registry.register_hook(SpectrumFlux2FeatureHook(state_manager), _SPECTRUM_FLUX2_FEATURE_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to Flux2Transformer2DModel with %d expected inference steps.",
            config.num_inference_steps,
        )
        return

    if isinstance(unwrapped_module, CosmosTransformer3DModel):
        if config.cosmos_runtime_state_callback is None:
            raise ValueError(
                "SPECTRUM Cosmos/Anima support requires cosmos_runtime_state_callback so guider ownership is explicit."
            )
        if (
            unwrapped_module.config.use_crossattn_projection
            or unwrapped_module.config.img_context_dim_in
            or unwrapped_module.config.controlnet_block_every_n is not None
        ):
            raise ValueError(
                "The current SPECTRUM Cosmos adapter is qualified only for Anima's text-only CosmosTransformer3DModel "
                "configuration without image-context projection or ControlNet block injection."
            )

        state_manager = StateManager(SpectrumCosmosState, init_args=(config,))
        root_registry = HookRegistry.check_if_exists_or_initialize(module)
        root_registry.register_hook(SpectrumCosmosDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

        feature_registry = HookRegistry.check_if_exists_or_initialize(unwrapped_module.norm_out)
        feature_registry.register_hook(SpectrumCosmosFeatureHook(state_manager), _SPECTRUM_COSMOS_FEATURE_HOOK)

        logger.debug(
            "Applied SPECTRUM cache to CosmosTransformer3DModel with %d expected inference steps.",
            config.num_inference_steps,
        )
        return

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
            "SpectrumCacheConfig currently supports FluxTransformer2DModel, Flux2Transformer2DModel, "
            "the qualified Qwen-Image-2512 T2I / Qwen-Image-Edit-2511 single-reference QwenImageTransformer2DModel routes, "
            "the qualified Z-Image standard T2I Base/Turbo ZImageTransformer2DModel route, "
            "the qualified Krea 2 Turbo/Raw standard T2I Krea2Transformer2DModel routes, "
            "the qualified Wan2.1 T2V 1.3B WanTransformer3DModel route, CosmosTransformer3DModel, "
            "and UNet2DConditionModel, "
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
