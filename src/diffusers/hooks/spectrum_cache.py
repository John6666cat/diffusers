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

# --- LTX-2.3 native SPECTRUM candidate -------------------------------------------------------------

_SPECTRUM_LTX2_DENOISER_HOOK = "spectrum_cache_ltx2_denoiser"
_SPECTRUM_LTX2_HEAD_BLOCK_HOOK = "spectrum_cache_ltx2_head_block"
_SPECTRUM_LTX2_BLOCK_HOOK = "spectrum_cache_ltx2_block"


class SpectrumLTX2State(BaseState):
    """Context-local paired video/audio SPECTRUM state for the qualified LTX-2.3 distilled route."""

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.schedule = SpectrumSchedule(config)
        self.video_forecaster = SpectrumForecaster(config)
        self.audio_forecaster = SpectrumForecaster(config)
        self.reset()

    def reset(self) -> None:
        self.schedule.reset()
        self.video_forecaster.reset()
        self.audio_forecaster.reset()
        self.step_index = -1
        self.should_compute = True
        self.bypass = False
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.real_block_executions = 0
        self.bypassed_block_executions = 0
        self.peak_history_bytes = 0

    def latch(self, reason: str) -> None:
        logical_step = max(int(self.step_index), 0)
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = logical_step
        record = {"step": logical_step, "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)

    def start_step(self) -> None:
        self.step_index += 1
        if self.step_index >= self.config.num_inference_steps:
            self.latch(
                f"logical step {self.step_index} outside configured inference-step range "
                f"{self.config.num_inference_steps}"
            )
            self.should_compute = True
            return
        if self.guard_latched:
            self.should_compute = True
            return
        self.should_compute = bool(self.schedule.decide(self.step_index))
        target = self.compute_steps if self.should_compute else self.forecast_steps
        target.append(self.step_index)

    def record_real_features(self, video_feature: torch.Tensor, audio_feature: torch.Tensor) -> None:
        self.video_forecaster.update(self.step_index, video_feature)
        self.audio_forecaster.update(self.step_index, audio_feature)
        total = self.video_forecaster.history_bytes() + self.audio_forecaster.history_bytes()
        self.peak_history_bytes = max(self.peak_history_bytes, total)

    def predict(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.video_forecaster.predict(self.step_index),
            self.audio_forecaster.predict(self.step_index),
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


class SpectrumLTX2DenoiserHook(ModelHook):
    """Fail-closed root adapter for the qualified LTX-2.3 distilled audiovisual route."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._argument_indices: dict[str, int] = {}

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        names = (
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "audio_encoder_hidden_states",
            "isolate_modalities",
            "spatio_temporal_guidance_blocks",
            "perturbation_mask",
            "use_cross_timestep",
            "attention_kwargs",
            "video_self_attention_mask",
            "video_keyframes_mask",
            "video_coords",
            "audio_coords",
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
        state: SpectrumLTX2State = self.state_manager.get_state()

        hidden_states = self._get_argument("hidden_states", args, kwargs)
        audio_hidden_states = self._get_argument("audio_hidden_states", args, kwargs)
        encoder_hidden_states = self._get_argument("encoder_hidden_states", args, kwargs)
        audio_encoder_hidden_states = self._get_argument("audio_encoder_hidden_states", args, kwargs)
        video_coords = self._get_argument("video_coords", args, kwargs)
        audio_coords = self._get_argument("audio_coords", args, kwargs)

        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "audio_hidden_states": _spectrum_tensor_signature(audio_hidden_states),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
            "audio_encoder_hidden_states": _spectrum_tensor_signature(audio_encoder_hidden_states),
            "video_coords": _spectrum_tensor_signature(video_coords),
            "audio_coords": _spectrum_tensor_signature(audio_coords),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local LTX-2 input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for LTX-2.3 SPECTRUM")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            state.latch("LTX-2 video hidden_states must be rank 3")
        if not torch.is_tensor(audio_hidden_states) or audio_hidden_states.ndim != 3:
            state.latch("LTX-2 audio_hidden_states must be rank 3")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            state.latch("LTX-2 video encoder_hidden_states must be rank 3")
        if not torch.is_tensor(audio_encoder_hidden_states) or audio_encoder_hidden_states.ndim != 3:
            state.latch("LTX-2 audio encoder_hidden_states must be rank 3")
        if bool(self._get_argument("isolate_modalities", args, kwargs)):
            state.latch("isolate_modalities route is not qualified")
        if self._get_argument("spatio_temporal_guidance_blocks", args, kwargs):
            state.latch("STG route is not qualified")
        if self._get_argument("perturbation_mask", args, kwargs) is not None:
            state.latch("perturbation-mask route is not qualified")
        if not bool(self._get_argument("use_cross_timestep", args, kwargs)):
            state.latch("legacy non-cross-timestep LTX route is not qualified")
        if self._get_argument("attention_kwargs", args, kwargs):
            state.latch("non-empty attention kwargs / LoRA route is not qualified")
        if self._get_argument("video_self_attention_mask", args, kwargs) is not None:
            state.latch("video self-attention mask / IC-LoRA route is not qualified")
        if self._get_argument("video_keyframes_mask", args, kwargs) is not None:
            state.latch("keyframe-token route is not qualified")
        # Standard LTX-2 modular T2VA precomputes and forwards RoPE coordinates.
        # Qualify the exact ordinary shape contract instead of fail-closing on their presence.
        if not torch.is_tensor(video_coords) or video_coords.ndim != 4:
            state.latch("standard modular LTX-2 video_coords tensor is required")
        elif (
            video_coords.shape[0] != hidden_states.shape[0]
            or video_coords.shape[1] != 3
            or video_coords.shape[2] != hidden_states.shape[1]
            or video_coords.shape[3] != 2
        ):
            state.latch("standard modular LTX-2 video_coords shape contract changed")
        if not torch.is_tensor(audio_coords) or audio_coords.ndim != 4:
            state.latch("standard modular LTX-2 audio_coords tensor is required")
        elif (
            audio_coords.shape[0] != audio_hidden_states.shape[0]
            or audio_coords.shape[1] != 1
            or audio_coords.shape[2] != audio_hidden_states.shape[1]
            or audio_coords.shape[3] != 2
        ):
            state.latch("standard modular LTX-2 audio_coords shape contract changed")

        state.bypass = state.guard_latched
        reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else None
        try:
            output = self.fn_ref.original_forward(*args, **kwargs)
        finally:
            state.bypass = False

        if state.guard_latched:
            state.records.append(
                {
                    "logical_step": max(int(state.step_index), 0),
                    "scheduled_full_compute": True,
                    "predicted_body_used": False,
                    "guard_latched": True,
                    "fallback_reason": reason,
                }
            )
        else:
            state.records.append(
                {
                    "logical_step": int(state.step_index),
                    "scheduled_full_compute": bool(state.should_compute),
                    "predicted_body_used": not bool(state.should_compute),
                    "guard_latched": False,
                    "fallback_reason": None,
                }
            )
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class _SpectrumLTX2BlockHookBase(ModelHook):
    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._hidden_index: int | None = None
        self._audio_index: int | None = None

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        self._hidden_index = parameters.index("hidden_states")
        self._audio_index = parameters.index("audio_hidden_states")
        return module

    def _inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        else:
            hidden_states = args[self._hidden_index]
        if "audio_hidden_states" in kwargs:
            audio_hidden_states = kwargs["audio_hidden_states"]
        else:
            audio_hidden_states = args[self._audio_index]
        return hidden_states, audio_hidden_states


class SpectrumLTX2HeadBlockHook(_SpectrumLTX2BlockHookBase):
    """Advance the eight-step schedule and inject paired forecasts at the first LTX-2 block."""

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTX2State = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        hidden_states, audio_hidden_states = self._inputs(args, kwargs)
        state.start_step()

        if state.should_compute:
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        predicted_video, predicted_audio = state.predict()
        predicted_video = predicted_video.to(device=hidden_states.device, dtype=hidden_states.dtype)
        predicted_audio = predicted_audio.to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
        if predicted_video.shape != hidden_states.shape:
            raise ValueError(
                f"SPECTRUM LTX-2 predicted video feature shape {predicted_video.shape} "
                f"does not match {hidden_states.shape}."
            )
        if predicted_audio.shape != audio_hidden_states.shape:
            raise ValueError(
                f"SPECTRUM LTX-2 predicted audio feature shape {predicted_audio.shape} "
                f"does not match {audio_hidden_states.shape}."
            )
        state.bypassed_block_executions += 1
        return predicted_video, predicted_audio


class SpectrumLTX2BlockHook(_SpectrumLTX2BlockHookBase):
    """Skip paired middle blocks and record both real tail streams."""

    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__(state_manager)
        self.is_tail = is_tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTX2State = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        if state.should_compute:
            state.real_block_executions += 1
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail:
                if not isinstance(output, tuple) or len(output) < 2:
                    raise RuntimeError("LTX-2 transformer block no longer returns paired video/audio states.")
                state.record_real_features(output[0], output[1])
            return output

        hidden_states, audio_hidden_states = self._inputs(args, kwargs)
        state.bypassed_block_executions += 1
        return hidden_states, audio_hidden_states


_apply_spectrum_cache_before_ltx2 = apply_spectrum_cache


def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, including the qualified LTX-2.3 distilled audiovisual adapter."""

    from ..models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, LTX2VideoTransformer3DModel):
        return _apply_spectrum_cache_before_ltx2(module, config)

    expected_signature = {
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "cross_attention_dim": 4096,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_cross_attention_dim": 2048,
        "caption_channels": 3840,
        "cross_attn_mod": True,
        "audio_cross_attn_mod": True,
        "gated_attn": True,
        "audio_gated_attn": True,
        "perturbed_attn": True,
        "rope_type": "split",
        "use_prompt_embeddings": False,
    }
    observed_signature = {key: getattr(unwrapped_module.config, key) for key in expected_signature}
    if observed_signature != expected_signature:
        raise ValueError(
            "The current SPECTRUM LTX-2 adapter is qualified only for the LTX-2.3 audiovisual "
            f"transformer architecture. Expected {expected_signature}, got {observed_signature}."
        )

    blocks = list(unwrapped_module.transformer_blocks)
    if len(blocks) != 48:
        raise ValueError("SPECTRUM LTX-2.3 support requires exactly 48 audiovisual transformer blocks.")

    state_manager = StateManager(SpectrumLTX2State, init_args=(config,))

    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    root_registry.register_hook(SpectrumLTX2DenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

    head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
    head_registry.register_hook(SpectrumLTX2HeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)

    for block in blocks[1:-1]:
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(SpectrumLTX2BlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)

    tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
    tail_registry.register_hook(SpectrumLTX2BlockHook(state_manager, is_tail=True), _SPECTRUM_BLOCK_HOOK)

    logger.debug(
        "Applied SPECTRUM cache to qualified LTX-2.3 audiovisual transformer with %d blocks and %d expected steps.",
        len(blocks),
        config.num_inference_steps,
    )

# --- end LTX-2.3 native SPECTRUM candidate ---------------------------------------------------------

# --- Neta Yume / Lumina2 native SPECTRUM candidate -----------------------------------------------


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

        state.prepare_call(_spectrum_tensor_identity(encoder_hidden_states))

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


_apply_spectrum_cache_before_netayume = apply_spectrum_cache


def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, including the qualified Neta Yume / Lumina2 dual-lane adapter."""

    from ..models.transformers.transformer_lumina2 import Lumina2Transformer2DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, Lumina2Transformer2DModel):
        return _apply_spectrum_cache_before_netayume(module, config)

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


# --- Pony V7 / AuraFlow native SPECTRUM candidate -----------------------------------------------


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


_apply_spectrum_cache_before_pony_v7 = apply_spectrum_cache


def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, including the qualified Pony V7 / AuraFlow adapter."""

    from ..models.transformers.auraflow_transformer_2d import AuraFlowTransformer2DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, AuraFlowTransformer2DModel):
        return _apply_spectrum_cache_before_pony_v7(module, config)

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

# --- historical LTX-Video 2B native SPECTRUM candidate ---------------------------------------------

class SpectrumLTXVideo2BState(BaseState):
    """Context-local SPECTRUM state for the qualified historical LTX-Video 2B standard T2V route."""

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
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.prediction_failures: list[dict[str, Any]] = []
        self.real_block_executions = 0
        self.bypassed_block_executions = 0
        self.peak_history_bytes = 0

    def latch(self, reason: str) -> None:
        logical_step = max(int(self.step_index), 0)
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = logical_step
        record = {"step": logical_step, "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)
        self.should_compute = True

    def start_step(self) -> None:
        self.step_index += 1
        if self.step_index >= self.config.num_inference_steps:
            self.latch(
                f"logical step {self.step_index} outside configured inference-step range "
                f"{self.config.num_inference_steps}"
            )
            return
        if self.guard_latched:
            self.should_compute = True
            return
        self.should_compute = bool(self.schedule.decide(self.step_index))

    def record_real_features(self, feature: torch.Tensor) -> None:
        self.forecaster.update(self.step_index, feature.detach())
        self.peak_history_bytes = max(self.peak_history_bytes, self.forecaster.history_bytes())

    def predict(self) -> torch.Tensor:
        return self.forecaster.predict(self.step_index)

    def summary(self) -> dict[str, Any]:
        return {
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "prediction_failures": list(self.prediction_failures),
            "prediction_call_count": sum(bool(r["predicted_body_used"]) for r in self.records),
            "full_call_count": sum(not bool(r["predicted_body_used"]) for r in self.records),
            "compute_steps": [int(r["logical_step"]) for r in self.records if not bool(r["predicted_body_used"])],
            "forecast_steps": [int(r["logical_step"]) for r in self.records if bool(r["predicted_body_used"])],
            "real_block_executions": self.real_block_executions,
            "bypassed_block_executions": self.bypassed_block_executions,
            "total_block_slots": self.real_block_executions + self.bypassed_block_executions,
            "peak_history_bytes": self.peak_history_bytes,
            "signature": self.signature,
            "records": list(self.records),
        }


class SpectrumLTXVideo2BDenoiserHook(ModelHook):
    """Fail-closed root adapter for the qualified historical LTX-Video 2B standard T2V route."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self._argument_indices: dict[str, int] = {}

    def initialize_hook(self, module: torch.nn.Module):
        parameters = list(inspect.signature(unwrap_module(module).__class__.forward).parameters)[1:]
        names = (
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "encoder_attention_mask",
            "num_frames",
            "height",
            "width",
            "rope_interpolation_scale",
            "video_coords",
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

    @staticmethod
    def _structure_signature(value: Any) -> Any:
        if torch.is_tensor(value):
            return _spectrum_tensor_signature(value)
        if isinstance(value, dict):
            return tuple(
                (str(key), SpectrumLTXVideo2BDenoiserHook._structure_signature(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(SpectrumLTXVideo2BDenoiserHook._structure_signature(item) for item in value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return type(value).__name__

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTXVideo2BState = self.state_manager.get_state()

        hidden_states = self._get_argument("hidden_states", args, kwargs)
        encoder_hidden_states = self._get_argument("encoder_hidden_states", args, kwargs)
        timestep = self._get_argument("timestep", args, kwargs)
        encoder_attention_mask = self._get_argument("encoder_attention_mask", args, kwargs)
        num_frames = self._get_argument("num_frames", args, kwargs)
        height = self._get_argument("height", args, kwargs)
        width = self._get_argument("width", args, kwargs)
        rope_interpolation_scale = self._get_argument("rope_interpolation_scale", args, kwargs)
        video_coords = self._get_argument("video_coords", args, kwargs)
        attention_kwargs = self._get_argument("attention_kwargs", args, kwargs)

        signature = {
            "hidden_states": self._structure_signature(hidden_states),
            "encoder_hidden_states": self._structure_signature(encoder_hidden_states),
            "timestep": self._structure_signature(timestep),
            "encoder_attention_mask": self._structure_signature(encoder_attention_mask),
            "num_frames": self._structure_signature(num_frames),
            "height": self._structure_signature(height),
            "width": self._structure_signature(width),
            "rope_interpolation_scale": self._structure_signature(rope_interpolation_scale),
            "video_coords": self._structure_signature(video_coords),
            "attention_kwargs": self._structure_signature(attention_kwargs),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local LTX-Video 2B input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for LTX-Video 2B SPECTRUM")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            state.latch("LTX-Video 2B hidden_states must be rank 3")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            state.latch("LTX-Video 2B encoder_hidden_states must be rank 3")
        if not torch.is_tensor(timestep) or timestep.ndim != 1:
            state.latch("per-token or non-1D LTX-Video timestep routes are not qualified")
        elif torch.is_tensor(hidden_states) and timestep.shape[0] != hidden_states.shape[0]:
            state.latch("LTX-Video 2B timestep batch contract changed")
        if encoder_attention_mask is not None and (
            not torch.is_tensor(encoder_attention_mask) or encoder_attention_mask.ndim != 2
        ):
            state.latch("LTX-Video 2B encoder attention mask contract changed")
        if video_coords is not None:
            state.latch("precomputed video_coords / conditioning routes are not qualified")
        if attention_kwargs:
            state.latch("non-empty attention kwargs / LoRA route is not qualified")
        if not all(isinstance(value, int) and value > 0 for value in (num_frames, height, width)):
            state.latch("standard LTX-Video latent geometry arguments are required")

        state.start_step()
        reason_before = state.guard_reasons[-1]["reason"] if state.guard_reasons else None
        output = self.fn_ref.original_forward(*args, **kwargs)

        predicted = not bool(state.should_compute) and not bool(state.guard_latched)
        fallback_reason = None
        if state.guard_latched:
            fallback_reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else reason_before
        state.records.append(
            {
                "logical_step": int(state.step_index),
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class _SpectrumLTXVideo2BBlockHookBase(ModelHook):
    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    @staticmethod
    def _hidden(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        if args:
            return args[0]
        return None


class SpectrumLTXVideo2BHeadBlockHook(_SpectrumLTXVideo2BBlockHookBase):
    """Predict the final block-stack feature at forecast steps and bypass the first heavy block."""

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTXVideo2BState = self.state_manager.get_state()
        if state.guard_latched or state.should_compute:
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        hidden_states = self._hidden(args, kwargs)
        try:
            if not torch.is_tensor(hidden_states):
                raise TypeError("missing LTX-Video hidden_states at forecast step")
            predicted = state.predict()
            if predicted.shape != hidden_states.shape:
                raise ValueError(
                    f"predicted feature shape {tuple(predicted.shape)} != input shape {tuple(hidden_states.shape)}"
                )
            if predicted.device != hidden_states.device or predicted.dtype != hidden_states.dtype:
                raise ValueError(
                    f"predicted feature placement {predicted.device}/{predicted.dtype} != "
                    f"input {hidden_states.device}/{hidden_states.dtype}"
                )
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("non-finite predicted LTX-Video body feature")
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            state.prediction_failures.append({"step": int(state.step_index), "error": reason})
            state.latch("forecast failure; sticky fail-closed")
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        state.bypassed_block_executions += 1
        return predicted


class SpectrumLTXVideo2BBlockHook(_SpectrumLTXVideo2BBlockHookBase):
    """Bypass middle/tail heavy blocks on forecast steps and record real tail features."""

    def __init__(self, state_manager: StateManager, is_tail: bool = False):
        super().__init__(state_manager)
        self.is_tail = is_tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTXVideo2BState = self.state_manager.get_state()
        hidden_states = self._hidden(args, kwargs)

        if state.guard_latched or state.should_compute:
            state.real_block_executions += 1
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.is_tail and not state.guard_latched:
                if not torch.is_tensor(output):
                    state.latch("LTX-Video 2B tail block no longer returns a tensor")
                else:
                    state.record_real_features(output)
            return output

        if not torch.is_tensor(hidden_states):
            state.latch("missing LTX-Video hidden_states while bypassing a heavy block")
            state.real_block_executions += 1
            return self.fn_ref.original_forward(*args, **kwargs)

        state.bypassed_block_executions += 1
        return hidden_states


_apply_spectrum_cache_before_ltxvideo2b = apply_spectrum_cache


def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, including the qualified historical LTX-Video 2B standard T2V adapter."""

    from ..models.transformers.transformer_ltx import LTXVideoTransformer3DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, LTXVideoTransformer3DModel):
        return _apply_spectrum_cache_before_ltxvideo2b(module, config)

    expected_signature = {
        "in_channels": 128,
        "out_channels": 128,
        "patch_size": 1,
        "patch_size_t": 1,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "cross_attention_dim": 2048,
        "num_layers": 28,
        "activation_fn": "gelu-approximate",
        "qk_norm": "rms_norm_across_heads",
        "norm_elementwise_affine": False,
        "norm_eps": 1e-6,
        "caption_channels": 4096,
        "attention_bias": True,
        "attention_out_bias": True,
    }
    observed_signature = {key: getattr(unwrapped_module.config, key) for key in expected_signature}
    if observed_signature != expected_signature:
        raise ValueError(
            "The current SPECTRUM LTX-Video adapter is qualified only for the historical 2B "
            f"28-block transformer architecture. Expected {expected_signature}, got {observed_signature}."
        )

    blocks = list(unwrapped_module.transformer_blocks)
    if len(blocks) != 28:
        raise ValueError("SPECTRUM LTX-Video 2B support requires exactly 28 transformer blocks.")

    state_manager = StateManager(SpectrumLTXVideo2BState, init_args=(config,))

    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    root_registry.register_hook(SpectrumLTXVideo2BDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK)

    head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
    head_registry.register_hook(SpectrumLTXVideo2BHeadBlockHook(state_manager), _SPECTRUM_HEAD_BLOCK_HOOK)

    for block in blocks[1:-1]:
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(SpectrumLTXVideo2BBlockHook(state_manager), _SPECTRUM_BLOCK_HOOK)

    tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
    tail_registry.register_hook(
        SpectrumLTXVideo2BBlockHook(state_manager, is_tail=True),
        _SPECTRUM_BLOCK_HOOK,
    )

    logger.debug(
        "Applied SPECTRUM cache to qualified historical LTX-Video 2B transformer with %d blocks and %d expected steps.",
        len(blocks),
        config.num_inference_steps,
    )

# --- end historical LTX-Video 2B native SPECTRUM candidate -----------------------------------------

# --- LTX-2.3 Full native SPECTRUM candidate --------------------------------------------------------

_SPECTRUM_LTX2_FULL_STG_PREFIX_END_EXCLUSIVE = 28
_SPECTRUM_LTX2_FULL_CONTEXTS = (
    "pred_cond",
    "pred_uncond",
    "pred_cond_stg",
    "pred_cond_modality",
)
_SPECTRUM_LTX2_FULL_BALANCED_FORECAST_STEPS = (13, 15, 17, 19, 21, 23, 25, 27, 29)


class SpectrumLTX2FullPrefixStore:
    """Exact cond post-block-27 activations for the immediately following real STG pass."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.step_index: int | None = None
        self.video: torch.Tensor | None = None
        self.audio: torch.Tensor | None = None
        self.store_count = 0
        self.consume_count = 0

    def store(self, step_index: int, video: torch.Tensor, audio: torch.Tensor) -> None:
        self.step_index = int(step_index)
        self.video = video.detach().clone()
        self.audio = audio.detach().clone()
        self.store_count += 1

    def take(self, step_index: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.step_index != int(step_index) or self.video is None or self.audio is None:
            return None
        out = (self.video, self.audio)
        self.step_index = None
        self.video = None
        self.audio = None
        self.consume_count += 1
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "pending_step": self.step_index,
            "store_count": self.store_count,
            "consume_count": self.consume_count,
        }


class SpectrumLTX2FullState(SpectrumLTX2State):
    """One context-local Full/SFT SPECTRUM history plus exact STG-prefix accounting."""

    def reset(self) -> None:
        super().reset()
        self.prefix_reuse_active = False
        self.prefix_reused_steps: list[int] = []
        self.prefix_reuse_misses: list[int] = []
        self.prefix_reused_block_executions = 0

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out.update(
            {
                "prefix_reused_steps": list(self.prefix_reused_steps),
                "prefix_reuse_misses": list(self.prefix_reuse_misses),
                "prefix_reused_block_executions": self.prefix_reused_block_executions,
                "total_block_slots": (
                    self.real_block_executions
                    + self.bypassed_block_executions
                    + self.prefix_reused_block_executions
                ),
            }
        )
        return out


class SpectrumLTX2FullDenoiserHook(SpectrumLTX2DenoiserHook):
    """Fail-closed root adapter for the qualified four-pass LTX-2.3 Full route."""

    def __init__(self, state_manager: StateManager, prefix_store: SpectrumLTX2FullPrefixStore):
        super().__init__(state_manager)
        self.prefix_store = prefix_store

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTX2FullState = self.state_manager.get_state()
        context = self.state_manager._current_context

        hidden_states = self._get_argument("hidden_states", args, kwargs)
        audio_hidden_states = self._get_argument("audio_hidden_states", args, kwargs)
        encoder_hidden_states = self._get_argument("encoder_hidden_states", args, kwargs)
        audio_encoder_hidden_states = self._get_argument("audio_encoder_hidden_states", args, kwargs)
        video_coords = self._get_argument("video_coords", args, kwargs)
        audio_coords = self._get_argument("audio_coords", args, kwargs)

        signature = {
            "hidden_states": _spectrum_tensor_signature(hidden_states),
            "audio_hidden_states": _spectrum_tensor_signature(audio_hidden_states),
            "encoder_hidden_states": _spectrum_tensor_signature(encoder_hidden_states),
            "audio_encoder_hidden_states": _spectrum_tensor_signature(audio_encoder_hidden_states),
            "video_coords": _spectrum_tensor_signature(video_coords),
            "audio_coords": _spectrum_tensor_signature(audio_coords),
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("context-local LTX-2 Full input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("autograd/training is not qualified for LTX-2.3 Full SPECTRUM")
        if context not in _SPECTRUM_LTX2_FULL_CONTEXTS:
            state.latch(f"unsupported LTX-2.3 Full cache context {context!r}")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 3:
            state.latch("LTX-2 Full video hidden_states must be rank 3")
        if not torch.is_tensor(audio_hidden_states) or audio_hidden_states.ndim != 3:
            state.latch("LTX-2 Full audio_hidden_states must be rank 3")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            state.latch("LTX-2 Full video encoder_hidden_states must be rank 3")
        if not torch.is_tensor(audio_encoder_hidden_states) or audio_encoder_hidden_states.ndim != 3:
            state.latch("LTX-2 Full audio encoder_hidden_states must be rank 3")

        stg_blocks = self._get_argument("spatio_temporal_guidance_blocks", args, kwargs)
        actual_stg = tuple(stg_blocks or ())
        isolate = bool(self._get_argument("isolate_modalities", args, kwargs))
        expected_flags = {
            "pred_cond": ((), False),
            "pred_uncond": ((), False),
            "pred_cond_stg": ((28,), False),
            "pred_cond_modality": ((), True),
        }
        if context in expected_flags:
            expected_stg, expected_isolate = expected_flags[context]
            if actual_stg != expected_stg:
                state.latch(
                    f"{context} STG block contract changed: expected {expected_stg}, got {actual_stg}"
                )
            if isolate != expected_isolate:
                state.latch(
                    f"{context} modality-isolation contract changed: expected {expected_isolate}, got {isolate}"
                )

        if self._get_argument("perturbation_mask", args, kwargs) is not None:
            state.latch("perturbation-mask route is not qualified for LTX-2.3 Full")
        if not bool(self._get_argument("use_cross_timestep", args, kwargs)):
            state.latch("legacy non-cross-timestep LTX-2.3 Full route is not qualified")
        if self._get_argument("attention_kwargs", args, kwargs):
            state.latch("non-empty attention kwargs / LoRA route is not qualified for LTX-2.3 Full")
        if self._get_argument("video_self_attention_mask", args, kwargs) is not None:
            state.latch("video self-attention mask / IC-LoRA route is not qualified for LTX-2.3 Full")
        if self._get_argument("video_keyframes_mask", args, kwargs) is not None:
            state.latch("keyframe-token route is not qualified for LTX-2.3 Full")

        if not torch.is_tensor(video_coords) or video_coords.ndim != 4:
            state.latch("standard modular LTX-2 Full video_coords tensor is required")
        elif (
            video_coords.shape[0] != hidden_states.shape[0]
            or video_coords.shape[1] != 3
            or video_coords.shape[2] != hidden_states.shape[1]
            or video_coords.shape[3] != 2
        ):
            state.latch("standard modular LTX-2 Full video_coords shape contract changed")

        if not torch.is_tensor(audio_coords) or audio_coords.ndim != 4:
            state.latch("standard modular LTX-2 Full audio_coords tensor is required")
        elif (
            audio_coords.shape[0] != audio_hidden_states.shape[0]
            or audio_coords.shape[1] != 1
            or audio_coords.shape[2] != audio_hidden_states.shape[1]
            or audio_coords.shape[3] != 2
        ):
            state.latch("standard modular LTX-2 Full audio_coords shape contract changed")

        state.bypass = state.guard_latched
        reason = state.guard_reasons[-1]["reason"] if state.guard_reasons else None
        try:
            output = self.fn_ref.original_forward(*args, **kwargs)
        finally:
            state.bypass = False

        if state.guard_latched:
            state.records.append(
                {
                    "logical_step": max(int(state.step_index), 0),
                    "scheduled_full_compute": True,
                    "predicted_body_used": False,
                    "guard_latched": True,
                    "fallback_reason": reason,
                }
            )
        else:
            state.records.append(
                {
                    "logical_step": int(state.step_index),
                    "scheduled_full_compute": bool(state.should_compute),
                    "predicted_body_used": not bool(state.should_compute),
                    "guard_latched": False,
                    "fallback_reason": None,
                }
            )
        return output

    def reset_state(self, module: torch.nn.Module):
        self.prefix_store.reset()
        self.state_manager.reset()
        return module


class _SpectrumLTX2FullBlockHookBase(_SpectrumLTX2BlockHookBase):
    def __init__(
        self,
        state_manager: StateManager,
        prefix_store: SpectrumLTX2FullPrefixStore,
        block_index: int,
    ):
        super().__init__(state_manager)
        self.prefix_store = prefix_store
        self.block_index = int(block_index)

    def _context(self) -> str | None:
        return self.state_manager._current_context


class SpectrumLTX2FullHeadBlockHook(_SpectrumLTX2FullBlockHookBase):
    """Advance Full schedule; inject a forecast or the exact cond prefix for a real STG pass."""

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTX2FullState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        hidden_states, audio_hidden_states = self._inputs(args, kwargs)
        state.start_step()
        state.prefix_reuse_active = False

        if not state.should_compute:
            predicted_video, predicted_audio = state.predict()
            predicted_video = predicted_video.to(device=hidden_states.device, dtype=hidden_states.dtype)
            predicted_audio = predicted_audio.to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
            if predicted_video.shape != hidden_states.shape:
                raise ValueError(
                    f"SPECTRUM LTX-2 Full predicted video feature shape {predicted_video.shape} "
                    f"does not match {hidden_states.shape}."
                )
            if predicted_audio.shape != audio_hidden_states.shape:
                raise ValueError(
                    f"SPECTRUM LTX-2 Full predicted audio feature shape {predicted_audio.shape} "
                    f"does not match {audio_hidden_states.shape}."
                )
            state.bypassed_block_executions += 1
            return predicted_video, predicted_audio

        if self._context() == "pred_cond_stg":
            prefix = self.prefix_store.take(state.step_index)
            if prefix is not None:
                prefix_video, prefix_audio = prefix
                prefix_video = prefix_video.to(device=hidden_states.device, dtype=hidden_states.dtype)
                prefix_audio = prefix_audio.to(device=audio_hidden_states.device, dtype=audio_hidden_states.dtype)
                if prefix_video.shape == hidden_states.shape and prefix_audio.shape == audio_hidden_states.shape:
                    state.prefix_reuse_active = True
                    state.prefix_reused_steps.append(int(state.step_index))
                    state.prefix_reused_block_executions += 1
                    return prefix_video, prefix_audio
            state.prefix_reuse_misses.append(int(state.step_index))

        state.real_block_executions += 1
        return self.fn_ref.original_forward(*args, **kwargs)


class SpectrumLTX2FullBlockHook(_SpectrumLTX2FullBlockHookBase):
    """Skip forecasts; reuse exact STG prefix through block 27; record real tail features."""

    def __init__(
        self,
        state_manager: StateManager,
        prefix_store: SpectrumLTX2FullPrefixStore,
        block_index: int,
        is_tail: bool = False,
    ):
        super().__init__(state_manager, prefix_store, block_index)
        self.is_tail = is_tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumLTX2FullState = self.state_manager.get_state()
        context = self._context()

        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        if not state.should_compute:
            hidden_states, audio_hidden_states = self._inputs(args, kwargs)
            state.bypassed_block_executions += 1
            return hidden_states, audio_hidden_states

        if (
            context == "pred_cond_stg"
            and state.prefix_reuse_active
            and self.block_index < _SPECTRUM_LTX2_FULL_STG_PREFIX_END_EXCLUSIVE
        ):
            hidden_states, audio_hidden_states = self._inputs(args, kwargs)
            state.prefix_reused_block_executions += 1
            return hidden_states, audio_hidden_states

        state.real_block_executions += 1
        output = self.fn_ref.original_forward(*args, **kwargs)

        if (
            context == "pred_cond"
            and self.block_index == _SPECTRUM_LTX2_FULL_STG_PREFIX_END_EXCLUSIVE - 1
        ):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("LTX-2 Full block 27 no longer returns paired video/audio states.")
            self.prefix_store.store(state.step_index, output[0], output[1])

        if self.is_tail:
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("LTX-2 Full transformer tail no longer returns paired video/audio states.")
            state.record_real_features(output[0], output[1])

        return output


_apply_spectrum_cache_before_ltx2_full = apply_spectrum_cache


def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Apply SPECTRUM, adding the qualified 30-step LTX-2.3 Full four-pass route."""

    from ..models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel

    unwrapped_module = unwrap_module(module)
    if not isinstance(unwrapped_module, LTX2VideoTransformer3DModel) or config.num_inference_steps != 30:
        return _apply_spectrum_cache_before_ltx2_full(module, config)

    if config.forecast_step_indices is None:
        raise ValueError("LTX-2.3 Full SPECTRUM currently requires explicit forecast_step_indices.")

    schedule = tuple(config.forecast_step_indices)
    allowed_schedules = {(), _SPECTRUM_LTX2_FULL_BALANCED_FORECAST_STEPS}
    profile_ok = (
        schedule in allowed_schedules
        and float(config.window_size) == 3.0
        and int(config.degree) == 2
        and float(config.ridge_lambda) == 0.1
        and float(config.blend_w) == 0.5
        and int(config.history_limit) == 8
        and float(config.coordinate_max) == 8.0
        and int(config.warmup_steps) == 0
        and float(config.flex_window) == 0.0
        and int(config.tail_actual_steps) == 0
    )
    if not profile_ok:
        raise ValueError(
            "The current LTX-2.3 Full native candidate is qualified only for the 30-step "
            "d2/window3/blend0.5/ridge0.1/c8 profile with forecast_step_indices=() "
            "or (13,15,17,19,21,23,25,27,29)."
        )

    expected_signature = {
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "cross_attention_dim": 4096,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_cross_attention_dim": 2048,
        "caption_channels": 3840,
        "cross_attn_mod": True,
        "audio_cross_attn_mod": True,
        "gated_attn": True,
        "audio_gated_attn": True,
        "perturbed_attn": True,
        "rope_type": "split",
        "use_prompt_embeddings": False,
    }
    observed_signature = {key: getattr(unwrapped_module.config, key) for key in expected_signature}
    if observed_signature != expected_signature:
        raise ValueError(
            "The current SPECTRUM LTX-2.3 Full adapter is qualified only for the pinned audiovisual "
            f"transformer architecture. Expected {expected_signature}, got {observed_signature}."
        )

    blocks = list(unwrapped_module.transformer_blocks)
    if len(blocks) != 48:
        raise ValueError("SPECTRUM LTX-2.3 Full support requires exactly 48 audiovisual transformer blocks.")

    state_manager = StateManager(SpectrumLTX2FullState, init_args=(config,))
    prefix_store = SpectrumLTX2FullPrefixStore()

    root_registry = HookRegistry.check_if_exists_or_initialize(module)
    root_registry.register_hook(
        SpectrumLTX2FullDenoiserHook(state_manager, prefix_store),
        _SPECTRUM_DENOISER_HOOK,
    )
    head_registry = HookRegistry.check_if_exists_or_initialize(blocks[0])
    head_registry.register_hook(
        SpectrumLTX2FullHeadBlockHook(state_manager, prefix_store, 0),
        _SPECTRUM_HEAD_BLOCK_HOOK,
    )
    for index, block in enumerate(blocks[1:-1], start=1):
        registry = HookRegistry.check_if_exists_or_initialize(block)
        registry.register_hook(
            SpectrumLTX2FullBlockHook(state_manager, prefix_store, index),
            _SPECTRUM_BLOCK_HOOK,
        )
    tail_registry = HookRegistry.check_if_exists_or_initialize(blocks[-1])
    tail_registry.register_hook(
        SpectrumLTX2FullBlockHook(state_manager, prefix_store, len(blocks) - 1, is_tail=True),
        _SPECTRUM_BLOCK_HOOK,
    )

    logger.debug(
        "Applied SPECTRUM cache to qualified LTX-2.3 Full transformer with %d blocks, %d steps, "
        "and exact STG prefix reuse through block %d.",
        len(blocks),
        config.num_inference_steps,
        _SPECTRUM_LTX2_FULL_STG_PREFIX_END_EXCLUSIVE - 1,
    )

# --- end LTX-2.3 Full native SPECTRUM candidate ----------------------------------------------------
