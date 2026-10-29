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
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..utils import logging
from ..utils.torch_utils import unwrap_module
from .hooks import BaseState, HookRegistry, ModelHook, StateManager


logger = logging.get_logger(__name__)

_SPECTRUM_DENOISER_HOOK = "spectrum_cache_denoiser"
_SPECTRUM_UNET_FEATURE_HOOK = "spectrum_cache_unet_feature"
_SPECTRUM_COSMOS_FEATURE_HOOK = "spectrum_cache_cosmos_feature"
_SPECTRUM_FLUX3_FEATURE_HOOK = "spectrum_cache_flux3_feature"
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
        predictor_backend (`str`, defaults to `"dense"`):
            Forecaster evaluation backend. `"dense"` retains the qualified coefficient-tensor reference path;
            `"chunked"` materializes coefficients only in fixed chunks with near-zero retained coefficient state;
            `"bounded"` retains at most a bounded contiguous native coefficient slab and evaluates the uncached suffix
            in chunks, preserving the current low-precision coefficient-quantization boundary.
        predictor_cache_bytes (`int`, defaults to `67108864`):
            Maximum native coefficient bytes retained by the `"bounded"` backend.
        predictor_chunk_size (`int`, defaults to `1048576`):
            Feature elements per temporary coefficient chunk while building the bounded slab and evaluating its suffix.
        coordinate_policy (`str`, defaults to `"legacy_fixed_max"`):
            Coordinate mapping used by the spectral predictor. `"legacy_fixed_max"` preserves the historical
            `coordinate_max` mapping. `"runtime_index_normalized"` maps logical denoising-step indices over the
            configured runtime step count so index 0 maps to -1 and index `N-1` maps to +1.
            `"runtime_horizon_normalized"` maps index `i` as `2*i/N - 1`; opt-in factories that select it carry
            their own route-specific qualification bounds.
        coordinate_max (`float`, defaults to `50.0`):
            Maximum coordinate used by `"legacy_fixed_max"`. Existing qualified profiles remain pinned to their
            historical value until separately requalified.
        tail_actual_steps (`int`, defaults to `0`):
            Number of final denoising steps forced to full compute. This is a generic quality guard; the default `0`
            preserves the source-faithful refresh schedule.
        max_consecutive_forecast_steps (`int`, *optional*):
            Maximum forecast-run length for the adaptive schedule. Longer runs are compiled before inference by
            promoting the midpoint of each overlong run to real compute until the limit is satisfied. The default
            `None` preserves every existing qualified schedule. This guard cannot be combined with an explicit
            `forecast_step_indices` schedule.
        forecast_step_indices (`tuple[int, ...]`, *optional*):
            Explicit denoising-step indices to forecast. When provided, these indices replace the adaptive refresh
            schedule while preserving all forecaster settings. This is useful for route-qualified sparse schedules.
        denoiser_call_topology (`str`, defaults to `"auto"`):
            Denoiser-call topology used by route-specific adapters. `"single"` means one denoiser forward per logical
            denoising step. `"paired_cfg"` means two sequential CFG forwards share one logical step and one cache
            context. `"auto"` preserves historical route inference for backward compatibility. The explicit topology
            marker is currently consumed by the Krea 2 adapter so variable-step Raw qualification does not depend on a
            fixed step count.
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
            Legacy runtime-state fallback for the Anima / `CosmosTransformer3DModel` adapter when the pipeline does
            not provide a complete native `CacheContext`. Native contexts are preferred. The legacy callback must
            return a mapping containing `step`, `num_inference_steps`, `num_conditions`, and `label` (`"cond"` or
            `"uncond"`), and may return `dynamic_conditioning=True` for sticky fail-closed behavior.

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
    predictor_backend: str = "dense"
    predictor_cache_bytes: int = 67_108_864
    predictor_chunk_size: int = 1_048_576
    coordinate_policy: str = "legacy_fixed_max"
    coordinate_max: float = 50.0
    tail_actual_steps: int = 0
    max_consecutive_forecast_steps: int | None = None
    forecast_step_indices: tuple[int, ...] | None = None
    denoiser_call_topology: str = "auto"
    sdxl_euler_post_grid_strict: bool = False
    allow_unet_controlnet_residuals: bool = False
    allow_unet_t2i_adapter_residuals: bool = False
    allow_unet_ip_adapter_image_embeds: bool = False
    cosmos_runtime_state_callback: Callable[[], Mapping[str, Any]] | None = None

    @classmethod

    def for_sdxl(
        cls,
        num_inference_steps: int = 24,
        *,
        conservative: bool = False,
        pag: bool = False,
    ) -> "SpectrumCacheConfig":
        """Build the route-qualified SDXL UNet profile used by this fork.

        The standard profile uses ``warmup=6 / window=2.0 / flex=0.75 / tail=3`` with
        ``degree=4 / ridge=0.1 / blend=0.60``. ``conservative=True`` selects the
        separately qualified ``warmup=5 / flex=0.25`` quality-oriented alternate.
        PAG routes use the standard coefficients with ``warmup=8``.

        ``conservative`` and ``pag`` are intentionally mutually exclusive because the
        conservative PAG combination has not been qualified.
        """
        if conservative and pag:
            raise ValueError("conservative and pag SDXL SPECTRUM profiles cannot be combined")

        warmup_steps = 8 if pag else (5 if conservative else 6)
        flex_window = 0.25 if conservative else 0.75
        return cls(
            num_inference_steps=num_inference_steps,
            warmup_steps=warmup_steps,
            window_size=2.0,
            flex_window=flex_window,
            degree=4,
            ridge_lambda=0.1,
            blend_w=0.60,
            coordinate_policy="legacy_fixed_max",
            coordinate_max=50.0,
            tail_actual_steps=3,
        )


    @classmethod
    def for_sdxl_variable(
        cls, num_inference_steps: int = 24, *, strict_post_grid: bool = False
    ) -> "SpectrumCacheConfig":
        """Build the opt-in ordinary SDXL variable-step profile qualified by Gate E.

        This profile retains the ordinary SDXL predictor coefficients while using
        ``runtime_index_normalized`` coordinates and a measured max-three consecutive
        forecast guard. The guard is compiled from the actual configured logical-step
        count and only promotes real forwards; it never adds forecast positions.

        Qualification covered ordinary SDXL Euler runs at 16/24/28/32/36/40 steps,
        with 20 steps used as a schedule-identity control. Other schedulers, PAG,
        conservative mode, distilled/few-step routes, and unmeasured step counts remain
        separately qualified or experimental.

        The historical :meth:`for_sdxl` profile remains unchanged and legacy-pinned.
        ``strict_post_grid`` is an opt-in SDXL text2img Euler ingress+grid guard;
        the default leaves existing source behavior unchanged.
        """
        return cls(
            num_inference_steps=num_inference_steps,
            warmup_steps=6,
            window_size=2.0,
            flex_window=0.75,
            degree=4,
            ridge_lambda=0.1,
            blend_w=0.60,
            coordinate_policy="runtime_index_normalized",
            coordinate_max=50.0,
            tail_actual_steps=3,
            max_consecutive_forecast_steps=3,
            sdxl_euler_post_grid_strict=strict_post_grid,
        )

    @classmethod
    def for_sd15(cls, num_inference_steps: int = 20) -> "SpectrumCacheConfig":
        """Build the ordinary route-qualified SD1.5 UNet profile used by this fork.

        The profile is ``warmup=6 / window=2.0 / flex=0.75 / tail=3`` with
        ``degree=4 / ridge=0.05 / blend=0.55``.

        This preset records the qualified ordinary SD1.5 route profile. It does not
        imply universal quality portability across every scheduler, prompt, seed, or
        low-step regime.
        """
        return cls(
            num_inference_steps=num_inference_steps,
            warmup_steps=6,
            window_size=2.0,
            flex_window=0.75,
            degree=4,
            ridge_lambda=0.05,
            blend_w=0.55,
            coordinate_policy="legacy_fixed_max",
            coordinate_max=50.0,
            tail_actual_steps=3,
        )


    @classmethod
    def for_sd15_variable(cls, num_inference_steps: int = 20) -> "SpectrumCacheConfig":
        """Build the opt-in ordinary SD1.5 variable-step Euler profile.

        This profile uses a locked path-aware forecast schedule selected offline and
        independently promotion-qualified on the pinned ordinary SD1.5 Euler route.
        The historical ``legacy_fixed_max`` predictor coordinate is preserved.

        Qualification covers exactly 20/24/28/32/36/40 denoiser calls. The 40-step
        profile uses a shorter history and lower-degree predictor selected by the
        locked operating table; the other qualified step counts retain the historical
        SD1.5 predictor coefficients.

        Other schedulers, step counts, distilled/few-step routes, and special
        conditioning paths remain separately qualified or experimental. The historical
        :meth:`for_sd15` profile remains unchanged and legacy-pinned.
        """
        forecast_steps_by_count = {
            20: (8, 10, 12, 13, 15, 16),
            24: (6, 8, 10, 11, 13, 14, 15, 17, 18, 20),
            28: (6, 8, 10, 11, 13, 15, 17, 19, 20, 22, 23, 24),
            32: (6, 8, 10, 11, 13, 14, 15, 17, 19, 20, 22, 23, 24, 25, 27, 28),
            36: (6, 8, 10, 11, 13, 15, 17, 18, 20, 22, 23, 25, 27, 29, 30, 31),
            40: (6, 8, 10, 11, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 31, 33, 34, 35, 36),
        }
        if num_inference_steps not in forecast_steps_by_count:
            raise ValueError(
                "for_sd15_variable is qualified only for Euler runs with "
                "num_inference_steps in {20, 24, 28, 32, 36, 40}"
            )

        predictor = (
            {"history_limit": 4, "degree": 3, "blend_w": 0.65}
            if num_inference_steps == 40
            else {"history_limit": 100, "degree": 4, "blend_w": 0.55}
        )
        return cls(
            num_inference_steps=num_inference_steps,
            warmup_steps=6,
            window_size=2.0,
            flex_window=0.75,
            degree=predictor["degree"],
            ridge_lambda=0.05,
            blend_w=predictor["blend_w"],
            history_limit=predictor["history_limit"],
            coordinate_policy="legacy_fixed_max",
            coordinate_max=50.0,
            tail_actual_steps=3,
            forecast_step_indices=forecast_steps_by_count[num_inference_steps],
        )

    @classmethod
    def for_anima_variable(cls, num_inference_steps: int = 30) -> "SpectrumCacheConfig":
        """Build the opt-in ordinary Anima variable-step profile.

        The profile uses ``runtime_horizon_normalized`` coordinates (`2*i/N - 1`) and the
        qualified ordinary-Anima predictor coefficients. The historical 30-step schedule
        remains the identity anchor. At 20 and 30 steps the existing ``warmup=8 / tail=5``
        protection floors are preserved. Above 30 steps, warmup and tail are scaled from
        the 30-step geometry to avoid increasing forecast pressure simply because the run
        is longer.

        Qualification covers the pinned ordinary Anima Base v1.0 route at 20/30/40 steps.
        Few-step/Turbo routes remain separate and this factory intentionally rejects
        values below 20.
        """
        if num_inference_steps < 20:
            raise ValueError("for_anima_variable requires num_inference_steps >= 20; few-step/Turbo routes are separate")

        warmup_steps = max(8, math.ceil(8 * num_inference_steps / 30))
        tail_actual_steps = max(5, math.ceil(5 * num_inference_steps / 30))
        return cls(
            num_inference_steps=num_inference_steps,
            warmup_steps=warmup_steps,
            window_size=2.0,
            flex_window=0.25,
            degree=4,
            ridge_lambda=0.1,
            blend_w=0.5,
            history_limit=100,
            coordinate_policy="runtime_horizon_normalized",
            coordinate_max=50.0,
            tail_actual_steps=tail_actual_steps,
        )

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
        if self.predictor_backend not in {"dense", "chunked", "bounded"}:
            raise ValueError('predictor_backend must be "dense", "chunked", or "bounded"')
        if self.predictor_cache_bytes < 0:
            raise ValueError("predictor_cache_bytes must be >= 0")
        if self.predictor_chunk_size < 1:
            raise ValueError("predictor_chunk_size must be >= 1")
        if self.coordinate_policy not in {
            "legacy_fixed_max",
            "runtime_index_normalized",
            "runtime_horizon_normalized",
        }:
            raise ValueError(
                'coordinate_policy must be "legacy_fixed_max", "runtime_index_normalized", '
                'or "runtime_horizon_normalized"'
            )
        if self.coordinate_max <= 0:
            raise ValueError("coordinate_max must be > 0")
        if self.tail_actual_steps < 0:
            raise ValueError("tail_actual_steps must be >= 0")
        if self.tail_actual_steps > self.num_inference_steps:
            raise ValueError("tail_actual_steps must be <= num_inference_steps")
        if self.max_consecutive_forecast_steps is not None:
            if (
                isinstance(self.max_consecutive_forecast_steps, bool)
                or not isinstance(self.max_consecutive_forecast_steps, int)
                or self.max_consecutive_forecast_steps < 1
            ):
                raise ValueError("max_consecutive_forecast_steps must be an integer >= 1 when provided")
            if self.forecast_step_indices is not None:
                raise ValueError(
                    "max_consecutive_forecast_steps cannot be combined with forecast_step_indices"
                )
        if self.denoiser_call_topology not in {"auto", "single", "paired_cfg"}:
            raise ValueError('denoiser_call_topology must be "auto", "single", or "paired_cfg"')
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
        self._guarded_forecast_steps = (
            self._compile_guarded_forecast_steps()
            if self._explicit_forecast_steps is None and config.max_consecutive_forecast_steps is not None
            else None
        )
        self.reset()

    def reset(self) -> None:
        self.cached_run_length = 0
        self.current_window = float(self.config.window_size)

    def _compile_guarded_forecast_steps(self) -> frozenset[int]:
        base_config = replace(self.config, max_consecutive_forecast_steps=None)
        base_schedule = SpectrumSchedule(base_config)
        compute_mask = [base_schedule.decide(step) for step in range(self.config.num_inference_steps)]
        max_run = int(self.config.max_consecutive_forecast_steps)

        while True:
            start = None
            overlong = None
            for step, should_compute in enumerate(compute_mask):
                if not should_compute and start is None:
                    start = step
                if should_compute and start is not None:
                    if step - start > max_run:
                        overlong = (start, step - 1)
                        break
                    start = None
            if overlong is None and start is not None:
                if len(compute_mask) - start > max_run:
                    overlong = (start, len(compute_mask) - 1)

            if overlong is None:
                break

            first, last = overlong
            compute_mask[(first + last) // 2] = True

        return frozenset(step for step, should_compute in enumerate(compute_mask) if not should_compute)

    def decide(self, step_index: int) -> bool:
        if self.config.tail_actual_steps and step_index >= self.config.num_inference_steps - self.config.tail_actual_steps:
            self.cached_run_length = 0
            return True

        if self._explicit_forecast_steps is not None:
            return step_index not in self._explicit_forecast_steps
        if self._guarded_forecast_steps is not None:
            return step_index not in self._guarded_forecast_steps

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

    _CHUNKED_COEFFICIENT_SIZE = 262_144

    def __init__(self, config: SpectrumCacheConfig):
        self.config = config
        self.steps: list[int] = []
        self.features: list[torch.Tensor] = []
        self._coef: torch.Tensor | None = None
        self._history_design: torch.Tensor | None = None
        self._history_chol: torch.Tensor | None = None
        self._coef_slab: torch.Tensor | None = None
        self._slab_elements = 0
        self._shape: torch.Size | None = None

    def reset(self) -> None:
        self.steps.clear()
        self.features.clear()
        self._coef = None
        self._history_design = None
        self._history_chol = None
        self._coef_slab = None
        self._slab_elements = 0
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
        self._history_design = None
        self._history_chol = None
        self._coef_slab = None
        self._slab_elements = 0

    def history_bytes(self) -> int:
        return sum(feature.numel() * feature.element_size() for feature in self.features)

    def predictor_state_bytes(self) -> int:
        tensors = (self._coef, self._history_design, self._history_chol, self._coef_slab)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)

    def _tau(self, steps: torch.Tensor) -> torch.Tensor:
        if self.config.coordinate_policy == "legacy_fixed_max":
            return (steps - self.config.coordinate_max / 2.0) * (2.0 / self.config.coordinate_max)
        if self.config.coordinate_policy == "runtime_index_normalized":
            if self.config.num_inference_steps == 1:
                return torch.zeros_like(steps)
            return 2.0 * steps / float(self.config.num_inference_steps - 1) - 1.0
        if self.config.coordinate_policy == "runtime_horizon_normalized":
            return 2.0 * steps / float(self.config.num_inference_steps) - 1.0
        raise RuntimeError(f"Unsupported SPECTRUM coordinate policy: {self.config.coordinate_policy}")

    def _design(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        columns = [torch.ones_like(tau)]
        if self.config.degree >= 1:
            columns.append(tau)
        for _ in range(2, self.config.degree + 1):
            columns.append(2 * tau * columns[-1] - columns[-2])
        return torch.cat(columns[: self.config.degree + 1], dim=1)

    def _ridge_cholesky(self, design: torch.Tensor) -> torch.Tensor:
        order = design.shape[1]
        design_t = design.transpose(0, 1)
        gram = design_t @ design
        identity = torch.eye(order, device=design.device, dtype=torch.float32)
        gram = gram + self.config.ridge_lambda * identity
        try:
            return torch.linalg.cholesky(gram)
        except RuntimeError:
            jitter = 1e-6 * gram.diag().mean().clamp_min(1e-12)
            return torch.linalg.cholesky(gram + jitter * identity)

    @torch.compiler.disable
    def _fit_dense(self) -> None:
        if self._coef is not None:
            return
        if not self.features:
            raise ValueError("SPECTRUM cannot predict before a full-compute feature has been recorded.")

        device = self.features[-1].device
        steps = torch.tensor(self.steps, device=device, dtype=torch.float32)
        design = self._design(self._tau(steps)).to(torch.float32)
        features = torch.stack([feature.reshape(-1) for feature in self.features], dim=0).to(torch.float32)
        design_t = design.transpose(0, 1)
        chol = self._ridge_cholesky(design)
        self._coef = torch.cholesky_solve(design_t @ features, chol).to(self.features[-1].dtype)

    @torch.compiler.disable
    def _fit_history_solver(self) -> None:
        if self._history_design is not None and self._history_chol is not None:
            return
        if not self.features:
            raise ValueError("SPECTRUM cannot predict before a full-compute feature has been recorded.")

        device = self.features[-1].device
        steps = torch.tensor(self.steps, device=device, dtype=torch.float32)
        design = self._design(self._tau(steps)).to(torch.float32)
        self._history_design = design
        self._history_chol = self._ridge_cholesky(design)

    def _local_first_order(self, step: int) -> torch.Tensor:
        if len(self.features) < 2:
            return self.features[-1]

        current, previous = self.features[-1], self.features[-2]
        current_step, previous_step = self.steps[-1], self.steps[-2]
        spacing = max(float(current_step - previous_step), 1e-8)
        scale = float(step - current_step) / spacing
        return current + scale * (current - previous)

    @torch.compiler.disable
    def _predict_dense(self, step: int) -> torch.Tensor:
        self._fit_dense()
        if self._coef is None or self._shape is None:
            raise RuntimeError("SPECTRUM dense forecaster fit did not produce coefficients.")

        target_step = torch.tensor([step], device=self.features[-1].device, dtype=torch.float32)
        design = self._design(self._tau(target_step)).to(self._coef.dtype)
        return (design @ self._coef).reshape(self._shape)

    @torch.compiler.disable
    def _predict_chunked(self, step: int) -> torch.Tensor:
        self._fit_history_solver()
        if self._history_design is None or self._history_chol is None or self._shape is None:
            raise RuntimeError("SPECTRUM chunked forecaster fit did not produce solver state.")

        feature = self.features[-1]
        target_step = torch.tensor([step], device=feature.device, dtype=torch.float32)
        target_design = self._design(self._tau(target_step)).to(feature.dtype)
        design_t = self._history_design.transpose(0, 1)
        flat_features = [item.reshape(-1) for item in self.features]
        predicted = torch.empty(feature.numel(), device=feature.device, dtype=feature.dtype)

        for start in range(0, feature.numel(), self._CHUNKED_COEFFICIENT_SIZE):
            end = min(start + self._CHUNKED_COEFFICIENT_SIZE, feature.numel())
            feature_chunk = torch.stack([item[start:end] for item in flat_features], dim=0).to(torch.float32)
            coefficient_chunk = torch.cholesky_solve(design_t @ feature_chunk, self._history_chol).to(feature.dtype)
            predicted[start:end] = (target_design @ coefficient_chunk).reshape(-1)

        return predicted.reshape(self._shape)

    @torch.compiler.disable
    def _fit_bounded_slab(self) -> None:
        self._fit_history_solver()
        if self._history_design is None or self._history_chol is None or self._shape is None:
            raise RuntimeError("SPECTRUM bounded forecaster fit did not produce solver state.")
        if self._coef_slab is not None:
            return

        feature = self.features[-1]
        order = self._history_design.shape[1]
        coefficient_bytes_per_element = order * feature.element_size()
        max_cached_elements = self.config.predictor_cache_bytes // max(coefficient_bytes_per_element, 1)
        self._slab_elements = int(min(feature.numel(), max_cached_elements))
        self._coef_slab = torch.empty(
            (order, self._slab_elements), device=feature.device, dtype=feature.dtype
        )
        if self._slab_elements == 0:
            return

        design_t = self._history_design.transpose(0, 1)
        flat_features = [item.reshape(-1) for item in self.features]
        for start in range(0, self._slab_elements, self.config.predictor_chunk_size):
            end = min(start + self.config.predictor_chunk_size, self._slab_elements)
            feature_chunk = torch.stack([item[start:end] for item in flat_features], dim=0).to(torch.float32)
            coefficient_chunk = torch.cholesky_solve(
                design_t @ feature_chunk, self._history_chol
            ).to(feature.dtype)
            self._coef_slab[:, start:end] = coefficient_chunk

    @torch.compiler.disable
    def _predict_bounded(self, step: int) -> torch.Tensor:
        self._fit_bounded_slab()
        if (
            self._coef_slab is None
            or self._history_design is None
            or self._history_chol is None
            or self._shape is None
        ):
            raise RuntimeError("SPECTRUM bounded forecaster fit did not produce bounded state.")

        feature = self.features[-1]
        target_step = torch.tensor([step], device=feature.device, dtype=torch.float32)
        target_design = self._design(self._tau(target_step)).to(feature.dtype)
        flat_features = [item.reshape(-1) for item in self.features]
        predicted = torch.empty(feature.numel(), device=feature.device, dtype=feature.dtype)

        if self._slab_elements:
            predicted[: self._slab_elements] = (target_design @ self._coef_slab).reshape(-1)

        design_t = self._history_design.transpose(0, 1)
        for start in range(self._slab_elements, feature.numel(), self.config.predictor_chunk_size):
            end = min(start + self.config.predictor_chunk_size, feature.numel())
            feature_chunk = torch.stack([item[start:end] for item in flat_features], dim=0).to(torch.float32)
            coefficient_chunk = torch.cholesky_solve(
                design_t @ feature_chunk, self._history_chol
            ).to(feature.dtype)
            predicted[start:end] = (target_design @ coefficient_chunk).reshape(-1)

        return predicted.reshape(self._shape)

    @torch.compiler.disable
    def predict(self, step: int) -> torch.Tensor:
        if self.config.predictor_backend == "dense":
            predicted = self._predict_dense(step)
        elif self.config.predictor_backend == "chunked":
            predicted = self._predict_chunked(step)
        elif self.config.predictor_backend == "bounded":
            predicted = self._predict_bounded(step)
        else:
            raise RuntimeError(f"Unsupported SPECTRUM predictor backend: {self.config.predictor_backend}")

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
        self._reset_coordinate_provenance()
        self.step_index = -1
        self.should_compute = True
        self.bypass = False
        self.bypass_latched = False
        self.prediction_failure_latched = False
        self.prediction_failure_latched_at: int | None = None
        self.predict_failures: list[dict[str, Any]] = []
        self.fallback_steps: list[int] = []
        self.compute_steps: list[int] = []
        self.forecast_steps: list[int] = []
        self.peak_history_bytes = 0

    def _reset_coordinate_provenance(self) -> None:
        self.coordinate_provenance_source: str | None = None
        self.coordinate_runtime_num_inference_steps: int | None = None
        self.coordinate_last_step_index: int | None = None
        self.coordinate_failure_latched = False
        self.coordinate_failure_latched_at: int | None = None
        self.coordinate_failures: list[dict[str, Any]] = []

    def coordinate_summary(self) -> dict[str, Any]:
        return {
            "coordinate_policy": self.config.coordinate_policy,
            "provenance_source": self.coordinate_provenance_source,
            "configured_num_inference_steps": int(self.config.num_inference_steps),
            "runtime_num_inference_steps": self.coordinate_runtime_num_inference_steps,
            "last_logical_step_index": self.coordinate_last_step_index,
            "failure_latched": self.coordinate_failure_latched,
            "failure_latched_at": self.coordinate_failure_latched_at,
            "failures": list(self.coordinate_failures),
        }

    def _fail_closed_coordinate_provenance(self, reason: str) -> bool:
        logical_step = max(int(self.step_index), 0)
        if not self.coordinate_failure_latched:
            self.coordinate_failure_latched = True
            self.coordinate_failure_latched_at = logical_step
        record = {"step": logical_step, "reason": str(reason)}
        if record not in self.coordinate_failures:
            self.coordinate_failures.append(record)
        self.should_compute = True
        self._reset_prediction_history()
        return False

    def _validate_coordinate_provenance(
        self,
        *,
        logical_step_index: int | None,
        runtime_num_inference_steps: int | None,
        provenance_source: str | None,
        allow_repeat_logical_step: bool = False,
    ) -> bool:
        if self.config.coordinate_policy not in {
            "runtime_index_normalized",
            "runtime_horizon_normalized",
        }:
            return True
        if self.coordinate_failure_latched:
            return False
        if logical_step_index is None:
            return self._fail_closed_coordinate_provenance("logical step provenance is missing")
        if runtime_num_inference_steps is None:
            return self._fail_closed_coordinate_provenance("runtime logical step count is missing")
        if not isinstance(provenance_source, str) or not provenance_source:
            return self._fail_closed_coordinate_provenance("coordinate provenance source is missing")
        if isinstance(logical_step_index, bool) or not isinstance(logical_step_index, int):
            return self._fail_closed_coordinate_provenance("logical step index must be an integer")
        if isinstance(runtime_num_inference_steps, bool) or not isinstance(runtime_num_inference_steps, int):
            return self._fail_closed_coordinate_provenance("runtime logical step count must be an integer")
        if runtime_num_inference_steps < 1:
            return self._fail_closed_coordinate_provenance("runtime logical step count must be >= 1")
        if logical_step_index < 0 or logical_step_index >= runtime_num_inference_steps:
            return self._fail_closed_coordinate_provenance(
                f"logical step {logical_step_index} outside runtime range {runtime_num_inference_steps}"
            )
        if runtime_num_inference_steps != self.config.num_inference_steps:
            return self._fail_closed_coordinate_provenance(
                f"runtime logical step count {runtime_num_inference_steps} != configured "
                f"{self.config.num_inference_steps}"
            )
        if logical_step_index != self.step_index:
            return self._fail_closed_coordinate_provenance(
                f"logical step provenance {logical_step_index} != route-owned logical step {self.step_index}"
            )
        if self.coordinate_provenance_source is not None and provenance_source != self.coordinate_provenance_source:
            return self._fail_closed_coordinate_provenance(
                f"coordinate provenance source changed {self.coordinate_provenance_source!r}->{provenance_source!r}"
            )
        if (
            self.coordinate_runtime_num_inference_steps is not None
            and runtime_num_inference_steps != self.coordinate_runtime_num_inference_steps
        ):
            return self._fail_closed_coordinate_provenance(
                f"runtime logical step count changed {self.coordinate_runtime_num_inference_steps}->"
                f"{runtime_num_inference_steps}"
            )

        previous = self.coordinate_last_step_index
        if previous is None:
            if logical_step_index != 0:
                return self._fail_closed_coordinate_provenance(
                    f"logical step provenance must start at 0, got {logical_step_index}"
                )
        elif allow_repeat_logical_step:
            if logical_step_index not in {previous, previous + 1}:
                return self._fail_closed_coordinate_provenance(
                    f"non-monotonic dual-lane logical step provenance {previous}->{logical_step_index}"
                )
        elif logical_step_index != previous + 1:
            return self._fail_closed_coordinate_provenance(
                f"non-monotonic logical step provenance {previous}->{logical_step_index}"
            )

        self.coordinate_provenance_source = provenance_source
        self.coordinate_runtime_num_inference_steps = runtime_num_inference_steps
        if previous is None or logical_step_index > previous:
            self.coordinate_last_step_index = logical_step_index
        return True

    def start_step(
        self,
        *,
        logical_step_index: int | None = None,
        runtime_num_inference_steps: int | None = None,
        provenance_source: str | None = None,
    ) -> None:
        self.step_index += 1
        if self.step_index >= self.config.num_inference_steps:
            if self.config.coordinate_policy == "legacy_fixed_max":
                raise ValueError(
                    f"SPECTRUM received denoising step {self.step_index}, but config expects "
                    f"{self.config.num_inference_steps} steps. Reset the cache state or use a matching config."
                )
            self._fail_closed_coordinate_provenance(
                f"route-owned logical step {self.step_index} outside configured inference-step range "
                f"{self.config.num_inference_steps}"
            )
        elif self.config.coordinate_policy in {
            "runtime_index_normalized",
            "runtime_horizon_normalized",
        }:
            self._validate_coordinate_provenance(
                logical_step_index=logical_step_index,
                runtime_num_inference_steps=runtime_num_inference_steps,
                provenance_source=provenance_source,
            )

        if self.coordinate_failure_latched:
            self.should_compute = True
            self.compute_steps.append(self.step_index)
            return

        if self.prediction_failure_latched:
            self.should_compute = True
            self.compute_steps.append(self.step_index)
            return

        self.should_compute = self.schedule.decide(self.step_index)
        target = self.compute_steps if self.should_compute else self.forecast_steps
        target.append(self.step_index)

    def fail_closed_prediction(self, error: Exception) -> None:
        reason = f"{type(error).__name__}: {error}"
        if not self.prediction_failure_latched:
            self.prediction_failure_latched = True
            self.prediction_failure_latched_at = int(self.step_index)
        self.predict_failures.append({"step": int(self.step_index), "error": reason})
        self.fallback_steps.append(int(self.step_index))
        self.should_compute = True
        # Once forecasting has failed, discard the old fit/history before the real
        # current step is allowed to seed a fresh, non-contaminated history.
        self._reset_prediction_history()

    def _reset_prediction_history(self) -> None:
        self.forecaster.reset()

    def record_real_feature(self, image_feature: torch.Tensor) -> None:
        if self.coordinate_failure_latched:
            return
        self.forecaster.update(self.step_index, image_feature)
        self.peak_history_bytes = max(self.peak_history_bytes, self.forecaster.history_bytes())

    def predict(self) -> torch.Tensor:
        if self.coordinate_failure_latched:
            raise RuntimeError("SPECTRUM coordinate provenance is fail-closed for this run.")
        return self.forecaster.predict(self.step_index)

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
        self.coordinate_runtime_source: str | None = None
        self.coordinate_runtime_num_inference_steps: int | None = None
        self.coordinate_last_step_index: int | None = None

    def latch(self, step: int, reason: str) -> None:
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = int(step)
        record = {"step": int(step), "reason": str(reason)}
        if record not in self.guard_reasons:
            self.guard_reasons.append(record)

    def _validate_runtime_coordinate_provenance(
        self,
        *,
        step: int,
        runtime_num_inference_steps: int,
        runtime_source: str,
    ) -> None:
        if self.config.coordinate_policy not in {
            "runtime_index_normalized",
            "runtime_horizon_normalized",
        }:
            return
        if runtime_source != "cache_context":
            self.latch(step, "runtime-normalized coordinate policy requires native CacheContext provenance")
            return
        if runtime_num_inference_steps != self.config.num_inference_steps:
            self.latch(
                step,
                f"runtime num_inference_steps {runtime_num_inference_steps} != configured "
                f"{self.config.num_inference_steps}",
            )
            return
        if self.coordinate_runtime_source is not None and runtime_source != self.coordinate_runtime_source:
            self.latch(
                step,
                f"runtime coordinate source changed {self.coordinate_runtime_source!r}->{runtime_source!r}",
            )
        if (
            self.coordinate_runtime_num_inference_steps is not None
            and runtime_num_inference_steps != self.coordinate_runtime_num_inference_steps
        ):
            self.latch(
                step,
                f"runtime num_inference_steps changed {self.coordinate_runtime_num_inference_steps}->"
                f"{runtime_num_inference_steps}",
            )

        previous = self.coordinate_last_step_index
        if previous is None:
            if step != 0:
                self.latch(step, f"runtime-coordinate logical step must start at 0, got {step}")
        elif step != previous + 1:
            self.latch(step, f"non-monotonic runtime-coordinate logical step {previous}->{step}")

        if self.guard_latched:
            return
        self.coordinate_runtime_source = runtime_source
        self.coordinate_runtime_num_inference_steps = int(runtime_num_inference_steps)
        self.coordinate_last_step_index = int(step)

    def prepare_call(
        self,
        *,
        step: int,
        label: str,
        num_conditions: int,
        dynamic_conditioning: bool,
        signature: dict[str, Any],
        runtime_num_inference_steps: int,
        runtime_source: str,
    ) -> None:
        self.current_step = int(step)
        self.current_label = label
        self._validate_runtime_coordinate_provenance(
            step=self.current_step,
            runtime_num_inference_steps=int(runtime_num_inference_steps),
            runtime_source=str(runtime_source),
        )

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
            "coordinate": {
                "policy": self.config.coordinate_policy,
                "runtime_source": self.coordinate_runtime_source,
                "runtime_num_inference_steps": self.coordinate_runtime_num_inference_steps,
                "last_logical_step_index": self.coordinate_last_step_index,
            },
            "logical_prediction_steps_used": logical_prediction_steps,
            "prediction_call_count": sum(record["predicted_body_used"] for record in self.records),
            "full_call_count": sum(not record["predicted_body_used"] for record in self.records),
            "predict_failures": list(self.predict_failures),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumCosmosStateManager(StateManager):
    """Context-partitioned Cosmos state with a legacy no-context fallback bucket."""

    _LEGACY_KEY = object()

    def get_state(self):
        if self._context is not None:
            name = self._context.name
            if not isinstance(name, str):
                raise TypeError("Cosmos SPECTRUM native CacheContext name must be a string.")
            if not name.strip():
                raise ValueError("Cosmos SPECTRUM native CacheContext name must be non-empty.")
        else:
            name = self._LEGACY_KEY
        if name not in self._state_cache:
            self._state_cache[name] = self._state_cls(*self._init_args, **self._init_kwargs)
        return self._state_cache[name]

class SpectrumCosmosDenoiserHook(ModelHook):
    """Root adapter for Anima's `CosmosTransformer3DModel`.

    Full-compute steps execute the model's original forward. Forecast steps reproduce only the source-faithful
    pre-block preparation and `norm_out -> proj_out -> unpatchify` tail around a predicted post-block feature.
    Guider ownership is read from native `CacheContext` when available, with the legacy runtime callback retained
    as a fallback. Unsupported or changing legacy conditions latch sticky fail-closed behavior.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _runtime_state(self, config: SpectrumCacheConfig) -> dict[str, Any]:
        context = self.state_manager._context
        if context is not None and context.step_index is not None and context.num_inference_steps is not None:
            if (
                isinstance(context.step_index, bool)
                or not isinstance(context.step_index, int)
                or isinstance(context.num_inference_steps, bool)
                or not isinstance(context.num_inference_steps, int)
            ):
                raise TypeError("Cosmos SPECTRUM native CacheContext step_index and num_inference_steps must be integers.")
            if not isinstance(context.name, str):
                raise TypeError("Cosmos SPECTRUM native CacheContext name must be a string.")
            label = context.name
            if not label.strip():
                raise ValueError("Cosmos SPECTRUM native CacheContext name must be non-empty.")
            return {
                "step": int(context.step_index),
                "num_inference_steps": int(context.num_inference_steps),
                "num_conditions": 1,
                "label": label,
                "dynamic_conditioning": False,
                "runtime_source": "cache_context",
                "timestep": context.timestep,
                "sigma": context.sigma,
            }

        callback = config.cosmos_runtime_state_callback
        if callback is None:
            missing = []
            if context is None or context.step_index is None:
                missing.append("step_index")
            if context is None or context.num_inference_steps is None:
                missing.append("num_inference_steps")
            raise ValueError(
                "Cosmos SPECTRUM requires a complete native CacheContext "
                f"(missing: {', '.join(missing)}) or cosmos_runtime_state_callback."
            )

        runtime = callback()
        if not isinstance(runtime, Mapping):
            raise TypeError("cosmos_runtime_state_callback must return a mapping.")
        required = ("step", "num_inference_steps", "num_conditions", "label")
        missing = [name for name in required if name not in runtime]
        if missing:
            raise ValueError(f"cosmos_runtime_state_callback is missing required keys: {missing}")
        for field in ("step", "num_inference_steps", "num_conditions"):
            if type(runtime[field]) is not int:
                raise TypeError(f"Legacy Cosmos runtime-state {field} must be a Python integer.")
        if runtime["num_conditions"] < 1:
            raise ValueError("Legacy Cosmos num_conditions must be >= 1.")
        label = str(runtime["label"])
        if label not in {"cond", "uncond"}:
            raise ValueError(f"Legacy Cosmos runtime-state label must be 'cond' or 'uncond', got {label!r}.")
        runtime = dict(runtime)
        runtime["runtime_source"] = "legacy_callback"
        return runtime

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

        def _i03_call_original():
            return self.fn_ref.original_forward(
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
            "timestep": _spectrum_tensor_signature(timestep),
            "encoder": _spectrum_tensor_signature(encoder_hidden_states),
            "attention_mask": _spectrum_tensor_signature(attention_mask),
            "padding_mask": _spectrum_tensor_signature(padding_mask),
            "input": _spectrum_tensor_signature(hidden_states),
        }
        state.prepare_call(
            step=step,
            label=label,
            num_conditions=num_conditions,
            dynamic_conditioning=dynamic_conditioning,
            signature=signature,
            runtime_num_inference_steps=num_inference_steps,
            runtime_source=str(runtime["runtime_source"]),
        )

        if state.guard_latched:
            state.bypass = True
            try:
                output = _i03_call_original()
            finally:
                state.bypass = False
            self._record(state, predicted=False)
            return output

        if state.should_compute or not state.can_predict():
            output = _i03_call_original()
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
            output = _i03_call_original()
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

        if state.config.coordinate_policy == "runtime_index_normalized":
            context = self.state_manager.context
            state.start_step(
                logical_step_index=context.step_index,
                runtime_num_inference_steps=context.num_inference_steps,
                provenance_source=f"cache_context:{context.name}",
            )
        else:
            state.start_step()
        if state.should_compute:
            return self.fn_ref.original_forward(*args, **kwargs)

        try:
            predicted = state.predict()
            if not torch.is_tensor(predicted):
                raise TypeError("SPECTRUM predictor did not return a tensor.")
            expected_shape = state.forecaster._shape
            if expected_shape is None or predicted.shape != expected_shape:
                raise ValueError(
                    f"SPECTRUM predicted UNet feature shape {predicted.shape} does not match history shape "
                    f"{expected_shape}."
                )
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("SPECTRUM predicted UNet feature contains non-finite values.")

            unwrapped = unwrap_module(module)
            if unwrapped.conv_norm_out is not None:
                predicted = unwrapped.conv_norm_out(predicted)
                predicted = unwrapped.conv_act(predicted)
            predicted = unwrapped.conv_out(predicted)
        except Exception as error:
            state.fail_closed_prediction(error)
            return self.fn_ref.original_forward(*args, **kwargs)

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


class SpectrumFlux3DenoiserHook(ModelHook):
    """Base FLUX.3 image-only forecast at the pre-norm_out image-token feature boundary.

    Uses native per-guider CacheContext and the existing per-label schedule/forecaster.
    Image-conditioned, distilled, LoRA-scaled, training and context-less calls compute normally.
    """

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        image_cond_hidden_states: torch.Tensor | None = None,
        image_cond_ids: torch.Tensor | None = None,
        target_timestep: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        kwargs = dict(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            image_cond_hidden_states=image_cond_hidden_states,
            image_cond_ids=image_cond_ids,
            target_timestep=target_timestep,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

        def original():
            return self.fn_ref.original_forward(**kwargs)

        context = self.state_manager._context
        if context is None or context.step_index is None or context.num_inference_steps is None:
            return original()
        if (type(context.step_index) is not int or type(context.num_inference_steps) is not int
                or not isinstance(context.name, str) or not context.name.strip()):
            return original()

        state: SpectrumCosmosState = self.state_manager.get_state()
        step = context.step_index
        if (torch.is_grad_enabled() or module.training or image_cond_hidden_states is not None
                or image_cond_ids is not None or target_timestep is not None or attention_kwargs
                or getattr(module, "peft_config", None)):
            state.latch(step, "FLUX.3 unsupported generation path or training")
        if context.num_inference_steps != state.config.num_inference_steps:
            state.latch(step, "FLUX.3 inference step count changed")

        state.prepare_call(
            step=step,
            label=context.name,
            num_conditions=1,
            dynamic_conditioning=False,
            signature={
                "latent": _spectrum_tensor_signature(hidden_states),
                "encoder": _spectrum_tensor_identity(encoder_hidden_states),
                "timestep": _spectrum_tensor_signature(timestep),
                "image_ids": _spectrum_tensor_identity(img_ids),
                "text_ids": _spectrum_tensor_identity(txt_ids),
            },
            runtime_num_inference_steps=context.num_inference_steps,
            runtime_source="cache_context",
        )

        if state.guard_latched:
            state.bypass = True
            try:
                result = original()
            finally:
                state.bypass = False
            self._record(state, predicted=False)
            return result
        if state.should_compute or not state.can_predict():
            result = original()
            self._record(state, predicted=False)
            return result

        try:
            predicted = state.predict()
            if predicted.shape != (hidden_states.shape[0], hidden_states.shape[1], module.inner_dim):
                raise ValueError("FLUX.3 forecast feature shape mismatch")
            if not torch.isfinite(predicted).all():
                raise FloatingPointError("non-finite FLUX.3 forecast")
        except Exception as error:
            state.latch(step, f"FLUX.3 forecast failed: {type(error).__name__}: {error}")
            state.should_compute = True
            state.bypass = True
            try:
                result = original()
            finally:
                state.bypass = False
            self._record(state, predicted=False)
            return result

        # Preserve the upstream final timestep-modulated projection; skip only transformer blocks.
        dtype = hidden_states.dtype
        image_temb = module.time_proj(timestep.to(dtype) * 1000)
        image_temb = module.timestep_embedder(image_temb.to(dtype))[:, None, :]
        image_temb = image_temb.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
        predicted = module.norm_out(predicted, image_temb)
        output = module.proj_out(predicted)
        self._record(state, predicted=True)
        if not return_dict:
            return (output,)
        from ..models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=output)

    @staticmethod
    def _record(state: SpectrumCosmosState, *, predicted: bool) -> None:
        state.records.append({"logical_step": state.current_step, "guider_label": state.current_label,
                              "predicted_body_used": predicted})

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumFlux3FeatureHook(ModelHook):
    """Capture the actual image-token feature immediately before the final normalization."""

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._context is None:
            return args, kwargs
        state: SpectrumCosmosState = self.state_manager.get_state()
        if not state.bypass and state.should_compute and not state.guard_latched:
            feature = args[0] if args else kwargs.get("hidden_states")
            if feature is None:
                raise RuntimeError("FLUX.3 norm_out did not receive image features")
            state.record_real_feature(feature)
        return args, kwargs



def apply_spectrum_cache(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Install opt-in SPECTRUM on supported focused family models.

    Other historical family adapters are retained only in the archived research source.
    """
    from ..models.transformers.transformer_cosmos import CosmosTransformer3DModel
    from ..models.transformers.transformer_flux3 import Flux3Transformer2DModel
    from ..models.unets.unet_2d_condition import UNet2DConditionModel

    unwrapped = unwrap_module(module)
    if isinstance(unwrapped, Flux3Transformer2DModel):
        if unwrapped.config.flow_map:
            raise ValueError("SPECTRUM FLUX.3 supports only the undistilled Base transformer")
        if config.coordinate_policy not in {"runtime_index_normalized", "runtime_horizon_normalized"}:
            raise ValueError("SPECTRUM FLUX.3 requires runtime-normalized native step coordinates")
        state_manager = SpectrumCosmosStateManager(SpectrumCosmosState, init_args=(config,))
        HookRegistry.check_if_exists_or_initialize(module).register_hook(
            SpectrumFlux3DenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK
        )
        HookRegistry.check_if_exists_or_initialize(unwrapped.norm_out).register_hook(
            SpectrumFlux3FeatureHook(state_manager), _SPECTRUM_FLUX3_FEATURE_HOOK
        )
        return

    if isinstance(unwrapped, CosmosTransformer3DModel):
        if (unwrapped.config.use_crossattn_projection or unwrapped.config.img_context_dim_in
                or unwrapped.config.controlnet_block_every_n is not None):
            raise ValueError("SPECTRUM Cosmos requires the qualified Anima text-only configuration.")
        state_manager = SpectrumCosmosStateManager(SpectrumCosmosState, init_args=(config,))
        HookRegistry.check_if_exists_or_initialize(module).register_hook(
            SpectrumCosmosDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK
        )
        HookRegistry.check_if_exists_or_initialize(unwrapped.norm_out).register_hook(
            SpectrumCosmosFeatureHook(state_manager), _SPECTRUM_COSMOS_FEATURE_HOOK
        )
        return

    if isinstance(unwrapped, UNet2DConditionModel):
        if unwrapped.conv_norm_out is None:
            raise ValueError("SPECTRUM UNet requires the post-body conv_norm_out feature boundary.")
        state_manager = StateManager(SpectrumState, init_args=(config,))
        HookRegistry.check_if_exists_or_initialize(module).register_hook(
            SpectrumUNetDenoiserHook(state_manager), _SPECTRUM_DENOISER_HOOK
        )
        HookRegistry.check_if_exists_or_initialize(unwrapped.conv_norm_out).register_hook(
            SpectrumUNetFeatureHook(state_manager), _SPECTRUM_UNET_FEATURE_HOOK
        )
        return

    raise ValueError("SPECTRUM focused 0.42.1 source supports FLUX.3 Base, Anima/Cosmos and SDXL/SD1.5 UNet only: "
                     f"got {type(unwrapped).__name__}.")
