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

"""SPECTRUM image-feature cache adapter for FLUX.1-dev.

This adapter deliberately leaves the current transformer block implementation intact.
Only the post-block image-stream feature is cached; on forecast steps it recomputes
the timestep/guidance embedding and the original norm/projection tail.
"""

from typing import Any

import torch

from ..models.modeling_outputs import Transformer2DModelOutput
from .hooks import ModelHook, StateManager
from .spectrum_cache import SpectrumFlux2State, _spectrum_tensor_identity, _spectrum_tensor_signature


_SPECTRUM_FLUX1_FEATURE_HOOK = "spectrum_cache_flux1_feature"


class SpectrumFlux1State(SpectrumFlux2State):
    """Independent context-local state for the 50-step FLUX.1-dev route."""


class SpectrumFlux1DenoiserHook(ModelHook):
    """Root hook; bypass only after validating native step and conditioning continuity."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def _record(self, state: SpectrumFlux1State, predicted: bool, reason: str | None = None) -> None:
        state.records.append(
            {
                "logical_step": state.step_index,
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": bool(predicted),
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": reason,
            }
        )

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        call_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
            "joint_attention_kwargs": joint_attention_kwargs,
            "controlnet_block_samples": controlnet_block_samples,
            "controlnet_single_block_samples": controlnet_single_block_samples,
            "return_dict": return_dict,
            "controlnet_blocks_repeat": controlnet_blocks_repeat,
        }
        try:
            context = self.state_manager.context
        except ValueError:
            # Direct transformer users without pipeline step metadata always run the original body.
            return self.fn_ref.original_forward(**call_kwargs)

        state: SpectrumFlux1State = self.state_manager.get_state()
        if context.step_index == 0 and state.step_index >= 0:
            # Also recovers after a previous denoising run exited before normal cleanup.
            state.reset()

        def conditioning_identity(value: Any) -> Any:
            if not torch.is_tensor(value):
                return _spectrum_tensor_identity(value)
            try:
                version = int(value._version)
            except RuntimeError:
                version = None
            return (*_spectrum_tensor_identity(value), version)

        state.start_call(
            {
                "latents": _spectrum_tensor_signature(hidden_states),
                "prompt": conditioning_identity(encoder_hidden_states),
                "pooled": conditioning_identity(pooled_projections),
                "img_ids": conditioning_identity(img_ids),
                "txt_ids": conditioning_identity(txt_ids),
                "guidance": conditioning_identity(guidance),
                "timestep": _spectrum_tensor_signature(timestep),
            }
        )

        if (
            not isinstance(context.step_index, int)
            or not isinstance(context.num_inference_steps, int)
            or context.step_index != state.step_index
            or context.num_inference_steps != state.config.num_inference_steps
        ):
            state.latch("missing, mismatched, or non-monotonic native FLUX.1 step context")
        if torch.is_grad_enabled() or module.training:
            state.latch("autograd/training is not qualified for FLUX.1 SPECTRUM")
        if (
            controlnet_block_samples is not None
            or controlnet_single_block_samples is not None
            or controlnet_blocks_repeat
        ):
            state.latch("ControlNet arguments are not qualified for FLUX.1 SPECTRUM")
        if joint_attention_kwargs:
            state.latch("non-empty attention kwargs/IP-Adapter/LoRA scale are not qualified for FLUX.1")
        if getattr(module, "peft_config", None):
            state.latch("active PEFT/LoRA adapters are not qualified for FLUX.1 SPECTRUM")
        if not all(
            torch.is_tensor(value)
            for value in (encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance)
        ):
            state.latch("qualified FLUX.1-dev text/guidance tensors are required")
        if any(
            torch.is_tensor(value) and torch.is_inference(value)
            for value in (encoder_hidden_states, pooled_projections, img_ids, txt_ids, guidance)
        ):
            state.latch("conditioning tensor mutation cannot be tracked in inference mode")

        if state.guard_latched:
            state.should_compute = True
        if state.should_compute or not state.can_predict():
            state.record_real_feature_enabled = not state.guard_latched
            try:
                result = self.fn_ref.original_forward(**call_kwargs)
            finally:
                state.record_real_feature_enabled = False
            self._record(state, predicted=False)
            return result

        try:
            predicted = state.predict()
            if not bool(torch.isfinite(predicted).all()):
                raise FloatingPointError("non-finite FLUX.1 forecast feature")

            # Current Diffusers FLUX.1 keeps text/image streams separate inside each block.
            # The cached feature is already the image stream directly before norm_out.
            projected = module.x_embedder(hidden_states)
            tail_timestep = timestep.to(projected.dtype) * 1000
            tail_guidance = guidance.to(projected.dtype) * 1000
            temb = module.time_text_embed(tail_timestep, tail_guidance, pooled_projections)
            output = module.proj_out(module.norm_out(predicted, temb))
            if not bool(torch.isfinite(output).all()):
                raise FloatingPointError("non-finite FLUX.1 forecast output")
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            state.predict_failures.append({"step": state.step_index, "error": reason})
            state.latch("forecast/tail failure; sticky fail-closed")
            state.should_compute = True
            state.record_real_feature_enabled = False
            result = self.fn_ref.original_forward(**call_kwargs)
            self._record(state, predicted=False, reason=reason)
            return result

        self._record(state, predicted=True)
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        return module


class SpectrumFlux1FeatureHook(ModelHook):
    """Record the post-single-block image stream at the original norm_out boundary."""

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
        try:
            state: SpectrumFlux1State = self.state_manager.get_state()
        except ValueError:
            return args, kwargs
        if state.record_real_feature_enabled:
            feature = args[0] if args else kwargs.get("hidden_states")
            if not torch.is_tensor(feature):
                raise RuntimeError("FLUX.1 norm_out image feature was not a tensor")
            state.record_real_feature(feature)
        return args, kwargs
