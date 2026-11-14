"""Experimental native SPECTRUM block adapter for text-only Wan2.1 T2V 1.3B.

The cache belongs to an individual pipeline guidance lane (cond/uncond), and
the logical denoising step comes from WanPipeline.cache_context(). The
post-block-29 hidden stream is forecast; patching, conditioning, output
normalization/projection, and scheduler math are always computed normally.
"""

from typing import Any

import torch

from .hooks import ModelHook, StateManager
from .spectrum_cache import SpectrumState, _spectrum_tensor_signature


class SpectrumWanState(SpectrumState):
    """Context-local Wan guard, coordinate ledger, and spectral feature history."""

    def reset(self) -> None:
        super().reset()
        self.guard_latched = False
        self.guard_latched_at: int | None = None
        self.guard_reasons: list[dict[str, Any]] = []
        self.signature: dict[str, Any] | None = None
        self.records: list[dict[str, Any]] = []

    def latch(self, reason: str) -> None:
        step = max(int(self.step_index) + 1, 0)
        if not self.guard_latched:
            self.guard_latched = True
            self.guard_latched_at = step
        entry = {"step": step, "reason": str(reason)}
        if entry not in self.guard_reasons:
            self.guard_reasons.append(entry)

    def summary(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate_summary(),
            "guard_latched": self.guard_latched,
            "guard_latched_at": self.guard_latched_at,
            "guard_reasons": list(self.guard_reasons),
            "prediction_failure_latched": self.prediction_failure_latched,
            "prediction_failure_latched_at": self.prediction_failure_latched_at,
            "predict_failures": list(self.predict_failures),
            "fallback_steps": list(self.fallback_steps),
            "prediction_call_count": sum(item["predicted_body_used"] for item in self.records),
            "full_call_count": sum(not item["predicted_body_used"] for item in self.records),
            "compute_steps": list(self.compute_steps),
            "forecast_steps": list(self.forecast_steps),
            "peak_history_bytes": self.peak_history_bytes,
            "records": list(self.records),
        }


class SpectrumWanDenoiserHook(ModelHook):
    """Reject unsupported conditioning and keep a per-lane runtime ledger."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self.latest_summary: dict[str, Any] | None = None

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
            state.latch("context-local Wan input signature changed")

        if module.training or torch.is_grad_enabled():
            state.latch("Wan autograd/training is not qualified")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 5:
            state.latch("Wan expects one 5D latent video tensor")
        if not torch.is_tensor(timestep) or timestep.ndim != 1:
            state.latch("Wan2.2 sequence timesteps are not qualified")
        if not torch.is_tensor(encoder_hidden_states) or encoder_hidden_states.ndim != 3:
            state.latch("Wan text-conditioning structure changed")
        if encoder_hidden_states_image is not None:
            state.latch("Wan I2V/image conditioning is not qualified")
        if attention_kwargs:
            state.latch("nonempty Wan attention kwargs are not qualified")

        output = self.fn_ref.original_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )
        fallback_reason = None
        if state.guard_latched:
            fallback_reason = state.guard_reasons[-1]["reason"]
        elif state.prediction_failure_latched and state.predict_failures:
            fallback_reason = state.predict_failures[-1]["error"]
        elif state.coordinate_failure_latched and state.coordinate_failures:
            fallback_reason = state.coordinate_failures[-1]["reason"]
        state.records.append(
            {
                "logical_step": max(int(state.step_index), 0),
                "scheduled_full_compute": bool(state.should_compute),
                "predicted_body_used": not state.should_compute,
                "guard_latched": bool(state.guard_latched),
                "fallback_reason": fallback_reason,
            }
        )
        self.latest_summary = state.summary()
        return output

    def reset_state(self, module: torch.nn.Module):
        self.state_manager.reset()
        self.latest_summary = None
        return module


class SpectrumWanBlockHook(ModelHook):
    """Forecast at the first block; skip middle blocks; sample at block 29."""

    def __init__(self, state_manager: StateManager, *, head: bool = False, tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.head = head
        self.tail = tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumWanState = self.state_manager.get_state()
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if self.head:
            context = self.state_manager.context
            state.start_step(
                logical_step_index=context.step_index,
                runtime_num_inference_steps=context.num_inference_steps,
                provenance_source="cache_context",
            )
            if state.guard_latched:
                # This lane stays full-compute even on a scheduled forecast step.
                state.should_compute = True
            if not state.should_compute:
                try:
                    predicted = state.predict()
                    if not torch.is_tensor(predicted):
                        raise TypeError("Wan forecast is not a tensor")
                    if (
                        predicted.shape != hidden_states.shape
                        or predicted.dtype != hidden_states.dtype
                        or predicted.device != hidden_states.device
                    ):
                        raise ValueError("Wan forecast feature tensor contract changed")
                    if not torch.isfinite(predicted).all():
                        raise FloatingPointError("Wan forecast contains non-finite values")
                    return predicted
                except Exception as error:
                    state.fail_closed_prediction(error)

        if state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.tail and not (state.guard_latched or state.coordinate_failure_latched):
                state.record_real_feature(output)
            return output
        # Every following WanTransformerBlock returns the hidden stream unchanged
        # on a successfully predicted step. Normalization/projection still run.
        return hidden_states
