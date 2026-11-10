"""Optional native SPECTRUM block adapter for Z-Image Base and Turbo text-to-image.

Both routes use native pipeline step provenance and explicitly enabled cache.
The owner's 0.42.1 GPU probes cover only narrow model/step/quality conditions.
"""

from typing import Any

import torch

from .hooks import ModelHook, StateManager
from .spectrum_cache import SpectrumQwenImageState, _spectrum_tensor_signature


class SpectrumZImageState(SpectrumQwenImageState):
    """The Z-Image route shares Qwen's context-local structural latch and ledger only."""


class SpectrumZImageDenoiserHook(ModelHook):
    """Keep only standard flat-list T2I with native step context eligible for forecasting."""

    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager
        self.latest_summary: dict[str, Any] | None = None

    @staticmethod
    def _sequence_signature(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return tuple(_spectrum_tensor_signature(item) for item in value)
        return _spectrum_tensor_signature(value)

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
        joint_attention_kwargs: dict[str, Any] | None = None,
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
            state.latch("autograd/training Z-Image path is not qualified")
        if not isinstance(x, (tuple, list)) or not x:
            state.latch("non-sequence Z-Image input is not qualified")
        elif not all(torch.is_tensor(item) for item in x):
            state.latch("nested-image or non-tensor Z-Image input is not qualified")
        if not isinstance(cap_feats, (tuple, list)) or not cap_feats:
            state.latch("non-sequence Z-Image captions are not qualified")
        elif not all(torch.is_tensor(item) for item in cap_feats):
            state.latch("nested caption-feature Z-Image path is not qualified")
        if controlnet_block_samples is not None or siglip_feats is not None or image_noise_mask is not None:
            state.latch("special-conditioning Z-Image path is not qualified")
        if patch_size != 2 or f_patch_size != 1:
            state.latch("non-default patch geometry is not qualified")
        if not torch.is_tensor(t) or t.ndim != 1:
            state.latch("non-1D Z-Image timestep is not qualified")
        if joint_attention_kwargs:
            state.latch("non-default LoRA/attention scale is not qualified")

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
            "joint_attention_kwargs": joint_attention_kwargs,
        }
        fallback_reason = None
        if state.guard_latched:
            state.bypass = True
            state.should_compute = True
            fallback_reason = state.guard_reasons[-1]["reason"]
        try:
            output = self.fn_ref.original_forward(**call_kwargs)
        finally:
            state.bypass = False
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
        return module


class SpectrumZImageBlockHook(ModelHook):
    """Forecast the post-refiner body at block zero, or capture it at block 29."""

    def __init__(self, state_manager: StateManager, *, head: bool = False, tail: bool = False):
        super().__init__()
        self.state_manager = state_manager
        self.head = head
        self.tail = tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        state: SpectrumZImageState = self.state_manager.get_state()
        if state.bypass:
            return self.fn_ref.original_forward(*args, **kwargs)

        x = args[0] if args else kwargs.get("x")
        if self.head:
            context = self.state_manager.context
            state.start_step(
                logical_step_index=context.step_index,
                runtime_num_inference_steps=context.num_inference_steps,
                provenance_source="cache_context",
            )
            if not state.should_compute:
                try:
                    predicted = state.predict()
                    if not torch.is_tensor(predicted):
                        raise TypeError("non-tensor Z-Image forecast")
                    if predicted.shape != x.shape or predicted.dtype != x.dtype or predicted.device != x.device:
                        raise ValueError("Z-Image forecast tensor contract changed")
                    if not torch.isfinite(predicted).all():
                        raise FloatingPointError("non-finite Z-Image forecast")
                    return predicted
                except Exception as error:
                    state.fail_closed_prediction(error)

        if state.should_compute:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.tail and not state.coordinate_failure_latched:
                state.record_real_feature(output)
            return output
        return x
