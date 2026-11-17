"""Opt-in SPECTRUM adapter for the MOSS-Image 42-block DiT.

The official transformer stays unchanged on the cache-OFF path. On a forecast
step the first block produces a predicted post-last-block feature; intermediate
blocks are bypassed, and the original timestep-dependent final head still runs.
"""

from typing import Any

import torch

from .hooks import HookRegistry, ModelHook, StateManager
from .spectrum_cache import (
    SpectrumCacheConfig,
    SpectrumState,
    _SPECTRUM_BLOCK_HOOK,
    _SPECTRUM_DENOISER_HOOK,
    _SPECTRUM_HEAD_BLOCK_HOOK,
    _spectrum_tensor_signature,
)


class SpectrumMOSSState(SpectrumState):
    """An isolated predictor and fail-closed ledger for one CFG/SLG lane."""

    def reset(self) -> None:
        super().reset()
        self.guard_reason: str | None = None
        self.signature: dict[str, Any] | None = None
        self.predicted_this_call = False
        self.records: list[dict[str, Any]] = []

    def latch(self, reason: str) -> None:
        if self.guard_reason is None:
            self.guard_reason = reason
        self.should_compute = True

    def summary(self) -> dict[str, Any]:
        return {
            "full_calls": sum(not row["predicted"] for row in self.records),
            "forecast_calls": sum(row["predicted"] for row in self.records),
            "full_steps": [row["step"] for row in self.records if not row["predicted"]],
            "forecast_steps": [row["step"] for row in self.records if row["predicted"]],
            "guard_reason": self.guard_reason,
            "coordinate": self.coordinate_summary(),
            "predict_failures": list(self.predict_failures),
            "peak_history_bytes": int(self.peak_history_bytes),
        }


class SpectrumMOSSDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(self, state_manager: StateManager):
        super().__init__()
        self.state_manager = state_manager

    def new_forward(
        self,
        module: torch.nn.Module,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
    ):
        if self.state_manager._context is None:
            # A direct transformer invocation outside MOSSImagePipeline must
            # remain an ordinary full forward even with the cache installed.
            return self.fn_ref.original_forward(
                hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict, skip_layers=skip_layers,
            )
        state: SpectrumMOSSState = self.state_manager.get_state()
        context = self.state_manager.context
        skip = tuple(sorted(skip_layers or ()))
        signature = {
            "hidden": _spectrum_tensor_signature(hidden_states),
            "text": _spectrum_tensor_signature(encoder_hidden_states),
            "mask": _spectrum_tensor_signature(encoder_attention_mask),
            "skip": skip,
        }
        if state.signature is None:
            state.signature = signature
        elif state.signature != signature:
            state.latch("MOSS conditioning or feature contract changed within a lane")

        if module.training or torch.is_grad_enabled():
            state.latch("MOSS SPECTRUM requires eval/no_grad")
        if not torch.is_tensor(hidden_states) or hidden_states.ndim != 4 or hidden_states.shape[0] != 1:
            state.latch("MOSS SPECTRUM supports only one image per denoiser call")
        if context.step_index is None or context.num_inference_steps is None:
            state.latch("native step provenance missing")
        state.predicted_this_call = False
        result = self.fn_ref.original_forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
            skip_layers=skip_layers,
        )
        state.records.append({
            "step": context.step_index,
            "predicted": bool(state.predicted_this_call),
        })
        return result

    def reset_state(self, module):
        self.state_manager.reset()
        return module


class SpectrumMOSSBlockHook(ModelHook):
    def __init__(self, state_manager: StateManager, *, head=False, tail=False):
        super().__init__()
        self.state_manager = state_manager
        self.head = head
        self.tail = tail

    def new_forward(self, module: torch.nn.Module, *args, **kwargs):
        if self.state_manager._context is None:
            return self.fn_ref.original_forward(*args, **kwargs)
        state: SpectrumMOSSState = self.state_manager.get_state()
        hidden = args[0] if args else kwargs["x"]

        if self.head and state.guard_reason is None:
            context = self.state_manager.context
            state.start_step(
                logical_step_index=context.step_index,
                runtime_num_inference_steps=context.num_inference_steps,
                provenance_source="moss_pipeline.cache_context",
            )
            if not state.should_compute and len(state.forecaster.features) < state.config.degree + 1:
                # Late-activated CFG/SLG branches must build their own real history.
                state.should_compute = True
            if not state.should_compute:
                try:
                    predicted = state.predict()
                    if not isinstance(predicted, torch.Tensor):
                        raise TypeError("forecast did not return a tensor")
                    if predicted.shape != hidden.shape or predicted.dtype != hidden.dtype or predicted.device != hidden.device:
                        raise ValueError("MOSS post-block feature shape/dtype/device changed")
                    if not bool(torch.isfinite(predicted).all()):
                        raise FloatingPointError("non-finite MOSS predicted feature")
                    state.predicted_this_call = True
                    return predicted
                except Exception as error:
                    state.fail_closed_prediction(error)

        if state.should_compute or state.guard_reason is not None:
            output = self.fn_ref.original_forward(*args, **kwargs)
            if self.tail and state.guard_reason is None and not state.coordinate_failure_latched:
                state.record_real_feature(output)
            return output
        return hidden


def apply_spectrum_cache_moss(module: torch.nn.Module, config: SpectrumCacheConfig) -> None:
    """Install an opt-in adapter with explicit MOSS topology checks."""
    from ..utils.torch_utils import unwrap_module

    model = unwrap_module(module)
    if (
        type(model).__name__ != "MOSSImageTransformer"
        or len(getattr(model, "layers", ())) != 42
        or getattr(model.config, "num_layers", None) != 42
        or getattr(model.config, "latent_channels", None) != 128
    ):
        raise ValueError("MOSS SPECTRUM requires the official 42-layer/128-channel DiT")
    if (
        config.coordinate_policy != "runtime_index_normalized"
        or config.tail_actual_steps < 1
        or config.num_inference_steps < 6
    ):
        raise ValueError("MOSS cache requires native normalized steps and >=1 real tail step")

    manager = StateManager(SpectrumMOSSState, init_args=(config,))
    HookRegistry.check_if_exists_or_initialize(module).register_hook(
        SpectrumMOSSDenoiserHook(manager), _SPECTRUM_DENOISER_HOOK
    )
    for index, block in enumerate(model.layers):
        HookRegistry.check_if_exists_or_initialize(block).register_hook(
            SpectrumMOSSBlockHook(manager, head=index == 0, tail=index == len(model.layers) - 1),
            _SPECTRUM_HEAD_BLOCK_HOOK if index == 0 else _SPECTRUM_BLOCK_HOOK,
        )


def get_moss_spectrum_summary(module: torch.nn.Module) -> dict[str, Any] | None:
    """Snapshot hook statistics before a pipeline resets the per-run states."""
    hook = HookRegistry.check_if_exists_or_initialize(module).get_hook(_SPECTRUM_DENOISER_HOOK)
    if not isinstance(hook, SpectrumMOSSDenoiserHook):
        return None
    return {name: state.summary() for name, state in hook.state_manager._state_cache.items()}
