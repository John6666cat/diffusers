# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only synthetic regression checks for bounded Qwen-Image SPECTRUM hooks."""

import pytest
import torch
from torch import nn

from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.hooks import HookRegistry, StateManager
from diffusers.hooks.spectrum_cache import (
    SpectrumCacheConfig,
    SpectrumQwenBlockHook,
    SpectrumQwenHeadBlockHook,
    SpectrumQwenImageDenoiserHook,
    SpectrumQwenImageState,
    _SPECTRUM_DENOISER_HOOK,
    _SPECTRUM_QWEN_BLOCK_HOOK,
    _SPECTRUM_QWEN_HEAD_BLOCK_HOOK,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel


class _ToyQwenBlock(nn.Module):
    """Same two-stream input/output order as a QwenImageTransformerBlock."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states):
        self.calls += 1
        return encoder_hidden_states + 0.1, hidden_states + 1.0


class _ToyQwenTransformer(nn.Module, CacheMixin):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_ToyQwenBlock() for _ in range(3)])
        self._cache_config = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        timestep=None,
        img_shapes=None,
        guidance=None,
        attention_kwargs=None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict=True,
    ):
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )
        return (hidden_states,)


def _config():
    return SpectrumCacheConfig(
        num_inference_steps=6,
        warmup_steps=2,
        forecast_step_indices=(3,),
        history_limit=8,
        degree=1,
        coordinate_policy="runtime_horizon_normalized",
    )


def _install_toy(*, route="t2i"):
    TransformerBlockRegistry.register(
        _ToyQwenBlock,
        TransformerBlockMetadata(return_hidden_states_index=1, return_encoder_hidden_states_index=0),
    )
    model = _ToyQwenTransformer().eval()
    config = _config()
    manager = StateManager(SpectrumQwenImageState, init_args=(config,))
    HookRegistry.check_if_exists_or_initialize(model).register_hook(
        SpectrumQwenImageDenoiserHook(manager, route), _SPECTRUM_DENOISER_HOOK
    )
    HookRegistry.check_if_exists_or_initialize(model.blocks[0]).register_hook(
        SpectrumQwenHeadBlockHook(manager), _SPECTRUM_QWEN_HEAD_BLOCK_HOOK
    )
    HookRegistry.check_if_exists_or_initialize(model.blocks[1]).register_hook(
        SpectrumQwenBlockHook(manager), _SPECTRUM_QWEN_BLOCK_HOOK
    )
    HookRegistry.check_if_exists_or_initialize(model.blocks[2]).register_hook(
        SpectrumQwenBlockHook(manager, is_tail=True), _SPECTRUM_QWEN_BLOCK_HOOK
    )
    model._cache_config = config
    return model, manager


def _run(model, step, *, lane="cond", route="t2i", extra=None, expected_steps=6):
    shapes = [[(1, 1, 2)]] if route == "t2i" else [[(1, 1, 1), (1, 1, 1)]]
    kwargs = {
        "hidden_states": torch.full((1, 2, 4), 0.1 * step),
        "encoder_hidden_states": torch.ones(1, 3, 4),
        "encoder_hidden_states_mask": torch.ones(1, 3, dtype=torch.bool),
        "timestep": torch.tensor([1.0 - 0.1 * step]),
        "img_shapes": shapes,
        "return_dict": False,
    }
    kwargs.update(extra or {})
    with model.cache_context(lane, step_index=step, num_inference_steps=expected_steps):
        return model(**kwargs)[0]


@pytest.mark.parametrize("steps", [16, 20, 24, 30, 40])
def test_qwen_variable_step_profile_is_bounded(steps):
    cfg = SpectrumCacheConfig.for_qwen_image_variable(steps)
    assert cfg.num_inference_steps == steps
    assert cfg.coordinate_policy == "runtime_horizon_normalized"
    assert cfg.warmup_steps == round(0.3 * steps)
    assert cfg.tail_actual_steps == round(0.2 * steps)
    assert cfg.history_limit == 8


@pytest.mark.parametrize("steps", [0, 15, 41, 60])
def test_unqualified_variable_step_profile_fails_closed(steps):
    with pytest.raises(ValueError, match="16 <= num_inference_steps <= 40"):
        SpectrumCacheConfig.for_qwen_image_variable(steps)


@pytest.mark.parametrize("route", ["t2i", "edit"])
@torch.no_grad()
def test_dual_stream_forecast_and_disable_are_reversible(route):
    model, manager = _install_toy(route=route)
    for step in range(6):
        for lane in ("cond", "uncond"):
            assert torch.isfinite(_run(model, step, lane=lane, route=route)).all()
    assert sum(block.calls for block in model.blocks) == 30  # 36 calls minus 2 forecast x 3 blocks
    for state in manager._state_cache.values():
        summary = state.summary()
        assert summary["prediction_call_count"] == 1
        assert summary["full_call_count"] == 5
        assert summary["forecast_steps"] == [3]
        assert not summary["guard_latched"]
        assert not summary["coordinate"]["failure_latched"]

    model.disable_cache()
    assert not model.is_cache_enabled
    before = [block.calls for block in model.blocks]
    _run(model, 0, route=route)
    assert [block.calls - count for block, count in zip(model.blocks, before)] == [1, 1, 1]


@torch.no_grad()
def test_qwen_changed_shape_latches_full_compute_without_forecast():
    model, manager = _install_toy()
    _run(model, 0)
    _run(model, 1, extra={"hidden_states": torch.ones(1, 3, 4), "img_shapes": [[(1, 1, 3)]]})
    for step in range(2, 6):
        _run(model, step)
    state = manager._state_cache["cond"]
    assert state.guard_latched
    assert state.summary()["prediction_call_count"] == 0
    model.disable_cache()


@pytest.mark.parametrize("extra", [{"guidance": torch.tensor([1.0])}, {"attention_kwargs": {"scale": 1.0}}])
@torch.no_grad()
def test_qwen_unqualified_inputs_latch_full_compute(extra):
    model, manager = _install_toy()
    for step in range(6):
        _run(model, step, extra=extra)
    assert manager._state_cache["cond"].guard_latched
    assert manager._state_cache["cond"].summary()["prediction_call_count"] == 0
    model.disable_cache()


@torch.no_grad()
def test_qwen_missing_step_provenance_stays_full_compute():
    model, manager = _install_toy()
    for step in range(6):
        _run(model, step, expected_steps=7)
    summary = manager._state_cache["cond"].summary()
    assert summary["coordinate"]["failure_latched"]
    assert summary["prediction_call_count"] == 0
    model.disable_cache()


@torch.no_grad()
def test_qwen_prediction_failure_falls_back_and_sticks():
    model, manager = _install_toy()
    for step in range(3):
        _run(model, step)
    state = manager._state_cache["cond"]

    def raise_forecast():
        raise FloatingPointError("test-injected failure")

    state.predict = raise_forecast
    for step in range(3, 6):
        assert torch.isfinite(_run(model, step)).all()
    assert state.prediction_failure_latched
    assert state.summary()["prediction_call_count"] == 0
    assert state.fallback_steps == [3]
    model.disable_cache()


@torch.no_grad()
def test_qwen_interrupted_context_reset_clears_history():
    model, manager = _install_toy()
    for step in range(4):
        _run(model, step)
    assert manager._state_cache["cond"].summary()["prediction_call_count"] == 1
    model._reset_stateful_cache()
    assert not manager._state_cache
    _run(model, 0)
    assert manager._state_cache["cond"].step_index == 0
    model.disable_cache()


def test_nonmatching_qwen_architecture_is_rejected():
    model = QwenImageTransformer2DModel(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        axes_dims_rope=(4, 4, 8),
    )
    with pytest.raises(ValueError, match="Qwen-Image-2512"):
        model.enable_cache(SpectrumCacheConfig.for_qwen_image_variable(20))
    assert not model.is_cache_enabled
