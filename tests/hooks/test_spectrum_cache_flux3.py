# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import torch
import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig
from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel


def _tiny_base(*, flow_map=False):
    return Flux3Transformer2DModel(
        in_channels=8,
        cond_in_channels=8,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        timestep_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
        flow_map=flow_map,
    ).eval()


def _inputs():
    return {
        "hidden_states": torch.randn(1, 4, 8),
        "encoder_hidden_states": torch.randn(1, 3, 32),
        "img_ids": torch.zeros(4, 4),
        "txt_ids": torch.zeros(3, 4),
    }


def _config(*, forecast_steps=(2,)):
    return SpectrumCacheConfig(
        num_inference_steps=4,
        warmup_steps=2,
        forecast_step_indices=forecast_steps,
        degree=1,
        coordinate_policy="runtime_horizon_normalized",
    )


def _run(model, inputs, *, step, slot="cond", extra=None):
    kwargs = dict(inputs)
    kwargs.update(extra or {})
    kwargs["timestep"] = torch.tensor([1.0 - 0.2 * step])
    with model.cache_context(slot, step_index=step, num_inference_steps=4, timestep=step):
        return model(**kwargs).sample


def _states(model):
    hook = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    return hook.state_manager._state_cache


@torch.no_grad()
def test_base_no_context_is_identical_and_disable_is_reversible():
    torch.manual_seed(123)
    model = _tiny_base()
    inputs = _inputs()
    before = model(**inputs, timestep=torch.tensor([0.8])).sample
    model.enable_cache(_config())
    during = model(**inputs, timestep=torch.tensor([0.8])).sample
    assert torch.equal(before, during)
    assert not _states(model)
    model.disable_cache()
    after = model(**inputs, timestep=torch.tensor([0.8])).sample
    assert not model.is_cache_enabled
    assert torch.equal(before, after)


@torch.no_grad()
def test_base_exact_all_real_compute():
    torch.manual_seed(123)
    model = _tiny_base()
    inputs = _inputs()
    expected = [model(**inputs, timestep=torch.tensor([1.0 - 0.2 * i])).sample for i in range(4)]
    model.enable_cache(_config(forecast_steps=()))
    actual = [_run(model, inputs, step=i) for i in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert _states(model)["cond"].summary()["prediction_call_count"] == 0
    model.disable_cache()


@torch.no_grad()
def test_base_cfg_is_partitioned_and_skips_real_block(monkeypatch):
    torch.manual_seed(123)
    model = _tiny_base()
    inputs = _inputs()
    other_inputs = dict(inputs)
    other_inputs["encoder_hidden_states"] = torch.randn_like(inputs["encoder_hidden_states"])
    block = model.single_transformer_blocks[0]
    forward = block.forward
    called = []

    def traced(*args, **kwargs):
        called.append(True)
        return forward(*args, **kwargs)

    monkeypatch.setattr(block, "forward", traced)
    model.enable_cache(_config())
    for i in range(4):
        for name, branch in (("cond", inputs), ("uncond", other_inputs)):
            assert torch.isfinite(_run(model, branch, step=i, slot=name)).all()
    assert len(called) == 6  # 8 CFG calls - 2 forecast calls
    assert set(_states(model)) == {"cond", "uncond"}
    for state in _states(model).values():
        result = state.summary()
        assert result["prediction_call_count"] == 1
        assert result["full_call_count"] == 3
        assert result["logical_prediction_steps_used"] == [2]
        assert not result["guard_latched"]
    model.disable_cache()


@torch.no_grad()
def test_changed_conditioning_identity_stays_real_compute():
    model = _tiny_base()
    inputs = _inputs()
    model.enable_cache(_config())
    _run(model, inputs, step=0)
    changed = dict(inputs)
    changed["encoder_hidden_states"] = inputs["encoder_hidden_states"].clone()
    for i in range(1, 4):
        _run(model, changed, step=i)
    state = _states(model)["cond"]
    assert state.guard_latched
    assert state.summary()["prediction_call_count"] == 0
    model.disable_cache()


@torch.no_grad()
def test_reference_image_path_latches_bypass():
    model = _tiny_base()
    inputs = _inputs()
    model.enable_cache(_config())
    _run(model, inputs, step=0)
    refs = {
        "image_cond_hidden_states": torch.randn(1, 2, 8),
        "image_cond_ids": torch.zeros(2, 4),
    }
    _run(model, inputs, step=1, extra=refs)
    _run(model, inputs, step=2)
    assert _states(model)["cond"].guard_latched
    assert _states(model)["cond"].summary()["prediction_call_count"] == 0
    model.disable_cache()


@torch.no_grad()
def test_prediction_error_fails_closed():
    model = _tiny_base()
    inputs = _inputs()
    model.enable_cache(_config())
    for i in (0, 1):
        _run(model, inputs, step=i)
    state = _states(model)["cond"]

    def fail_prediction():
        raise FloatingPointError("injected")

    state.predict = fail_prediction
    output = _run(model, inputs, step=2)
    assert torch.isfinite(output).all()
    assert state.guard_latched
    assert state.summary()["prediction_call_count"] == 0
    model.disable_cache()


@torch.no_grad()
def test_reset_erases_interrupted_history():
    model = _tiny_base()
    inputs = _inputs()
    model.enable_cache(_config())
    for i in (0, 1, 2):
        _run(model, inputs, step=i)
    assert _states(model)["cond"].summary()["prediction_call_count"] == 1
    model._reset_stateful_cache()
    assert not _states(model)
    changed = _inputs()
    _run(model, changed, step=0)
    assert _states(model)["cond"].summary()["prediction_call_count"] == 0
    assert _states(model)["cond"].current_step == 0
    model.disable_cache()


def test_unsupported_config_is_not_silently_enabled():
    model = _tiny_base(flow_map=True)
    with pytest.raises(ValueError, match="undistilled Base"):
        model.enable_cache(_config())
    assert not model.is_cache_enabled
    model = _tiny_base()
    with pytest.raises(ValueError, match="runtime-normalized"):
        model.enable_cache(SpectrumCacheConfig(num_inference_steps=4))
    assert not model.is_cache_enabled
