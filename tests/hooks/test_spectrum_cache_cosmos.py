# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from diffusers import CosmosTransformer3DModel
from diffusers.hooks.hooks import CacheContext
from diffusers.hooks.spectrum_cache import (
    SpectrumCacheConfig,
    SpectrumCosmosState,
    SpectrumCosmosStateManager,
)


def make_model():
    model = CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=2,
        mlp_ratio=2,
        text_embed_dim=16,
        adaln_lora_dim=4,
        max_size=(4, 32, 32),
        patch_size=(1, 2, 2),
        rope_scale=(2.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type="learnable",
    )
    return model.eval()


def make_inputs(seed=0):
    generator = torch.Generator("cpu").manual_seed(seed)
    return {
        "hidden_states": torch.randn((1, 4, 1, 16, 16), generator=generator),
        "timestep": torch.tensor([0.5]),
        "encoder_hidden_states": torch.randn((1, 12, 16), generator=generator),
        "padding_mask": torch.zeros(1, 1, 16, 16),
        "return_dict": False,
    }


def make_config(runtime):
    return SpectrumCacheConfig(
        num_inference_steps=4,
        warmup_steps=2,
        window_size=2.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=4.0,
        tail_actual_steps=0,
        cosmos_runtime_state_callback=lambda: dict(runtime),
    )


def root_state(model):
    hook = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    return hook.state_manager.get_state()


@torch.no_grad()
def test_spectrum_cosmos_static_single_slot_skips_body():
    model = make_model()
    runtime = {
        "step": 0,
        "num_inference_steps": 4,
        "num_conditions": 1,
        "label": "cond",
        "dynamic_conditioning": False,
    }
    model.enable_cache(make_config(runtime))
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    with model.cache_context("anima"):
        for step in range(4):
            runtime["step"] = step
            output = model(**make_inputs(seed=step))[0]
            assert torch.isfinite(output).all()
        state = root_state(model)
        summary = state.summary()
        assert summary["logical_prediction_steps_used"] == [2]
        assert summary["prediction_call_count"] == 1
        assert summary["full_call_count"] == 3
        assert calls["count"] == 3

    handle.remove()
    model.disable_cache()


@torch.no_grad()
def test_spectrum_cosmos_static_cfg_partitions_cond_uncond_history():
    model = make_model()
    runtime = {
        "step": 0,
        "num_inference_steps": 4,
        "num_conditions": 2,
        "label": "cond",
        "dynamic_conditioning": False,
    }
    model.enable_cache(make_config(runtime))
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    with model.cache_context("anima-cfg"):
        for step in range(4):
            runtime["step"] = step
            for label in ("cond", "uncond"):
                runtime["label"] = label
                output = model(**make_inputs(seed=100 + step))[0]
                assert torch.isfinite(output).all()
        state = root_state(model)
        summary = state.summary()
        assert summary["logical_prediction_steps_used"] == [2]
        assert summary["prediction_call_count"] == 2
        assert summary["full_call_count"] == 6
        assert set(state.forecasters) == {"cond", "uncond"}
        assert calls["count"] == 6

    handle.remove()
    model.disable_cache()


@torch.no_grad()
def test_spectrum_cosmos_declared_dynamic_conditioning_is_sticky_fail_closed():
    model = make_model()
    runtime = {
        "step": 0,
        "num_inference_steps": 4,
        "num_conditions": 2,
        "label": "cond",
        "dynamic_conditioning": True,
    }
    model.enable_cache(make_config(runtime))
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    with model.cache_context("anima-dynamic"):
        for step in range(4):
            runtime["step"] = step
            for label in ("cond", "uncond"):
                runtime["label"] = label
                model(**make_inputs(seed=200 + step))
        summary = root_state(model).summary()
        assert summary["guard_latched"]
        assert summary["guard_latched_at"] == 0
        assert summary["prediction_call_count"] == 0
        assert calls["count"] == 8

    handle.remove()
    model.disable_cache()


@torch.no_grad()
def test_spectrum_cosmos_condition_count_transition_latches_before_forecast():
    model = make_model()
    runtime = {
        "step": 0,
        "num_inference_steps": 4,
        "num_conditions": 1,
        "label": "cond",
        "dynamic_conditioning": False,
    }
    model.enable_cache(make_config(runtime))
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    with model.cache_context("anima-transition"):
        for step in range(4):
            runtime["step"] = step
            labels = ("cond",) if step < 2 else ("cond", "uncond")
            runtime["num_conditions"] = len(labels)
            for label in labels:
                runtime["label"] = label
                model(**make_inputs(seed=300 + step))
        summary = root_state(model).summary()
        assert summary["guard_latched"]
        assert summary["guard_latched_at"] == 2
        assert summary["prediction_call_count"] == 0
        assert calls["count"] == 6

    handle.remove()
    model.disable_cache()

# EFF03R: Native and legacy ingress regressions for independent A/B/C/D guards.
# These intentionally exercise production hook entry and state manager ownership.
@pytest.mark.parametrize(
    "step_index,num_inference_steps",
    [
        (False, 4),
        (1.75, 4),
        ("1", 4),
        (0, True),
        (0, 4.5),
        (0, "4"),
    ],
)
@torch.no_grad()
def test_spectrum_cosmos_native_coordinate_types_fail_before_forward(step_index, num_inference_steps):
    model = make_model()
    model.enable_cache(make_config(dict(step=0, num_inference_steps=4, num_conditions=1, label="cond")))
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )
    try:
        with model.cache_context("cond", step_index=step_index, num_inference_steps=num_inference_steps):
            with pytest.raises(TypeError, match="must be integers"):
                model(**make_inputs())
        assert calls["count"] == 0
    finally:
        handle.remove()
        model.disable_cache()


@pytest.mark.parametrize("name", ["cond", "uncond", "anima/route-A"])
@torch.no_grad()
def test_spectrum_cosmos_valid_native_names_and_integer_steps_still_execute(name):
    model = make_model()
    model.enable_cache(make_config(dict(step=0, num_inference_steps=4, num_conditions=1, label="cond")))
    try:
        with model.cache_context(name, step_index=0, num_inference_steps=4):
            output = model(**make_inputs())[0]
            assert torch.isfinite(output).all()
    finally:
        model.disable_cache()


@pytest.mark.parametrize("name", [None, False, 0, b"cond"])
@torch.no_grad()
def test_spectrum_cosmos_native_nonstring_names_fail_before_forward(name):
    model = make_model()
    model.enable_cache(make_config(dict(step=0, num_inference_steps=4, num_conditions=1, label="cond")))
    calls = {"count": 0}
    handle = model.transformer_blocks[0].register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )
    try:
        with model.cache_context(name, step_index=0, num_inference_steps=4):
            with pytest.raises(TypeError, match="name must be a string"):
                model(**make_inputs())
        assert calls["count"] == 0
    finally:
        handle.remove()
        model.disable_cache()


@pytest.mark.parametrize("name", ["", " ", "\t"])
@torch.no_grad()
def test_spectrum_cosmos_native_blank_names_fail_before_forward(name):
    model = make_model()
    model.enable_cache(make_config(dict(step=0, num_inference_steps=4, num_conditions=1, label="cond")))
    try:
        with model.cache_context(name, step_index=0, num_inference_steps=4):
            with pytest.raises(ValueError, match="name must be non-empty"):
                model(**make_inputs())
    finally:
        model.disable_cache()


def test_spectrum_cosmos_legacy_bucket_cannot_alias_native_name():
    config = make_config(dict(step=0, num_inference_steps=4, num_conditions=1, label="cond"))
    manager = SpectrumCosmosStateManager(SpectrumCosmosState, init_args=(config,))
    legacy = manager.get_state()
    manager.set_context(CacheContext(name="__legacy_cosmos__", step_index=0, num_inference_steps=4))
    native = manager.get_state()
    assert legacy is not native
    assert manager.get_state() is native
    manager.set_context(None)
    assert manager.get_state() is legacy
    manager.reset()
    assert manager._state_cache == {}


@pytest.mark.parametrize(
    "field,value,error_type",
    [
        ("step", True, TypeError),
        ("step", 1.75, TypeError),
        ("num_inference_steps", False, TypeError),
        ("num_inference_steps", "4", TypeError),
        ("num_conditions", True, TypeError),
        ("num_conditions", 2.5, TypeError),
        ("num_conditions", 0, ValueError),
        ("num_conditions", -1, ValueError),
    ],
)
@torch.no_grad()
def test_spectrum_cosmos_legacy_invalid_numeric_ingress_fail_before_forward(field, value, error_type):
    runtime = dict(step=0, num_inference_steps=4, num_conditions=1, label="cond")
    runtime[field] = value
    model = make_model()
    model.enable_cache(make_config(runtime))
    try:
        with pytest.raises(error_type):
            model(**make_inputs())
    finally:
        model.disable_cache()


@pytest.mark.parametrize("label", ["cond", "uncond"])
@torch.no_grad()
def test_spectrum_cosmos_legacy_valid_integer_callback_still_executes(label):
    runtime = dict(step=0, num_inference_steps=4, num_conditions=1, label=label)
    model = make_model()
    model.enable_cache(make_config(runtime))
    try:
        output = model(**make_inputs())[0]
        assert torch.isfinite(output).all()
    finally:
        model.disable_cache()


@pytest.mark.parametrize("mutation", ["none", "hidden_dtype", "timestep_dtype"])
@torch.no_grad()
def test_spectrum_cosmos_signature_metadata_change_is_fail_closed(mutation):
    """Retain changing values, but fail closed when input or timestep metadata drifts."""
    from types import SimpleNamespace

    from diffusers.hooks.spectrum_cache import SpectrumCosmosDenoiserHook

    config = SpectrumCacheConfig(
        num_inference_steps=4, warmup_steps=4, window_size=2.0, flex_window=0.0,
        degree=2, ridge_lambda=0.1, blend_w=0.5, history_limit=8,
        coordinate_max=4.0, tail_actual_steps=0,
    )
    manager = SpectrumCosmosStateManager(SpectrumCosmosState, init_args=(config,))
    hook = SpectrumCosmosDenoiserHook(manager)
    hook.fn_ref = SimpleNamespace(original_forward=lambda **kwargs: (kwargs["hidden_states"],))
    module = torch.nn.Identity().eval()

    for step in range(2):
        manager.set_context(CacheContext("pred_cond", step_index=step, num_inference_steps=4))
        hidden_dtype = torch.bfloat16 if step == 1 and mutation == "hidden_dtype" else torch.float32
        timestep_dtype = torch.float64 if step == 1 and mutation == "timestep_dtype" else torch.float32
        output = hook.new_forward(
            module, hidden_states=torch.zeros(1, 4, 1, 2, 2, dtype=hidden_dtype),
            timestep=torch.tensor([0.5 + 0.1 * step], dtype=timestep_dtype),
            encoder_hidden_states=torch.zeros(1, 3, 4),
            padding_mask=torch.zeros(1, 1, 2, 2), return_dict=False,
        )
        assert len(output) == 1

    state = manager._state_cache["pred_cond"]
    assert state.guard_latched == (mutation != "none")
    assert len(state.records) == 2
    if mutation != "none":
        assert state.guard_latched_at == 1
        assert any("conditioning/input signature changed" in row["reason"] for row in state.guard_reasons)
