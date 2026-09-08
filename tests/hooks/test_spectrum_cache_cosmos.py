# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from diffusers import CosmosTransformer3DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


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
