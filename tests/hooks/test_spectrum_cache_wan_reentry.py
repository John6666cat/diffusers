"""Synthetic CPU gates for the opt-in Wan2.1 T2V 1.3B research profiles.

No large Wan weights are needed: this checks the strict architecture gate,
separate CFG-lane state, actual skipped block calls, fail-closed guards, and
hook cleanup. Actual T2V quality and NF4 memory require a separate GPU probe.
"""

import types
from dataclasses import replace

import pytest
import torch

from diffusers.hooks.hooks import HookRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel


WAN21_13B_SIGNATURE = {
    "num_layers": 30,
    "num_attention_heads": 12,
    "attention_head_dim": 128,
    "ffn_dim": 8960,
    "text_dim": 4096,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": (1, 2, 2),
    "image_dim": None,
    "added_kv_proj_dim": None,
}

PRACTICAL_50_STEP_PROFILES = (
    (
        "for_wan21_13b_nfe12_nested",
        (0, 1, 2, 3, 7, 11, 18, 26, 34, 38, 42, 49),
    ),
    (
        "for_wan21_13b_nfe11_add5",
        (0, 1, 2, 3, 5, 11, 18, 26, 34, 42, 49),
    ),
)


class TinyWanBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        self.calls += 1
        return hidden_states + 0.125


def make_tiny_wan():
    model = WanTransformer3DModel.__new__(WanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.register_to_config(**WAN21_13B_SIGNATURE)
    model.blocks = torch.nn.ModuleList(TinyWanBlock() for _ in range(30))

    def fake_forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
        attention_kwargs=None,
    ):
        # Exercise the real Wan block-call signature with tiny feature maps.
        feature = hidden_states.flatten(2).transpose(1, 2).contiguous()
        feature = feature + timestep.reshape(-1, 1, 1) * 0.01
        for block in self.blocks:
            feature = block(feature, encoder_hidden_states, timestep, None)
        return (feature,) if not return_dict else {"sample": feature}

    model.forward = types.MethodType(fake_forward, model)
    model.eval()
    return model


@pytest.mark.parametrize("steps", (40, 45, 50, 55, 60))
def test_profile_scheduled_counts_and_bounds(steps):
    config = SpectrumCacheConfig.for_wan21_13b_variable(steps)
    early = int(0.22 * steps + 0.5)
    tail = int(0.20 * steps + 0.5)
    forecast = tuple(index for index in range(early, steps - tail) if (index - early) % 3 != 1)
    assert config.forecast_step_indices == forecast
    assert config.warmup_steps == early
    assert config.tail_actual_steps == tail
    assert config.coordinate_policy == "runtime_horizon_normalized"
    assert config.degree == 4 and config.ridge_lambda == 0.1 and config.blend_w == 0.5


@pytest.mark.parametrize("steps", (39, 61))
def test_profile_outside_measured_range_rejected(steps):
    with pytest.raises(ValueError, match="40 <= num_inference_steps <= 60"):
        SpectrumCacheConfig.for_wan21_13b_variable(steps)


@pytest.mark.parametrize("factory_name, actual_steps", PRACTICAL_50_STEP_PROFILES)
def test_practical_50_step_explicit_schedule(factory_name, actual_steps):
    config = getattr(SpectrumCacheConfig, factory_name)()
    schedule = SpectrumSchedule(config)
    assert config.num_inference_steps == 50
    assert config.warmup_steps == 4
    assert config.tail_actual_steps == 1
    assert config.coordinate_policy == "runtime_horizon_normalized"
    assert config.predictor_backend == "dense"
    assert config.degree == 4 and config.ridge_lambda == 0.1 and config.blend_w == 0.5
    assert tuple(i for i in range(50) if schedule.decide(i)) == actual_steps
    assert tuple(i for i in range(50) if i not in config.forecast_step_indices) == actual_steps


@pytest.mark.parametrize(
    "config_factory",
    (
        SpectrumCacheConfig.for_wan21_13b_variable,
        SpectrumCacheConfig.for_wan21_13b_nfe12_nested,
        SpectrumCacheConfig.for_wan21_13b_nfe11_add5,
    ),
)
def test_50_step_cond_uncond_skip_and_cleanup(config_factory):
    model = make_tiny_wan()
    config = config_factory()
    model.enable_cache(config)

    latent = torch.ones((1, 16, 1, 4, 4))
    text = torch.ones((1, 5, 8))
    with torch.no_grad():
        for step in range(50):
            for label in ("cond", "uncond"):
                with model.cache_context(label, step_index=step, num_inference_steps=50):
                    result = model(
                        hidden_states=latent,
                        timestep=torch.tensor([step], dtype=torch.float32),
                        encoder_hidden_states=text,
                        return_dict=False,
                    )
                    assert isinstance(result, tuple)

    root = HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser")
    assert sorted(root.state_manager._state_cache) == ["cond", "uncond"]
    for label in ("cond", "uncond"):
        summary = root.state_manager._state_cache[label].summary()
        assert not summary["guard_latched"], summary["guard_reasons"]
        assert not summary["coordinate"]["failure_latched"]
        assert not summary["prediction_failure_latched"], summary["predict_failures"]
        assert summary["forecast_steps"] == list(config.forecast_step_indices)
        assert summary["prediction_call_count"] == len(config.forecast_step_indices)
        assert summary["full_call_count"] == 50 - len(config.forecast_step_indices)
    for block in model.blocks:
        assert block.calls == 2 * (50 - len(config.forecast_step_indices))

    model.disable_cache()
    assert not model.is_cache_enabled
    assert HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser") is None


def test_image_conditioning_is_sticky_full_compute():
    model = make_tiny_wan()
    config = SpectrumCacheConfig.for_wan21_13b_variable(40)
    model.enable_cache(config)
    with torch.no_grad():
        for step in range(40):
            with model.cache_context("cond", step_index=step, num_inference_steps=40):
                model(
                    hidden_states=torch.ones((1, 16, 1, 4, 4)),
                    timestep=torch.tensor([step], dtype=torch.float32),
                    encoder_hidden_states=torch.ones((1, 5, 8)),
                    encoder_hidden_states_image=torch.ones((1, 3, 8)) if step == 0 else None,
                    return_dict=False,
                )
    root = HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert summary["full_call_count"] == 40
    assert all(block.calls == 40 for block in model.blocks)
    model.disable_cache()


def test_architecture_and_coordinate_gates():
    model = make_tiny_wan()
    config = SpectrumCacheConfig.for_wan21_13b_variable()
    model.register_to_config(num_layers=29)
    with pytest.raises(ValueError, match="architecture"):
        model.enable_cache(config)
    model.register_to_config(num_layers=30)
    with pytest.raises(ValueError, match="explicit native-context"):
        model.enable_cache(replace(config, coordinate_policy="legacy_fixed_max"))
    assert not model.is_cache_enabled
