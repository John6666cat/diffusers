"""Opt-in N1–80 research horizons; no GPU/Owner quality qualification is implied."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule
from diffusers.hooks.spectrum_qwenimage21 import SpectrumQwen21State


# Digests pin every dataclass field for the exact old Source parent daded36e.
HISTORICAL = {
    "zimage_p12": (20, 40, "184216abc7a42ea92d43ec5c8424b29fb3dec55f7e41e9062736abc5cc834138"),
    "qwen_legacy": (16, 40, "a725a2ade1f74cf4024e304b8ea30a8d06ab148c0b85b032f42885bb3c86e342"),
    "qwen21_55": (16, 40, "5a0386a3f02ecad1e1d68883890dc34b47a48b5eb77ad86ec0a9d906dc9d2acc"),
    "qwen21_60": (16, 40, "68248ac683867188dc2f4339bc6fce5d08837ea0b68de2157398eb629d6f5cf8"),
}


def _factory(name, n):
    if name == "zimage_p12":
        return SpectrumCacheConfig.for_zimage_base_variable(n)
    if name == "qwen_legacy":
        return SpectrumCacheConfig.for_qwen_image_variable(n)
    return SpectrumCacheConfig.for_qwen_image21_variable(n, forecast_fraction=0.60 if name == "qwen21_60" else 0.55)


@pytest.mark.parametrize("name", HISTORICAL)
def test_historical_full_field_digest_unchanged(name):
    lo, hi, expected = HISTORICAL[name]
    digest = hashlib.sha256()
    for n in range(lo, hi + 1):
        obj = _factory(name, n)
        digest.update(json.dumps(asdict(obj), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    assert digest.hexdigest() == expected


def test_old_8_step_turbo_and_n28_spread_unchanged():
    for config, digest in (
        (SpectrumCacheConfig.for_zimage_turbo(), "06dbab8cdae0d4040cdd619df945f0a3bee68a5b1da7dbfbab95211c13250995"),
        (SpectrumCacheConfig.for_zimage_base_variable(28, profile="p8_spread"), "c08e9421f337cd12a805598807988c45e74bff63cfd2f6f39dd84c4c531895fd"),
    ):
        actual = hashlib.sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert actual == digest


@pytest.mark.parametrize("name", ("zimage_p12", "qwen_legacy", "qwen21_55"))
def test_all_n1_80_schedule_and_realtime_guards(name):
    for n in range(1, 81):
        obj = _factory(name, n)
        schedule = SpectrumSchedule(obj)
        mask = tuple(i for i in range(n) if not schedule.decide(i))
        assert obj.num_inference_steps == n
        assert all(0 <= x < n for x in mask)
        if n <= 11:
            assert mask == ()
        else:
            assert len(mask) > 0
            assert not any(i < obj.warmup_steps or i >= n - obj.tail_actual_steps for i in mask)
        if name == "qwen21_55" and not 16 <= n <= 40:
            run = "".join("1" if i in mask else "0" for i in range(n))
            assert max(map(len, run.split("0"))) <= 3
        if name == "qwen_legacy" and not 16 <= n <= 40 and n >= 12:
            run = "".join("1" if i in mask else "0" for i in range(n))
            assert max(map(len, run.split("0"))) <= 3


@pytest.mark.parametrize("name", ("zimage_p12", "qwen_legacy", "qwen21_55"))
def test_reject_outside_n1_80_or_noninteger(name):
    for invalid in (True, False, -1, 0, 81, 50.0, "50", None):
        with pytest.raises((ValueError, TypeError)):
            _factory(name, invalid)


def test_spread_profile_stays_at_n28_only():
    for n in (1, 8, 11, 12, 20, 27, 29, 40, 80):
        with pytest.raises(ValueError, match="only at N28"):
            SpectrumCacheConfig.for_zimage_base_variable(n, profile="p8_spread")


@torch.no_grad()
@pytest.mark.parametrize("n", (11, 12, 41, 80))
def test_qwen21_target_only_native_kv_state_and_guards(n):
    """Exercise native-KV state without real weights or nontrivial pipeline dependencies."""
    cfg = SpectrumCacheConfig.for_qwen_image21_variable(n)
    state = SpectrumQwen21State(cfg)
    module = torch.nn.Linear(1, 1).eval()
    native_cache = object()
    mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    scheduled = set(cfg.forecast_step_indices)

    for step in range(n):
        hidden = torch.full((1, 3, 4), 0.25 + step / max(1, n))
        kwargs = {
            "hidden_states": hidden,
            "target_token_mask": mask,
            "kv_cache_mode": "extract" if step == 0 else "cached",
            "layer_cache": native_cache,
        }
        for i in range(3):
            state.block(module, i, 3, lambda **k: k["hidden_states"] + 0.01, (), kwargs)
    report = state.summary()
    assert report["observed_steps"] == n
    assert report["guard_reason"] is None, report
    assert report["prediction_call_count"] == len(scheduled)
    assert report["block_real_calls"] + report["block_skipped_calls"] == n * 3


@torch.no_grad()
def test_qwen21_change_in_cache_identity_latches_real():
    cfg = SpectrumCacheConfig.for_qwen_image21_variable(12)
    state = SpectrumQwen21State(cfg)
    module = torch.nn.Linear(1, 1).eval()
    mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    for step in range(12):
        kwargs = dict(hidden_states=torch.ones((1, 3, 4)),target_token_mask=mask,
                      kv_cache_mode="extract" if step == 0 else "cached",
                      layer_cache=object() if step >= 1 else object())
        for i in range(3):
            state.block(module, i, 3, lambda **k: k["hidden_states"] + 0.01, (), kwargs)
    assert state.summary()["prediction_call_count"] == 0
    assert "cache identity changed" in state.summary()["guard_reason"]

@torch.no_grad()
@pytest.mark.parametrize("n", (11, 12, 41, 80))
def test_qwen21_native_kv_full_enable_path_at_new_n(n):
    """One full opt-in dispatch with 32 tiny blocks, native prefill and target-only cached phases."""
    import types
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from diffusers.hooks.hooks import HookRegistry

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states, **kwargs):
            self.calls += 1
            return hidden_states + 0.002

    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.register_to_config(
        patch_size=1, in_channels=64, out_channels=64, num_layers=32, causal_condition=True
    )
    model._cache_config = None
    model.transformer_blocks = torch.nn.ModuleList([TinyBlock() for _ in range(32)])

    def fake_forward(self, hidden_states, target_token_mask, kv_cache, kv_cache_mode, layer_cache, **kwargs):
        output = hidden_states
        for block in self.transformer_blocks:
            output = block(
                hidden_states=output,
                target_token_mask=target_token_mask,
                kv_cache_mode=kv_cache_mode,
                layer_cache=layer_cache,
            )
        return (output,)

    model.forward = types.MethodType(fake_forward, model)
    model.eval()
    config = SpectrumCacheConfig.for_qwen_image21_variable(n)
    model.enable_cache(config)
    assert model.is_cache_enabled
    native_cache = object()
    target_mask = torch.tensor([[False, True, True]], dtype=torch.bool)
    with torch.no_grad():
        for step in range(n):
            with model.cache_context("t2i", step_index=step, num_inference_steps=n):
                out = model(
                    hidden_states=torch.full((1, 3, 4), 0.25 + step / n),
                    target_token_mask=target_mask,
                    kv_cache=native_cache,
                    kv_cache_mode="extract" if step == 0 else "cached",
                    layer_cache=native_cache,
                )[0]
                assert torch.isfinite(out).all()
    hook = HookRegistry.check_if_exists_or_initialize(model).get_hook("spectrum_cache_denoiser")
    summaries = [state.summary() for state in hook.state_manager._state_cache.values()]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["guard_reason"] is None, summary
    assert summary["prediction_call_count"] == len(config.forecast_step_indices)
    assert summary["block_real_calls"] + summary["block_skipped_calls"] == 32 * n
    model.disable_cache()
    assert not model.is_cache_enabled


@torch.no_grad()
def test_qwen21_unreviewed_n_rejects_mutated_config_at_install():
    import types
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.register_to_config(patch_size=1, in_channels=64, out_channels=64, num_layers=32, causal_condition=True)
    model._cache_config = None
    model.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
    model.forward = types.MethodType(lambda self, **kwargs: (kwargs["hidden_states"],), model)
    cfg = SpectrumCacheConfig.for_qwen_image21_variable(41)
    with pytest.raises(ValueError, match="exact N1–80 research native-KV"):
        model.enable_cache(replace(cfg, blend_w=0.9))
    assert not model.is_cache_enabled
