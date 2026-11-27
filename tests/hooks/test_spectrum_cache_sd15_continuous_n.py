"""SD1.5 ordinary Euler continuous-N: isolated CPU contract checks, not GPU quality."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule
from diffusers.pipelines.stable_diffusion._spectrum_euler_gate import (
    _EULER_CONFIG,
    _spectrum_sd15_euler_lifecycle,
    _spectrum_sd15_euler_maybe_bypass,
    _spectrum_sd15_euler_reasons,
)


class EulerDiscreteScheduler:
    __module__ = "diffusers.schedulers.scheduling_euler_discrete"

    def __init__(self, config=None):
        self.config = dict(_EULER_CONFIG if config is None else config)
        self.timesteps = torch.tensor([])
        self.sigmas = torch.tensor([])

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.float32, device=device)
        self.sigmas = torch.linspace(1.0, 0.0, n + 1, device=device)


class FakeUNet:
    def __init__(self, config):
        self._cache_config = config
        self.config = SimpleNamespace(cross_attention_dim=768, sample_size=64, in_channels=4,
                                      out_channels=4, time_cond_proj_dim=None)
        self.peft_config = None
        self.disabled = 0
        self.reset = 0
        self._diffusers_hook = None

    def parameters(self):
        yield SimpleNamespace(dtype=torch.float16, device=SimpleNamespace(type="cuda"))

    def modules(self):
        return [self]

    def disable_cache(self):
        self.disabled += 1
        self._cache_config = None

    def _reset_stateful_cache(self):
        self.reset += 1


def make_pipe(n, config):
    pipe = type("StableDiffusionPipeline", (), {})()
    pipe.unet = FakeUNet(config)
    pipe.scheduler = EulerDiscreteScheduler()
    pipe.scheduler.set_timesteps(n)
    pipe._num_timesteps = n
    pipe.text_encoder = SimpleNamespace(peft_config=None, config=SimpleNamespace(hidden_size=768))
    return pipe


def decisions(config):
    s = SpectrumSchedule(config)
    return tuple(s.decide(i) for i in range(config.num_inference_steps))


def max_run(mask):
    count = result = 0
    for actual in mask:
        count = 0 if actual else count + 1
        result = max(count, result)
    return result


@pytest.mark.parametrize("n", range(1, 161))
def test_any_integer_n_has_safe_schedule(n):
    config = SpectrumCacheConfig.for_sd15_continuous_n(n)
    mask = decisions(config)
    assert len(mask) == n
    assert config.num_inference_steps == n
    assert config.sd15_euler_continuous_n_candidate
    assert all(mask[:min(6, n)])
    assert all(mask[-min(3, n):])
    assert max_run(mask) <= 4
    assert mask == decisions(config)
    if n < 10:
        assert mask == (True,) * n
    if n not in (20, 24, 28, 32, 36, 40):
        assert config.coordinate_policy == "runtime_index_normalized"
    else:
        assert config.coordinate_policy == "legacy_fixed_max"


@pytest.mark.parametrize("n", [20, 24, 28, 32, 36, 40])
def test_historical_discrete_configs_bit_identical_except_opt_in(n):
    historical = SpectrumCacheConfig.for_sd15_variable(n)
    candidate = SpectrumCacheConfig.for_sd15_continuous_n(n)
    assert replace(candidate, sd15_euler_continuous_n_candidate=False) == historical
    assert decisions(candidate) == decisions(historical)
    assert historical.sd15_euler_continuous_n_candidate is False


@pytest.mark.parametrize("invalid", [0, -1, True, False, 20.0, None, "20"])
def test_invalid_n_rejected(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        SpectrumCacheConfig.for_sd15_continuous_n(invalid)


@pytest.mark.parametrize("n", [1, 4, 8, 9, 10, 12, 16, 17, 19, 20, 21, 23, 24, 25, 31, 32, 36, 39, 40, 41, 48, 60, 96, 128, 192, 256])
@torch.no_grad()
def test_runtime_gate_accepts_exact_candidate(n):
    p = make_pipe(n, SpectrumCacheConfig.for_sd15_continuous_n(n))
    assert _spectrum_sd15_euler_reasons(
        p, p.scheduler.timesteps, n, requested_steps=n, plain_request=True
    ) == ()


@pytest.mark.parametrize("n", [12, 20, 24, 40, 60])
@torch.no_grad()
def test_old_profile_is_not_silently_admitted(n):
    if n in (20, 24, 40):
        config = SpectrumCacheConfig.for_sd15_variable(n)
    else:
        config = SpectrumCacheConfig.for_sd15(n)
    p = make_pipe(n, config)
    assert "not_sd15_continuous_config" in _spectrum_sd15_euler_reasons(
        p, p.scheduler.timesteps, n, requested_steps=n, plain_request=True
    )
    assert _spectrum_sd15_euler_maybe_bypass(
        p, p.scheduler.timesteps, n, requested_steps=n, plain_request=True
    ) == ()
    assert p.unet.disabled == p.unet.reset == 0


@pytest.mark.parametrize("mutate", [
    "sigma", "config", "wrong_n", "effective", "plain", "type", "offload", "adapters",
    "dtype", "autograd", "bad_mask", "bad_degree", "bad_steps", "bad_request", "architecture",
])
def test_fail_closed_before_first_unet_call(mutate):
    n = 27
    p = make_pipe(n, SpectrumCacheConfig.for_sd15_continuous_n(n))
    effective = requested = n
    plain = True
    if mutate == "sigma":
        p.scheduler.sigmas[3] = 4
    elif mutate == "config":
        p.scheduler.config["steps_offset"] = 0
    elif mutate == "wrong_n":
        requested = 28
    elif mutate == "effective":
        effective = 26
    elif mutate == "plain":
        plain = False
    elif mutate == "type":
        p.scheduler = type("FakeScheduler", (EulerDiscreteScheduler,), {})()
        p.scheduler.set_timesteps(n)
    elif mutate == "offload":
        p.unet._hf_hook = object()
    elif mutate == "adapters":
        p.text_encoder.peft_config = {"a": 1}
    elif mutate == "architecture":
        p.unet.config.cross_attention_dim = 1024
    elif mutate == "dtype":
        p.unet.parameters = lambda: iter([SimpleNamespace(dtype=torch.float32, device=SimpleNamespace(type="cpu"))])
    elif mutate == "autograd":
        pass
    elif mutate == "bad_mask":
        p.unet._cache_config = replace(p.unet._cache_config, forecast_step_indices=(7,))
    elif mutate == "bad_degree":
        p.unet._cache_config = replace(p.unet._cache_config, degree=2)
    elif mutate == "bad_steps":
        p._num_timesteps = n-1
    elif mutate == "bad_request":
        requested = 0
    if mutate != "autograd":
        with torch.no_grad():
            reasons = _spectrum_sd15_euler_maybe_bypass(
                p, p.scheduler.timesteps, effective, requested_steps=requested, plain_request=plain
            )
    else:
        with torch.enable_grad():
            reasons = _spectrum_sd15_euler_maybe_bypass(
                p, p.scheduler.timesteps, effective, requested_steps=requested, plain_request=plain
            )
    assert reasons
    assert p.unet.disabled == p.unet.reset == 1


def test_lifecycle_cleans_up_on_success_and_exception():
    for fails in (False, True):
        p = make_pipe(27, SpectrumCacheConfig.for_sd15_continuous_n(27))
        @_spectrum_sd15_euler_lifecycle
        def call(self):
            if fails:
                raise RuntimeError("original failure")
            return "OK"
        if fails:
            with pytest.raises(RuntimeError, match="original failure"):
                call(p)
        else:
            assert call(p) == "OK"
        assert p.unet.disabled == 1
        assert p.unet.reset == 2


@pytest.mark.parametrize("n", [11, 12, 16, 19, 21, 25, 39, 41, 48, 60, 90, 128, 192, 256])
def test_no_extra_forecasts_are_introduced_by_guard(n):
    cfg = SpectrumCacheConfig.for_sd15_continuous_n(n)
    anchors = (20, 24, 28, 32, 36, 40)
    anchor_n = min(anchors, key=lambda a:(abs(a-n), a))
    anchor = SpectrumCacheConfig.for_sd15_variable(anchor_n)
    from diffusers.hooks.spectrum_cache import _spectrum_retime_qualified_mask
    # Non-anchor candidate may PROMOTE real calls, never add speculative calls.
    projected = set(_spectrum_retime_qualified_mask(anchor.forecast_step_indices, anchor_n, n))
    assert set(cfg.forecast_step_indices).issubset(projected)


@pytest.mark.parametrize("n", [1, 6, 8, 9, 20, 24, 60, 128])
def test_repeatable_schedules(n):
    a = SpectrumCacheConfig.for_sd15_continuous_n(n)
    b = SpectrumCacheConfig.for_sd15_continuous_n(n)
    assert a == b
    assert decisions(a) == decisions(b)


def test_n14_selected_mask_and_exact_gate():
    cfg = SpectrumCacheConfig.for_sd15_continuous_n(14)
    assert cfg.forecast_step_indices == (7, 9, 10)
    assert tuple(i for i, actual in enumerate(decisions(cfg)) if not actual) == (7, 9, 10)
    with torch.no_grad():
        p = make_pipe(14, cfg)
        assert _spectrum_sd15_euler_reasons(
            p, p.scheduler.timesteps, 14, requested_steps=14, plain_request=True
        ) == ()
        for mask in ((7, 8, 9, 10), (7, 10)):
            p = make_pipe(14, replace(cfg, forecast_step_indices=mask))
            assert _spectrum_sd15_euler_maybe_bypass(
                p, p.scheduler.timesteps, 14, requested_steps=14, plain_request=True
            )
            assert p.unet.disabled == p.unet.reset == 1


@pytest.mark.parametrize(
    "n, expected, previous",
    [
        (
            30,
            (
                12, 14, 16, 18, 20, 21, 24, 25, 26,
            ),
            (
                6, 9, 11, 12, 14, 16, 18, 20, 21, 24, 25, 26,
            ),
        ),
        (
            50,
            (
                14, 16, 19, 21, 22, 24, 25, 27, 28, 30, 31, 34, 35, 36, 39, 41,
                42, 44, 45,
            ),
            (
                7, 8, 10, 12, 13, 14, 16, 19, 21, 22, 24, 25, 27, 28, 30, 31,
                34, 35, 36, 39, 41, 42, 44, 45,
            ),
        ),
        (
            64,
            (
                24, 25, 27, 28, 29, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44,
                46, 47, 50, 53, 54, 56, 57, 58,
            ),
            (
                10, 13, 16, 17, 18, 21, 24, 25, 27, 28, 29, 31, 32, 33, 35, 36,
                37, 39, 40, 41, 43, 44, 46, 47, 50, 53, 54, 56, 57, 58,
            ),
        ),
        (
            80,
            (
                26, 27, 30, 31, 34, 35, 36, 38, 39, 40, 41, 44, 45, 46, 48, 49,
                50, 51, 54, 55, 57, 58, 59, 62, 63, 66, 67, 68, 70, 71, 72, 73,
            ),
            (
                12, 13, 16, 17, 20, 21, 22, 23, 26, 27, 30, 31, 34, 35, 36, 38,
                39, 40, 41, 44, 45, 46, 48, 49, 50, 51, 54, 55, 57, 58, 59, 62,
                63, 66, 67, 68, 70, 71, 72, 73,
            ),
        ),
    ],
)
@torch.no_grad()
def test_early_protect_exact_gate(n, expected, previous):
    cfg = SpectrumCacheConfig.for_sd15_continuous_n(n)
    assert cfg.forecast_step_indices == expected
    assert cfg.max_consecutive_forecast_steps is None
    assert tuple(i for i, actual in enumerate(decisions(cfg)) if not actual) == expected
    pipe = make_pipe(n, cfg)
    assert _spectrum_sd15_euler_reasons(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    ) == ()
    for mask in (previous, expected[1:]):
        pipe = make_pipe(n, replace(cfg, forecast_step_indices=mask))
        assert _spectrum_sd15_euler_maybe_bypass(
            pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
        )
        assert pipe.unet.disabled == pipe.unet.reset == 1
