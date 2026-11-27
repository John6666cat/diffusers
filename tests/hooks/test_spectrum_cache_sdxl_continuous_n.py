"""SDXL continuous-integer-N research gate; synthetic CPU only, no quality claim."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule
from diffusers.pipelines.stable_diffusion_xl._spectrum_euler_gate import (
    _EULER_CONFIG,
    _spectrum_sdxl_euler_lifecycle,
    _spectrum_sdxl_euler_maybe_bypass,
    _spectrum_sdxl_euler_reasons,
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


def fake_pipe(n, config):
    pipe = type("StableDiffusionXLPipeline", (), {})()
    pipe.unet = FakeUNet(config)
    pipe.scheduler = EulerDiscreteScheduler()
    pipe.scheduler.set_timesteps(n)
    pipe._num_timesteps = n
    pipe.text_encoder = SimpleNamespace(peft_config=None)
    pipe.text_encoder_2 = SimpleNamespace(peft_config=None)
    return pipe


def decisions(config):
    schedule = SpectrumSchedule(config)
    return tuple(schedule.decide(i) for i in range(config.num_inference_steps))


@pytest.mark.parametrize("n", range(1, 81))
def test_every_integer_n_has_safe_schedule(n):
    config = SpectrumCacheConfig.for_sdxl_continuous_n(n)
    actual = decisions(config)
    assert config.num_inference_steps == n
    assert config.sdxl_euler_post_grid_strict
    assert config.sdxl_euler_continuous_n_candidate
    assert config.coordinate_policy == "runtime_index_normalized"
    assert all(actual[:min(n, 6)])
    assert all(actual[-min(n, 3):])
    assert all(any(actual[i : i + 4]) for i in range(n - 3))
    if n <= 9 or n == 20:
        assert config.forecast_step_indices == ()
        assert all(actual)
    else:
        assert any(not x for x in actual)
    assert actual == decisions(config)  # new schedule instance is deterministic


@pytest.mark.parametrize("n", [24, 32])
def test_legacy_24_32_profiles_unchanged(n):
    previous = SpectrumCacheConfig.for_sdxl_variable(n, strict_post_grid=True)
    candidate = SpectrumCacheConfig.for_sdxl_continuous_n(n)
    assert previous.sdxl_euler_continuous_n_candidate is False
    assert replace(candidate, sdxl_euler_continuous_n_candidate=False) == previous
    assert decisions(previous) == decisions(candidate)


@pytest.mark.parametrize("invalid", [0, -1, True, False, 24.0, None, "24"])
def test_continuous_factory_rejects_out_of_scope(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        SpectrumCacheConfig.for_sdxl_continuous_n(invalid)


@pytest.mark.parametrize("n", [1, 4, 8, 9, 10, 12, 16, 17, 19, 20, 21, 23, 24, 25, 31, 32, 39, 40, 48, 60, 96, 128])
@torch.no_grad()
def test_runtime_gate_admits_exact_new_candidate(n):
    config = SpectrumCacheConfig.for_sdxl_continuous_n(n)
    pipe = fake_pipe(n, config)
    assert _spectrum_sdxl_euler_reasons(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    ) == ()


@pytest.mark.parametrize("n", [24, 32])
@torch.no_grad()
def test_legacy_strict_gate_24_32_remains_accepted(n):
    pipe = fake_pipe(n, SpectrumCacheConfig.for_sdxl_variable(n, strict_post_grid=True))
    assert _spectrum_sdxl_euler_reasons(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    ) == ()


@pytest.mark.parametrize("n", [16, 20, 40])
@torch.no_grad()
def test_old_strict_gate_remains_closed_outside_24_32(n):
    pipe = fake_pipe(n, SpectrumCacheConfig.for_sdxl_variable(n, strict_post_grid=True))
    reasons = _spectrum_sdxl_euler_reasons(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    )
    assert "unqualified_step_count" in reasons


@torch.no_grad()
def test_profile_tampering_fails_closed_and_resets():
    pipe = fake_pipe(27, replace(SpectrumCacheConfig.for_sdxl_continuous_n(27), degree=3))
    reasons = _spectrum_sdxl_euler_maybe_bypass(
        pipe, pipe.scheduler.timesteps, 27, requested_steps=27, plain_request=True
    )
    assert "unexpected_spectrum_profile" in reasons
    assert pipe.unet.disabled == pipe.unet.reset == 1


@pytest.mark.parametrize("mutate", ["sigma", "scheduler", "effective", "plain", "offload", "nonpositive_n", "mismatched_n"])
@torch.no_grad()
def test_scheduler_and_request_mismatches_fail_closed(mutate):
    n = 27
    pipe = fake_pipe(n, SpectrumCacheConfig.for_sdxl_continuous_n(n))
    requested = n
    effective = n
    plain = True
    if mutate == "sigma":
        pipe.scheduler.sigmas[4] = 3.0
    elif mutate == "scheduler":
        pipe.scheduler.config["steps_offset"] = 0
    elif mutate == "effective":
        effective = 26
    elif mutate == "plain":
        plain = False
    elif mutate == "offload":
        pipe.unet._hf_hook = object()
    elif mutate == "nonpositive_n":
        requested = 0
    elif mutate == "mismatched_n":
        requested = 48
    reasons = _spectrum_sdxl_euler_maybe_bypass(
        pipe, pipe.scheduler.timesteps, effective, requested_steps=requested, plain_request=plain
    )
    assert reasons
    assert pipe.unet.disabled == pipe.unet.reset == 1


def test_lifecycle_preserves_original_exception_and_cleans_up():
    pipe = fake_pipe(27, SpectrumCacheConfig.for_sdxl_continuous_n(27))

    @_spectrum_sdxl_euler_lifecycle
    def erroring_call(self):
        raise RuntimeError("primary failure")

    with pytest.raises(RuntimeError, match="primary failure"):
        erroring_call(pipe)
    assert pipe.unet.disabled == 1
    assert pipe.unet.reset == 2


@pytest.mark.parametrize("n", [96, 128, 192, 256])
def test_high_n_schedule_is_generated_without_fixed_upper_limit(n):
    config = SpectrumCacheConfig.for_sdxl_continuous_n(n)
    mask = decisions(config)
    assert len(mask) == n
    assert all(mask[:6]) and all(mask[-3:])
    assert all(any(mask[i:i+4]) for i in range(n - 3))
    assert any(not actual for actual in mask)


@pytest.mark.parametrize("n", [1, 5, 9, 20])
def test_low_n_identity_uses_no_forecasts(n):
    config = SpectrumCacheConfig.for_sdxl_continuous_n(n)
    assert config.forecast_step_indices == ()
    assert decisions(config) == (True,) * n


@pytest.mark.parametrize("n", [10, 12, 14, 48, 60, 96])
@torch.no_grad()
def test_old_strict_gate_does_not_implicitly_accept_new_n(n):
    pipe = fake_pipe(n, SpectrumCacheConfig.for_sdxl_variable(n, strict_post_grid=True))
    reasons = _spectrum_sdxl_euler_reasons(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    )
    assert "unqualified_step_count" in reasons


@torch.no_grad()
def test_switching_n_requires_exact_new_factory_and_revisiting_is_deterministic():
    original = None
    for n in (12, 24, 48, 60, 12):
        config = SpectrumCacheConfig.for_sdxl_continuous_n(n)
        pipe = fake_pipe(n, config)
        assert _spectrum_sdxl_euler_reasons(
            pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
        ) == ()
        mask = decisions(config)
        if n == 12:
            if original is None:
                original = mask
            else:
                assert original == mask


@pytest.mark.parametrize(
    "n, expected, previous",
    [
        (
            30,
            (
                11, 13, 14, 15, 17, 19, 20, 22, 24, 25,
            ),
            (
                6, 8, 10, 11, 13, 14, 15, 17, 19, 20, 22, 24, 25,
            ),
        ),
        (
            50,
            (
                14, 15, 17, 19, 20, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37,
                38, 40, 41, 42, 44, 45, 46,
            ),
            (
                6, 8, 10, 11, 13, 14, 15, 17, 19, 20, 22, 24, 25, 27, 28, 30,
                31, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46,
            ),
        ),
        (
            64,
            (
                15, 17, 19, 20, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 38,
                40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60,
            ),
            (
                6, 8, 10, 11, 13, 14, 15, 17, 19, 20, 22, 24, 25, 27, 28, 30,
                31, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52,
                53, 54, 56, 57, 59, 60,
            ),
        ),
        (
            80,
            (
                17, 19, 20, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 38, 40,
                41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 62,
                63, 65, 67, 68, 70, 72, 73, 75, 76,
            ),
            (
                6, 8, 10, 11, 13, 14, 15, 17, 19, 20, 22, 24, 25, 27, 28, 30,
                31, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52,
                53, 54, 56, 57, 58, 60, 62, 63, 65, 67, 68, 70, 72, 73, 75, 76,
            ),
        ),
    ],
)
@torch.no_grad()
def test_early_protect_exact_gate(n, expected, previous):
    cfg = SpectrumCacheConfig.for_sdxl_continuous_n(n)
    assert cfg.forecast_step_indices == expected
    assert cfg.max_consecutive_forecast_steps is None
    assert tuple(i for i, actual in enumerate(decisions(cfg)) if not actual) == expected
    pipe = fake_pipe(n, cfg)
    assert _spectrum_sdxl_euler_reasons(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    ) == ()
    for mask in (previous, expected[1:]):
        pipe = fake_pipe(n, replace(cfg, forecast_step_indices=mask))
        assert _spectrum_sdxl_euler_maybe_bypass(
            pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
        )
        assert pipe.unet.disabled == pipe.unet.reset == 1
    old = replace(SpectrumCacheConfig.for_sdxl_variable(n, strict_post_grid=True),
                  sdxl_euler_continuous_n_candidate=True)
    pipe = fake_pipe(n, old)
    assert _spectrum_sdxl_euler_maybe_bypass(
        pipe, pipe.scheduler.timesteps, n, requested_steps=n, plain_request=True
    )
    assert pipe.unet.disabled == pipe.unet.reset == 1
