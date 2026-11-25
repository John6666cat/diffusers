import math
from dataclasses import asdict, replace

import pytest

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumSchedule


def forecast_steps(config):
    schedule = SpectrumSchedule(config)
    return tuple(i for i in range(config.num_inference_steps) if not schedule.decide(i))


@pytest.mark.parametrize("checkpoint", ["base", "turbo"])
@pytest.mark.parametrize("n", range(1, 81))
def test_anima_base_turbo_variable_schedule(checkpoint, n):
    config = SpectrumCacheConfig.for_anima_base_turbo_variable(n, checkpoint=checkpoint)
    steps = forecast_steps(config)
    assert config.num_inference_steps == n
    assert config.coordinate_policy == "runtime_horizon_normalized"
    assert config.coordinate_max == 50.0
    assert config.denoiser_call_topology == "auto"
    assert config.predictor_backend == "dense"
    if n <= (8 if checkpoint == "base" else 5):
        assert config.forecast_step_indices == () and steps == ()
        assert config.warmup_steps == n and config.tail_actual_steps == 0
    else:
        assert steps and steps == tuple(sorted(set(steps)))
        assert config.warmup_steps >= config.degree + 1
        assert config.history_limit >= config.degree + 1
        assert steps[0] >= config.warmup_steps and steps[-1] < n - config.tail_actual_steps
    if n >= 30 and checkpoint == "base":
        ordinary = SpectrumCacheConfig.for_anima_variable(n)
        assert asdict(config) == asdict(replace(ordinary, history_limit=6))
        assert steps == forecast_steps(ordinary)
    elif n >= 20:
        warmup = max(5, math.ceil(5 * n / 19))
        tail = max(2, math.ceil(2 * n / 19))
        stride = 2 if checkpoint == "base" else 3
        expected = tuple(i for i in range(warmup, n - tail) if (i - warmup) % stride != stride - 1)
        assert steps == expected
        assert config.history_limit == 6


@pytest.mark.parametrize(
    "checkpoint,n,expected", [
        ("base", 9, (6,)),
        ("base", 10, (6,)),
        ("base", 11, (7,)),
        ("base", 12, (5, 7, 9)),
        ("base", 13, (5, 7, 8, 10)),
        ("base", 14, (5, 7, 9, 11)),
        ("base", 15, (5, 8, 10, 12)),
        ("base", 16, (5, 7, 9, 11, 13)),
        ("base", 17, (5, 7, 9, 11, 14)),
        ("base", 18, (5, 7, 9, 11, 13, 15)),
        ("base", 19, (5, 8, 10, 12, 14, 16)),
        ("turbo", 6, (3,)),
        ("turbo", 7, (3, 4)),
        ("turbo", 8, (3, 5)),
        ("turbo", 9, (4, 6)),
        ("turbo", 10, (5, 7)),
        ("turbo", 11, (5, 7)),
        ("turbo", 12, (5, 7, 9)),
        ("turbo", 13, (5, 8, 10)),
        ("turbo", 14, (5, 7, 9, 11)),
        ("turbo", 15, (5, 6, 8, 9, 11, 12)),
        ("turbo", 16, (5, 7, 9, 11, 13)),
        ("turbo", 17, (5, 6, 7, 9, 10, 11, 13, 14)),
        ("turbo", 18, (5, 6, 7, 9, 10, 11, 13, 14, 15)),
        ("turbo", 19, (5, 7, 8, 10, 11, 12, 14, 15, 16)),
    ],
)
def test_anima_low_step_exact_masks(checkpoint, n, expected):
    config = SpectrumCacheConfig.for_anima_base_turbo_variable(n, checkpoint=checkpoint)
    assert forecast_steps(config) == expected
    short_history = checkpoint == "turbo" and n <= 9
    assert (config.degree, config.history_limit, config.warmup_steps, config.tail_actual_steps) == (
        (2, 3, 3, 2) if short_history else (4, 6, 5, 2)
    )


@pytest.mark.parametrize("n", [0, -1, 81, 1.5, 20.0, True, "30", None])
def test_anima_invalid_n(n):
    with pytest.raises(ValueError):
        SpectrumCacheConfig.for_anima_base_turbo_variable(n, checkpoint="base")


@pytest.mark.parametrize("checkpoint", [None, "", "other", "BASE", 1])
def test_anima_invalid_checkpoint(checkpoint):
    with pytest.raises(ValueError):
        SpectrumCacheConfig.for_anima_base_turbo_variable(30, checkpoint=checkpoint)


@pytest.mark.parametrize("n", [20, 30, 40, 80, 81, 120])
def test_anima_ordinary_factory_unchanged(n):
    config = SpectrumCacheConfig.for_anima_variable(n)
    assert config.history_limit == 100
    assert config.forecast_step_indices is None
    assert config.warmup_steps == max(8, math.ceil(8 * n / 30))
    assert config.tail_actual_steps == max(5, math.ceil(5 * n / 30))
