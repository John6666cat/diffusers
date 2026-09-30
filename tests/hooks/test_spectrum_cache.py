# coding=utf-8
# Copyright 2026 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumForecaster, SpectrumSchedule


def test_spectrum_default_refresh_schedule():
    config = SpectrumCacheConfig()
    schedule = SpectrumSchedule(config)

    compute_steps = [step for step in range(config.num_inference_steps) if schedule.decide(step)]

    assert compute_steps == [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 25, 31, 38, 46]
    assert len(compute_steps) == 14


def test_spectrum_tail_actual_steps_preserves_default_and_forces_final_steps():
    default = SpectrumCacheConfig(num_inference_steps=24)
    default_schedule = SpectrumSchedule(default)
    default_compute = [step for step in range(24) if default_schedule.decide(step)]
    assert default_compute == [0, 1, 2, 3, 4, 6, 8, 11, 15, 20]

    guarded = SpectrumCacheConfig(num_inference_steps=24, tail_actual_steps=3)
    guarded_schedule = SpectrumSchedule(guarded)
    guarded_compute = [step for step in range(24) if guarded_schedule.decide(step)]
    assert guarded_compute == [0, 1, 2, 3, 4, 6, 8, 11, 15, 20, 21, 22, 23]


def test_spectrum_forecaster_preserves_shape_dtype_and_history_limit():
    config = SpectrumCacheConfig(degree=2, history_limit=3)
    forecaster = SpectrumForecaster(config)

    for step in range(5):
        feature = torch.full((1, 4, 8), float(step), dtype=torch.float32)
        forecaster.update(step, feature)

    assert forecaster.steps == [2, 3, 4]
    assert len(forecaster.features) == 3

    predicted = forecaster.predict(5)
    assert predicted.shape == (1, 4, 8)
    assert predicted.dtype == torch.float32
    assert torch.isfinite(predicted).all()


def test_spectrum_forecaster_rejects_shape_change():
    forecaster = SpectrumForecaster(SpectrumCacheConfig())
    forecaster.update(0, torch.zeros(1, 4, 8))

    with pytest.raises(ValueError, match="feature shape changed"):
        forecaster.update(1, torch.zeros(1, 5, 8))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_inference_steps": 0},
        {"warmup_steps": -1},
        {"window_size": 0},
        {"flex_window": -0.1},
        {"degree": -1},
        {"ridge_lambda": -0.1},
        {"blend_w": -0.1},
        {"blend_w": 1.1},
        {"history_limit": 0},
        {"coordinate_max": 0},
        {"tail_actual_steps": -1},
        {"num_inference_steps": 4, "tail_actual_steps": 5},
    ],
)
def test_spectrum_config_validation(kwargs):
    with pytest.raises(ValueError):
        SpectrumCacheConfig(**kwargs)


def test_spectrum_sdxl_profile_factories_and_schedules():
    standard = SpectrumCacheConfig.for_sdxl()
    conservative = SpectrumCacheConfig.for_sdxl(conservative=True)
    pag = SpectrumCacheConfig.for_sdxl(pag=True)

    assert (standard.degree, standard.ridge_lambda, standard.blend_w) == (4, 0.1, 0.60)
    assert (standard.warmup_steps, standard.window_size, standard.flex_window, standard.tail_actual_steps) == (
        6,
        2.0,
        0.75,
        3,
    )
    standard_schedule = SpectrumSchedule(standard)
    conservative_schedule = SpectrumSchedule(conservative)
    pag_schedule = SpectrumSchedule(pag)

    assert [step for step in range(24) if standard_schedule.decide(step)] == [
        0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 22, 23
    ]
    assert [step for step in range(24) if conservative_schedule.decide(step)] == [
        0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 22, 23
    ]
    assert [step for step in range(24) if pag_schedule.decide(step)] == [
        0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 18, 21, 22, 23
    ]

    with pytest.raises(ValueError, match="cannot be combined"):
        SpectrumCacheConfig.for_sdxl(conservative=True, pag=True)


def test_spectrum_sd15_profile_factory_and_schedule():
    config = SpectrumCacheConfig.for_sd15()

    assert (config.degree, config.ridge_lambda, config.blend_w) == (4, 0.05, 0.55)
    assert (config.warmup_steps, config.window_size, config.flex_window, config.tail_actual_steps) == (
        6,
        2.0,
        0.75,
        3,
    )

    schedule = SpectrumSchedule(config)
    assert [step for step in range(20) if schedule.decide(step)] == [
        0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 17, 18, 19
    ]

def test_spectrum_hunyuan_video15_profile_factory():
    config = SpectrumCacheConfig.for_hunyuan_video15()
    assert (config.num_inference_steps, config.history_limit) == (50, 8)
    assert (config.degree, config.ridge_lambda, config.blend_w) == (4, 0.1, 0.5)
    assert config.forecast_step_indices == (20, 22, 24, 27, 32, 34, 36, 38, 40, 42)
@pytest.mark.parametrize(
    ("steps", "expected_compute"),
    [
        (16, [0, 1, 2, 3, 4, 5, 7, 9, 12, 13, 14, 15]),
        (20, [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 17, 18, 19]),
        (24, [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 18, 21, 22, 23]),
        (28, [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 18, 21, 25, 26, 27]),
        (32, [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 18, 21, 23, 26, 29, 30, 31]),
        (36, [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 18, 21, 23, 26, 29, 32, 33, 34, 35]),
        (40, [0, 1, 2, 3, 4, 5, 7, 9, 12, 16, 18, 21, 23, 26, 29, 32, 34, 37, 38, 39]),
    ],
)
def test_spectrum_sdxl_variable_profile_reproduces_gate_e_mc3_schedule(steps, expected_compute):
    legacy = SpectrumCacheConfig.for_sdxl(num_inference_steps=steps)
    config = SpectrumCacheConfig.for_sdxl_variable(num_inference_steps=steps)

    assert legacy.coordinate_policy == "legacy_fixed_max"
    assert legacy.max_consecutive_forecast_steps is None
    assert config.coordinate_policy == "runtime_index_normalized"
    assert config.max_consecutive_forecast_steps == 3
    assert config.forecast_step_indices is None

    schedule = SpectrumSchedule(config)
    compute = [step for step in range(steps) if schedule.decide(step)]
    assert compute == expected_compute

    forecast = [step for step in range(steps) if step not in set(compute)]
    run = 0
    for step in range(steps):
        if step in forecast:
            run += 1
            assert run <= 3
        else:
            run = 0


def test_spectrum_max_consecutive_forecast_guard_validation():
    with pytest.raises(ValueError, match="integer >= 1"):
        SpectrumCacheConfig(max_consecutive_forecast_steps=0)

    with pytest.raises(ValueError, match="cannot be combined"):
        SpectrumCacheConfig(
            num_inference_steps=4,
            forecast_step_indices=(1,),
            max_consecutive_forecast_steps=3,
        )


@pytest.mark.parametrize(
    ("steps", "expected_forecast", "expected_predictor", "expected_body_calls"),
    [
        (20, (8, 10, 12, 13, 15, 16), (100, 4, 0.55), 14),
        (24, (6, 8, 10, 11, 13, 14, 15, 17, 18, 20), (100, 4, 0.55), 14),
        (28, (6, 8, 10, 11, 13, 15, 17, 19, 20, 22, 23, 24), (100, 4, 0.55), 16),
        (32, (6, 8, 10, 11, 13, 14, 15, 17, 19, 20, 22, 23, 24, 25, 27, 28), (100, 4, 0.55), 16),
        (36, (6, 8, 10, 11, 13, 15, 17, 18, 20, 22, 23, 25, 27, 29, 30, 31), (100, 4, 0.55), 20),
        (40, (6, 8, 10, 11, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 31, 33, 34, 35, 36), (4, 3, 0.65), 18),
    ],
)
def test_spectrum_sd15_variable_euler_profile_reproduces_locked_operating_table(
    steps, expected_forecast, expected_predictor, expected_body_calls
):
    config = SpectrumCacheConfig.for_sd15_variable(num_inference_steps=steps)

    assert config.coordinate_policy == "legacy_fixed_max"
    assert config.coordinate_max == 50.0
    assert config.tail_actual_steps == 3
    assert config.max_consecutive_forecast_steps is None
    assert config.forecast_step_indices == expected_forecast
    assert (config.history_limit, config.degree, config.blend_w) == expected_predictor
    assert config.ridge_lambda == 0.05

    schedule = SpectrumSchedule(config)
    compute = [step for step in range(steps) if schedule.decide(step)]
    forecast = [step for step in range(steps) if step not in set(compute)]

    assert forecast == list(expected_forecast)
    assert len(compute) == expected_body_calls
    assert compute[:6] == list(range(6))
    assert compute[-3:] == list(range(steps - 3, steps))


@pytest.mark.parametrize("steps", [16, 18, 21, 30, 44])
def test_spectrum_sd15_variable_euler_profile_rejects_unqualified_step_counts(steps):
    with pytest.raises(ValueError, match="qualified only for Euler runs"):
        SpectrumCacheConfig.for_sd15_variable(num_inference_steps=steps)
