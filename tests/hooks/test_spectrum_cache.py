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
    ],
)
def test_spectrum_config_validation(kwargs):
    with pytest.raises(ValueError):
        SpectrumCacheConfig(**kwargs)
