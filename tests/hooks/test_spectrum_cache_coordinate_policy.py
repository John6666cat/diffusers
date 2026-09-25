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

from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumForecaster


def test_coordinate_policy_default_is_legacy_fixed_max():
    config = SpectrumCacheConfig()
    assert config.coordinate_policy == "legacy_fixed_max"


def test_qualified_factories_remain_explicitly_legacy_pinned():
    configs = [
        SpectrumCacheConfig.for_sdxl(),
        SpectrumCacheConfig.for_sd15(),
        SpectrumCacheConfig.for_hunyuan_video15(),
        SpectrumCacheConfig.for_cogvideox5b_t2v(),
    ]
    assert all(config.coordinate_policy == "legacy_fixed_max" for config in configs)


def test_legacy_fixed_max_preserves_historical_mapping():
    forecaster = SpectrumForecaster(
        SpectrumCacheConfig(coordinate_policy="legacy_fixed_max", coordinate_max=50.0)
    )
    steps = torch.tensor([0.0, 25.0, 50.0])
    assert torch.equal(forecaster._tau(steps), torch.tensor([-1.0, 0.0, 1.0]))


def test_runtime_index_normalized_maps_runtime_endpoints():
    forecaster = SpectrumForecaster(
        SpectrumCacheConfig(
            num_inference_steps=5,
            coordinate_policy="runtime_index_normalized",
            coordinate_max=999.0,
        )
    )
    steps = torch.tensor([0.0, 2.0, 4.0])
    assert torch.equal(forecaster._tau(steps), torch.tensor([-1.0, 0.0, 1.0]))


def test_runtime_index_normalized_single_step_is_zero():
    forecaster = SpectrumForecaster(
        SpectrumCacheConfig(num_inference_steps=1, coordinate_policy="runtime_index_normalized")
    )
    assert torch.equal(forecaster._tau(torch.tensor([0.0])), torch.tensor([0.0]))


def test_coordinate_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="coordinate_policy"):
        SpectrumCacheConfig(coordinate_policy="scheduler_magic")


def test_explicit_legacy_prediction_matches_default_legacy_prediction():
    common = {
        "num_inference_steps": 24,
        "degree": 4,
        "ridge_lambda": 0.1,
        "blend_w": 0.5,
        "history_limit": 6,
        "coordinate_max": 50.0,
    }
    default = SpectrumForecaster(SpectrumCacheConfig(**common))
    explicit = SpectrumForecaster(
        SpectrumCacheConfig(**common, coordinate_policy="legacy_fixed_max")
    )
    generator = torch.Generator().manual_seed(29)
    for step in (0, 1, 2, 4, 7, 10):
        feature = torch.randn((2, 16, 32), generator=generator)
        default.update(step, feature)
        explicit.update(step, feature)

    assert torch.equal(default.predict(12), explicit.predict(12))
