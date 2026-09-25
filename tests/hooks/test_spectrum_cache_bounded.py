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


def _feed_pair(dense, bounded):
    generator = torch.Generator().manual_seed(7)
    for step in range(5):
        feature = torch.randn((2, 32, 64), generator=generator, dtype=torch.float32)
        dense.update(step, feature)
        bounded.update(step, feature)


def test_spectrum_bounded_backend_matches_dense_float32_and_bounds_retained_coefficients():
    common = {"degree": 4, "ridge_lambda": 0.1, "blend_w": 0.5, "history_limit": 5}
    dense = SpectrumForecaster(SpectrumCacheConfig(**common, predictor_backend="dense"))
    bounded = SpectrumForecaster(
        SpectrumCacheConfig(
            **common,
            predictor_backend="bounded",
            predictor_cache_bytes=4096,
            predictor_chunk_size=257,
        )
    )
    _feed_pair(dense, bounded)

    dense_predicted = dense.predict(7)
    bounded_predicted = bounded.predict(7)

    assert torch.allclose(bounded_predicted, dense_predicted, atol=1e-5, rtol=1e-5)
    assert bounded._coef is None
    assert bounded._coef_slab is not None
    assert bounded._coef_slab.numel() * bounded._coef_slab.element_size() <= 4096
    assert bounded.predictor_state_bytes() < dense.predictor_state_bytes()


def test_spectrum_bounded_backend_zero_cache_behaves_like_chunked():
    common = {
        "degree": 2,
        "ridge_lambda": 0.1,
        "blend_w": 0.5,
        "history_limit": 4,
        "predictor_chunk_size": 1024,
    }
    bounded = SpectrumForecaster(
        SpectrumCacheConfig(**common, predictor_backend="bounded", predictor_cache_bytes=0)
    )
    chunked = SpectrumForecaster(SpectrumCacheConfig(**common, predictor_backend="chunked"))
    generator = torch.Generator().manual_seed(11)
    for step in range(4):
        feature = torch.randn((1, 16, 32), generator=generator, dtype=torch.float32)
        bounded.update(step, feature)
        chunked.update(step, feature)

    assert torch.allclose(bounded.predict(6), chunked.predict(6), atol=1e-5, rtol=1e-5)
    assert bounded._coef_slab is not None
    assert bounded._coef_slab.numel() == 0


def test_spectrum_bounded_backend_invalidates_slab_on_update():
    forecaster = SpectrumForecaster(
        SpectrumCacheConfig(
            degree=2,
            history_limit=3,
            predictor_backend="bounded",
            predictor_cache_bytes=4096,
            predictor_chunk_size=256,
        )
    )
    for step in range(3):
        forecaster.update(step, torch.full((1, 8, 16), float(step)))

    _ = forecaster.predict(4)
    assert forecaster._coef_slab is not None

    forecaster.update(4, torch.full((1, 8, 16), 4.0))
    assert forecaster._coef_slab is None
    assert forecaster._slab_elements == 0
    assert forecaster._history_design is None
    assert forecaster._history_chol is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"predictor_backend": "unknown"},
        {"predictor_cache_bytes": -1},
        {"predictor_chunk_size": 0},
    ],
)
def test_spectrum_bounded_config_validation(kwargs):
    with pytest.raises(ValueError):
        SpectrumCacheConfig(**kwargs)
