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


def test_spectrum_chunked_backend_matches_dense_float32_and_avoids_retained_dense_coefficients():
    common = {"degree": 4, "ridge_lambda": 0.1, "blend_w": 0.5, "history_limit": 5}
    dense = SpectrumForecaster(SpectrumCacheConfig(**common, predictor_backend="dense"))
    chunked = SpectrumForecaster(SpectrumCacheConfig(**common, predictor_backend="chunked"))

    generator = torch.Generator().manual_seed(0)
    for step in range(5):
        feature = torch.randn((2, 8, 16), generator=generator, dtype=torch.float32)
        dense.update(step, feature)
        chunked.update(step, feature)

    dense_predicted = dense.predict(7)
    chunked_predicted = chunked.predict(7)

    assert torch.allclose(chunked_predicted, dense_predicted, atol=1e-5, rtol=1e-5)
    assert chunked._coef is None
    assert chunked._history_design is not None
    assert chunked._history_chol is not None
    assert chunked._history_design.device == chunked.features[-1].device
    assert chunked._history_chol.device == chunked.features[-1].device
    assert chunked.predictor_state_bytes() < dense.predictor_state_bytes()


def test_spectrum_chunked_backend_preserves_shape_dtype_and_invalidates_solver_on_update():
    forecaster = SpectrumForecaster(SpectrumCacheConfig(degree=2, history_limit=3, predictor_backend="chunked"))
    for step in range(3):
        forecaster.update(step, torch.full((1, 4, 8), float(step), dtype=torch.float32))

    predicted = forecaster.predict(4)
    assert predicted.shape == (1, 4, 8)
    assert predicted.dtype == torch.float32
    assert torch.isfinite(predicted).all()
    assert forecaster._history_design is not None
    assert forecaster._history_chol is not None

    forecaster.update(4, torch.full((1, 4, 8), 4.0, dtype=torch.float32))
    assert forecaster._history_design is None
    assert forecaster._history_chol is None
    assert forecaster._coef is None


def test_spectrum_config_rejects_unknown_predictor_backend():
    with pytest.raises(ValueError, match="predictor_backend"):
        SpectrumCacheConfig(predictor_backend="unknown")
