from unittest.mock import Mock

import pytest
import torch

from diffusers import SpectrumCacheConfig
from diffusers.hooks.spectrum_cache import (
    _SPECTRUM_BLOCK_HOOK,
    _SPECTRUM_DENOISER_HOOK,
    _SPECTRUM_HEAD_BLOCK_HOOK,
)
from diffusers.utils.testing_utils import torch_device
from tests.models.transformers.test_models_transformer_flux import FluxTransformerTesterConfig


class TestFluxTransformerSpectrumCache(FluxTransformerTesterConfig):
    """Native-style SPECTRUM cache tests for Flux Transformer."""

    def _get_spectrum_model(self):
        model = self.model_class(**self.get_init_dict()).to(torch_device)
        model.eval()
        return model

    @torch.no_grad()
    def test_spectrum_cache_enable_disable_and_forecast(self):
        model = self._get_spectrum_model()
        inputs = self.get_dummy_inputs()
        config = SpectrumCacheConfig(
            num_inference_steps=2,
            warmup_steps=1,
            window_size=2.0,
            flex_window=0.75,
            degree=1,
            ridge_lambda=0.1,
            blend_w=0.5,
        )

        model.enable_cache(config)
        assert model.is_cache_enabled

        registered = []
        for submodule in model.modules():
            if not hasattr(submodule, "_diffusers_hook"):
                continue
            for hook_name in (_SPECTRUM_DENOISER_HOOK, _SPECTRUM_HEAD_BLOCK_HOOK, _SPECTRUM_BLOCK_HOOK):
                if submodule._diffusers_hook.get_hook(hook_name) is not None:
                    registered.append(hook_name)

        assert _SPECTRUM_DENOISER_HOOK in registered
        assert _SPECTRUM_HEAD_BLOCK_HOOK in registered
        assert _SPECTRUM_BLOCK_HOOK in registered

        with model.cache_context("spectrum_test"):
            output_first = model(**inputs, return_dict=False)[0]

            inputs_second = inputs.copy()
            inputs_second["hidden_states"] = inputs_second["hidden_states"] + torch.randn_like(
                inputs_second["hidden_states"]
            )
            output_cached = model(**inputs_second, return_dict=False)[0]

        root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)
        state = root_hook.state_manager._state_cache["spectrum_test"]
        assert state.compute_steps == [0]
        assert state.forecast_steps == [1]
        assert torch.isfinite(output_first).all()
        assert torch.isfinite(output_cached).all()

        model.disable_cache()
        assert not model.is_cache_enabled

        output_uncached = model(**inputs_second, return_dict=False)[0]
        assert not torch.allclose(output_cached, output_uncached, atol=1e-5)

        for submodule in model.modules():
            if not hasattr(submodule, "_diffusers_hook"):
                continue
            for hook_name in (_SPECTRUM_DENOISER_HOOK, _SPECTRUM_HEAD_BLOCK_HOOK, _SPECTRUM_BLOCK_HOOK):
                assert submodule._diffusers_hook.get_hook(hook_name) is None

    @torch.no_grad()
    def test_spectrum_cache_context_isolation_and_reenable(self):
        model = self._get_spectrum_model()
        inputs = self.get_dummy_inputs()
        config = SpectrumCacheConfig(num_inference_steps=2, warmup_steps=1, degree=1)

        model.enable_cache(config)
        with model.cache_context("context_1"):
            output_1 = model(**inputs, return_dict=False)[0]
        with model.cache_context("context_2"):
            output_2 = model(**inputs, return_dict=False)[0]

        assert torch.allclose(output_1, output_2, atol=1e-5)

        model.disable_cache()
        model.enable_cache(config)
        with model.cache_context("context_after_reenable"):
            output_3 = model(**inputs, return_dict=False)[0]

        assert torch.allclose(output_1, output_3, atol=1e-5)
        model.disable_cache()

    @torch.no_grad()
    def test_spectrum_cache_reset_stateful_cache(self):
        model = self._get_spectrum_model()
        inputs = self.get_dummy_inputs()
        config = SpectrumCacheConfig(num_inference_steps=2, warmup_steps=1, degree=1)

        model.enable_cache(config)
        with model.cache_context("spectrum_test"):
            _ = model(**inputs, return_dict=False)[0]

        root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)
        assert root_hook.state_manager._state_cache

        model._reset_stateful_cache()
        assert not root_hook.state_manager._state_cache
        model.disable_cache()

    @torch.no_grad()
    def test_spectrum_cache_fails_closed_for_controlnet_samples(self):
        model = self._get_spectrum_model()
        inputs = self.get_dummy_inputs()
        config = SpectrumCacheConfig(num_inference_steps=2, warmup_steps=1, degree=1)
        model.enable_cache(config)

        # Flux expects control residuals in the embedded hidden-state width.
        control_sample = torch.zeros(
            inputs["hidden_states"].shape[0],
            inputs["hidden_states"].shape[1],
            model.inner_dim,
            device=inputs["hidden_states"].device,
            dtype=inputs["hidden_states"].dtype,
        )
        controlled_inputs = inputs.copy()
        controlled_inputs["controlnet_block_samples"] = [control_sample]

        with model.cache_context("control"):
            output_cached_model = model(**controlled_inputs, return_dict=False)[0]

        root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)
        state = root_hook.state_manager._state_cache["control"]
        assert state.compute_steps == []
        assert state.forecast_steps == []

        model.disable_cache()
        output_uncached_model = model(**controlled_inputs, return_dict=False)[0]
        assert torch.allclose(output_cached_model, output_uncached_model, atol=1e-5)


@pytest.mark.parametrize("failure_kind", ["exception", "nonfinite", "shape"])
@torch.no_grad()
def test_spectrum_cache_forecast_failure_falls_back_and_latches(failure_kind):
    tester = TestFluxTransformerSpectrumCache()
    model = tester._get_spectrum_model()
    inputs = tester.get_dummy_inputs()
    second_inputs = inputs.copy()
    second_inputs["hidden_states"] = inputs["hidden_states"] + 0.2
    third_inputs = inputs.copy()
    third_inputs["hidden_states"] = inputs["hidden_states"] - 0.15

    expected_second = model(**second_inputs, return_dict=False)[0]
    expected_third = model(**third_inputs, return_dict=False)[0]

    config = SpectrumCacheConfig(
        num_inference_steps=3,
        warmup_steps=1,
        degree=1,
        forecast_step_indices=(1, 2),
    )
    model.enable_cache(config)
    root_hook = model._diffusers_hook.get_hook(_SPECTRUM_DENOISER_HOOK)

    with model.cache_context("forecast-failure"):
        _ = model(**inputs, return_dict=False)[0]
        state = root_hook.state_manager.get_state()
        feature = state.forecaster.features[-1]

        if failure_kind == "exception":
            state.predict = Mock(side_effect=RuntimeError("injected predictor failure"))
        elif failure_kind == "nonfinite":
            state.predict = Mock(return_value=torch.full_like(feature, float("nan")))
        else:
            shape = (*feature.shape[:-1], feature.shape[-1] + 1)
            state.predict = Mock(return_value=torch.zeros(shape, device=feature.device, dtype=feature.dtype))
        fallback = model(**second_inputs, return_dict=False)[0]
        sticky = model(**third_inputs, return_dict=False)[0]

        assert state.prediction_failure_latched is True
        assert state.prediction_failure_latched_at == 1
        assert state.fallback_steps == [1]
        assert len(state.predict_failures) == 1
        assert state.predict_failures[0]["step"] == 1
        assert state.forecast_steps == [1]
        assert state.compute_steps == [0, 2]
        assert state.forecaster.steps == [1, 2]
        assert torch.allclose(fallback, expected_second, atol=1e-5)
        assert torch.allclose(sticky, expected_third, atol=1e-5)

    model._reset_stateful_cache()
    with model.cache_context("forecast-failure"):
        state = root_hook.state_manager.get_state()
        assert state.prediction_failure_latched is False
        assert state.predict_failures == []
        assert state.fallback_steps == []

    model.disable_cache()
