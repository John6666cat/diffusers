import torch

from diffusers import Krea2Transformer2DModel
from diffusers.hooks._helpers import TransformerBlockRegistry
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumKrea2RawState, SpectrumSchedule


KREA2_TURBO_CONFIG_SIGNATURE = {
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": (32, 48, 48),
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}


def make_model():
    model = Krea2Transformer2DModel(
        in_channels=4,
        num_layers=28,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=12,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        text_intermediate_size=32,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
        rope_theta=1000.0,
        norm_eps=1e-5,
    ).eval()
    # Tiny mechanics graph; the adapter itself remains fail-closed to the exact production config.
    model.register_to_config(**KREA2_TURBO_CONFIG_SIGNATURE)
    return model


def make_unqualified_model():
    return Krea2Transformer2DModel(
        in_channels=4,
        num_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=12,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        text_intermediate_size=32,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()


def make_inputs(step=0, *, image_tokens=16, attention_kwargs=None):
    text_tokens = 4
    return {
        "hidden_states": torch.randn(1, image_tokens, 4),
        "encoder_hidden_states": torch.randn(1, text_tokens, 12, 8),
        "timestep": torch.tensor([1.0 - step / 8.0]),
        "position_ids": torch.zeros(text_tokens + image_tokens, 3),
        "encoder_attention_mask": torch.ones(1, text_tokens, dtype=torch.bool),
        "attention_kwargs": attention_kwargs,
        "return_dict": True,
    }


def make_config():
    return SpectrumCacheConfig(
        num_inference_steps=8,
        warmup_steps=6,
        window_size=2.0,
        flex_window=0.0,
        degree=1,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=50.0,
        tail_actual_steps=1,
    )


def assert_finite_sample(output):
    assert torch.is_tensor(output.sample)
    assert torch.isfinite(output.sample).all()


def test_krea2_block_registry():
    from diffusers.models.transformers.transformer_krea2 import Krea2TransformerBlock

    metadata = TransformerBlockRegistry.get(Krea2TransformerBlock)
    assert metadata.hidden_states_argument_name == "hidden_states"
    assert metadata.return_hidden_states_index == 0
    assert metadata.return_encoder_hidden_states_index is None


@torch.no_grad()
def test_spectrum_krea2_turbo_exact_selected_step6_and_block_accounting():
    model = make_model()
    calls = [0 for _ in model.transformer_blocks]
    hooks = [
        block.attn.register_forward_hook(
            lambda *args, i=i: calls.__setitem__(i, calls[i] + 1)
        )
        for i, block in enumerate(model.transformer_blocks)
    ]
    try:
        model.enable_cache(make_config())
        for step in range(8):
            with model.cache_context("cond"):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["cond"].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 4, 5, 7]
        assert summary["forecast_steps"] == [6]
        assert summary["prediction_call_count"] == 1
        assert summary["full_call_count"] == 7
        assert not summary["guard_latched"]
        assert calls == [7] * 28

        model.disable_cache()
        assert not model.is_cache_enabled
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_spectrum_krea2_contexts_partition_history():
    model = make_model()
    model.enable_cache(make_config())
    for step in range(8):
        for label in ("cond", "alternate"):
            with model.cache_context(label):
                output = model(**make_inputs(step))
            assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert set(root.state_manager._state_cache) == {"cond", "alternate"}
    for label in ("cond", "alternate"):
        summary = root.state_manager._state_cache[label].summary()
        assert summary["compute_steps"] == [0, 1, 2, 3, 4, 5, 7]
        assert summary["forecast_steps"] == [6]
        assert summary["prediction_call_count"] == 1
        assert not summary["guard_latched"]


@torch.no_grad()
def test_spectrum_krea2_shape_change_latches_sticky_failclosed():
    model = make_model()
    model.enable_cache(make_config())

    with model.cache_context("cond"):
        output = model(**make_inputs(0))
    assert_finite_sample(output)

    with model.cache_context("cond"):
        output = model(**make_inputs(1, image_tokens=9))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("signature changed" in item["reason"] for item in summary["guard_reasons"])


@torch.no_grad()
def test_spectrum_krea2_nonempty_attention_kwargs_latch_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    with model.cache_context("cond"):
        output = model(**make_inputs(0, attention_kwargs={"scale": 1.0}))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("attention kwargs" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_krea2_autograd_latches_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    with model.cache_context("cond"):
        output = model(**make_inputs(0))
    assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["cond"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("autograd/training" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_krea2_rejects_unqualified_architecture_at_enable():
    model = make_unqualified_model()
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "Krea 2 standard T2I transformer architecture" in str(error)
    else:
        raise AssertionError("Expected unqualified Krea 2 architecture to be rejected.")



def make_raw_config():
    return SpectrumCacheConfig(
        num_inference_steps=52,
        forecast_step_indices=(7, 21, 30, 35, 42, 44),
        window_size=2.0,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=100.0,
        tail_actual_steps=0,
    )


def test_spectrum_explicit_forecast_step_indices_schedule():
    config = SpectrumCacheConfig(num_inference_steps=10, forecast_step_indices=(2, 5, 8))
    schedule = SpectrumSchedule(config)
    actual, forecast = [], []
    for step in range(10):
        (actual if schedule.decide(step) else forecast).append(step)
    assert actual == [0, 1, 3, 4, 6, 7, 9]
    assert forecast == [2, 5, 8]


def test_spectrum_explicit_forecast_step_indices_validation():
    for kwargs in (
        {"num_inference_steps": 10, "forecast_step_indices": (2, 2)},
        {"num_inference_steps": 10, "forecast_step_indices": (-1, 2)},
        {"num_inference_steps": 10, "forecast_step_indices": (2, 10)},
        {"num_inference_steps": 10, "forecast_step_indices": (8,), "tail_actual_steps": 2},
    ):
        try:
            SpectrumCacheConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid explicit forecast schedule to fail: {kwargs}")


@torch.no_grad()
def test_spectrum_krea2_raw_dual_lane_selected_profile_and_block_accounting():
    model = make_model()
    calls = [0 for _ in model.transformer_blocks]
    hooks = [
        block.attn.register_forward_hook(
            lambda *args, i=i: calls.__setitem__(i, calls[i] + 1)
        )
        for i, block in enumerate(model.transformer_blocks)
    ]
    positive = torch.randn(1, 4, 12, 8)
    negative = torch.randn(1, 4, 12, 8)
    try:
        model.enable_cache(make_raw_config())
        for step in range(52):
            for encoder_hidden_states in (positive, negative):
                inputs = make_inputs(step)
                inputs["encoder_hidden_states"] = encoder_hidden_states
                with model.cache_context("raw"):
                    output = model(**inputs)
                assert_finite_sample(output)

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        state = root.state_manager._state_cache["raw"]
        summary = state.summary()
        assert summary["compute_steps"] == [
            step for step in range(52) if step not in (7, 21, 30, 35, 42, 44)
        ]
        assert summary["forecast_steps"] == [7, 21, 30, 35, 42, 44]
        assert summary["prediction_call_count"] == 12
        assert summary["full_call_count"] == 92
        assert summary["lane_prediction_call_count"] == {"positive": 6, "negative": 6}
        assert not summary["guard_latched"]
        assert calls == [92] * 28
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_spectrum_krea2_raw_missing_cfg_latches_failclosed():
    model = make_model()
    config = make_raw_config()
    model.enable_cache(config)
    shared = torch.randn(1, 4, 12, 8)

    for step in range(2):
        inputs = make_inputs(step)
        inputs["encoder_hidden_states"] = shared
        with model.cache_context("raw"):
            output = model(**inputs)
        assert_finite_sample(output)

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["raw"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("positive/negative conditioning identities are not distinct" in item["reason"] for item in summary["guard_reasons"])


def test_spectrum_krea2_raw_requires_explicit_schedule():
    model = make_model()
    try:
        model.enable_cache(SpectrumCacheConfig(num_inference_steps=52))
    except ValueError as error:
        assert "requires explicit forecast_step_indices" in str(error)
    else:
        raise AssertionError("Expected Krea 2 Raw 52-step route without an explicit schedule to fail.")



def test_spectrum_krea2_raw_prediction_failure_latches_both_lanes_and_resets():
    config = SpectrumCacheConfig(num_inference_steps=2, forecast_step_indices=(1,), degree=1)
    state = SpectrumKrea2RawState(config)
    positive = torch.ones(1, 2, 3)
    negative = torch.full((1, 2, 3), 2.0)

    state.prepare_call("positive")
    state.start_step()
    state.record_real_feature(positive)
    state.prepare_call("negative")
    state.start_step()
    state.record_real_feature(negative)

    state.prepare_call("positive")
    state.start_step()
    assert state.should_compute is False
    state.fail_closed_prediction(RuntimeError("injected predictor failure"))
    assert state.prediction_failure_latched is True
    assert state.prediction_failure_latched_at == 1
    assert state.fallback_steps == [1]
    summary = state.summary()
    assert summary["prediction_failure_latched"] is True
    assert summary["predict_failures"][0]["step"] == 1
    assert summary["fallback_steps"] == [1]
    assert all(forecaster.features == [] for forecaster in state.forecasters.values())

    state.record_real_feature(positive + 1)
    state.prepare_call("negative")
    state.start_step()
    assert state.should_compute is True
    state.record_real_feature(negative + 1)
    assert state.forecasters["positive"].steps == [1]
    assert state.forecasters["negative"].steps == [1]

    state.reset()
    assert state.prediction_failure_latched is False
    assert state.predict_failures == []
    assert state.fallback_steps == []


def _krea2_selected_state_cls(config):
    model = make_model()
    model.enable_cache(config)
    try:
        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        return root.state_manager._state_cls
    finally:
        model.disable_cache()


def test_spectrum_krea2_raw_dual_lane_runtime_coordinate_provenance_accepts_repeated_outer_step():
    config = SpectrumCacheConfig(
        num_inference_steps=3,
        forecast_step_indices=(2,),
        degree=1,
        history_limit=3,
        coordinate_policy="runtime_index_normalized",
        coordinate_max=3.0,
    )
    state = SpectrumKrea2RawState(config)
    positive = torch.ones(1, 2, 3)
    negative = torch.full((1, 2, 3), 2.0)

    for step in range(3):
        for lane, feature in (("positive", positive + step), ("negative", negative + step)):
            state.prepare_call(lane)
            state.start_step(
                logical_step_index=step,
                runtime_num_inference_steps=3,
                provenance_source="cache_context:krea",
            )
            if state.should_compute:
                state.record_real_feature(feature)

    summary = state.summary()
    assert summary["coordinate"]["failure_latched"] is False
    assert summary["coordinate"]["runtime_num_inference_steps"] == 3
    assert summary["coordinate"]["last_logical_step_index"] == 2
    assert summary["compute_steps"] == [0, 1]
    assert summary["forecast_steps"] == [2]


def test_spectrum_krea2_explicit_paired_cfg_topology_selects_raw_state_independent_of_step_count():
    for steps, forecast in ((20, (7,)), (28, (7, 21)), (40, (7, 21, 30))):
        config = SpectrumCacheConfig(
            num_inference_steps=steps,
            forecast_step_indices=forecast,
            denoiser_call_topology="paired_cfg",
        )
        assert _krea2_selected_state_cls(config) is SpectrumKrea2RawState


def test_spectrum_krea2_explicit_single_topology_overrides_legacy_52_step_raw_heuristic():
    config = SpectrumCacheConfig(
        num_inference_steps=52,
        forecast_step_indices=(7, 21, 30, 35, 42, 44),
        denoiser_call_topology="single",
    )
    assert _krea2_selected_state_cls(config).__name__ == "SpectrumKrea2State"


def test_spectrum_krea2_auto_topology_preserves_historical_route_identity():
    raw = SpectrumCacheConfig(
        num_inference_steps=52,
        forecast_step_indices=(7, 21, 30, 35, 42, 44),
    )
    turbo = make_config()
    assert _krea2_selected_state_cls(raw) is SpectrumKrea2RawState
    assert _krea2_selected_state_cls(turbo).__name__ == "SpectrumKrea2State"


def test_spectrum_krea2_paired_cfg_requires_explicit_schedule_at_any_step_count():
    model = make_model()
    try:
        model.enable_cache(
            SpectrumCacheConfig(
                num_inference_steps=28,
                denoiser_call_topology="paired_cfg",
            )
        )
    except ValueError as error:
        assert "paired-CFG route requires explicit forecast_step_indices" in str(error)
    else:
        raise AssertionError("Expected paired-CFG Krea 2 route without explicit schedule to fail.")


def test_spectrum_denoiser_call_topology_validation():
    try:
        SpectrumCacheConfig(denoiser_call_topology="mystery")
    except ValueError as error:
        assert "denoiser_call_topology" in str(error)
    else:
        raise AssertionError("Expected invalid denoiser_call_topology to fail.")


def test_spectrum_krea2_raw_runtime_horizon_provenance_accepts_repeated_outer_step():
    config = SpectrumCacheConfig(
        num_inference_steps=3,
        forecast_step_indices=(2,),
        degree=1,
        history_limit=3,
        coordinate_policy="runtime_horizon_normalized",
        coordinate_max=3.0,
        denoiser_call_topology="paired_cfg",
    )
    state = SpectrumKrea2RawState(config)
    positive = torch.ones(1, 2, 3)
    negative = torch.full((1, 2, 3), 2.0)

    for step in range(3):
        for lane, feature in (("positive", positive + step), ("negative", negative + step)):
            state.prepare_call(lane)
            state.start_step(
                logical_step_index=step,
                runtime_num_inference_steps=3,
                provenance_source="cache_context",
            )
            if state.should_compute:
                state.record_real_feature(feature)

    summary = state.summary()
    assert summary["coordinate"]["failure_latched"] is False
    assert summary["coordinate"]["runtime_num_inference_steps"] == 3
    assert summary["coordinate"]["last_logical_step_index"] == 2
    assert summary["compute_steps"] == [0, 1]
    assert summary["forecast_steps"] == [2]


def test_spectrum_krea2_raw_variable_factory_locked_operating_table():
    expected = {
        28: (4, 16, 23),
        40: (5, 8, 21, 23, 24, 27, 32, 34),
        52: (7, 10, 21, 28, 30, 32, 35, 37, 42, 44),
    }
    for steps, schedule in expected.items():
        config = SpectrumCacheConfig.for_krea2_raw_variable(steps)
        assert config.num_inference_steps == steps
        assert config.forecast_step_indices == schedule
        assert config.warmup_steps == 0
        assert config.window_size == 2.0
        assert config.flex_window == 0.0
        assert config.degree == 4
        assert config.ridge_lambda == 0.1
        assert config.blend_w == 0.5
        assert config.history_limit == 8
        assert config.predictor_backend == "dense"
        assert config.coordinate_policy == "runtime_horizon_normalized"
        assert config.coordinate_max == 100.0
        assert config.tail_actual_steps == 0
        assert config.denoiser_call_topology == "paired_cfg"
        assert _krea2_selected_state_cls(config) is SpectrumKrea2RawState

    for steps in (20, 30, 51, 60):
        try:
            SpectrumCacheConfig.for_krea2_raw_variable(steps)
        except ValueError as error:
            assert "{28, 40, 52}" in str(error)
        else:
            raise AssertionError(f"Expected unqualified Krea Raw step count {steps} to be rejected")
