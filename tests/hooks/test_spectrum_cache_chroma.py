import torch

from diffusers import ChromaTransformer2DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig


CHROMA_HD_SIGNATURE = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "axes_dims_rope": (16, 56, 56),
    "approximator_num_channels": 64,
    "approximator_hidden_dim": 5120,
    "approximator_layers": 5,
}


def make_model():
    model = ChromaTransformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=19,
        num_single_layers=38,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        approximator_num_channels=16,
        approximator_hidden_dim=32,
        approximator_layers=1,
    ).eval()
    model.register_to_config(**CHROMA_HD_SIGNATURE)
    return model


def make_inputs(step, encoder_hidden_states, *, attention_mask=None, joint_attention_kwargs=None):
    image_tokens = 2
    text_tokens = encoder_hidden_states.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones(1, text_tokens + image_tokens, dtype=torch.bool)
    return {
        "hidden_states": torch.randn(1, image_tokens, 4),
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": torch.tensor([1.0 - step / 40.0]),
        "img_ids": torch.zeros(image_tokens, 3),
        "txt_ids": torch.zeros(text_tokens, 3),
        "attention_mask": attention_mask,
        "joint_attention_kwargs": joint_attention_kwargs,
        "return_dict": True,
    }


def make_config(schedule=(7, 10, 13, 16, 19, 22, 25, 28, 31, 34), coordinate_max=50.0):
    return SpectrumCacheConfig(
        num_inference_steps=40,
        warmup_steps=7,
        window_size=2.0,
        flex_window=0.75,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=100,
        coordinate_max=coordinate_max,
        tail_actual_steps=5,
        forecast_step_indices=schedule,
    )


@torch.no_grad()
def test_spectrum_chroma_hd_spread10_dual_lane_skips_body_and_disables_cleanly():
    model = make_model()
    positive = torch.randn(1, 3, 12)
    negative = torch.randn(1, 3, 12)
    dual_calls = [0] * len(model.transformer_blocks)
    single_calls = [0] * len(model.single_transformer_blocks)
    hooks = []
    for i, block in enumerate(model.transformer_blocks):
        hooks.append(block.register_forward_hook(lambda *args, i=i: dual_calls.__setitem__(i, dual_calls[i] + 1)))
    for i, block in enumerate(model.single_transformer_blocks):
        hooks.append(block.register_forward_hook(lambda *args, i=i: single_calls.__setitem__(i, single_calls[i] + 1)))

    try:
        model.enable_cache(make_config())
        for step in range(40):
            for encoder_hidden_states in (positive, negative):
                with model.cache_context("chroma-hd"):
                    output = model(**make_inputs(step, encoder_hidden_states)).sample
                assert torch.isfinite(output).all()

        root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
        summary = root.state_manager._state_cache["chroma-hd"].summary()
        expected_forecast = [7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
        assert summary["forecast_steps"] == expected_forecast
        assert summary["compute_steps"] == [x for x in range(40) if x not in expected_forecast]
        assert summary["prediction_call_count"] == 20
        assert summary["full_call_count"] == 60
        assert summary["lane_prediction_call_count"] == {"positive": 10, "negative": 10}
        assert summary["predict_failures"] == []
        assert not summary["guard_latched"]
        assert dual_calls == [60] * 19
        assert single_calls == [60] * 38

        model.disable_cache()
        assert not model.is_cache_enabled
        assert model._diffusers_hook.get_hook("spectrum_cache_denoiser") is None
        assert model.norm_out._diffusers_hook.get_hook("spectrum_cache_chroma_feature") is None
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def test_spectrum_chroma_hd_non_bool_mask_latches_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    positive = torch.randn(1, 3, 12)
    negative = torch.randn(1, 3, 12)
    mask = torch.ones(1, 5, dtype=torch.float32)
    for step in range(8):
        for encoder_hidden_states in (positive, negative):
            with model.cache_context("bad-mask"):
                output = model(**make_inputs(step, encoder_hidden_states, attention_mask=mask)).sample
            assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["bad-mask"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("boolean attention mask" in row["reason"] for row in summary["guard_reasons"])


@torch.no_grad()
def test_spectrum_chroma_hd_missing_external_cfg_latches_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    shared = torch.randn(1, 3, 12)
    for step in range(8):
        with model.cache_context("no-cfg"):
            output = model(**make_inputs(step, shared)).sample
        assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["no-cfg"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("identities are not distinct" in row["reason"] for row in summary["guard_reasons"])


@torch.no_grad()
def test_spectrum_chroma_hd_nonempty_attention_kwargs_latch_failclosed():
    model = make_model()
    model.enable_cache(make_config())
    positive = torch.randn(1, 3, 12)
    negative = torch.randn(1, 3, 12)
    for encoder_hidden_states in (positive, negative):
        with model.cache_context("kwargs"):
            output = model(**make_inputs(0, encoder_hidden_states, joint_attention_kwargs={"scale": 1.0})).sample
        assert torch.isfinite(output).all()

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    summary = root.state_manager._state_cache["kwargs"].summary()
    assert summary["guard_latched"]
    assert summary["prediction_call_count"] == 0
    assert any("joint_attention_kwargs" in row["reason"] for row in summary["guard_reasons"])


def test_spectrum_chroma_hd_rejects_unqualified_profile():
    model = make_model()
    bad = SpectrumCacheConfig(
        num_inference_steps=40,
        warmup_steps=7,
        window_size=2.0,
        flex_window=0.75,
        degree=4,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=100,
        coordinate_max=50.0,
        tail_actual_steps=5,
        forecast_step_indices=(8, 10, 15, 17, 22, 24, 29, 31),
    )
    try:
        model.enable_cache(bad)
    except ValueError as error:
        assert "qualified 40-step external-CFG profiles" in str(error)
    else:
        raise AssertionError("Expected the dominated/unqualified paired8 schedule to be rejected.")
