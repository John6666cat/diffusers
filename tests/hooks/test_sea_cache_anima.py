import torch

from diffusers import CosmosTransformer3DModel, SeaCacheConfig
from diffusers.hooks._helpers import TransformerBlockRegistry
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock


def make_model():
    return CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=2,
        mlp_ratio=2,
        text_embed_dim=16,
        adaln_lora_dim=4,
        max_size=(4, 32, 32),
        patch_size=(1, 2, 2),
        rope_scale=(2.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type="learnable",
    ).eval()


def make_inputs(seed=0, *, fps=None, condition_mask=None):
    generator = torch.Generator("cpu").manual_seed(seed)
    latent_channels = 3 if condition_mask is not None else 4
    data = {
        "hidden_states": torch.randn((1, latent_channels, 1, 16, 16), generator=generator),
        "timestep": torch.tensor([0.5]),
        "encoder_hidden_states": torch.randn((1, 12, 16), generator=generator),
        "padding_mask": torch.zeros(1, 1, 16, 16),
        "return_dict": False,
    }
    if fps is not None:
        data["fps"] = fps
    if condition_mask is not None:
        data["condition_mask"] = condition_mask
    return data


def config():
    return SeaCacheConfig(
        threshold=1e6,
        residual_order=0,
        retention_steps=1,
        cache_end_steps=0,
        max_consecutive_cached=0,
    )


def root_hook(model):
    return model._diffusers_hook.get_hook("sea_cache_root")


def run_context(model, label, step, inputs):
    with model.cache_context(
        label,
        step_index=step,
        num_inference_steps=4,
        timestep=torch.tensor([float(step)]),
        sigma=0.8 - 0.1 * step,
    ):
        return model(**inputs)[0]


def test_cosmos_transformer_block_metadata_is_registered_for_generic_seacache():
    metadata = TransformerBlockRegistry.get(CosmosTransformerBlock)
    assert metadata.return_hidden_states_index == 0
    assert metadata.return_encoder_hidden_states_index is None
    assert metadata.hidden_states_argument_name == "hidden_states"


@torch.no_grad()
def test_anima_like_cosmos_uses_builtin_raw_vision_and_partitions_cache_history():
    model = make_model()
    baseline_inputs = make_inputs(seed=123)
    baseline = model(**baseline_inputs)[0]

    model.enable_cache(config())
    calls = {"count": 0}
    handle = model.transformer_blocks[0].attn1.register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    for step in range(4):
        for label in ("pred_cond", "pred_uncond"):
            output = run_context(model, label, step, make_inputs(seed=123))
            assert torch.isfinite(output).all()

    manager = root_hook(model).state_manager
    assert set(manager._state_cache) == {"pred_cond", "pred_uncond"}
    for state in manager._state_cache.values():
        assert len(state.history) == 1
        assert state.consecutive_cached == 3

    # One actual leader-block body execution per trajectory; later steps reuse the residual.
    assert calls["count"] == 2
    handle.remove()

    model.disable_cache()
    restored = model(**make_inputs(seed=123))[0]
    torch.testing.assert_close(restored, baseline, rtol=0.0, atol=0.0)


@torch.no_grad()
def test_anima_seacache_unqualified_video_fps_route_fails_open():
    model = make_model()
    model.enable_cache(config())

    calls = {"count": 0}
    handle = model.transformer_blocks[0].attn1.register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )

    for step in range(4):
        output = run_context(model, "pred_cond", step, make_inputs(seed=77, fps=24))
        assert torch.isfinite(output).all()

    # Raw-vision adapter rejects fps/video mode, so every call stays full-compute.
    assert calls["count"] == 4
    handle.remove()
    model.disable_cache()


@torch.no_grad()
def test_anima_seacache_unqualified_condition_mask_route_fails_open():
    model = make_model()
    model.enable_cache(config())

    calls = {"count": 0}
    handle = model.transformer_blocks[0].attn1.register_forward_pre_hook(
        lambda _module, _args: calls.__setitem__("count", calls["count"] + 1)
    )
    mask = torch.zeros(1, 1, 1, 16, 16)

    for step in range(4):
        output = run_context(model, "pred_cond", step, make_inputs(seed=81, condition_mask=mask))
        assert torch.isfinite(output).all()

    assert calls["count"] == 4
    handle.remove()
    model.disable_cache()


def test_cosmos_img_context_configuration_is_rejected_before_generic_block_skip_contract():
    model = make_model()
    model.register_to_config(img_context_dim_in=8)
    try:
        model.enable_cache(config())
    except ValueError as error:
        assert "image-context" in str(error)
    else:
        raise AssertionError("Expected image-context Cosmos SeaCache configuration to be rejected.")
