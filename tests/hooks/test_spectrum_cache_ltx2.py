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

import torch

from diffusers import LTX2VideoTransformer3DModel
from diffusers.hooks.spectrum_cache import SpectrumCacheConfig, SpectrumLTX2State


QUALIFIED_SIGNATURE = {
    "in_channels": 128,
    "out_channels": 128,
    "num_layers": 48,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "cross_attention_dim": 4096,
    "audio_in_channels": 128,
    "audio_out_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "caption_channels": 3840,
    "cross_attn_mod": True,
    "audio_cross_attn_mod": True,
    "gated_attn": True,
    "audio_gated_attn": True,
    "perturbed_attn": True,
    "rope_type": "split",
    "use_prompt_embeddings": False,
}


def make_config():
    return SpectrumCacheConfig(
        num_inference_steps=8,
        warmup_steps=0,
        window_size=3.0,
        flex_window=0.0,
        degree=2,
        ridge_lambda=0.1,
        blend_w=0.5,
        history_limit=8,
        coordinate_max=8.0,
        tail_actual_steps=0,
        forecast_step_indices=(3,),
    )


def make_tiny_model(num_layers=48):
    return LTX2VideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=8,
        cross_attention_dim=16,
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=8,
        num_layers=num_layers,
        caption_channels=16,
        rope_double_precision=False,
        use_prompt_embeddings=False,
    )


def test_spectrum_ltx2_selected_profile_paired_state():
    state = SpectrumLTX2State(make_config())
    for step in range(8):
        state.start_step()
        if step == 3:
            assert not state.should_compute
            predicted_video, predicted_audio = state.predict()
            assert predicted_video.shape == (1, 4, 8)
            assert predicted_audio.shape == (1, 2, 6)
        else:
            assert state.should_compute
            state.record_real_features(
                torch.full((1, 4, 8), float(step)),
                torch.full((1, 2, 6), float(step) + 0.25),
            )

    summary = state.summary()
    assert summary["compute_steps"] == [0, 1, 2, 4, 5, 6, 7]
    assert summary["forecast_steps"] == [3]
    assert not summary["guard_latched"]
    assert summary["peak_history_bytes"] > 0


def test_spectrum_ltx2_rejects_unqualified_architecture():
    model = make_tiny_model(num_layers=2)
    try:
        model.enable_cache(make_config())
    except ValueError as error:
        assert "qualified only for the LTX-2.3 audiovisual transformer architecture" in str(error)
    else:
        raise AssertionError("Expected an unqualified LTX-2 architecture to be rejected.")


def test_spectrum_ltx2_qualified_signature_registers_full_stack_hooks():
    model = make_tiny_model(num_layers=48)
    model.register_to_config(**QUALIFIED_SIGNATURE)
    model.enable_cache(make_config())

    root = model._diffusers_hook.get_hook("spectrum_cache_denoiser")
    assert root is not None
    assert model.transformer_blocks[0]._diffusers_hook.get_hook("spectrum_cache_head_block") is not None
    for block in model.transformer_blocks[1:]:
        assert block._diffusers_hook.get_hook("spectrum_cache_block") is not None
