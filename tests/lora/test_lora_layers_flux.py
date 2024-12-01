# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import gc
import os
import sys
import tempfile
import unittest

import numpy as np
import safetensors.torch
import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel

from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
from diffusers.utils import logging
from diffusers.utils.testing_utils import (
    CaptureLogger,
    floats_tensor,
    is_peft_available,
    nightly,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_peft_version_greater,
    require_torch_gpu,
    slow,
    torch_device,
)


if is_peft_available():
    from peft.utils import get_peft_model_state_dict

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


@require_peft_backend
class FluxLoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = FluxPipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler()
    scheduler_kwargs = {}
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    transformer_kwargs = {
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 16,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "axes_dims_rope": [4, 4, 8],
    }
    transformer_cls = FluxTransformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 1,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    has_two_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_2_cls, tokenizer_2_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    text_encoder_2_cls, text_encoder_2_id = T5EncoderModel, "hf-internal-testing/tiny-random-t5"

    @property
    def output_shape(self):
        return (1, 8, 8, 3)

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "height": 8,
            "width": 8,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    def get_dummy_tensor_inputs(self, device=None):
        batch_size = 1
        num_latent_channels = 4
        num_image_channels = 3
        height = width = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = torch.randn((batch_size, height * width, num_latent_channels)).to(torch_device)
        encoder_hidden_states = torch.randn((batch_size, sequence_length, embedding_dim)).to(torch_device)
        pooled_prompt_embeds = torch.randn((batch_size, embedding_dim)).to(torch_device)
        text_ids = torch.randn((sequence_length, num_image_channels)).to(torch_device)
        image_ids = torch.randn((height * width, num_image_channels)).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": image_ids,
            "timestep": timestep,
        }

    def test_with_alpha_in_state_dict(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == self.output_shape)

        pipe.transformer.add_adapter(denoiser_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in transformer")

        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            denoiser_state_dict = get_peft_model_state_dict(pipe.transformer)
            self.pipeline_class.save_lora_weights(tmpdirname, transformer_lora_layers=denoiser_state_dict)

            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname, "pytorch_lora_weights.safetensors"))

            # modify the state dict to have alpha values following
            # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA/blob/main/jon_snow.safetensors
            state_dict_with_alpha = safetensors.torch.load_file(
                os.path.join(tmpdirname, "pytorch_lora_weights.safetensors")
            )
            alpha_dict = {}
            for k, v in state_dict_with_alpha.items():
                # only do for `transformer` and for the k projections -- should be enough to test.
                if "transformer" in k and "to_k" in k and "lora_A" in k:
                    alpha_dict[f"{k}.alpha"] = float(torch.randint(10, 100, size=()))
            state_dict_with_alpha.update(alpha_dict)

        images_lora_from_pretrained = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(check_if_lora_correctly_set(pipe.transformer), "Lora not correctly set in denoiser")

        pipe.unload_lora_weights()
        pipe.load_lora_weights(state_dict_with_alpha)
        images_lora_with_alpha = pipe(**inputs, generator=torch.manual_seed(0)).images

        self.assertTrue(
            np.allclose(images_lora, images_lora_from_pretrained, atol=1e-3, rtol=1e-3),
            "Loading from saved checkpoints should give same results.",
        )
        self.assertFalse(np.allclose(images_lora_with_alpha, images_lora, atol=1e-3, rtol=1e-3))

    def test_with_norm_in_state_dict(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_tensor_inputs(torch_device)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.INFO)

        with torch.no_grad():
            original_output = pipe.transformer(**inputs)[0]

        for norm_layer in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]:
            norm_state_dict = {}
            for name, module in pipe.transformer.named_modules():
                if norm_layer not in name or not hasattr(module, "weight") or module.weight is None:
                    continue
                norm_state_dict[f"transformer.{name}.weight"] = torch.randn(
                    module.weight.shape, device=module.weight.device, dtype=module.weight.dtype
                )

            with torch.no_grad():
                with CaptureLogger(logger) as cap_logger:
                    pipe.load_lora_weights(norm_state_dict)
                    lora_load_output = pipe.transformer(**inputs)[0]
                self.assertTrue(
                    cap_logger.out.startswith(
                        "The provided state dict contains normalization layers in addition to LoRA layers"
                    )
                )

                pipe.unload_lora_weights()
                lora_unload_output = pipe.transformer(**inputs)[0]

            self.assertTrue(pipe.transformer._transformer_norm_layers is None)
            self.assertFalse(np.allclose(original_output, lora_load_output, atol=1e-5, rtol=1e-5))
            self.assertTrue(np.allclose(original_output, lora_unload_output, atol=1e-5, rtol=1e-5))

        with CaptureLogger(logger) as cap_logger:
            for key in list(norm_state_dict.keys()):
                norm_state_dict[key.replace("norm", "norm_k_something_random")] = norm_state_dict.pop(key)
            pipe.load_lora_weights(norm_state_dict)

        self.assertTrue(
            cap_logger.out.startswith("Unsupported keys found in state dict when trying to load normalization layers")
        )

    def test_lora_parameter_expanded_shapes(self):
        components, _, _ = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_tensor_inputs(torch_device)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.DEBUG)

        with torch.no_grad():
            original_output = pipe.transformer(**inputs)[0]

        out_features, in_features = pipe.transformer.x_embedder.weight.shape
        rank = 4

        dummy_lora_A = torch.nn.Linear(2 * in_features, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        with CaptureLogger(logger) as cap_logger:
            pipe.load_lora_weights(lora_state_dict, "adapter-1")
        inputs["hidden_states"] = torch.cat([inputs["hidden_states"]] * 2, dim=2)
        with torch.no_grad():
            expanded_output = pipe.transformer(**inputs)[0]
        pipe.delete_adapters("adapter-1")
        self.assertTrue(cap_logger.out.startswith("Expanding the nn.Linear input/output features for module"))
        self.assertFalse(np.allclose(original_output, expanded_output, atol=1e-3, rtol=1e-3))

        components, _, _ = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        dummy_lora_A = torch.nn.Linear(1, rank, bias=False)
        dummy_lora_B = torch.nn.Linear(rank, out_features, bias=False)
        lora_state_dict = {
            "transformer.x_embedder.lora_A.weight": dummy_lora_A.weight,
            "transformer.x_embedder.lora_B.weight": dummy_lora_B.weight,
        }
        # We should error out because lora input features is less than original. We only
        # support expanding the module, not shrinking it
        with self.assertRaises(NotImplementedError):
            pipe.load_lora_weights(lora_state_dict, "adapter-1")

    @require_peft_version_greater("0.13.2")
    def test_lora_B_bias(self):
        components, _, denoiser_lora_config = self.get_dummy_components(FlowMatchEulerDiscreteScheduler)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_tensor_inputs(torch_device)

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(logging.INFO)

        with torch.no_grad():
            original_output = pipe.transformer(**inputs)[0]

        denoiser_lora_config.lora_bias = False
        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        with torch.no_grad():
            lora_bias_false_output = pipe.transformer(**inputs)[0]
        pipe.delete_adapters("adapter-1")

        denoiser_lora_config.lora_bias = True
        pipe.transformer.add_adapter(denoiser_lora_config, "adapter-1")
        with torch.no_grad():
            lora_bias_true_output = pipe.transformer(**inputs)[0]

        self.assertFalse(np.allclose(original_output, lora_bias_false_output, atol=1e-3, rtol=1e-3))
        self.assertFalse(np.allclose(original_output, lora_bias_true_output, atol=1e-3, rtol=1e-3))
        self.assertFalse(np.allclose(lora_bias_false_output, lora_bias_true_output, atol=1e-3, rtol=1e-3))

    @unittest.skip("Not supported in Flux.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in Flux.")
    def test_modify_padding_mode(self):
        pass


@slow
@nightly
@require_torch_gpu
@require_peft_backend
@unittest.skip("We cannot run inference on this model with the current CI hardware")
# TODO (DN6, sayakpaul): move these tests to a beefier GPU
class FluxLoRAIntegrationTests(unittest.TestCase):
    """internal note: The integration slices were obtained on audace.

    torch: 2.6.0.dev20241006+cu124 with CUDA 12.5. Need the same setup for the
    assertions to pass.
    """

    num_inference_steps = 10
    seed = 0

    def setUp(self):
        super().setUp()

        gc.collect()
        torch.cuda.empty_cache()

        self.pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    def tearDown(self):
        super().tearDown()

        gc.collect()
        torch.cuda.empty_cache()

    def test_flux_the_last_ben(self):
        self.pipeline.load_lora_weights("TheLastBen/Jon_Snow_Flux_LoRA", weight_name="jon_snow.safetensors")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "jon snow eating pizza with ketchup"

        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=4.0,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images
        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.1855, 0.1855, 0.1836, 0.1855, 0.1836, 0.1875, 0.1777, 0.1758, 0.2246])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_kohya(self):
        self.pipeline.load_lora_weights("Norod78/brain-slug-flux")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "The cat with a brain slug earring"
        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=4.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.6367, 0.6367, 0.6328, 0.6367, 0.6328, 0.6289, 0.6367, 0.6328, 0.6484])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_kohya_with_text_encoder(self):
        self.pipeline.load_lora_weights("cocktailpeanut/optimus", weight_name="optimus.safetensors")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "optimus is cleaning the house with broomstick"
        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=4.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images

        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.4023, 0.4023, 0.4023, 0.3965, 0.3984, 0.3965, 0.3926, 0.3906, 0.4219])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_xlabs(self):
        self.pipeline.load_lora_weights("XLabs-AI/flux-lora-collection", weight_name="disney_lora.safetensors")
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "A blue jay standing on a large basket of rainbow macarons, disney style"

        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=3.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images
        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.3965, 0.4180, 0.4434, 0.4082, 0.4375, 0.4590, 0.4141, 0.4375, 0.4980])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3

    def test_flux_xlabs_load_lora_with_single_blocks(self):
        self.pipeline.load_lora_weights(
            "salinasr/test_xlabs_flux_lora_with_singleblocks", weight_name="lora.safetensors"
        )
        self.pipeline.fuse_lora()
        self.pipeline.unload_lora_weights()
        self.pipeline.enable_model_cpu_offload()

        prompt = "a wizard mouse playing chess"

        out = self.pipeline(
            prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=3.5,
            output_type="np",
            generator=torch.manual_seed(self.seed),
        ).images
        out_slice = out[0, -3:, -3:, -1].flatten()
        expected_slice = np.array(
            [0.04882812, 0.04101562, 0.04882812, 0.03710938, 0.02929688, 0.02734375, 0.0234375, 0.01757812, 0.0390625]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), out_slice)

        assert max_diff < 1e-3
