import types

import pytest
import torch

from diffusers import ChromaImg2ImgPipeline, ChromaInpaintPipeline, ChromaPipeline


class _DummyTokenizer:
    def __call__(self, prompt, **kwargs):
        batch = len(prompt)
        input_ids = torch.tensor([[10, 11, 0, 0]], dtype=torch.long).repeat(batch, 1)
        attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long).repeat(batch, 1)
        return types.SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)


class _DummyTextEncoder:
    dtype = torch.float32

    def __call__(self, input_ids, output_hidden_states=False, attention_mask=None):
        assert attention_mask.dtype in (torch.int64, torch.long)
        batch, seq = input_ids.shape
        return (torch.zeros(batch, seq, 8, dtype=self.dtype, device=input_ids.device),)


class _PromptHarness:
    tokenizer = _DummyTokenizer()
    text_encoder = _DummyTextEncoder()


@pytest.mark.parametrize(
    "pipeline_cls",
    [ChromaPipeline, ChromaImg2ImgPipeline, ChromaInpaintPipeline],
)
def test_chroma_t5_prompt_mask_is_boolean_and_keeps_one_padding_token(pipeline_cls):
    harness = _PromptHarness()
    _, mask = pipeline_cls._get_t5_prompt_embeds(
        harness,
        prompt=["short"],
        num_images_per_prompt=1,
        max_sequence_length=4,
        device=torch.device("cpu"),
    )

    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, True, False]]


@pytest.mark.parametrize(
    "pipeline_cls",
    [ChromaPipeline, ChromaImg2ImgPipeline, ChromaInpaintPipeline],
)
def test_chroma_joint_attention_mask_stays_boolean(pipeline_cls):
    prompt_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    mask = pipeline_cls._prepare_attention_mask(
        None,
        batch_size=1,
        sequence_length=2,
        dtype=torch.float32,
        attention_mask=prompt_mask,
    )

    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, False, True, True]]
