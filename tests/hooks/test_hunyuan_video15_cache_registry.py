import inspect

from diffusers.hooks._helpers import TransformerBlockRegistry
from diffusers.models.transformers.transformer_hunyuan_video15 import HunyuanVideo15TransformerBlock


def test_hunyuan_video15_block_registry_metadata():
    metadata = TransformerBlockRegistry.get(HunyuanVideo15TransformerBlock)

    assert metadata.return_hidden_states_index == 0
    assert metadata.return_encoder_hidden_states_index == 1
    assert metadata.hidden_states_argument_name == "hidden_states"

    parameters = inspect.signature(HunyuanVideo15TransformerBlock.forward).parameters
    assert "hidden_states" in parameters
    assert "encoder_hidden_states" in parameters
