# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
import inspect
from functools import partial
from typing import Dict, List, Optional, Union

import torch.nn as nn

from ..utils import (
    MIN_PEFT_VERSION,
    USE_PEFT_BACKEND,
    check_peft_version,
    convert_unet_state_dict_to_peft,
    delete_adapter_layers,
    get_adapter_name,
    get_peft_kwargs,
    is_accelerate_available,
    is_peft_available,
    is_peft_version,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from .lora_base import _fetch_state_dict
from .unet_loader_utils import _maybe_expand_lora_scales


if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)

_SET_ADAPTER_SCALE_FN_MAPPING = {
    "UNet2DConditionModel": _maybe_expand_lora_scales,
    "UNetMotionModel": _maybe_expand_lora_scales,
    "SD3Transformer2DModel": lambda model_cls, weights: weights,
    "FluxTransformer2DModel": lambda model_cls, weights: weights,
    "CogVideoXTransformer3DModel": lambda model_cls, weights: weights,
}


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them in a base model, check out the PEFT
    [documentation](https://huggingface.co/docs/peft/index).

    Install the latest version of PEFT, and use this mixin to:

    - Attach new adapters in the model.
    - Attach multiple adapters and iteratively activate/deactivate them.
    - Activate/deactivate all adapters from the model.
    - Get a list of the active adapters.
    """

    _hf_peft_config_loaded = False

    @classmethod
    # Copied from diffusers.loaders.lora_base.LoraBaseMixin._optionally_disable_offloading
    def _optionally_disable_offloading(cls, _pipeline):
        """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.

        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        if _pipeline is not None and _pipeline.hf_device_map is None:
            for _, component in _pipeline.components.items():
                if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                    if not is_model_cpu_offload:
                        is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                    if not is_sequential_cpu_offload:
                        is_sequential_cpu_offload = (
                            isinstance(component._hf_hook, AlignDevicesHook)
                            or hasattr(component._hf_hook, "hooks")
                            and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
                        )

                    logger.info(
                        "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                    )
                    remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

        return (is_model_cpu_offload, is_sequential_cpu_offload)

    def load_lora_adapter(self, pretrained_model_name_or_path_or_dict, prefix="transformer", **kwargs):
        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        adapter_name = kwargs.pop("adapter_name", None)
        _pipeline = kwargs.pop("_pipeline", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        allow_pickle = False

        if low_cpu_mem_usage and is_peft_version("<=", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        state_dict = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )

        keys = list(state_dict.keys())
        transformer_keys = [k for k in keys if k.startswith(prefix)]
        if len(transformer_keys) > 0:
            state_dict = {k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in transformer_keys}

        if len(state_dict.keys()) > 0:
            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                state_dict = convert_unet_state_dict_to_peft(state_dict)

            if adapter_name in getattr(self, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
                )

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=state_dict)
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                    raise ValueError(
                        "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                    )
                else:
                    lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(self)

            # <Unsafe code
            # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
            # Now we remove any existing hooks to `_pipeline`.

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error
            is_model_cpu_offload, is_sequential_cpu_offload = self._optionally_disable_offloading(_pipeline)
            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            inject_adapter_in_model(lora_config, self, adapter_name=adapter_name, **peft_kwargs)
            incompatible_keys = set_peft_model_state_dict(self, state_dict, adapter_name, **peft_kwargs)

            warn_msg = ""
            if incompatible_keys is not None:
                # Check only for unexpected keys.
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    lora_unexpected_keys = [k for k in unexpected_keys if "lora_" in k and adapter_name in k]
                    if lora_unexpected_keys:
                        warn_msg = (
                            f"Loading adapter weights from state_dict led to unexpected keys found in the model:"
                            f" {', '.join(lora_unexpected_keys)}. "
                        )

                # Filter missing keys specific to the current adapter.
                missing_keys = getattr(incompatible_keys, "missing_keys", None)
                if missing_keys:
                    lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
                    if lora_missing_keys:
                        warn_msg += (
                            f"Loading adapter weights from state_dict led to missing keys in the model:"
                            f" {', '.join(lora_missing_keys)}."
                        )

            if warn_msg:
                logger.warning(warn_msg)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]
        weights = scale_expansion_fn(self, weights)

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        r"""
        Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
        to the adapter to follow the convention of the PEFT library.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_config (`[~peft.PeftConfig]`):
                The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
                methods.
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        from peft import PeftConfig, inject_adapter_in_model

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # Unlike transformers, here we don't need to retrieve the name_or_path of the unet as the loading logic is
        # handled by the `load_lora_layers` or `StableDiffusionLoraLoaderMixin`. Therefore we set it to `None` here.
        adapter_config.base_model_name_or_path = None
        inject_adapter_in_model(adapter_config, self, adapter_name)
        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """
        Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_name (Union[str, List[str]])):
                The list of adapters to set or the adapter name in the case of a single adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        missing = set(adapter_name) - set(self.peft_config)
        if len(missing) > 0:
            raise ValueError(
                f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                f" current loaded adapters are: {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                # Previous versions of PEFT does not support multi-adapter inference
                elif not hasattr(module, "set_adapter") and len(adapter_name) != 1:
                    raise ValueError(
                        "You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT."
                        " `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`"
                    )
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        Disable all adapters attached to the model and fallback to inference with the base model only.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model uses `self.active_adapters()` to retrieve the list of
        adapters to enable.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = False

    def active_adapters(self) -> List[str]:
        """
        Gets the current list of active adapters of the model.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        [documentation](https://huggingface.co/docs/peft).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                return module.active_adapter

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `fuse_lora()`.")

        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        from peft.tuners.tuners_utils import BaseTunerLayer

        merge_kwargs = {"safe_merge": self._safe_fusing}

        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)

            # For BC with prevous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)

    def unfuse_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `unfuse_lora()`.")
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(module, BaseTunerLayer):
            module.unmerge()

    def unload_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `unload_lora()`.")

        from ..utils import recurse_remove_peft_layers

        recurse_remove_peft_layers(self)
        if hasattr(self, "peft_config"):
            del self.peft_config

    def disable_lora(self):
        """
        Disables the active LoRA layers of the underlying model.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enables the active LoRA layers of the underlying model.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the underlying model.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)

            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)
