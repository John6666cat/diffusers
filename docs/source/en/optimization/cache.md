<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching

Caching accelerates inference by storing and reusing intermediate outputs of different layers, such as attention and feedforward layers, instead of performing the entire computation at each inference step. It significantly improves generation speed at the expense of more memory and doesn't require additional training.

This guide shows you how to use the caching methods supported in Diffusers.

## Pyramid Attention Broadcast

[Pyramid Attention Broadcast (PAB)](https://huggingface.co/papers/2408.12588) is based on the observation that attention outputs aren't that different between successive timesteps of the generation process. The attention differences are smallest in the cross attention layers and are generally cached over a longer timestep range. This is followed by temporal attention and spatial attention layers.

> [!TIP]
> Not all video models have three types of attention (cross, temporal, and spatial)!

PAB can be combined with other techniques like sequence parallelism and classifier-free guidance parallelism (data parallelism) for near real-time video generation.

Set up and pass a [`PyramidAttentionBroadcastConfig`] to a pipeline's transformer to enable it. The `spatial_attention_block_skip_range` controls how often to skip attention calculations in the spatial attention blocks and the `spatial_attention_timestep_skip_range` is the range of timesteps to skip. Take care to choose an appropriate range because a smaller interval can lead to slower inference speeds and a larger interval can result in lower generation quality.

```python
import torch
from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

pipeline = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=torch.bfloat16)
pipeline.to("cuda")  # or "mps", "xpu", "cpu"

config = PyramidAttentionBroadcastConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(100, 800),
    current_timestep_callback=lambda: pipe.current_timestep,
)
pipeline.transformer.enable_cache(config)
```

## FasterCache

[FasterCache](https://huggingface.co/papers/2410.19355) caches and reuses attention features similar to [PAB](#pyramid-attention-broadcast) since output differences are small for each successive timestep.

This method may also choose to skip the unconditional branch prediction, when using classifier-free guidance for sampling (common in most base models), and estimate it from the conditional branch prediction if there is significant redundancy in the predicted latent outputs between successive timesteps.

Set up and pass a [`FasterCacheConfig`] to a pipeline's transformer to enable it.

```python
import torch
from diffusers import CogVideoXPipeline, FasterCacheConfig

pipe line= CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", dtype=torch.bfloat16)
pipeline.to("cuda")  # or "mps", "xpu", "cpu"

config = FasterCacheConfig(
    spatial_attention_block_skip_range=2,
    spatial_attention_timestep_skip_range=(-1, 681),
    current_timestep_callback=lambda: pipe.current_timestep,
    attention_weight_callback=lambda _: 0.3,
    unconditional_batch_skip_range=5,
    unconditional_batch_timestep_skip_range=(-1, 781),
    tensor_format="BFCHW",
)
pipeline.transformer.enable_cache(config)
```

## FirstBlockCache

[FirstBlock Cache](https://huggingface.co/docs/diffusers/main/en/api/cache#diffusers.FirstBlockCacheConfig) checks how much the early layers of the denoiser changes from one timestep to the next. If the change is small, the model skips the expensive later layers and reuses the previous output.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", dtype=torch.bfloat16
)
apply_first_block_cache(pipeline.transformer, FirstBlockCacheConfig(threshold=0.2))
```
## TaylorSeer Cache

[TaylorSeer Cache](https://huggingface.co/papers/2403.06923) accelerates diffusion inference by using Taylor series expansions to approximate and cache intermediate activations across denoising steps. The method predicts future outputs based on past computations, reusing them at specified intervals to reduce redundant calculations.

This caching mechanism delivers strong results with minimal additional memory overhead. For detailed performance analysis, see [our findings here](https://github.com/huggingface/diffusers/pull/12648#issuecomment-3610615080).

To enable TaylorSeer Cache, create a [`TaylorSeerCacheConfig`] and pass it to your pipeline's transformer:

- `cache_interval`: Number of steps to reuse cached outputs before performing a full forward pass
- `disable_cache_before_step`: Initial steps that use full computations to gather data for approximations
- `max_order`: Approximation accuracy (in theory, higher values improve quality but increase memory usage but we recommend it should be set to `1`)

```python
import torch
from diffusers import FluxPipeline, TaylorSeerCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    dtype=torch.bfloat16,
).to("cuda")  # or "mps", "xpu", "cpu"

config = TaylorSeerCacheConfig(
    cache_interval=5,
    max_order=1,
    disable_cache_before_step=10,
    taylor_factors_dtype=torch.bfloat16,
)
pipe.transformer.enable_cache(config)
```

## SPECTRUM

SPECTRUM accelerates diffusion inference by forecasting a late denoiser feature from features recorded at earlier full-compute steps. On a forecast step, the model-specific SPECTRUM adapter skips the qualified expensive denoiser body and executes the remaining native output path.

Set up and pass a [`SpectrumCacheConfig`] to a supported denoiser to enable it.

```python
import torch
from diffusers import FluxPipeline, SpectrumCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    dtype=torch.bfloat16,
).to("cuda")

config = SpectrumCacheConfig(num_inference_steps=50)
pipe.transformer.enable_cache(config)

with pipe.transformer.cache_context("generation"):
    image = pipe(
        "A small cabin beside a frozen lake at sunrise",
        num_inference_steps=50,
    ).images[0]

pipe.transformer.disable_cache()
```

The main scheduling parameters are:

- `warmup_steps`: number of initial full-compute steps used to establish feature history.
- `window_size`: initial spacing between full-compute refreshes.
- `flex_window`: amount added to the adaptive refresh window after each refresh.
- `degree`: degree of the Chebyshev polynomial used by the forecaster.
- `blend_w`: blend between the spectral prediction and local first-order extrapolation.
- `history_limit`: maximum number of full-compute feature snapshots retained per cache context.
- `forecast_step_indices`: optional explicit denoising-step indices to forecast; when supplied, they replace the adaptive refresh schedule.
- `tail_actual_steps`: number of final denoising steps forced to full compute.

> [!WARNING]
> SPECTRUM profiles are route-specific. Match `num_inference_steps` to the number of denoiser forwards observed by SPECTRUM, which may differ from a pipeline's nominal scheduler step count. A profile qualified for one model, scheduler, guidance topology, or model variant should not be assumed to work for another route.

SPECTRUM state is partitioned by [`~CacheMixin.cache_context`]. Use separate context names for denoising trajectories that must not share forecast history.

Supported adapters validate model- and route-specific invariants. When a qualified adapter detects an incompatible runtime state, such as a changed input signature or an unsupported conditioning path, it falls back to the original full-compute forward instead of forecasting from incompatible history. Some adapters latch this fail-closed behavior for the remainder of the cache context.

For UNet routes, special conditioning is disabled by default. The following opt-ins are available for paths that have been separately validated:

```python
config = SpectrumCacheConfig(
    num_inference_steps=30,
    allow_unet_controlnet_residuals=True,
    allow_unet_t2i_adapter_residuals=True,
    allow_unet_ip_adapter_image_embeds=True,
)
```

Model variants can share an integration skeleton without sharing a SPECTRUM schedule. In particular, a profile measured on a distilled or Turbo model should be requalified before it is used on a Base or Raw sibling.

The qualified Krea 2 Raw standard text-to-image route uses 52 denoising steps with classifier-free guidance and an explicit sparse schedule. Its conservative default profile is:

```python
config = SpectrumCacheConfig(
    num_inference_steps=52,
    forecast_step_indices=(7, 21, 30, 35, 42, 44),
    degree=4,
    history_limit=8,
    coordinate_max=100.0,
)
```

Krea 2 Raw keeps positive and negative CFG forecast histories separate and fails closed if the expected dual-lane conditioning pattern changes. Krea 2 Turbo uses a different route-specific schedule.

An optional aggressive Krea 2 Raw profile is available for the exact L4/NF4 standard text-to-image setup used to validate it:

```python
config = SpectrumCacheConfig(
    num_inference_steps=52,
    forecast_step_indices=(7, 10, 21, 28, 30, 32, 35, 37, 42, 44),
    degree=4,
    history_limit=8,
    coordinate_max=100.0,
)
```

> [!WARNING]
> The aggressive profile is experimental and is not the default. It trades additional approximation for speed and was only validated on the current Krea 2 Raw L4/NF4 standard text-to-image representation. Revalidate output quality and fail-closed behavior before using it with a different checkpoint, precision or quantization mode, accelerator or backend, conditioning route, scheduler, or step/guidance settings.



### Wan2.1 T2V 1.3B profiles

The native Wan2.1 adapter is qualified for the text-only `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` route with 50 denoising steps and classifier-free guidance. Conditional and unconditional forecast histories remain separate.

The conservative profile remains the default:

```python
config = SpectrumCacheConfig(
    num_inference_steps=50,
    warmup_steps=10,
    window_size=2.0,
    flex_window=0.0,
    degree=4,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=10,
)
```

A faster opt-in profile was qualified by forecasting 19 explicit middle-trajectory steps while keeping step 10 and the final 10 steps at full compute:

```python
config = SpectrumCacheConfig(
    num_inference_steps=50,
    forecast_step_indices=(11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38),
    degree=4,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=10,
)
```

The opt-in profile was rechecked with two seeds on an L4/NF4 representation over both the native 480x832 / 17-frame visual lane and the 192x320 / 81-frame temporal lane. It is not a portable speed or fidelity guarantee.

> [!WARNING]
> The opt-in Wan profile is qualified only for the current 1.3B text-only T2V route. Revalidate when changing checkpoint or model size, I2V/VACE/image conditioning, scheduler or timestep semantics, guidance topology, precision/quantization, accelerator/backend, attention kwargs, LoRA/adapters, or denoising-step count. Wan 14B and Wan2.2 are not covered by this profile.


### Neta Yume profiles

The native Neta Yume adapter is qualified for the pinned `duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40` standard CFG route with 50 denoising steps. Positive and negative CFG forecast histories remain separate.

The conservative six-forecast profile remains the default:

```python
config = SpectrumCacheConfig(
    num_inference_steps=50,
    forecast_step_indices=(15, 20, 23, 25, 27, 36),
    window_size=3.0,
    flex_window=0.0,
    degree=2,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=8,
    coordinate_max=50.0,
    warmup_steps=0,
    tail_actual_steps=0,
)
```

A faster opt-in profile was qualified by spreading nine forecast steps across the middle trajectory:

```python
config = SpectrumCacheConfig(
    num_inference_steps=50,
    forecast_step_indices=(15, 18, 20, 22, 24, 26, 28, 30, 36),
    window_size=3.0,
    flex_window=0.0,
    degree=2,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=8,
    coordinate_max=50.0,
    warmup_steps=0,
    tail_actual_steps=0,
)
```

The opt-in profile was checked on an NVIDIA L4 in BF16 at the project-qualified creator route (`width=1536`, `height=1024`, 50 steps, CFG 4, `cfg_trunc_ratio=6`, `cfg_normalization=False`) over three prompt/seed cases. It averaged about 1.19x end-to-end speedup versus full compute in that environment, compared with about 1.12x for the conservative profile. These measurements are environment-specific and are not portable speed or fidelity guarantees.

> [!WARNING]
> The Neta Yume profiles are qualified only for the pinned standard route above. Revalidate when changing checkpoint or Lumina variant, image geometry/orientation, scheduler semantics, denoising-step count, CFG settings, precision, accelerator/backend, batch size, attention kwargs, LoRA/adapters, or conditioning topology. The upstream model card's portrait example uses the opposite image orientation from the project-qualified landscape route.

### Anima family profiles

The native Cosmos adapter is qualified for the standard Anima text-to-image route when the runtime callback reports a stable sequential CFG label (`cond` or `uncond`) and condition count. The callback should expose the current logical denoising step and keep conditional and unconditional forecast histories separate.

```python
from diffusers import SpectrumCacheConfig


def anima_runtime_state(pipe):
    guider_state = pipe.guider.get_state()
    return {
        "step": int(guider_state["step"]),
        "num_inference_steps": int(guider_state["num_inference_steps"]),
        "num_conditions": int(guider_state["num_conditions"]),
        "label": "cond" if pipe.guider.is_conditional else "uncond",
        "dynamic_conditioning": False,
    }
```

For the 30-step standard route, the following quality/speed knee was requalified on Anima Base, Anima 2.9B, and Anima Aesthetic v1.1. The measured speedup was about 2.1x in the L4/NF4 qualification environment; treat that number as environment-specific rather than a portable guarantee.

```python
config = SpectrumCacheConfig(
    num_inference_steps=30,
    warmup_steps=5,
    window_size=2.0,
    flex_window=0.75,
    degree=4,
    ridge_lambda=0.1,
    blend_w=0.25,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=3,
    cosmos_runtime_state_callback=lambda: anima_runtime_state(pipe),
)
```

A more aggressive 30-step profile was also qualified for those routes, with roughly 2.4x measured speedup in the same environment. It trades more approximation for speed and should remain opt-in:

```python
config = SpectrumCacheConfig(
    num_inference_steps=30,
    warmup_steps=5,
    window_size=2.0,
    flex_window=1.0,
    degree=4,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=2,
    cosmos_runtime_state_callback=lambda: anima_runtime_state(pipe),
)
```

Anima Turbo v1.1 is a separate distilled route. Do not reuse the 30-step profiles. The qualified point uses CFG 1, 12 denoising steps, and forecasts only steps 8 and 10:

```python
config = SpectrumCacheConfig(
    num_inference_steps=12,
    forecast_step_indices=(8, 10),
    degree=1,
    ridge_lambda=0.1,
    blend_w=0.5,
    history_limit=100,
    coordinate_max=50.0,
    tail_actual_steps=1,
    cosmos_runtime_state_callback=lambda: anima_runtime_state(pipe),
)
```

> [!WARNING]
> Anima 3.8B v1.1 is **not** covered by the standard Cosmos adapter above. Its bundled Semantic Connector v2 is timestep-aware and must run for both CFG branches on every denoising step. Current research only qualifies forecasting the post-connector 52-block DiT body with branch-asymmetric schedules. Do not use the standard root Cosmos SPECTRUM hook for that receiver until Diffusers owns a native Semantic Connector v2 runtime and exposes the post-connector body boundary without bypassing the connector.

> [!NOTE]
> Model-family measurements are route-specific. Revalidate when changing checkpoint, scheduler, guidance topology, precision/quantization mode, accelerator/backend, or denoising-step count.

## MagCache

[MagCache](https://github.com/Zehong-Ma/MagCache) accelerates inference by skipping transformer blocks based on the magnitude of the residual update. It observes that the magnitude of updates (Output - Input) decays predictably over the diffusion process. By accumulating an "error budget" based on pre-computed magnitude ratios, it dynamically decides when to skip computation and reuse the previous residual.

MagCache relies on **Magnitude Ratios** (`mag_ratios`), which describe this decay curve. These ratios are specific to the model checkpoint and scheduler.

To use MagCache, you typically follow a two-step process: **Calibration** and **Inference**.

1.  **Calibration**: Run inference once with `calibrate=True`. The hook will measure the residual magnitudes and print the calculated ratios to the console.
2.  **Inference**: Pass these ratios to `MagCacheConfig` to enable acceleration.

```python
import torch
from diffusers import FluxPipeline, MagCacheConfig

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    dtype=torch.bfloat16
).to("cuda")  # or "mps", "xpu", "cpu"

# 1. Calibration Step
# Run full inference to measure model behavior.
calib_config = MagCacheConfig(calibrate=True, num_inference_steps=4)
pipe.transformer.enable_cache(calib_config)

# Run a prompt to trigger calibration
pipe("A cat playing chess", num_inference_steps=4)
# Logs will print something like: "MagCache Calibration Results: [1.0, 1.37, 0.97, 0.87]"

# 2. Inference Step
# Apply the specific ratios obtained from calibration for optimized speed.
# Note: For Flux models, you can also import defaults: 
# from diffusers.hooks.mag_cache import FLUX_MAG_RATIOS
mag_config = MagCacheConfig(
    mag_ratios=[1.0, 1.37, 0.97, 0.87],
    num_inference_steps=4
)

pipe.transformer.enable_cache(mag_config) 

image = pipe("A cat playing chess", num_inference_steps=4).images[0]
```

> [!NOTE]
> `mag_ratios` represent the model's intrinsic magnitude decay curve. Ratios calibrated for a high number of steps (e.g., 50) can be reused for lower step counts (e.g., 20). The implementation uses interpolation to map the curve to the current number of inference steps.

> [!TIP]
> For pipelines that run Classifier-Free Guidance sequentially (like Kandinsky 5.0), the calibration log might print two arrays: one for the Conditional pass and one for the Unconditional pass. In most cases, you should use the first array (Conditional).

> [!TIP]
> For pipelines that run Classifier-Free Guidance in a **batched** manner (like SDXL or Flux), the `hidden_states` processed by the model contain both conditional and unconditional branches concatenated together. The calibration process automatically accounts for this, producing a single array of ratios that represents the joint behavior. You can use this resulting array directly without modification.
