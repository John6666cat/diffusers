<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Caching methods

Cache methods speedup diffusion transformers by storing and reusing intermediate outputs of specific layers, such as attention and feedforward layers, instead of recalculating them at each inference step.

## CacheMixin

[[autodoc]] CacheMixin

## PyramidAttentionBroadcastConfig

[[autodoc]] PyramidAttentionBroadcastConfig

[[autodoc]] apply_pyramid_attention_broadcast

## FasterCacheConfig

[[autodoc]] FasterCacheConfig

[[autodoc]] apply_faster_cache

## FirstBlockCacheConfig

[[autodoc]] FirstBlockCacheConfig

[[autodoc]] apply_first_block_cache

## TaylorSeerCacheConfig

[[autodoc]] TaylorSeerCacheConfig

[[autodoc]] apply_taylorseer_cache

## SpectrumCacheConfig

[[autodoc]] SpectrumCacheConfig

[[autodoc]] apply_spectrum_cache

### Qualified 2D UNet profiles in this fork

The global `SpectrumCacheConfig()` defaults remain the original FLUX.1 research profile.
For qualified Stable Diffusion UNet routes, use the explicit profile factories instead.

```python
from diffusers import SpectrumCacheConfig

# SDXL ordinary route profile
config = SpectrumCacheConfig.for_sdxl()

# SDXL conservative quality-oriented alternate
conservative = SpectrumCacheConfig.for_sdxl(conservative=True)

# SDXL PAG routes
pag_config = SpectrumCacheConfig.for_sdxl(pag=True)

# Stable Diffusion 1.5 ordinary route profile
sd15_config = SpectrumCacheConfig.for_sd15()
```

The SDXL standard profile is `degree=4`, `ridge_lambda=0.1`, `blend_w=0.60`,
`warmup_steps=6`, `window_size=2.0`, `flex_window=0.75`, and
`tail_actual_steps=3`. The conservative SDXL alternate uses `warmup_steps=5`
and `flex_window=0.25`; PAG uses `warmup_steps=8`.

The SD1.5 ordinary profile is `degree=4`, `ridge_lambda=0.05`, `blend_w=0.55`,
`warmup_steps=6`, `window_size=2.0`, `flex_window=0.75`, and
`tail_actual_steps=3`.

These are measured profiles rather than universal lossless guarantees. SDXL has
been exercised across ordinary generation, 1024px runs, partial trajectories,
ControlNet, T2I-Adapter, IP-Adapter, PAG, compositions, and ControlNet Union. Mixed ControlNet + IP-Adapter remains intentionally fail-closed.

For SD1.5, the ordinary profile has passed full-route, composition, static-LoRA,
resolution/aspect, and representative-route qualification. A fresh scheduler
portability sweep remained structurally valid on Euler, DDIM, DPM++ 2M, UniPC,
and PNDM, but isolated Euler/DPM++/PNDM cases fell slightly below the 20 dB
research PSNR floor. Do not interpret `for_sd15()` as a universal scheduler
quality guarantee. The tested LCM-LoRA 4/6/8-step forecast profile was not
promoted.

### Qualified HunyuanVideo 1.5 profile in this fork

For the measured 480p T2V 50-step route, use the explicit factory:

```python
from diffusers import SpectrumCacheConfig

config = SpectrumCacheConfig.for_hunyuan_video15()
```

The qualified profile forecasts steps `(20, 22, 24, 27, 32, 34, 36, 38, 40, 42)` on each CFG lane
and retains eight real feature snapshots per lane. In the pinned 480p T2V qualification, this
reduced first-block calls from 100 to 80 and measured about 1.23x-1.25x wall-time speedup on two
prompts. The adapter remains fail-closed for I2V, mean-flow, training/autograd, non-empty attention
kwargs / LoRA, unsupported cache-context lanes, and non-qualified profile values.

## MagCacheConfig

[[autodoc]] MagCacheConfig

[[autodoc]] apply_mag_cache

## SeaCacheConfig

[[autodoc]] SeaCacheConfig

[[autodoc]] apply_sea_cache
